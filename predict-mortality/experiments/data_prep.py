import pandas as pd
import numpy as np
from copy import deepcopy
from utilsforminds.containers import merge_dictionaries
import random
import os
from collections import OrderedDict
from tqdm import tqdm
import utilsforminds
import group_finder
import skimage
import cv2
from lungmask import mask
import SimpleITK as sitk
import nibabel as nib

pd.set_option('display.max_columns', 100)


def reorder_snp(basic_data, snp_data):
    """Reorder the SNP Data to be correctly associated with the other .xlsx data frames"""
    tmp = snp_data['ID'].apply(lambda x: x[-4:])
    tmp.rename('sortID', inplace=True)

    snp_data = pd.concat([tmp, snp_data], axis=1)
    snp_data.sort_values(by='sortID', inplace=True)
    snp_data.drop(columns=['sortID'], inplace=True)

    all_patients = basic_data["SubjID"]
    snp_final = pd.merge(all_patients.to_frame(), snp_data,
                         left_on='SubjID', right_on='ID', how='outer')

    # Remove 018_S_0055 (because it doesn't exist in longitudinal data
    snp_final = snp_final[snp_final.ID != '018_S_0055']

    # Remove unwanted ID labels
    snp_final.drop(columns=['SubjID', 'ID'], inplace=True)
    return snp_final


def rand_gen_with_range(range_=None):
    if range_ is None:
        range_ = [0., 1.]
    assert(range_[0] <= range_[1])
    return range_[0] + (range_[1] - range_[0]) * random.random()


class Dataset():
    def __init__(self, path_to_dataset, dataset_kind="covid", init_number_for_unobserved_entry=-1.0, static_data_input_mapping=None, kwargs_dataset_specific=None):
        """Preprocess the input raw tabular dataset.

        The most important attributes are self.dicts and self.input_to_features_map.

        Attributes
        ----------
        dicts : list of dict
            List of dictionaries of patients records where each dictionary for each patient.
        static_data_input_mapping : dict
            Static feature -> input, let the user to decide which static feature feed which input. For example, static_data_input_mapping = dict(RNN_vectors = [], static_encoder = ["SNP"], raw = ["BL_age", "Gender"]).
        prediction_labels_bag : list
            The set of possible target labels, for example, for COVID-19 dataset: [0, 1], for Alz dataset: [1, 2, 3].
        input_to_features_map : dict
            The feature names of each input. For example, looks like a dict(RNN_vectors= [], static_encoder= [], raw= [], dynamic_vectors_decoder= []). The stack order of numpy array data follows the order of self.input_to_features_map, except the static feature precedes the dynamic features in RNN input.
        groups_of_features_info : 
            Used to calculate the feature importance of each plotting group. For example, self.groups_of_features_info[feature_group = "FS"][feature_name] = {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": [], "mean": None, "std": None}.
        """

        self.this_dataset_is_splitted = False  # Not yet k-fold splitted.
        self.shape_2D_records = None  # What is the dimension of image input if it has?
        self.init_number_for_unobserved_entry = init_number_for_unobserved_entry
        self.ID = utilsforminds.helpers.get_current_utc_timestamp()
        self.dataset_kind = dataset_kind
        self.path_to_dataset = path_to_dataset
        self.input_to_features_map = dict(RNN_vectors=[], static_encoder=[], raw=[
        ], dynamic_vectors_decoder=[])  # Feature names for each input.

        # Set class names for ROC-curve plot, sequence is decided by one-hot encoded form, such that  self.class_names = [name of [1, 0, 0], name of [0, 1, 0], name of [0, 0, 1]]
        if self.dataset_kind == "challenge":
            self.class_names = ["survival", "death"]
        elif self.dataset_kind == "alz":
            self.class_names = ["AD", "MCI", "HC"]
        else:
            self.class_names = None

        if dataset_kind == "covid":
            # Set default option.
            self.static_data_input_mapping = merge_dictionaries([dict(RNN_vectors=[], static_encoder=[], raw=[
                                                                "age", "gender", 'Admission time', 'Discharge time']), static_data_input_mapping])
            # All the information regardless of whether it is used by models or not.
            self.static_column_names = [
                'PATIENT_ID', "age", "gender", 'Admission time', 'Discharge time', "outcome"]
            # Information actually used by the models. \in self.static_column_names.
            self.input_to_features_map = merge_dictionaries(
                [self.input_to_features_map, self.static_data_input_mapping])
            self.dynamic_feature_names = []

            # Set dataframe.
            self.path_to_dataset = path_to_dataset
            self.dataframe_raw = pd.read_excel(path_to_dataset)
            self.dataframe = self.dataframe_raw.copy()

            # Set dynamic features names.
            # Should not include target label, such as 'outcome'.
            for name in self.dataframe_raw.columns:
                if name not in self.static_column_names + kwargs_dataset_specific["excluded_features"]:
                    self.dynamic_feature_names.append(name)
                    self.input_to_features_map["RNN_vectors"].append(name)

            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(
                self.input_to_features_map["RNN_vectors"])
            # decoder labels = LSTM labels - RE_DATE.
            self.input_to_features_map["dynamic_vectors_decoder"].remove(
                "RE_DATE")
            self.dynamic_feature_names.remove("RE_DATE")
            # for name in self.decoder_labels_feature_names:
            #     if name in self.static_column_names:
            #         self.decoder_labels_feature_names_is_static.append(1.)
            #     else:
            #         self.decoder_labels_feature_names_is_static.append(0.)
            # self.decoder_labels_feature_names_is_static = np.array(self.decoder_labels_feature_names_is_static)

            self.scale_times()
            self.min_max_scale()
            self.set_observed_numbers_pool_for_each_group_of_features()

        elif dataset_kind == "chestxray":
            self.shape_2D_records = kwargs_dataset_specific["image_shape"]
            self.apply_segment = kwargs_dataset_specific["apply_segment"]
            # Set dataframe.
            self.path_to_dataset = path_to_dataset
            self.dataframe_raw = pd.read_csv(
                os.path.join(path_to_dataset, "metadata.csv"))
            self.dataframe = self.dataframe_raw.copy()
            self.dataframe = self.dataframe.drop(
                columns=kwargs_dataset_specific["excluded_features"])
            categorical_columns = [name for name in ["sex", "finding", "RT_PCR_positive", "survival", "intubated", "intubation_present", "went_icu",
                                                     "in_icu", "needed_supplemental_O2", "extubated", "view", "modality"] if name not in kwargs_dataset_specific["excluded_features"]]

            # One-hot encoding
            self.dataframe = pd.get_dummies(
                self.dataframe, columns=categorical_columns)
            df_columns = self.dataframe.columns.to_list()

            # Set default option.
            self.static_data_input_mapping = merge_dictionaries([dict(RNN_vectors=[], static_encoder=[], raw=[
                                                                "sex", "age", "RT_PCR_positive"]), static_data_input_mapping])
            self.static_column_names = ['patientid', "sex", "age", "RT_PCR_positive", "finding", "survival", "intubated", "went_icu",
                                        "needed_supplemental_O2", "extubated", "filename"]  # All the information regardless of whether it is used by models or not.
            # Information actually used by the models. \in self.static_column_names.
            self.input_to_features_map = merge_dictionaries(
                [self.input_to_features_map, self.static_data_input_mapping])
            self.dynamic_feature_names = []

            # One-hot encoded colmun names.
            for dict_ in [self.static_data_input_mapping, self.input_to_features_map]:
                for key in dict_.keys():
                    dict_[key] = change_names_from_starting_names(
                        dict_[key], df_columns)
            self.static_column_names = change_names_from_starting_names(
                self.static_column_names, df_columns)

            # Set dynamic features names.
            for name in df_columns:  # Should not include target label, such as 'outcome'.
                if name not in self.static_column_names:
                    self.dynamic_feature_names.append(name)
                    self.input_to_features_map["RNN_vectors"].append(name)

            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(
                self.input_to_features_map["RNN_vectors"])
            # decoder labels = LSTM labels - RE_DATE.
            self.input_to_features_map["dynamic_vectors_decoder"].remove(
                "offset")
            self.dynamic_feature_names.remove("offset")

            # One-hot encode for categorical columns
            self.min_max_scale()
            self.set_observed_numbers_pool_for_each_group_of_features()

        elif dataset_kind == "challenge":
            # Set default option.
            self.static_data_input_mapping = merge_dictionaries([dict(RNN_vectors=[], static_encoder=[], raw=[
                                                                "Age", "Gender", 'Height', "Weight", "CCU", "CSRU", "SICU"]), static_data_input_mapping])
            # All the information regardless of whether it is used by models or not.
            self.static_column_names = ["SAPS-I", "SOFA", "Length_of_stay", "Survival",
                                        "In-hospital_death", "Age", "Gender", 'Height', "Weight", "CCU", "CSRU", "SICU", ]
            # Information actually used by the models. \in self.static_column_names.
            self.input_to_features_map = merge_dictionaries(
                [self.input_to_features_map, self.static_data_input_mapping])
            self.dynamic_feature_names = []

            # Set dataframe.
            self.path_to_dataset = path_to_dataset
            self.dataframe_static = pd.read_csv(os.path.join(
                path_to_dataset, "PhysionetChallenge2012-static-set-a.csv"))
            self.dataframe_dynamic = pd.read_csv(os.path.join(
                path_to_dataset, "PhysionetChallenge2012-temporal-set-a.csv"))

            # Set dynamic features names.
            # Should not include target label, such as 'outcome'.
            for name in self.dataframe_dynamic.columns:
                if name != "recordid":
                    self.dynamic_feature_names.append(name)
                    self.input_to_features_map["RNN_vectors"].append(name)

            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(
                self.input_to_features_map["RNN_vectors"])
            # decoder labels = LSTM labels - RE_DATE.
            self.input_to_features_map["dynamic_vectors_decoder"].remove(
                "time")
            self.dynamic_feature_names.remove("time")

            # self.scale_times()
            self.set_observed_numbers_pool_for_each_group_of_features(
                excluded_features=["time"])
            # self.dataframe_static = self.dataframe_static.fillna(init_number_for_unobserved_entry)
            # self.dataframe_dynamic = self.dataframe_dynamic.fillna(init_number_for_unobserved_entry)
            self.min_max_scale()

        elif dataset_kind == "toy":
            # raise Exception("Deprecated.")
            # Just to prevent exception, actually do nothing.
            self.set_observed_numbers_pool_for_each_group_of_features()
            pass

        elif dataset_kind == "alz":
            # Set default keyword arguments.
            kwargs_dataset_specific = merge_dictionaries([dict(TIMES=["BL", "M6", "M12", "M18", "M24", "M36"], target_label="CurrentDX_032911", dynamic_modalities_to_use=[
                                                         "FS", "VBM"]), kwargs_dataset_specific])  # You can add cognitive tests to dynamic_modalities_to_use.
            # dict(RNN_vectors = [], static_encoder = ["SNP"], raw = ["BL_Age", "Gender"]) ## You can add age/gender.
            self.static_data_input_mapping = merge_dictionaries(
                [dict(RNN_vectors=[], static_encoder=["SNP"], raw=[]), static_data_input_mapping])

            # Features extracted from the participant's basic info.
            info_feature_names = []
            for list_ in self.static_data_input_mapping.values():
                for feature in list_:
                    if feature != "SNP":
                        info_feature_names.append(feature)

            # --- Set SNPs info dataframe for SNPs plots.
            # Get the indices of SNPs group.
            SNP_identified_group_df = pd.read_excel(
                path_to_dataset + 'CanSNPs_Top40Genes_org.xlsx', 'CanSNPs_Top40Genes_LD', usecols=['L1', 'L2', 'r^2'])
            SNP_label_dict = utilsforminds.biomarkers.get_SNP_name_sequence_dict(
                path_to_dataset + 'snp_labels.csv')
            adjacency_matrix = utilsforminds.biomarkers.get_adjacency_matrix_from_pairwise_value(
                SNP_label_dict, SNP_identified_group_df, ['L1', 'L2'], 'r^2', 0.2)
            group_finder_inst = group_finder.Group_finder(adjacency_matrix)
            # snp_group_idc_lst = [[141, 423, 58, ...], [indices of second group], ...]
            snp_group_idc_lst = group_finder_inst.connected_components()

            # %% Reorder SNP data following groups
            snps_popular_group_df = pd.read_excel(
                path_to_dataset + 'CanSNPs_Top40Genes_org.xlsx', sheet_name="CanSNPs_Top40Genes_Annotation")
            reordered_SNPs_info_list = []
            col_names = ["index_original", "identified_group",
                         "SNP", "chr", "AlzGene", "location"]
            # for indices(in original SNPs array) of each SNPs group
            for idc_lst, group_idx in zip(snp_group_idc_lst, range(len(snp_group_idc_lst))):
                for idx in idc_lst:
                    reordered_SNPs_info_list.append(
                        [idx, group_idx] + list(snps_popular_group_df.loc[idx, ["SNP", "chr", "AlzGene", "location"]]))
            # For SNP group-wise plots.
            self.reordered_SNPs_info_df = pd.DataFrame(
                reordered_SNPs_info_list, columns=col_names)
            # self.static_feature_names = self.reordered_SNPs_info_df['SNP'].tolist()
            # True if 'chr' column is not 'X'.
            series_obj = self.reordered_SNPs_info_df.apply(
                lambda x: True if x['chr'] != 'X' else False, axis=1)
            # list of indices where 'chr' column is not 'X'.
            self.idc_chr_not_X_list = list(
                series_obj[series_obj == True].index)
            self.reordered_SNPs_info_df = self.reordered_SNPs_info_df.loc[
                self.idc_chr_not_X_list, :]
            self.colors_list = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue",
                                "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"]
            for group_column in ['chr', 'AlzGene', 'location', 'identified_group']:
                self.reordered_SNPs_info_df = utilsforminds.helpers.add_column_conditional(
                    self.reordered_SNPs_info_df, group_column, self.colors_list, new_column_name=group_column + "_colors")  # Adds color column for each group.

            # Actual dataframe preprocess.
            # Set time stamp.
            self.TIMES = deepcopy(kwargs_dataset_specific["TIMES"])
            # self.TIMES_TO_RE_DATE = {self.TIMES[i]: (i + 1) / len(self.TIMES) for i in range(len(self.TIMES))} ## Normalize the time stamp.
            # Normalize the time stamp.
            self.TIMES_TO_RE_DATE = dict(
                BL=0.0, M6=0.1, M12=0.2, M18=0.3, M24=0.4, M36=0.6,)
            # ["FS", "VBM", "ADAS", "FLU", "MMSE", "RAVLT", "TRAILS"]
            self.dynamic_modalities_sequence = deepcopy(
                kwargs_dataset_specific["dynamic_modalities_to_use"])

            # --- --- Set dataframes of dataset.
            self.df_dict = {}
            # Set dynamic dataframes.
            file_and_sheet_name_base_dict = dict(FS=dict(file="longitudinal imaging measures_FS_final.xlsx", sheet_name="FS_"),
                                                 VBM=dict(
                                                     file="longitudinal imaging measures_VBM_mod_final.xlsx", sheet_name="VBM_mod_"),
                                                 RAVLT=dict(
                                                     file="longitudinal cognitive scores_RAVLT.xlsx", sheet_name="RAVLT_"),
                                                 ADAS=dict(
                                                     file="longitudinal cognitive scores_ADAS.xlsx", sheet_name="ADAS_"),
                                                 FLU=dict(
                                                     sheet_name="FLU_", file="longitudinal cognitive scores_FLU.xlsx"),
                                                 MMSE=dict(sheet_name="MMSE_",
                                                           file="longitudinal cognitive scores_MMSE.xlsx"),
                                                 TRAILS=dict(sheet_name="TRAILS_", file="longitudinal cognitive scores_TRAILS.xlsx"),)
            df_dict_of_dynamic_temp = {time: {modality: pd.read_excel(
                path_to_dataset + file_and_sheet_name_base_dict[modality]["file"], sheet_name=file_and_sheet_name_base_dict[modality]["sheet_name"] + time, header=0) for modality in self.dynamic_modalities_sequence} for time in self.TIMES}

            # Set static dataframes.
            # Reorder the order of SNPs features, [["ID"] + self.static_feature_names]
            self.df_dict["snp"] = pd.read_excel(
                path_to_dataset + "CanSNPs_Top40Genes_org.xlsx", sheet_name="CanSNPs_Top40Genes_org", header=0)
            self.df_dict["info"] = pd.read_excel(
                path_to_dataset + "longitudinal basic infos.xlsx", sheet_name="Sheet1", header=0)
            # Reorder the order of SNPs features, [["ID"] + self.static_feature_names], the indices of SNPs become consistent with the patient's sequence of other biomarkers.
            self.df_dict["snp"] = reorder_snp(
                self.df_dict["info"], self.df_dict["snp"])
            self.df_dict["static"] = self.df_dict["info"][info_feature_names]
            self.df_dict["outcome"] = pd.read_excel(
                path_to_dataset + "longitudinal diagnosis.xlsx", sheet_name="Sheet1", header=0)[kwargs_dataset_specific["target_label"]]

            # Set static feature names, for RNN, static precedes the dynamic.
            # To handle grouped SNP features and individual info static feature.
            self.static_modality_to_features_map = {
                "SNP": list(self.df_dict["snp"].columns)[1:]}
            for input_ in self.static_data_input_mapping.keys():  # input for static.
                for modality in self.static_data_input_mapping[input_]:
                    if modality == "SNP":
                        self.input_to_features_map[input_] = deepcopy(
                            self.input_to_features_map[input_] + list(self.df_dict["snp"].columns)[1:])  # Get rid of the ID label
                    else:
                        self.input_to_features_map[input_].append(modality)
                        # Non-group, just single modality.
                        self.static_modality_to_features_map[modality] = [
                            modality]

            # Set feature names of dynamic modality.

            self.dynamic_feature_names_for_each_modality = {
                modality: [] for modality in self.dynamic_modalities_sequence}
            # --- Set 'feature name' of FS and VBM using baseline timepoint.
            # FS, VBM, or other cognitive scores.
            for modality in self.dynamic_modalities_sequence:
                # Use column names of BL timepoint.
                for column_name in df_dict_of_dynamic_temp["BL"][modality].columns:
                    # Set feature name.
                    # Rename dynamic column names.
                    def converter_column_name(column_name):
                        column_name_splitted = column_name.split('_')
                        if len(column_name_splitted) >= 3:
                            # Use last TWO blocks for dynamic feature name.
                            return f"{modality}_{column_name_splitted[-2]}_{column_name_splitted[-1]}"
                        else:
                            # [1:] to remove time label.
                            return f"{modality}_" + "_".join(column_name_splitted[1:])
                    converted_column_name = converter_column_name(column_name)

                    # [(static features), dynamic data, RE_DATE] or [(static features), dynamic data, RE_DATE, (static masks), dynamic mask, RE_DATE]
                    self.input_to_features_map["RNN_vectors"].append(
                        converted_column_name)
                    self.dynamic_feature_names_for_each_modality[modality].append(
                        converted_column_name)
                # Simplify the column names of dataframe of dynamic.
                for time in self.TIMES:
                    df_dict_of_dynamic_temp[time][modality] = df_dict_of_dynamic_temp[time][modality].rename(
                        mapper=converter_column_name, axis=1)  # Be careful for the scope of modality, this is hard to debug if bug.
            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(
                self.input_to_features_map["RNN_vectors"])
            # Decoder label = RNN label - "RE_DATE".
            self.input_to_features_map["RNN_vectors"].append("RE_DATE")

            # --- Set the dummy nan rows for consistent number of rows = number of patients.
            self.num_participants_in_dirty_dataset = len(self.df_dict["snp"])
            # --- For dynamic.
            # Makes concatenated dataframe with consistent length for all dynamic modalities.
            self.df_dict["dynamic"] = {}
            for time in self.TIMES:
                concatenated_dynamic_dfs = []
                for modality in self.dynamic_modalities_sequence:
                    df = df_dict_of_dynamic_temp[time][modality]
                    df_dict_of_dynamic_temp[time][modality] = pd.concat([df, pd.DataFrame([[np.nan for i in range(df.shape[1])] for j in range(
                        self.num_participants_in_dirty_dataset - len(df))], columns=df.columns)], ignore_index=True)  # add i column and j row of nans.
                    concatenated_dynamic_dfs.append(
                        df_dict_of_dynamic_temp[time][modality])
                # Combine all dynamic modalities, the sequence follows self.dynamic_modalities_sequence.
                self.df_dict["dynamic"][time] = pd.concat(
                    concatenated_dynamic_dfs, axis=1)
            # --- For static, because actual static data is sliced with column names, the order of columns in static dataframe does not affect the order of actual input of model.
            self.df_dict["static"] = pd.concat([self.df_dict["static"], pd.DataFrame([[np.nan for i in range(self.df_dict["static"].shape[1])] for j in range(
                self.num_participants_in_dirty_dataset - len(self.df_dict["static"]))], columns=self.df_dict["static"].columns)])
            # static feature sequence : [SNP, other features], combine all static modalities.
            self.df_dict["static"] = pd.concat(
                [self.df_dict["snp"], self.df_dict["static"]], axis=1)

            # Get patient's valid (intersection of two list of indices) indices from SNP/static features and Target Label.
            # Example code : df_dict_of_dynamic_temp["M12"]["FS"][~df_dict_of_dynamic_temp["M12"]["FS"].isna().any(axis = 1)].index.tolist()
            self.valid_indices = {}
            self.valid_indices["static"] = self.df_dict["static"][~self.df_dict["static"].isna(
            ).any(axis=1)].index.tolist()  # ~ means 'not'.
            self.valid_indices["outcome"] = self.df_dict["outcome"][~self.df_dict["outcome"].isna(
            )].index.tolist()
            self.valid_indices["intersect"] = []
            for static_valid_index in self.valid_indices["static"]:
                if static_valid_index in self.valid_indices["outcome"]:
                    # static_valid_index is included in snp and outcome both.
                    self.valid_indices["intersect"].append(static_valid_index)
            self.min_max_scale()
            self.set_observed_numbers_pool_for_each_group_of_features()
        else:
            raise Exception(NotImplementedError)

    def scale_times(self):
        """Convert datatime into seconds of floating numbers. The absolute RE_DATE time is changed to the relative time elapsed from the admission time"""

        for column_name in ["RE_DATE", "Admission time", 'Discharge time']:
            self.dataframe[column_name] = self.dataframe[column_name].apply(
                lambda x: x.timestamp() if not pd.isnull(x) else None)
        # Use RE_DATE as relative time elapsed after admission.
        self.dataframe["RE_DATE"] = self.dataframe["RE_DATE"] - \
            self.dataframe["Admission time"]

    def min_max_scale(self):
        """Scale the data of dataframes range of each feature into [0, 1]"""

        self.feature_min_max_dict = {}
        if self.dataset_kind == "covid":
            for column in self.dataframe.columns:
                if column not in ['PATIENT_ID', "gender", "outcome"]:
                    min_ = self.dataframe[column].min()
                    max_ = self.dataframe[column].max()
                    assert(max_ >= min_)
                    range_ = max_ - min_ if max_ > min_ else 1e-16
                    self.dataframe[column] = (
                        self.dataframe[column] - min_) / range_
                    self.feature_min_max_dict[column] = dict(
                        max=max_, min=min_)

        elif self.dataset_kind == "chestxray":
            for column in self.dataframe.columns:
                if column not in ["filename", "patientid"]:
                    min_ = self.dataframe[column].min()
                    max_ = self.dataframe[column].max()
                    assert(max_ >= min_)
                    range_ = max_ - min_ if max_ > min_ else 1e-16
                    self.dataframe[column] = (
                        self.dataframe[column] - min_) / range_
                    self.feature_min_max_dict[column] = dict(
                        max=max_, min=min_)

        elif self.dataset_kind == "challenge":
            for dataframe in [self.dataframe_static, self.dataframe_dynamic]:
                for column in dataframe.columns:
                    if column not in ["In-hospital_death", "Gender", "CCU", "CSRU", "SICU", "MechVent"]:
                        min_ = dataframe[column].dropna().min()
                        max_ = dataframe[column].dropna().max()
                        assert(max_ >= min_)
                        range_ = max_ - min_
                        dataframe[column] = (dataframe[column] - min_) / range_
                        self.feature_min_max_dict[column] = dict(
                            max=max_, min=min_)

        elif self.dataset_kind == "alz":
            # For static,
            for static_feature_name in self.df_dict["static"].columns:
                self.feature_min_max_dict[static_feature_name] = {"max": self.df_dict["static"][static_feature_name].max(
                ), "min": self.df_dict["static"][static_feature_name].min()}
                if self.feature_min_max_dict[static_feature_name]["max"] != self.feature_min_max_dict[static_feature_name]["min"]:
                    self.df_dict["static"][static_feature_name] = (self.df_dict["static"][static_feature_name] - self.feature_min_max_dict[static_feature_name]["min"]) / (
                        self.feature_min_max_dict[static_feature_name]["max"] - self.feature_min_max_dict[static_feature_name]["min"])
            for modality in self.dynamic_modalities_sequence:  # For dynamic,
                # min, max for each dynamic feature.
                for feature_name in self.dynamic_feature_names_for_each_modality[modality]:
                    assert(feature_name not in self.feature_min_max_dict.keys())
                    max_, min_ = -1e+30, +1e+30
                    for time in self.TIMES:  # min, max across all the time points.
                        if self.df_dict["dynamic"][time][feature_name].max() > max_:
                            max_ = self.df_dict["dynamic"][time][feature_name].max(
                            )
                        if self.df_dict["dynamic"][time][feature_name].min() < min_:
                            min_ = self.df_dict["dynamic"][time][feature_name].min(
                            )
                    self.feature_min_max_dict[feature_name] = {
                        "max": max_, "min": min_}
                    for time in self.TIMES:
                        if max_ > min_:
                            self.df_dict["dynamic"][time][feature_name] = (
                                self.df_dict["dynamic"][time][feature_name] - min_) / (max_ - min_)
        else:
            raise Exception(NotImplementedError)

    def set_observed_numbers_pool_for_each_group_of_features(self, excluded_features=None):
        """Prepare the group of features used to plot feature importance, "input" means which input where that feature exists. Each group is plotted separately.

        Set the groups_of_features_info.

        Attributes
        ----------
        groups_of_features_info : dict
            For example, {group_1: {feature_1: {idx: 3, observed_numbers: [1.3, 5.3, ...], mean: 3.5, std: 1.4}, feature_2: {...}, ...}, ...}
        """

        if excluded_features is None:
            excluded_features = []
        self.groups_of_features_info = {}
        self.feature_to_input_map = {}
        # "dynamic_vectors_decoder" has no feature importance.
        for input_ in ["RNN_vectors", "static_encoder", "raw"]:
            for feature in self.input_to_features_map[input_]:
                # assert(feature not in self.feature_to_input_map.keys()) ## Avoid duplication, but dynamic decoder/RNN can be duplicated, so comment out.
                self.feature_to_input_map[feature] = input_

        # dynamic and all groups.
        if self.dataset_kind == "covid" or self.dataset_kind == "chestxray":
            self.groups_of_features_info["all"] = {}
            self.groups_of_features_info["dynamic"] = {}
            for input_ in ["raw", "RNN_vectors", "static_encoder"]:
                for feature in self.input_to_features_map[input_]:
                    self.groups_of_features_info["all"][feature] = dict(
                        idx=self.input_to_features_map[input_].index(feature), input=input_)
                    if feature not in self.static_column_names:
                        self.groups_of_features_info["dynamic"][feature] = dict(
                            idx=self.input_to_features_map[input_].index(feature), input=input_)

            for feature_group in self.groups_of_features_info.keys():
                for feature_name in self.groups_of_features_info[feature_group].keys():
                    self.groups_of_features_info[feature_group][feature_name]["observed_numbers"] = list(
                        self.dataframe[feature_name].dropna())
                    self.groups_of_features_info[feature_group][feature_name]["mean"] = np.mean(
                        self.groups_of_features_info[feature_group][feature_name]["observed_numbers"])
                    self.groups_of_features_info[feature_group][feature_name]["std"] = np.std(
                        self.groups_of_features_info[feature_group][feature_name]["observed_numbers"])

        elif self.dataset_kind == "challenge":
            self.groups_of_features_info["static"] = {}
            self.groups_of_features_info["dynamic"] = {}

            for input_ in ["raw", "static_encoder"]:
                for feature in self.input_to_features_map[input_]:
                    if feature not in excluded_features:
                        self.groups_of_features_info["static"][feature] = dict(
                            idx=self.input_to_features_map[input_].index(feature), input=input_)
            for feature in self.input_to_features_map["RNN_vectors"]:
                if feature not in excluded_features:
                    self.groups_of_features_info["dynamic"][feature] = dict(
                        idx=self.input_to_features_map["RNN_vectors"].index(feature), input="RNN_vectors")

            for feature_group, dataframe in zip(["static", "dynamic"], [self.dataframe_static, self.dataframe_dynamic]):
                for feature_name in self.groups_of_features_info[feature_group].keys():
                    if feature_name not in excluded_features:
                        self.groups_of_features_info[feature_group][feature_name]["observed_numbers"] = list(
                            dataframe[feature_name].dropna())
                        self.groups_of_features_info[feature_group][feature_name]["mean"] = np.mean(
                            self.groups_of_features_info[feature_group][feature_name]["observed_numbers"])
                        self.groups_of_features_info[feature_group][feature_name]["std"] = np.std(
                            self.groups_of_features_info[feature_group][feature_name]["observed_numbers"])

        elif self.dataset_kind == "alz":
            # Prepare result of the group of features used to plot feature importance, "input" (means where that feature exists in the input) can be different with self.separate_dynamic_static_features. Each group is plotted separately.
            self.groups_of_features_info = {
                "FS": {name: {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": [], "mean": None, "std": None} for name in self.dynamic_feature_names_for_each_modality["FS"]},
                "VBM": {name: {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": [], "mean": None, "std": None} for name in self.dynamic_feature_names_for_each_modality["VBM"]}}
            # Static
            self.groups_of_features_info["SNP"] = {name: {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(
                name), "observed_numbers": self.df_dict["static"][name].dropna().tolist(), "mean": None, "std": None} for name in list(self.df_dict["snp"].columns)[1:]}
            # Dynamic
            for time in self.TIMES:  # Collect observations across the time points.
                for modality in ["FS", "VBM"]:
                    for feature_name in self.dynamic_feature_names_for_each_modality[modality]:
                        self.groups_of_features_info[modality][feature_name]["observed_numbers"] += deepcopy(
                            self.df_dict["dynamic"][time][feature_name].dropna().tolist())
            # Calculate the mean, std for each group.
            for group in self.groups_of_features_info.keys():
                for feature_name in self.groups_of_features_info[group].keys():
                    self.groups_of_features_info[group][feature_name]["mean"] = np.mean(
                        self.groups_of_features_info[group][feature_name]["observed_numbers"])
                    self.groups_of_features_info[group][feature_name]["std"] = np.std(
                        self.groups_of_features_info[group][feature_name]["observed_numbers"])

        # Delete RE_DATE from the feature, you may/may not want to include RE_DATE.
        for group in self.groups_of_features_info.keys():
            if "RE_DATE" in self.groups_of_features_info[group].keys():
                del self.groups_of_features_info[group]["RE_DATE"]

    def set_dicts(self, kwargs_toy_dataset=None, prediction_label_mapping=None, observation_density_threshold=0.1):
        """Load the pandas dataframe datset into the dictionaries of Numpy arrays for Encoder or Decoder

        Attributes
        ----------
        observation_density_threshold : 0. < float < 1.
            The minimum threshold of observed entries divided by the total entries. The larger observation_density_threshold, result the more strict selection, thus the smaller number of records.
        """

        # outcome -> prediction label, mapping.
        if prediction_label_mapping is None:
            # Set prediction label length.
            if self.dataset_kind == "covid" or self.dataset_kind == "toy" or self.dataset_kind == "challenge" or self.dataset_kind == "chestxray":
                self.prediction_labels_bag = [0, 1]
            elif self.dataset_kind == "alz":
                self.prediction_labels_bag = [1, 2, 3]
            else:
                raise Exception(NotImplementedError)

            self.prediction_label_mapping = self.prediction_label_mapping_default_funct

        # list of dictionaries where each dictionary for each patient.
        self.dicts = []
        if self.dataset_kind == "covid":  # --- --- COVID-19 DATASET.
            for index, row in self.dataframe.iterrows():
                if pd.isnull(row["PATIENT_ID"]):  # Middle row of patient.
                    for column_name in self.static_column_names:
                        if column_name != 'PATIENT_ID':
                            # Static data should not change.
                            assert(patient_dict[column_name]
                                   == row[column_name])

                    for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                        patient_dict[input_]["data"].append(row[self.input_to_features_map[input_]].fillna(
                            self.init_number_for_unobserved_entry).tolist())
                        patient_dict[input_]["mask"].append(
                            (1. - row[self.input_to_features_map[input_]].isna()).tolist())
                        patient_dict[input_]["concat"].append(
                            deepcopy(patient_dict[input_]["data"][-1] + patient_dict[input_]["mask"][-1]))
                    patient_dict["RE_DATE"].append(row["RE_DATE"])
                    # Final row of patient.
                    if (index == len(self.dataframe) - 1 or not pd.isnull(self.dataframe.iloc[index + 1]["PATIENT_ID"])) and not pd.isnull(patient_dict["RE_DATE"][0]):
                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                            for data_type in ["data", "mask", "concat"]:
                                # Dummy dimension for time series.
                                patient_dict[input_][data_type] = np.array(
                                    [patient_dict[input_][data_type]])
                        for input_ in ["static_encoder", "raw"]:
                            for data_type in ["data", "mask", "concat"]:
                                # There is no dummy dimension for static features.
                                patient_dict[input_][data_type] = np.array(
                                    patient_dict[input_][data_type])
                        # time stamp should be increasing.
                        for date_idx in range(len(patient_dict["RE_DATE"]) - 1):
                            assert(patient_dict["RE_DATE"][date_idx]
                                   <= patient_dict["RE_DATE"][date_idx + 1])
                        patient_dict["RE_DATE"] = np.array(
                            [[[RE_DATE] for RE_DATE in patient_dict["RE_DATE"]]])
                        patient_dict["predictor label"] = self.prediction_label_mapping(
                            patient_dict["outcome"], inverse=False)
                        self.dicts.append(patient_dict)
                else:  # First row of patient.
                    patient_dict = {feature: row[feature]
                                    for feature in self.static_column_names}
                    for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "static_encoder", "raw"]:
                        patient_dict[input_] = {}
                        patient_dict[input_]["data"] = [row[self.input_to_features_map[input_]].fillna(
                            self.init_number_for_unobserved_entry).tolist()]
                        patient_dict[input_]["mask"] = [
                            (1. - row[self.input_to_features_map[input_]].isna()).tolist()]
                        patient_dict[input_]["concat"] = [
                            deepcopy(patient_dict[input_]["data"][0] + patient_dict[input_]["mask"][0])]
                    patient_dict["RE_DATE"] = [row["RE_DATE"]]

            self.num_patients = len(self.dicts)

        # --- --- physionet-challenge DATASET.
        elif self.dataset_kind == "challenge":
            self.dicts = {}  # Later, we will convert dict to list.
            for index, row in self.dataframe_static.iterrows():  # static
                patient_dict = {feature: row[feature]
                                for feature in self.static_column_names}
                patient_dict["outcome"] = patient_dict["In-hospital_death"]
                # Note that "RNN_vectors", "dynamic_vectors_decoder", are deleted, so feeding static data to LSTM will not be supported.
                for input_ in ["static_encoder", "raw"]:
                    patient_dict[input_] = {}
                    patient_dict[input_]["data"] = [row[self.input_to_features_map[input_]].fillna(
                        self.init_number_for_unobserved_entry).tolist()]
                    patient_dict[input_]["mask"] = [
                        (1. - row[self.input_to_features_map[input_]].isna()).tolist()]
                    patient_dict[input_]["concat"] = [
                        deepcopy(patient_dict[input_]["data"][0] + patient_dict[input_]["mask"][0])]
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    patient_dict[input_] = {}
                    patient_dict[input_]["data"] = []
                    patient_dict[input_]["mask"] = []
                    patient_dict[input_]["concat"] = []
                patient_dict["RE_DATE"] = []
                self.dicts[row["recordid"]] = patient_dict

            for index, row in self.dataframe_dynamic.iterrows():  # dynamic
                recordid = row["recordid"]
                assert(row["recordid"] in self.dicts.keys())
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    self.dicts[recordid][input_]["data"].append(row[self.input_to_features_map[input_]].fillna(
                        self.init_number_for_unobserved_entry).tolist())
                    self.dicts[recordid][input_]["mask"].append(
                        (1. - row[self.input_to_features_map[input_]].isna()).tolist())
                    self.dicts[recordid][input_]["concat"].append(deepcopy(
                        self.dicts[recordid][input_]["data"][-1] + self.dicts[recordid][input_]["mask"][-1]))
                self.dicts[recordid]["RE_DATE"].append(row["time"])

            # remove participant with zero dynamic record.
            self.dicts = {key: val for key,
                          val in self.dicts.items() if len(val["RE_DATE"]) > 0}

            for recordid in self.dicts.keys():  # finalize/clean up records.
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    for data_type in ["data", "mask", "concat"]:
                        # Dummy dimension for time series.
                        self.dicts[recordid][input_][data_type] = np.array(
                            [self.dicts[recordid][input_][data_type]])
                    for input_ in ["static_encoder", "raw"]:
                        for data_type in ["data", "mask", "concat"]:
                            # There is no dummy dimension for static features.
                            self.dicts[recordid][input_][data_type] = np.array(
                                self.dicts[recordid][input_][data_type])
                # time stamp should be increasing.
                for date_idx in range(len(self.dicts[recordid]["RE_DATE"]) - 1):
                    assert(self.dicts[recordid]["RE_DATE"][date_idx]
                           <= self.dicts[recordid]["RE_DATE"][date_idx + 1])
                self.dicts[recordid]["RE_DATE"] = np.array(
                    [[[RE_DATE] for RE_DATE in self.dicts[recordid]["RE_DATE"]]])
                self.dicts[recordid]["predictor label"] = self.prediction_label_mapping(
                    self.dicts[recordid]["In-hospital_death"], inverse=False)

            self.dicts = list(self.dicts.values())
            self.num_patients = len(self.dicts)

        elif self.dataset_kind == "chestxray":  # --- --- chestxray dataset.
            self.dicts = {}  # Later, we will convert dict to list.
            # for index, row in self.dataframe.iloc[:101].iterrows(): ## static
            for index, row in self.dataframe.iterrows():  # static
                if not row["filename"].endswith("nii.gz"):
                    patientid = row["patientid"]
                    img_arr, mask_arr = get_arr_of_image_from_path(path_to_img=os.path.join(
                        self.path_to_dataset, "images", row["filename"]), shape=self.shape_2D_records, nopostprocess=False, apply_segment=self.apply_segment)
                    if patientid not in self.dicts.keys():  # first time.
                        self.dicts[patientid] = {feature: row[feature]
                                                 for feature in self.static_column_names}
                        patient_dict = self.dicts[patientid]  # shallow copy.
                        patient_dict["outcome"] = patient_dict["survival_Y"]
                        for input_ in ["static_encoder", "raw", "RNN_vectors", "dynamic_vectors_decoder"]:
                            patient_dict[input_] = {}
                            patient_dict[input_]["data"] = [row[self.input_to_features_map[input_]].fillna(
                                self.init_number_for_unobserved_entry).tolist()]
                            patient_dict[input_]["mask"] = [
                                (1. - row[self.input_to_features_map[input_]].isna()).tolist()]
                            patient_dict[input_]["concat"] = [
                                deepcopy(patient_dict[input_]["data"][0] + patient_dict[input_]["mask"][0])]
                        patient_dict["RE_DATE"] = [self.init_number_for_unobserved_entry if np.isnan(
                            row["offset"]) else row["offset"]]
                        # 2D image input
                        patient_dict["RNN_2D"] = {}
                        patient_dict["RNN_2D"]["data"] = [img_arr]
                        patient_dict["RNN_2D"]["mask"] = [mask_arr]
                        patient_dict["RNN_2D"]["concat"] = [
                            np.concatenate([img_arr, mask_arr])]
                    else:
                        # for feature in self.static_column_names:
                        #     if feature not in ["filename", "age"]: assert(row[feature] == self.dicts[patientid][feature]) ## static data should not change.
                        patient_dict = self.dicts[patientid]  # shallow copy.
                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:  # dynamic data
                            patient_dict[input_]["data"].append(row[self.input_to_features_map[input_]].fillna(
                                self.init_number_for_unobserved_entry).tolist())
                            patient_dict[input_]["mask"].append(
                                (1. - row[self.input_to_features_map[input_]].isna()).tolist())
                            patient_dict[input_]["concat"].append(
                                deepcopy(patient_dict[input_]["data"][-1] + patient_dict[input_]["mask"][-1]))
                        patient_dict["RE_DATE"].append(self.init_number_for_unobserved_entry if np.isnan(
                            row["offset"]) else row["offset"])
                        # 2D image input
                        patient_dict["RNN_2D"]["data"].append(img_arr)
                        patient_dict["RNN_2D"]["mask"].append(mask_arr)
                        patient_dict["RNN_2D"]["concat"].append(
                            np.concatenate([img_arr, mask_arr]))

            self.dicts = {key: val for key, val in self.dicts.items() if len(val["RE_DATE"]) > 0 and len(
                val["RNN_2D"]["data"]) > 0}  # remove participant with zero dynamic record.

            for patientid in self.dicts.keys():  # finalize/clean up records.
                # patient_dict["RE_DATE"], *[patient_dict[input_][kinds_] for input_ in ["RNN_vectors", "dynamic_vectors_decoder"] for kinds_ in ["data", "mask", "concat"]] = sorted_multiple_lists(patient_dict["RE_DATE"], *[patient_dict[input_][kinds_] for input_ in ["RNN_vectors", "dynamic_vectors_decoder"] for kinds_ in ["data", "mask", "concat"]])
                # self.dicts[patientid]["RE_DATE"], self.dicts[patientid]["RNN_vectors"]["data"], self.dicts[patientid]["RNN_vectors"]["mask"], self.dicts[patientid]["RNN_vectors"]["concat"], self.dicts[patientid]["dynamic_vectors_decoder"]["data"], self.dicts[patientid]["dynamic_vectors_decoder"]["mask"], self.dicts[patientid]["dynamic_vectors_decoder"]["concat"], self.dicts[patientid]["RNN_2D"]["data"], self.dicts[patientid]["RNN_2D"]["mask"], self.dicts[patientid]["RNN_2D"]["concat"] = sorted_multiple_lists(self.dicts[patientid]["RE_DATE"], self.dicts[patientid]["RNN_vectors"]["data"], self.dicts[patientid]["RNN_vectors"]["mask"], self.dicts[patientid]["RNN_vectors"]["concat"], self.dicts[patientid]["dynamic_vectors_decoder"]["data"], self.dicts[patientid]["dynamic_vectors_decoder"]["mask"], self.dicts[patientid]["dynamic_vectors_decoder"]["concat"], self.dicts[patientid]["RNN_2D"]["data"], self.dicts[patientid]["RNN_2D"]["mask"], self.dicts[patientid]["RNN_2D"]["concat"])
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "RNN_2D"]:
                    for data_type in ["data", "mask", "concat"]:
                        # Dummy dimension for time series.
                        self.dicts[patientid][input_][data_type] = np.array(
                            [self.dicts[patientid][input_][data_type]])
                    for input_ in ["static_encoder", "raw"]:
                        for data_type in ["data", "mask", "concat"]:
                            # There is no dummy dimension for static features.
                            self.dicts[patientid][input_][data_type] = np.array(
                                self.dicts[patientid][input_][data_type])
                # for date_idx in range(len(self.dicts[patientid]["RE_DATE"]) - 1): ## time stamp should be increasing.
                #     assert(self.dicts[patientid]["RE_DATE"][date_idx] <= self.dicts[patientid]["RE_DATE"][date_idx + 1])
                self.dicts[patientid]["RE_DATE"] = np.array(
                    [[[RE_DATE] for RE_DATE in self.dicts[patientid]["RE_DATE"]]])
                self.dicts[patientid]["predictor label"] = self.prediction_label_mapping(
                    self.dicts[patientid]["outcome"], inverse=False)

            self.dicts = list(self.dicts.values())
            self.num_patients = len(self.dicts)

        elif self.dataset_kind == "toy":  # --- --- TOY DATASET.
            # Set default arguments.
            kwargs_copy = merge_dictionaries([{"num_patients": 385, "max_time_steps": 80, "num_features_static_dict": dict(raw=1, RNN_vectors=1, static_encoder=1), "num_features_dynamic": 2, "time_interval_range": [0., 0.05], "static_data_range": [0., 1.], "sin function info ranges dict": {
                                             "amplitude": [0., 1.], "displacement_along_x_axis": [0., 1.], "frequency": [0.5, 1.], "displacement_along_y_axis": [1., 2.]}, "observation noise range": [0., 0.], "missing probability": 0.0, "class_0_proportion": [0.5]}, kwargs_toy_dataset])
            num_features_static_dict = kwargs_copy["num_features_static_dict"]

            # Change column names from the names of real dataset to the names of toy dataset.
            for input_ in num_features_static_dict.keys():
                for i in range(num_features_static_dict[input_]):
                    self.input_to_features_map[input_].append(
                        f"static_{input_}_{i}")
            for input_ in ["RNN_vectors"]:
                for i in range(kwargs_copy["num_features_dynamic"]):
                    self.input_to_features_map[input_].append(f"dynamic_{i}")
            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(
                self.input_to_features_map["RNN_vectors"])
            # Last feature of LSTM input records is RE_DATE.
            self.input_to_features_map["RNN_vectors"].append(f"RE_DATE")

            # Set the classification threshold for each class.
            # outcome_class_decision_threshold = [kwargs_copy[num_features_static] * (kwargs_copy["static_data_range"][1] -  kwargs_copy["static_data_range"][0])]
            # for range_ in kwargs_copy["sin function info ranges dict"].items():
            #     outcome_class_decision_threshold[0] += (range_[1] - range_[0]) * kwargs_copy["num_features_dynamic"]
            # outcome_class_decision_threshold[0] = outcome_class_decision_threshold[0] / 2.

            # Stack 2D time series table.
            self.num_patients = kwargs_copy["num_patients"]
            for index in range(self.num_patients):
                time_steps = random.randint(1, kwargs_copy["max_time_steps"])

                # Initialize the records. shape = (time_steps, num_features)
                num_features_RNN = len(
                    self.input_to_features_map["RNN_vectors"])

                patient_dict = {}
                patient_dict["time_steps"] = time_steps
                patient_dict["RE_DATE"] = []
                patient_dict["predictor label"] = [[]]
                patient_dict["outcome_decision_values"] = [0.]
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    patient_dict[input_] = {}
                    patient_dict[input_]["data"] = [[None for i in range(
                        len(self.input_to_features_map[input_]))] for j in range(time_steps)]
                    patient_dict[input_]["mask"] = [[None for i in range(
                        len(self.input_to_features_map[input_]))] for j in range(time_steps)]
                    patient_dict[input_]["concat"] = [[]
                                                      for j in range(time_steps)]
                    for i in range(num_features_static_dict["RNN_vectors"]):
                        static_feature = rand_gen_with_range(
                            kwargs_copy["static_data_range"])
                        patient_dict["outcome_decision_values"][0] += static_feature
                        for j in range(time_steps):
                            patient_dict[input_]["data"][j][i] = static_feature
                            patient_dict[input_]["mask"][j][i] = 1.

                # Set RE_DATE
                patient_dict["RE_DATE"] = [rand_gen_with_range(
                    kwargs_copy["time_interval_range"])]
                for i in range(1, time_steps):
                    patient_dict["RE_DATE"].append(
                        patient_dict["RE_DATE"][i - 1] + rand_gen_with_range(kwargs_copy["time_interval_range"]))

                # Set static features.
                for input_ in ["raw", "static_encoder"]:
                    patient_dict[input_] = {}
                    patient_dict[input_]["data"] = [rand_gen_with_range(
                        kwargs_copy["static_data_range"]) for i in range(num_features_static_dict[input_])]
                    patient_dict[input_]["mask"] = [
                        1. for i in range(num_features_static_dict[input_])]
                    patient_dict[input_]["concat"] = deepcopy(
                        patient_dict[input_]["data"] + patient_dict[input_]["mask"])
                    patient_dict["outcome_decision_values"][0] += sum(
                        patient_dict[input_]["data"])

                # Create records matrix
                # Starting from num_features_static_dict["RNN_vectors"] to discard static features, -1 for discarded RE_DATE
                for feature_idx in range(num_features_static_dict["RNN_vectors"], num_features_RNN - 1):
                    amplitude = rand_gen_with_range(
                        kwargs_copy["sin function info ranges dict"]["amplitude"])
                    frequency = rand_gen_with_range(
                        kwargs_copy["sin function info ranges dict"]["frequency"])
                    displacement_along_x_axis = rand_gen_with_range(
                        kwargs_copy["sin function info ranges dict"]["displacement_along_x_axis"])
                    displacement_along_y_axis = rand_gen_with_range(
                        kwargs_copy["sin function info ranges dict"]["displacement_along_y_axis"])
                    # Static feature contribution to outcome_decision_values is already applied.
                    patient_dict["outcome_decision_values"][0] += amplitude + \
                        frequency + displacement_along_x_axis + displacement_along_y_axis

                    # Create record vector
                    for t in range(time_steps):
                        # Observed case
                        if random.random() > kwargs_copy["missing probability"]:
                            data_gen = amplitude * np.sin((patient_dict["RE_DATE"][t] - displacement_along_x_axis) / frequency) + \
                                displacement_along_y_axis + \
                                rand_gen_with_range(
                                    kwargs_copy["observation noise range"])
                            mask_gen = 1.
                        else:  # Unobserved case
                            data_gen = self.init_number_for_unobserved_entry
                            mask_gen = 0.
                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                            patient_dict[input_]["data"][t][feature_idx] = data_gen
                            patient_dict[input_]["mask"][t][feature_idx] = mask_gen

                # Last feature of each time step is RE_DATE.
                for t in range(time_steps):
                    assert(patient_dict["RNN_vectors"]["data"][t][-1]
                           is None and patient_dict["RNN_vectors"]["mask"][t][-1] is None)
                    patient_dict["RNN_vectors"]["data"][t][-1] = patient_dict["RE_DATE"][t]
                    patient_dict["RNN_vectors"]["mask"][t][-1] = 1.

                # Format data to fit with Keras Input.
                for date_idx in range(len(patient_dict["RE_DATE"]) - 1):
                    assert(patient_dict["RE_DATE"][date_idx]
                           <= patient_dict["RE_DATE"][date_idx + 1])
                # patient_dict["RE_DATE"] = np.array([patient_dict["RE_DATE"]])
                patient_dict["RE_DATE"] = np.array(
                    [[[RE_DATE] for RE_DATE in patient_dict["RE_DATE"]]])

                # list -> array
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    for t in range(time_steps):
                        patient_dict[input_]["concat"][t] = deepcopy(
                            patient_dict[input_]["data"][t] + patient_dict[input_]["mask"][t])
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "raw", "static_encoder"]:
                    for type_ in ["data", "mask", "concat"]:
                        # Adds dummy batch dimension.
                        patient_dict[input_][type_] = np.array(
                            [patient_dict[input_][type_]])
                # patient_dict['LSTM inputs data and mask concatenated'] = np.array([np.concatenate([patient_dict['LSTM inputs data'][0], patient_dict['LSTM inputs mask'][0]], axis = 1)])
                self.dicts.append(patient_dict)

            # Set target labels.
            # Supports multi-class label.
            for class_0_proportion_idx in range(len(kwargs_copy["class_0_proportion"])):
                rank = round(
                    self.num_patients * kwargs_copy["class_0_proportion"][class_0_proportion_idx])
                self.dicts.sort(
                    key=lambda x: x["outcome_decision_values"][class_0_proportion_idx])
                for patient_idx in range(0, rank):
                    self.dicts[patient_idx]["outcome"] = 0.
                    # self.dicts[patient_idx]["predictor label"][0].append(0.)
                for patient_idx in range(rank, self.num_patients):
                    self.dicts[patient_idx]["outcome"] = 1.
                    # self.dicts[patient_idx]["predictor label"][0].append(1.)
            for patient_idx in range(self.num_patients):
                # Set outcome to be label of first class, change here if you want to adds multi classes.
                self.dicts[patient_idx]["predictor label"] = self.prediction_label_mapping(
                    self.dicts[patient_idx]["outcome"], inverse=False)
                # self.dicts[patient_idx]["predictor label"] = np.array(self.dicts[patient_idx]["predictor label"])
            np.random.shuffle(self.dicts)

        elif self.dataset_kind == "alz":  # --- --- ADNI dataset.
            self.dicts = []
            # Stack the actual data which will be fed into Autoencoder.
            for valid_idx in tqdm(self.valid_indices["intersect"]):
                # only for this patient.
                patient_dict = {"valid_idx": valid_idx, "RE_DATE": []}
                # --- Set static info.
                for input_ in ["static_encoder", "raw"]:
                    patient_dict[input_] = dict(
                        data=[self.df_dict["static"][self.input_to_features_map[input_]].iloc[valid_idx].fillna(
                            self.init_number_for_unobserved_entry).tolist()],
                        mask=[(1. - self.df_dict["static"][self.input_to_features_map[input_]
                                                           ].iloc[valid_idx].isna()).tolist()],
                        concat=[self.df_dict["static"][self.input_to_features_map[input_]].iloc[valid_idx].fillna(self.init_number_for_unobserved_entry).tolist(
                        ) + (1. - self.df_dict["static"][self.input_to_features_map[input_]].iloc[valid_idx].isna()).tolist()]
                    )

                # --- Set target label.
                patient_dict["outcome"] = self.df_dict["outcome"].iloc[valid_idx]
                patient_dict["ID"] = self.df_dict["info"]["SubjID"].iloc[valid_idx]
                patient_dict["predictor label"] = self.prediction_label_mapping(
                    patient_dict["outcome"], inverse=False)

                # --- Set dynamic info
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    patient_dict[input_] = dict(
                        data=[[]], mask=[[]], concat=[[]])
                patient_dict["RE_DATE"] = [[]]  # [[[RE_DATE], [RE_DATE], ...]]

                # Set static features for RNN input.
                static_for_RNN = {"data": [], "mask": []}
                # appends this list before dynamic feature list, and follows the sequence in self.static_data_input_mapping["RNN_vectors"].
                for static_modality in self.static_data_input_mapping["RNN_vectors"]:
                    static_for_RNN["data"] = deepcopy(static_for_RNN["data"] + self.df_dict["static"]
                                                      [self.static_modality_to_features_map[static_modality]].iloc[valid_idx].fillna(self.init_number_for_unobserved_entry).tolist())
                    static_for_RNN["mask"] = deepcopy(static_for_RNN["mask"] + (
                        1. - self.df_dict["static"][self.static_modality_to_features_map[static_modality]].iloc[valid_idx].isna()).tolist())

                for time in self.TIMES:
                    RE_DATE = [self.TIMES_TO_RE_DATE[time]]
                    record_mask = (
                        1. - self.df_dict["dynamic"][time].iloc[valid_idx].isna()).tolist()
                    # Filter out sparse record.
                    if sum(record_mask) / max(len(record_mask), 1) >= observation_density_threshold:
                        record_data = self.df_dict["dynamic"][time].iloc[valid_idx].fillna(
                            self.init_number_for_unobserved_entry).tolist()
                        patient_dict["dynamic_vectors_decoder"]["data"][0].append(
                            deepcopy(static_for_RNN["data"] + record_data))
                        patient_dict["dynamic_vectors_decoder"]["mask"][0].append(
                            deepcopy(static_for_RNN["mask"] + record_mask))
                        patient_dict["dynamic_vectors_decoder"]["concat"][0].append(deepcopy(
                            static_for_RNN["data"] + record_data + static_for_RNN["mask"] + record_mask))
                        patient_dict["RNN_vectors"]["data"][0].append(
                            deepcopy(static_for_RNN["data"] + record_data + RE_DATE))
                        patient_dict["RNN_vectors"]["mask"][0].append(
                            deepcopy(static_for_RNN["mask"] + record_mask + [1.]))
                        patient_dict["RNN_vectors"]["concat"][0].append(deepcopy(
                            static_for_RNN["data"] + record_data + RE_DATE + static_for_RNN["mask"] + record_mask + [1.]))
                        patient_dict["RE_DATE"][0].append(RE_DATE)
                # patient_dict["RNN_vectors"]["concat"] = np.array([np.concatenate([patient_dict["RNN_vectors"]["data"][0], patient_dict["RNN_vectors"]["mask"][0]], axis = 1)])
                # --- list to array.
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "static_encoder", "raw"]:
                    for data_type in ["data", "mask", "concat"]:
                        patient_dict[input_][data_type] = np.array(
                            patient_dict[input_][data_type])
                patient_dict["RE_DATE"] = np.array(patient_dict["RE_DATE"])
                self.dicts.append(patient_dict)

            self.num_patients = len(self.dicts)  # Finish this participiant.
            # --- SANITY TEST for Alz datset: pick participant: 011_S_0002 (valid_index: 0, snp and diagnosis both exists).
            # static data sanity.
            if len(self.static_data_input_mapping["static_encoder"]) > 0 and self.static_data_input_mapping["static_encoder"][0] == "SNP" and 0 in self.valid_indices["intersect"]:
                assert(self.dicts[0]["static_encoder"]["concat"][0][0] * (self.feature_min_max_dict["rs4846048"]["max"] -
                                                                          self.feature_min_max_dict["rs4846048"]["min"]) + self.feature_min_max_dict["rs4846048"]["min"] == 1.0)
                assert(self.dicts[0]["static_encoder"]["concat"][0][2] * (self.feature_min_max_dict["rs1476413"]["max"] -
                                                                          self.feature_min_max_dict["rs1476413"]["min"]) + self.feature_min_max_dict["rs1476413"]["min"] == 0.0)
            # dynamic data sanity.
            assert(self.dicts[0]["RNN_vectors"]["concat"][0][0][self.input_to_features_map["RNN_vectors"].index("VBM_mod_LCalcarine")] * (self.feature_min_max_dict["VBM_mod_LCalcarine"]
                                                                                                                                          ["max"] - self.feature_min_max_dict["VBM_mod_LCalcarine"]["min"]) + self.feature_min_max_dict["VBM_mod_LCalcarine"]["min"] == 0.403619369811371)
            assert(abs(self.dicts[0]["RNN_vectors"]["concat"][0][0][self.input_to_features_map["RNN_vectors"].index("FS_MPavg_RTransvTemporal")] * (self.feature_min_max_dict["FS_MPavg_RTransvTemporal"]
                                                                                                                                                    ["max"] - self.feature_min_max_dict["FS_MPavg_RTransvTemporal"]["min"]) + self.feature_min_max_dict["FS_MPavg_RTransvTemporal"]["min"] - 1.85876451391478) < 1e-5)
            assert(self.dicts[0]["outcome"] == 3.)

        else:
            raise Exception(NotImplementedError)

    def split_train_test(self, train_proportion=None, k_fold=2, whether_training_is_large=False, shuffle=True, balance_label_distribution=False):
        """Set the indices of test and training set for each k-fold split.

        Parameters
        ----------
        k_fold : int
            The number of splits in k-fold cross validation. This is ignored when train_proportion is not None.
        whether_training_is_large : bool
            Whether the training set picks the remaining splits of k-folded indices. his is ignored when train_proportion is not None.
        """

        assert(k_fold > 1)
        self.this_dataset_is_splitted = True
        self.train_proportion = train_proportion
        self.k_fold = k_fold
        self.whether_training_is_large = whether_training_is_large
        # [{"train": [...], "test": [...]}, {"train": [...], "test": [...]}, {"train": [...], "test": [...]}, ...].
        self.splits_train_test = []
        # {"x_train": [k= 0, k= 1, ...], "y_train": [k= 0, k= 1, ...], "x_test": [k= 0, k= 1, ...], "y_test": [k= 0, k= 1, ...]}
        self.most_recent_record_dict = {
            "x_train": [], "y_train": [], "x_test": [], "y_test": []}
        num_patients_each_split = round(self.num_patients / k_fold)
        self.indices_patients_splits = []  # [[indices], [indices], ...].
        self.indices_patients_shuffled = list(
            range(self.num_patients))  # [indices of all]
        if shuffle:
            np.random.shuffle(self.indices_patients_shuffled)

        if train_proportion is None:  # k_fold cross validation is not applied.
            # Stack self.indices_patients_splits : [[indices], [indices], ...].
            split_idx = 0
            for k in range(k_fold - 1):
                self.indices_patients_splits.append(
                    self.indices_patients_shuffled[split_idx * num_patients_each_split: (split_idx + 1) * num_patients_each_split])
                split_idx += 1
            if split_idx * num_patients_each_split < self.num_patients:
                self.indices_patients_splits.append(
                    self.indices_patients_shuffled[split_idx * num_patients_each_split:])
            assert(len(self.indices_patients_splits) == k_fold)

            # self.splits_train_test : [{"train": [...], "test": [...]}, {"train": [...], "test": [...]}, {"train": [...], "test": [...]}, ...].
            for k in range(k_fold):
                one_piece = self.indices_patients_splits[k]
                remaining_pieces = []
                for k_ in range(k_fold):
                    if k_ != k:
                        remaining_pieces = remaining_pieces + \
                            self.indices_patients_splits[k_]
                if whether_training_is_large:
                    self.splits_train_test.append(
                        {"train": deepcopy(remaining_pieces), "test": deepcopy(one_piece)})
                else:
                    self.splits_train_test.append(
                        {"train": deepcopy(one_piece), "test": deepcopy(remaining_pieces)})

        else:  # k_fold cross validation is applied.
            num_train = round(train_proportion * self.num_patients)
            self.splits_train_test.append({"train": deepcopy(
                self.indices_patients_shuffled[:num_train]), "test": deepcopy(self.indices_patients_shuffled[num_train:])})

        # Set self.most_recent_record_dict for baseline models.
        for k in range(len(self.splits_train_test)):  # range(k_fold).
            x_train, y_train, x_test, y_test = [], [], [], []
            for patient_idx_train in self.splits_train_test[k]["train"]:
                x_train.append(np.concatenate([self.dicts[patient_idx_train]["RNN_vectors"]["data"][0, -1], self.dicts[patient_idx_train]
                                               ["static_encoder"]["data"][0], self.dicts[patient_idx_train]["raw"]["data"][0]], axis=0))
                # 0 for dummy batch dimension.
                y_train.append(
                    self.dicts[patient_idx_train]["predictor label"][0])
            for patient_idx_test in self.splits_train_test[k]["test"]:
                x_test.append(np.concatenate([self.dicts[patient_idx_test]["RNN_vectors"]["data"][0, -1], self.dicts[patient_idx_test]
                                              ["static_encoder"]["data"][0], self.dicts[patient_idx_test]["raw"]["data"][0]], axis=0))
                # 0 for dummy batch dimension.
                y_test.append(
                    self.dicts[patient_idx_test]["predictor label"][0])
            # List -> Array.
            self.most_recent_record_dict["x_train"].append(np.array(x_train))
            self.most_recent_record_dict["y_train"].append(np.array(y_train))
            self.most_recent_record_dict["x_test"].append(np.array(x_test))
            self.most_recent_record_dict["y_test"].append(np.array(y_test))

        # For unbalanced dataset, upsample the minor-classes.
        if balance_label_distribution:
            for split in self.splits_train_test:
                list_of_train_indices_orig = split["train"]
                list_of_train_indices_augmented = deepcopy(split["train"])
                counts_dict = utilsforminds.containers.get_items_counts_dict_from_container(
                    container=list_of_train_indices_orig, access_to_item_funct=lambda x: self.dicts[x]["predictor label"])
                num_largest_class = utilsforminds.containers.get_max_with_accessor(
                    counts_dict, accessor=lambda x: counts_dict[x])

                for train_index in list_of_train_indices_orig:
                    predictor_label = tuple(
                        map(tuple, self.dicts[train_index]["predictor label"]))
                    num_copies = (
                        num_largest_class - counts_dict[predictor_label]) / counts_dict[predictor_label]
                    for i in range(round(num_copies)):
                        list_of_train_indices_augmented.append(train_index)
                    if random.random() < num_copies - round(num_copies):
                        list_of_train_indices_augmented.append(train_index)
                if shuffle:
                    np.random.shuffle(list_of_train_indices_augmented)
                split["train"] = deepcopy(list_of_train_indices_augmented)

    def set_observabilities(self, indices_patients_train, indices_patients_test):
        for patient_idx in indices_patients_train:
            # np.array([[1.]]), in training set.
            self.dicts[patient_idx]["observability"] = np.array([[1.]])
        for patient_idx in indices_patients_test:
            # in test set.
            self.dicts[patient_idx]["observability"] = np.array([[0.]])

    def export_to_csv(self, dir_path_to_export="./datasets/datasets_in_csv/"):
        """Exports ONLY RNN input features, to csv file of each participant.

        """

        os.mkdir(dir_path_to_export)
        for patient_dict_idx, patient_dict in zip(range(len(self.dicts)), self.dicts):
            for type_ in ["data", "mask"]:
                df = pd.DataFrame(
                    patient_dict["RNN_vectors"][type_][0], columns=self.input_to_features_map["RNN_vectors"])
                df["outcome"] = [patient_dict["outcome"]
                                 for i in range(patient_dict["time_steps"])]
                df.to_csv(
                    f"{dir_path_to_export}{patient_dict_idx}_{type_}.csv")
            # if self.separate_dynamic_static_features == "separate raw" or self.separate_dynamic_static_features == "separate enrich":
            #     df = pd.DataFrame(patient_dict[f'static_features_data'], columns = self.static_feature_names)
            #     df.to_csv(f"{dir_path_to_export}{patient_dict_idx}_static_data.csv")

    def clear_misc(self):
        """Remove non-necessary data"""

        self.dicts = None

    def prediction_label_mapping_default_funct(self, label, inverse=False):
        """If label is scalar -> vector (1, n), else if label is vector (n, ) -> scalar"""

        if self.dataset_kind == "covid" or self.dataset_kind == "toy" or self.dataset_kind == "challenge" or self.dataset_kind == "chestxray":
            if not inverse:
                if round(label) == 0:
                    return np.array([[1., 0.]])
                elif round(label) == 1:
                    return np.array([[0., 1.]])
                else:
                    raise Exception(NotImplementedError)
            else:
                if np.argmax(label) == 0:
                    return 0
                elif np.argmax(label) == 1:
                    return 1
                else:
                    raise Exception(NotImplementedError)
        elif self.dataset_kind == "alz":
            if not inverse:
                if round(label) == 1:
                    return np.array([[1., 0., 0.]])
                elif round(label) == 2:
                    return np.array([[0., 1., 0.]])
                elif round(label) == 3:
                    return np.array([[0., 0., 1.]])
                else:
                    raise Exception(NotImplementedError)
            else:
                if np.argmax(label) == 0:
                    return 1
                elif np.argmax(label) == 1:
                    return 2
                elif np.argmax(label) == 2:
                    return 3
                else:
                    raise Exception(NotImplementedError)
        else:
            raise Exception(NotImplementedError)

    def reduce_number_of_participants(self, reduced_number_of_participants):
        """Reduce the number of samples in this dataset.

        Usually used to create dataset for debugging purpose.

        Args:
            reduced_number_of_participants ([type]): [description]
        """

        # Should not be k-fold splitted yet.
        assert(not self.this_dataset_is_splitted)
        if reduced_number_of_participants < self.num_patients:
            self.dicts = random.sample(
                self.dicts, reduced_number_of_participants)
            print(
                f"The number of participants is changed from {self.num_patients} to {reduced_number_of_participants}")
            self.num_patients = reduced_number_of_participants
        else:
            print("WARNING: The requested number of participants is larger than current number of participants, so do nothing.")

    def get_data_collections_from_all_samples(self, data_access_function):
        """Get the list of collections.

        Parameters
        ----------
        data_access_function : callable
            For example, lambda x: x['outcome'].
        """

        collections = []
        for idx in range(len(self.dicts)):
            collections.append(data_access_function(self.dicts[idx]))
        return collections

    def get_measures_of_feature(self, feature_name, group_name="static", reverse_min_max_scale=False):
        """

        Parameters
        ----------
        feature_access_funct : callable
        Function to access the features given self.groups_of_features_info. For example lambda x: x["static"]["Age"]["observed_numbers"].

        Examples
        --------
        """

        observed_numbers = np.array(
            self.groups_of_features_info[group_name][feature_name]["observed_numbers"])
        if reverse_min_max_scale and self.feature_min_max_dict[feature_name]["max"] > self.feature_min_max_dict[feature_name]["min"]:
            observed_numbers = observed_numbers * \
                (self.feature_min_max_dict[feature_name]["max"] - self.feature_min_max_dict[feature_name]
                 ["min"]) + self.feature_min_max_dict[feature_name]["min"]
        return observed_numbers


def change_names_from_starting_names(names: list, starting_names: list):
    """Useful when you change column names form the column names changed by pd.get_dummies.

    Examples
    --------
    names = ["gender", "age", "cells"]
    starting_names = ["gender_male", "gender_female", "age", "cells_1", "cells_2", "cells_3", "additional_column"]
    print(change_names_from_starting_names(names, starting_names))
    >>> ['gender_male', 'gender_female', 'age', 'cells_1', 'cells_2', 'cells_3']

    """
    changed_names = []
    for name in names:
        if name in starting_names:
            changed_names.append(name)
        else:
            for starting_name in starting_names:
                if starting_name.startswith(name):
                    changed_names.append(starting_name)
    return changed_names


def get_arr_of_image_from_path(path_to_img, shape=None, nopostprocess=False, apply_segment=True):
    if shape is None:
        shape = (1051, 1024)
    noHU = True

    if path_to_img.endswith("nii.gz"):
        input_image = nib.load(path_to_img)
        input_image = skimage.color.rgb2gray(input_image.get_fdata())
    else:
        # param 0 means gray-mode image load.
        input_image = cv2.imread(path_to_img, 0)
    input_image_arr = cv2.equalizeHist(input_image)
    input_image_equalized = sitk.GetImageFromArray(input_image_arr)

    if apply_segment:
        model = mask.get_model("unet", 'R231CovidWeb')
        result = mask.apply(input_image_equalized, model=model, noHU=noHU,
                            volume_postprocessing=not nopostprocess)  # default model is U-net(R231)

    if not apply_segment or type(result) == str:
        if apply_segment:
            print(
                f"Segment failure with reason: {result}, segmentation will not applied to image.")
        img_arr_tosave = input_image_arr
        mask_arr = np.ones(shape=shape)
    else:
        if noHU:
            file_ending = path_to_img.split('.')[-1]
            if file_ending in ['jpg', 'jpeg', 'png']:
                result = (result/(result.max())*255).astype(np.uint8)
            result = result[0]

        if len(input_image_arr.shape) == 3:
            raise Exception(
                f"Should be gray image already by parameter 0 in cv2.imread(img_path, 0)")
            grey_img = skimage.color.rgb2gray(input_image_arr)
        elif len(input_image_arr.shape) == 2:
            grey_img = input_image_arr
        else:
            raise Exception(f"Unsupported dimension: {input_image_arr.shape}")

        if grey_img.max() > 1.0:
            grey_img = grey_img / 255

        mask_arr = result
        img_arr_tosave = grey_img * mask_arr

    # Resize Image
    img_arr_tosave = skimage.transform.resize(img_arr_tosave, list(
        shape) + [1])  # Adds [1] for (gray) channel dimension
    # Adds [1] for (gray) channel dimension
    mask_arr = skimage.transform.resize(mask_arr, list(shape) + [1])

    # Normalize
    if img_arr_tosave.max() > 1.0:
        img_arr_tosave = img_arr_tosave / 255
    if mask_arr.max() > 1.0:
        mask_arr = mask_arr / 255

    return img_arr_tosave, mask_arr


def sorted_multiple_lists(list_sort_baseline, *other_lists, reverse=False):
    """

    Examples
    --------
    foo = ["c", "b", "a"]
    bar = [1, 2, 3]
    too = [5, 4, 6]
    foo, bar, too = sorted_multiple_lists(foo, bar, too)
    print(foo, bar, too)
    >>> ('a', 'b', 'c') (3, 2, 1) (6, 4, 5)
    foo, bar, too = sorted_multiple_lists(foo, bar, too, reverse = True)
    print(foo, bar, too)
    >>> ('c', 'b', 'a') (1, 2, 3) (5, 4, 6)
    too, bar, foo = sorted_multiple_lists(too, bar, foo)
    print(foo, bar, too)
    >>> ('b', 'c', 'a') (2, 1, 3) (4, 5, 6)
    [foo, bar, too] = sorted_multiple_lists(*[foo, bar, too])
    print(foo, bar, too)
    >>> ('a', 'b', 'c') (3, 2, 1) (6, 4, 5)
    foo, *[bar, too] = sorted_multiple_lists(foo, *[bar, too])
    print(foo, bar, too)
    >>> ('a', 'b', 'c') (3, 2, 1) (6, 4, 5) 
    """

    return zip(*sorted(zip(list_sort_baseline, *other_lists), reverse=reverse, key=lambda x: x[0]))


if __name__ == "__main__":
    foo = ["c", "b", "a"]
    bar = [1, 2, 3]
    too = [5, 4, 6]
    foo, *[bar, too] = sorted_multiple_lists(foo, *[bar, too])
    print(foo, bar, too)

    print("END")
