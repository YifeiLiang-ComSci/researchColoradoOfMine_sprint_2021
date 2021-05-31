import pickle
import utilsforminds
import os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from utilsforminds.math import mean, std
from utilsforminds.containers import GridSearch

from outer_sources.clean_labels import VBMDataCleaner
from outer_sources.roi_map import VBMRegionOfInterestMap
from outer_sources.roi_map import FSRegionOfInterestMap


class Experiment():
    """Experiment object to predict something from dataset object and estimator object."""

    def __init__(self, base_name):
        """

        Attributes
        ----------
        base_name : str
            Directory path to save the experimental result.
        """

        # Set the parent directory to save the experimental results.
        self.base_name = base_name
        parent_dir_name = utilsforminds.helpers.getNewDirectoryName(
            "outputs/", base_name)
        self.parent_path = "./outputs/" + parent_dir_name
        os.mkdir(Path(self.parent_path + "/"))

    def set_experimental_result(self, experiment_settings_list, save_result=True, note: str = None, dataset=None, dataset_path: str = None):
        """Conduct the experiment and save the experimental results.

        Attributes
        ----------
        experiment_settings_list : list of (dict or GirdSearch)
            Each dictionary contains each model represented by it's keyword arguments or GridSearch object, for example,
            dict(model_class = autoencoder.Autoencoder,
            init = dict(debug = 1, verbose = 1, use_mask_in_LSTM_input = True),
            fit = dict(...) (optional),
            predict = dict(...),
            fit_on = 'train',
            name = 'semi-supervised AE' (optional))

        use_mask_in_LSTM_input : bool
            Whether to concatenate mask (observabilities on the features) to input.
        """

        # Load dataset
        if dataset is None:
            assert(dataset_path is not None)
            dataset = pickle.load(open(dataset_path, "rb"))
        self.dataset_path = dataset_path
        self.dataset_ID = dataset.ID
        # Save the dataset parameters.
        with open(self.parent_path + "/parameters.txt", "a") as txt_file:
            if note is not None:
                txt_file.write(f"\tNote: \n{note}\n")
            txt_file.write(
                f"\t=================== Dataset ===================\n")
            txt_file.write(utilsforminds.strings.container_to_str(dataset.__dict__, recursive_prints=False,
                                                                  whether_print_object=False, limit_number_of_prints_in_container=20) + "\n")

        # For debugging: The indices of patients should not be changed.
        indices_of_patients_copy = deepcopy(dataset.indices_patients_shuffled)

        # Expand the GridSearch object if it exists.
        experiment_settings_list_expanded = []
        list_of_paths_to_grids = []
        for experiment_setting in experiment_settings_list:
            if isinstance(experiment_setting, GridSearch):
                experiment_settings = experiment_setting.get_list_of_grids()
                for idx in range(len(experiment_settings)):
                    experiment_settings_list_expanded.append(
                        experiment_settings[idx])
                    list_of_paths_to_grids.append(
                        deepcopy(experiment_setting.list_of_paths_to_grids))
            else:
                experiment_settings_list_expanded.append(experiment_setting)
                list_of_paths_to_grids.append(None)

        # Conduct experiments.
        # {"model_1": [result for each split], ...}
        self.experiment_results = {}
        # For each model.
        for experiment_setting, experiment_idx in zip(experiment_settings_list_expanded, range(len(experiment_settings_list_expanded))):
            # Set model name.
            if "name" in experiment_setting.keys():
                experiment_name = experiment_setting["name"]
            else:
                experiment_name = experiment_setting["model_class"].name
            # [dict_for_split_1, dict_for_split_2, ...]
            self.experiment_results[experiment_name] = []

            # Save the model parameters.
            with open(self.parent_path + "/parameters.txt", "a") as txt_file:
                txt_file.write(
                    f"\t============================{experiment_name}============================\n")
                txt_file.write(utilsforminds.strings.container_to_str(
                    experiment_setting, file_format="txt", paths_to_emphasize=list_of_paths_to_grids[experiment_idx]) + "\n")

            # Set indices of patients to fit and predict
            # split_train_test = {"train": [...], "test": [...]}.
            for split_train_test, split_train_test_idx in zip(dataset.splits_train_test, range(len(dataset.splits_train_test))):
                experiment_result_on_each_split = {}  # Result for this split.

                # Set the indices of participants to learn.
                if experiment_setting["fit_on"] == "train":
                    experiment_result_on_each_split["fit_on"] = deepcopy(
                        split_train_test["train"])
                elif experiment_setting["fit_on"] == "test":
                    experiment_result_on_each_split["fit_on"] = deepcopy(
                        split_train_test["test"])
                elif experiment_setting["fit_on"] == "both":
                    experiment_result_on_each_split["fit_on"] = deepcopy(
                        split_train_test["train"] + split_train_test["test"])

                # Set observabilities only for Autoencoder model.
                if experiment_setting["model_class"].name == "Autoencoder":
                    dataset.set_observabilities(
                        indices_patients_train=split_train_test["train"], indices_patients_test=split_train_test["test"])
                    # Sanity check.
                    for pateint_idx in split_train_test["test"]:
                        assert(dataset.dicts[pateint_idx]
                               ["observability"][0][0] == 0.)
                    for pateint_idx in split_train_test["train"]:
                        assert(dataset.dicts[pateint_idx]
                               ["observability"][0][0] == 1.)

                # Train and Predict.
                model_instance = experiment_setting["model_class"](
                    **experiment_setting["init"])

                # Train if trainable.
                if "fit" in experiment_setting.keys():
                    experiment_result_on_each_split["loss_dict"] = model_instance.fit(dataset=dataset, indices_of_patients=experiment_result_on_each_split["fit_on"], x_train=dataset.most_recent_record_dict[
                                                                                      "x_train"][split_train_test_idx], y_train=dataset.most_recent_record_dict["y_train"][split_train_test_idx], **experiment_setting["fit"])

                # Predict.
                # possible keys: "enriched_vectors_stack", "predicted_labels_stack", "reconstructed_vectors_stack", "feature_importance_dict".
                experiment_result_on_each_split["predictions_dict"] = model_instance.predict(
                    dataset=dataset, indices_of_patients=split_train_test["test"], x_test=dataset.most_recent_record_dict["x_test"][split_train_test_idx], **experiment_setting["predict"])
                # experiment_result_on_each_split["predictions_dict"] is dictionary which should contain at least key "predicted_labels_stack".
                assert(
                    "predicted_labels_stack" in experiment_result_on_each_split["predictions_dict"].keys())

                self.experiment_results[experiment_name].append(
                    experiment_result_on_each_split)  # [dict_for_split_1, dict_for_split_2, ...]

        # For debugging: The indices of patients should not be changed.
        for idx in range(dataset.num_patients):
            assert(indices_of_patients_copy[idx] ==
                   dataset.indices_patients_shuffled[idx])

        if save_result:
            with open(self.parent_path + "/experiment.obj", "wb") as experiment_file:
                # self.model_instance.clear_model()
                pickle.dump(self, experiment_file)


def plot_experimental_results(experiment, dataset=None, num_loss_plotting_points=200, num_top_features=None, verbose=1):
    """Plot the experimental results from the experiment object. We separate the visualization from the experiment.

    Parameters
    ----------
    experiment : Experiment
        The experiment object containing the results.
    dataset : Dataset
        The same dataset used in the experiment object.
    num_loss_plotting_points : int
        The number of plotting points for losses of model.
    num_top_features : int
        The number of most important features to plot.
    """

    visualizations_patent_path = experiment.parent_path + "/visualizations"
    os.mkdir(visualizations_patent_path + "/")
    os.mkdir(visualizations_patent_path + "/losses/")
    os.mkdir(visualizations_patent_path + "/feature_importance/")
    os.mkdir(visualizations_patent_path + "/ROC_curve/")

    # Calculate the prediction accuracy.
    # Load dataset.
    if dataset is None:
        dataset = pickle.load(open(experiment.dataset_path, "rb"))
    # To check whether the dataset used in the experiment is same as this dataset.
    assert(dataset.ID == experiment.dataset_ID)

    # For ROC plot
    y_true = []
    for split_train_test in dataset.splits_train_test:
        for patient_idx in split_train_test["test"]:
            y_true.append(dataset.dicts[patient_idx]["predictor label"][0])
    y_true = np.array(y_true)
    list_of_y_pred = []
    list_of_model_names = []

    # Plot the results.
    with open(experiment.parent_path + "/detail_scores.txt", "a") as txt_file_detail, open(experiment.parent_path + "/short_scores.txt", "a") as txt_file_short:
        for experiment_name in experiment.experiment_results.keys():  # For each model,
            txt_file_detail.write(
                f"\t============================{experiment_name}============================\n")
            txt_file_short.write(
                f"\t============================{experiment_name}============================\n")
            if verbose >= 1:
                print(
                    f"\t============================{experiment_name}============================")
            scores_dict_short = {"accuracy": [], "precision": {prediction_label: [] for prediction_label in dataset.prediction_labels_bag}, "recall": {prediction_label: [
            ] for prediction_label in dataset.prediction_labels_bag}, "F1 score": {prediction_label: [] for prediction_label in dataset.prediction_labels_bag}}  # "all" for all k splits.
            feature_importance_dict_merged_across_splits = None
            list_of_model_names.append(experiment_name)
            y_pred = []

            # split_train_test = {"train": [...], "test": [...]}.
            for split_train_test, split_train_test_idx in zip(dataset.splits_train_test, range(len(dataset.splits_train_test))):
                txt_file_detail.write(
                    f"\t==============split: {split_train_test_idx}==============\n")
                if verbose >= 2:
                    print(
                        f"\t==============split: {split_train_test_idx}==============")

                # Set the result for this split.
                list_patient_idx = split_train_test["test"]
                predictions_dict = experiment.experiment_results[
                    experiment_name][split_train_test_idx]["predictions_dict"]

                # [vec, vec, ...] -> [scalar, scalar, ...], one-hot encoded vectors to scalar labels.
                if type(predictions_dict["predicted_labels_stack"]) == type([]):
                    # When list, Note that 'inverse = True'.
                    predicted_labels_rounded = [dataset.prediction_label_mapping(
                        prediction, inverse=True) for prediction in predictions_dict["predicted_labels_stack"]]
                else:
                    predicted_labels_rounded = [dataset.prediction_label_mapping(predictions_dict["predicted_labels_stack"][i], inverse=True) for i in range(
                        predictions_dict["predicted_labels_stack"].shape[0])]  # When array.

                # result dict only for this split.
                true_pred_counts = {}  # {'true label 1': {'pred label 1': 4, 'pred label 2': 7, ...}}
                for label_in_bag in dataset.prediction_labels_bag:  # true
                    # pred in true.
                    true_pred_counts[label_in_bag] = {
                        label_in_bag_: 0 for label_in_bag_ in dataset.prediction_labels_bag}

                # Counts the match/mismatch cases.
                for i, patient_idx in zip(range(len(list_patient_idx)), list_patient_idx):
                    true_label = round(dataset.dicts[patient_idx]["outcome"])
                    predicted_label = predicted_labels_rounded[i]
                    true_pred_counts[true_label][predicted_label] += 1
                    y_pred.append(
                        predictions_dict["predicted_labels_stack"][i])

                # Calculate the metrics only for this split.
                accuracy_of_this_split = sum([true_pred_counts[label_in_bag][label_in_bag]
                                              for label_in_bag in dataset.prediction_labels_bag]) / len(list_patient_idx)  # Calculate Accuracy
                precision_of_this_split = {}
                recall_of_this_split = {}
                # label_in_bag is integer; 0 or 1 for COVID; 1, 2, 3 for Alz.
                for label_in_bag_i in dataset.prediction_labels_bag:
                    # Calculate Precision
                    pred_i_sum = sum([true_pred_counts[label_in_bag_j][label_in_bag_i]
                                      for label_in_bag_j in dataset.prediction_labels_bag])
                    if pred_i_sum > 0:
                        precision_of_this_split[label_in_bag_i] = true_pred_counts[label_in_bag_i][label_in_bag_i] / pred_i_sum
                    else:
                        precision_of_this_split[label_in_bag_i] = np.nan
                    # Calculate Recall
                    true_i_sum = sum([true_pred_counts[label_in_bag_i][label_in_bag_j]
                                      for label_in_bag_j in dataset.prediction_labels_bag])
                    if true_i_sum > 0:
                        recall_of_this_split[label_in_bag_i] = true_pred_counts[label_in_bag_i][label_in_bag_i] / true_i_sum
                    else:
                        recall_of_this_split[label_in_bag_i] = np.nan
                result_of_this_split_str = f"\tAccuracy: {accuracy_of_this_split},\n\tPrecision: {precision_of_this_split},\n\tRecall: {recall_of_this_split}"

                if verbose >= 2:
                    print(result_of_this_split_str)
                txt_file_detail.write(
                    f"{result_of_this_split_str}\n{true_pred_counts}\n")
                scores_dict_short["accuracy"].append(accuracy_of_this_split)
                # Precision, Recall for each label.
                for label_in_bag in dataset.prediction_labels_bag:
                    scores_dict_short["precision"][label_in_bag].append(
                        precision_of_this_split[label_in_bag])
                    scores_dict_short["recall"][label_in_bag].append(
                        recall_of_this_split[label_in_bag])
                    scores_dict_short["F1 score"][label_in_bag].append((2 * precision_of_this_split[label_in_bag] * recall_of_this_split[label_in_bag] / (
                        precision_of_this_split[label_in_bag] + recall_of_this_split[label_in_bag])) if (
                        precision_of_this_split[label_in_bag] + recall_of_this_split[label_in_bag]) != 0 else 0)

                # Collect and merge the feature importance, for this split.
                if num_top_features is not None and "feature_importance_dict" in predictions_dict.keys():
                    if feature_importance_dict_merged_across_splits is None:
                        feature_importance_dict_merged_across_splits = deepcopy(
                            predictions_dict["feature_importance_dict"])
                    else:
                        for method in predictions_dict["feature_importance_dict"].keys():
                            for feature_group in predictions_dict["feature_importance_dict"][method].keys():
                                for feature_name in predictions_dict["feature_importance_dict"][method][feature_group].keys():
                                    feature_importance_dict_merged_across_splits[method][feature_group][feature_name] = feature_importance_dict_merged_across_splits[
                                        method][feature_group][feature_name] + deepcopy(predictions_dict["feature_importance_dict"][method][feature_group][feature_name])

                # Plot loss graph to see convergency, if exists.
                if "loss_dict" in experiment.experiment_results[experiment_name][split_train_test_idx].keys() and experiment.experiment_results[experiment_name][split_train_test_idx]["loss_dict"] is not None and num_loss_plotting_points >= 0:
                    loss_dict = experiment.experiment_results[experiment_name][split_train_test_idx]["loss_dict"]
                    dir_to_save_loss = visualizations_patent_path + \
                        f"/losses/{experiment_name}_{split_train_test_idx}/"
                    os.mkdir(dir_to_save_loss)
                    for loss_name, loss_list in loss_dict.items():
                        squeezed_list = utilsforminds.containers.squeeze_list_of_numbers_with_average_of_each_range(
                            list_of_numbers=loss_list, num_points_in_list_out=num_loss_plotting_points)
                        utilsforminds.visualization.plot_xy_lines(range(len(squeezed_list)), [
                                                                  {"label": loss_name, "ydata": squeezed_list}], f'{dir_to_save_loss}{loss_name}.eps', save_tikz=False)

            # For ROC curve plot, for this model.
            list_of_y_pred.append(np.array(y_pred))

            # Calculates the statistics for all the splits and for this model. scores_dict_short = {"accuracy": [...], "precision": {1: [...], 2: [...]}, "recall": {1: [...], 2: [...]}}
            formatted_str = f"\taccuracy. mean: {round(mean(scores_dict_short['accuracy']), 4)}, std: {round(std(scores_dict_short['accuracy']), 6)}, scores: {scores_dict_short['accuracy']}\n"
            if verbose >= 1:
                print(formatted_str)
            txt_file_detail.write(formatted_str)
            txt_file_short.write(formatted_str)
            for metric in ["precision", "recall", "F1 score"]:
                for label_in_bag in dataset.prediction_labels_bag:
                    formatted_str = f"\t{metric}_{label_in_bag}. mean: {round(mean(scores_dict_short[metric][label_in_bag]), 4)}, std: {round(std(scores_dict_short[metric][label_in_bag]), 6)}, scores: {scores_dict_short[metric][label_in_bag]}\n"
                    if verbose >= 1:
                        print(formatted_str)
                    txt_file_detail.write(formatted_str)
                    txt_file_short.write(formatted_str)

            # --- --- Plot Feature Importance.
            # feature_importance_dict_merged_across_splits[method][feature_group][feature_name] = [changes_on_prediction, changes_on_prediction, ...]
            if feature_importance_dict_merged_across_splits is not None:
                # --- Common plotting for all dataset.
                # if dataset.dataset_kind == "covid" or dataset.dataset_kind == "alz":
                for method in feature_importance_dict_merged_across_splits.keys():
                    for feature_group in feature_importance_dict_merged_across_splits[method].keys():
                        feature_importance_dict_this_group = feature_importance_dict_merged_across_splits[
                            method][feature_group]
                        # Plot if at least one importance exists.
                        if len(next(iter(feature_importance_dict_this_group.values()))) >= 1:
                            # Get name, mean, std of each feature importance.
                            num_top_features_local = min(num_top_features, len(
                                feature_importance_dict_this_group))
                            name_mean_std_list = [[feature_name, mean(feature_importance_dict_this_group[feature_name], default=0.), std(
                                feature_importance_dict_this_group[feature_name], default=0.)] for feature_name in feature_importance_dict_this_group.keys()]  # [[name, mean, std], [name, mean, std], ...]
                            # Cut top num_top_features features.
                            name_mean_std_list.sort(
                                key=lambda x: x[1], reverse=True)  # sort by mean
                            feature_importance_names = [
                                name_mean_std_list[i][0] for i in range(num_top_features_local)]
                            feature_importance_means = [
                                name_mean_std_list[i][1] for i in range(num_top_features_local)]
                            # Standard Deviation across the patients in test set.
                            feature_importance_stds = [
                                name_mean_std_list[i][2] for i in range(num_top_features_local)]

                            utilsforminds.visualization.plot_bar_charts(path_to_save=f"{visualizations_patent_path}/feature_importance/{experiment_name}_{feature_group}_{method}_with_std.eps", name_numbers={"Mean": feature_importance_means}, xlabels=feature_importance_names, xlabels_for_names=None, xtitle=None, ytitle="Importance", bar_width='auto',
                                                                        alpha=0.8, colors_dict=None, format='eps', diagonal_xtickers=False, name_errors={"Mean": feature_importance_stds}, name_to_show_percentage=None, name_not_to_show_percentage_legend=None, fontsize=12, title=None, figsize=None, ylim=None, fix_legend=True, plot_legend=False, save_tikz=False, horizontal_bars=True)

                # --- Additional plotting dedicated to the specific dataset.
                if dataset.dataset_kind == "alz":
                    for method in feature_importance_dict_merged_across_splits.keys():
                        # --- --- SNP plot.
                        feature_importance_dict_snp = feature_importance_dict_merged_across_splits[
                            method]["SNP"]
                        names_SNPs = list(dataset.df_dict["snp"].columns)[1:]
                        # Plot if at least one importance exists.
                        if len(next(iter(feature_importance_dict_snp.values()))) >= 1:
                            # plottable SNPs < 1224 total SNPs.
                            importance_of_snp_at_idc_chr_not_X_list = []
                            for snp_idx in dataset.idc_chr_not_X_list:  # For each snp feature.
                                importance_of_snp_at_idc_chr_not_X_list.append(
                                    mean(feature_importance_dict_snp[names_SNPs[snp_idx]]))
                            weights_appended_SNPs_info_df = dataset.reordered_SNPs_info_df.copy()
                            weights_appended_SNPs_info_df["Importance"] = importance_of_snp_at_idc_chr_not_X_list

                            # --- Actual plots.
                            # Scatter plots.
                            utilsforminds.visualization.plot_group_scatter(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/chromosome_{experiment_name}_{method}.eps",
                                                                           xlabel='Chromosome', group_column="chr", y_column="Importance", color_column="chr_colors", group_sequence=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 17, 19, 20, 21])

                            utilsforminds.visualization.plot_group_scatter(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/alzgene_{experiment_name}_{method}.eps", xlabel='AlzGene', group_column="AlzGene", y_column="Importance", color_column=None, rotation_xtickers=45, group_sequence=[
                                                                           'MTHFR', 'ECE1', 'CR1', 'LDLR', 'IL1B', 'BIN1', 'TF', 'NEDD9', 'LOC651924', 'TFAM', 'CLU', 'IL33', 'DAPK1', 'SORCS1', 'GAB2', 'PICALM', 'SORL1', 'ADAM10', 'ACE', 'PRNP'])

                            utilsforminds.visualization.plot_group_scatter(
                                weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/location_{experiment_name}_{method}.eps", xlabel='location', group_column="location", y_column="Importance", color_column=None, rotation_xtickers=45)

                            utilsforminds.visualization.plot_group_scatter(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/identified_group_{experiment_name}_{method}.eps",
                                                                           xlabel='Identified Group', group_column="identified_group", y_column="Importance", color_column=None, rotation_xtickers=45, num_truncate_small_groups=140)

                            # Bar charts plots for individual SNPs.
                            for group_column in ['chr', 'AlzGene', 'location', 'identified_group']:
                                utilsforminds.visualization.plot_top_bars_with_rows(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/topbars_SNP_{group_column}_{experiment_name}_{method}.eps", colors_rotation=None, color_column=group_column +
                                                                                    "_colors", order_by="Importance", x_column="SNP", show_group_size=False, xlabel="SNP", ylabel="Importance", num_bars=10, num_rows=2, re_range_max_min_proportion=None, rotation_xtickers=45, save_tikz=False)

                            utilsforminds.visualization.plot_top_bars_with_rows(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/topbars_SNP_{experiment_name}_{method}.eps", colors_rotation=[
                                                                                "blue"], order_by="Importance", x_column="SNP", show_group_size=False, xlabel="SNP", ylabel="Importance", num_bars=10, num_rows=3, re_range_max_min_proportion=None, rotation_xtickers=45)

                            # Bar charts plots for each group.
                            for color_column, group_column, xlabel, rotation_xtickers in zip(['chr_colors', 'AlzGene_colors', 'location_colors', 'identified_group_colors'], ["chr", "AlzGene", "location", "identified_group"], ["Chromosome", "AlzGene", "Location", "Identified Group"], [45, 45, 45, 0]):
                                utilsforminds.visualization.plot_top_bars_with_rows(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/topbars_{xlabel}_{experiment_name}_{method}.eps", color_column=color_column,
                                                                                    order_by="Importance", group_column=group_column, xlabel=xlabel, ylabel="Importance", num_bars=10, num_rows=2, re_range_max_min_proportion=None, rotation_xtickers=rotation_xtickers)

                        # MRI Region Of Interest plot.
                        labels_path_dict = {"FS": f"{dataset.path_to_dataset}/fs_atlas_labels.csv",
                                            "VBM": f"{dataset.path_to_dataset}/longitudinal imaging measures_VBM_mod_final.xlsx"}
                        for dynamic_modality in ["FS", "VBM"]:
                            feature_importance_dict_dynamic_modality = feature_importance_dict_merged_across_splits[
                                method][dynamic_modality]
                            # Plot if at least one importance exists.
                            if len(next(iter(feature_importance_dict_dynamic_modality.values()))) >= 1:
                                # For each dynamic feature.
                                feature_importance_weights = [mean(feature_importance_dict_dynamic_modality[feature_name])
                                                              for feature_name in dataset.dynamic_feature_names_for_each_modality[dynamic_modality]]
                                draw_brains(weights_of_rois=feature_importance_weights,
                                            path_to_save=f"{visualizations_patent_path}/feature_importance/{dynamic_modality}_{experiment_name}_{method}.png", path_to_labels=labels_path_dict[dynamic_modality], modality=dynamic_modality)
    # Plot the ROC curve
    utilsforminds.visualization.plot_ROC(path_to_save=visualizations_patent_path + "/ROC_curve/roc_curve", y_true=y_true, list_of_y_pred=list_of_y_pred, list_of_model_names=list_of_model_names,
                                         list_of_class_names=dataset.class_names, title='Receiver operating characteristic', xlabel='False Positive Rate', ylabel='True Positive Rate', colors=None, linewidth=1, extension="eps")

    print(f"Experimental results are saved in {experiment.parent_path}")


def vector_with_one_hot_encoding(vector):
    """Find the index of maximum element, and set it to 1, and set others to 0

    Examples
    --------
    print(vector_with_one_hot_encoding([0.3, -0.1, 0., 0.5, 0.2]))
        [0, 0, 0, 1, 0]
    print(vector_with_one_hot_encoding([1.5, -0.4, 1.5, 0.5, 0.4]))
        [1, 0, 0, 0, 0]
    """

    if type(vector) == type([]) or type(vector) == type(()):  # list or tuple
        num_elements = len(vector)
    else:  # Array
        num_elements = vector.shape[0]
    encoded = [0 for i in range(num_elements)]
    encoded[np.argmax(vector)] = 1
    return encoded


def draw_brains(weights_of_rois, path_to_save, path_to_labels, modality="FS"):
    """Plot the Region Of Interests from FS or VBM rois in ADNI dataset.

    Attributes
    ----------
    path_to_labels : str
        Possibly one of "./inputs/alz/data/fs_atlas_labels.csv" or "./inputs/alz/data/longitudinal imaging measures_VBM_mod_final.xlsx"
    weights_of_rois : list
        List of weights on 90 rois of VBM or FS.
    """
    if any(pd.isna(weights_of_rois)):
        print(f"WARNING: This function draw_brains encounters invalid numbers such as nan/inf, so will be passed and do nothing.")
        return None

    if modality == "FS":
        fs_roi_map = FSRegionOfInterestMap()
        fs_label_df = pd.read_csv(path_to_labels)
        for index, row in fs_label_df.iterrows():
            atlas = row["Atlas"]
            rois = row[atlas].split("+")
            for roi in rois:
                fs_roi_map.add_roi(roi, weights_of_rois[index], atlas)
        fs_roi_map.build_map(smoothed=True)
        fs_roi_map.save(path_to_save, "FS")  # label at upper left box.
    elif modality == "VBM":
        # Load vbm labels.
        vbm_cleaner = VBMDataCleaner()
        vbm_cleaner.load_data(path_to_labels)
        vbm_labels = vbm_cleaner.clean()

        # plot VBM
        vbm_roi_map = VBMRegionOfInterestMap()
        for label, weight in zip(vbm_labels, weights_of_rois):
            vbm_roi_map.add_roi(label, weight)

        vbm_roi_map.build_map(smoothed=True)
        # vbm_roi_map.plot(time)
        vbm_roi_map.save(path_to_save, "VBM")  # label at upper left box.
    else:
        raise Exception(NotImplementedError)


if __name__ == "__main__":
    pass
    print(1.0 == 1)
