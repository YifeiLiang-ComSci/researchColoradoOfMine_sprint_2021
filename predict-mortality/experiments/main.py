from keras.layers import ReLU, Dropout, BatchNormalization, Concatenate, RepeatVector, Dense, LSTM, Input, LeakyReLU, Lambda, GRU, SimpleRNN, Flatten, Conv2DTranspose, ConvLSTM2D, MaxPooling3D, TimeDistributed, Reshape, Subtract, Add, Multiply
import numpy as np
from keras import activations
from keras import layers
from keras.layers import LeakyReLU, ReLU
from keras import regularizers
from tensorflow.keras import optimizers
from utilsforminds.containers import GridSearch, Grid
import estimator
import utils
import pickle
import autoencoder
import data_prep
import tensorflow
from numpy.random import seed
seeds = {"np": 2, "tf": 4}
seed(seeds["np"])
tensorflow.random.set_seed(seeds["tf"])


basic_regularizer = None  # regularizers.l1(0.000)
# activations.tanh, lambda x: activations.relu(x, alpha = 0.1)
def basic_activation(x): return activations.relu(x, alpha=0.1)


num_neurons_increase_factor = 1
# ["Dropout", {"rate": 0.5}], ["BatchNormalization", {}]
dropout_or_batchnormalization_layer = None
# Do NOT use dropout or BatchNormalization, they don't work with train_on_batch.
assert(dropout_or_batchnormalization_layer is None)
# assert(basic_regularizer is None) ## Do not use regularizer, slightly better improvements.


dataset_kind = "chestxray"  # covid, challenge, alz, toy, chestxray

dataset_path = f"./datasets/{dataset_kind}.obj"

if False:  # Whether you want to create and save dataset.
    # --- --- Create and Save dataset, comment below if you want to wait dataset creation.
    # Best hyper-parameters found

    if dataset_kind == "covid":

        # --- COVID DATASET.
        dataset_obj = data_prep.Dataset(path_to_dataset="./inputs/time_series_375_prerpocess_en.xlsx", dataset_kind=dataset_kind, init_number_for_unobserved_entry=-1.0, static_data_input_mapping=dict(RNN_vectors=[], static_encoder=[], raw=["age", "gender"]), kwargs_dataset_specific=dict(
            excluded_features=[]))  # separate_dynamic_static_features = "combine", "separate raw", "separate enrich", raw = ["age", "gender", 'Admission time', 'Discharge time'], excluded_features= ["High sensitivity C-reactive protein", "Lactate dehydrogenase", "(%)lymphocyte"]

    elif dataset_kind == "chestxray":
        apply_segment = False
        # --- COVID DATASET.
        dataset_obj = data_prep.Dataset(path_to_dataset="/Users/yifeiliang/Downloads/covid-chestxray-dataset", dataset_kind=dataset_kind, init_number_for_unobserved_entry=-10.0, static_data_input_mapping=dict(RNN_vectors=[], static_encoder=[], raw=["sex", "age", "RT_PCR_positive"]),
                                        kwargs_dataset_specific=dict(excluded_features=["intubation_present", "in_icu", "date", "location", "folder", "doi", "url", "license", "clinical_notes", "other_notes", "Unnamed: 29"], image_shape=(400, 400), apply_segment=apply_segment))  # excluded_features= ["Unnamed: 29"]
        if apply_segment:
            dataset_path = dataset_path[:-4] + "_segment.obj"
        else:
            dataset_path = dataset_path[:-4] + "_non-segment.obj"

    elif dataset_kind == "challenge":
        # --- CHALLENGE DATASET
        dataset_obj = data_prep.Dataset(path_to_dataset="./inputs/physionet-challenge", dataset_kind=dataset_kind, init_number_for_unobserved_entry=0.0,
                                        static_data_input_mapping=dict(RNN_vectors=[], static_encoder=[], raw=["Age", "Gender", 'Height', "CCU", "CSRU", "SICU"]), kwargs_dataset_specific=dict())

    elif dataset_kind == "alz":
        # --- ALZ DATASET.
        dataset_obj = data_prep.Dataset(path_to_dataset="./inputs/alz/data/", dataset_kind=dataset_kind, init_number_for_unobserved_entry=-10.0, static_data_input_mapping=dict(RNN_vectors=[], static_encoder=["SNP"], raw=[
                                        "BL_Age", "Gender"]), kwargs_dataset_specific=dict(TIMES=["BL", "M6", "M12", "M18", "M24"], target_label="M36_DX", dynamic_modalities_to_use=["FS", "VBM", "RAVLT", "ADAS", "FLU", "MMSE", "TRAILS"]))

    # --- COMMON for all DATASET.
    # For toy dataset: kwargs_toy_dataset = {"num_patients": 385, "max_time_steps": 80, "num_features_static_dict": dict(raw = 1, RNN = 1, static_encoder = 1), "num_features_dynamic": 2, "time_interval_range": [0., 0.05], "static_data_range": [0., 1.], "sin function info ranges dict": {"amplitude": [0., 1.], "displacement_along_x_axis": [0., 1.], "frequency": [0.5, 1.], "displacement_along_y_axis": [1., 2.]}, "observation noise range": [0., 0.], "missing probability": 0.0, "class_0_proportion": [0.5]}
    dataset_obj.set_dicts(kwargs_toy_dataset=None)
    # Save dataset.
    with open(dataset_path, "wb") as dataset_file:
        pickle.dump(dataset_obj, dataset_file)
        print("Done Uploading")

# Load dataset.
dataset_obj = pickle.load(open("datasets/chestxray_non-segment.obj", "rb"))
# dataset_obj.export_to_csv(dir_path_to_export = "./datasets/datasets_in_csv/")

# split the dataset.
# dataset_obj.reduce_number_of_participants(reduced_number_of_participants = 150) ## Reduce the number of participants for saving time. This can be used for fast debugging or grid search, not for the actual experiment. Comment out this line if you don't want to reduce.
dataset_obj.split_train_test(train_proportion=None, k_fold=5,
                             whether_training_is_large=False, shuffle=True, balance_label_distribution=True)

# Getting statistics
# sum([i * (dataset_obj.feature_min_max_dict["BL_Age"]["max"] - dataset_obj.feature_min_max_dict["BL_Age"]["min"]) + dataset_obj.feature_min_max_dict["BL_Age"]["min"] for i in dataset_obj.get_data_collections_from_all_samples(data_access_function = lambda x: x["raw"]["data"][0][0])]) / 379
# for group, feature in zip(["static"], ["Age"]):
#     observed_numbers = dataset_obj.get_measures_of_feature(feature_name = feature, group_name = group)
#     print(f"{feature}, mean: {observed_numbers.mean()}, std: {observed_numbers.std()}")
# for group, feature in zip(["static"], ["Gender"]):
#     observed_numbers = dataset_obj.get_measures_of_feature(feature_name = feature, group_name = group)
#     unique, counts = np.unique(observed_numbers, return_counts=True)
#     print(f"{feature}, item: occurence = {dict(zip(unique, counts))}")
# unique, counts = np.unique(np.array(dataset_obj.dataframe_static["In-hospital_death"]), return_counts=True)
# print(f"In-hospital_death, item: occurence = {dict(zip(unique, counts))}")

# --- --- Conduct the Experiment
experiment = utils.Experiment(base_name=dataset_kind + "-")

# ------------------------- COVID Dataset Learning -------------------------
# Best hyper-parameters found
# learning_rate= 0.00001, +++ 0.0003 +++
# factors_dict = {"reconstruction loss": 5 * 1e-3, "prediction": 1e-1} 90.91%
# Decoder First layer "activation": lambda x: activations.relu(x, alpha = 0.1).
# Recon, Predictor loss: 2, 2 > 3, 3 (97) > 2, 3. (88.61) > 2, 2. (86) > 3, 3 (86) > 2, binary cross entropy
# kwargs_RNN_vectors= {"units": 105}, kwargs_RNN_vectors= {"units": 80} > {"units": 100}
# factors_dict = {"reconstruction loss": 1 * 1e-1}

if dataset_kind == "covid":
    experiment_settings_list = [
        GridSearch(dict(model_class=autoencoder.Autoencoder, init=dict(debug=1, verbose=1, whether_use_mask=True),
                        fit=dict(
            iters=200, factors_dict={"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer=optimizers.Adam(learning_rate=0.0003), loss_kind_dict={"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

            predictor_structure_list=[["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], [
                "Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list=[["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha=0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], [
                "Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            model_name_RNN_vectors="LSTM",
            kwargs_RNN_vectors={"units": 60, "activity_regularizer": basic_regularizer,
                                "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"},

            static_encoder_decoder_structure_dict=None, loss_wrapper_funct=None
        ),
            predict=dict(perturbation_info_for_feature_importance={"center": 0., "proportion": 0.1}, swap_info_for_feature_importance={"proportion": 0.7}, feature_importance_calculate_prob_dict={"dynamic": 1.0, "static": 1.0}), fit_on="both", name='SAE'
        ), key_of_name="name"),

        GridSearch(dict(model_class=autoencoder.Autoencoder, init=dict(debug=1, verbose=1, whether_use_mask=True),
                        fit=dict(
            iters=200, factors_dict={"dynamic vectors reconstruction loss": 0., "static reconstruction loss": 0., "prediction": 1e-1}, optimizer=optimizers.Adam(learning_rate=0.0003), loss_kind_dict={"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

            predictor_structure_list=[["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], [
                "Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list=[["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha=0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], [
                "Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            model_name_RNN_vectors="LSTM",
            kwargs_RNN_vectors={"units": 60, "activity_regularizer": basic_regularizer,
                                "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"},

            static_encoder_decoder_structure_dict=None, loss_wrapper_funct=None
        ),
            predict=dict(perturbation_info_for_feature_importance={"center": 0., "proportion": 0.1}, swap_info_for_feature_importance={"proportion": 0.7}, feature_importance_calculate_prob_dict={"dynamic": 1.0, "static": 1.0}), fit_on="train", name='BLSTM'
        ), key_of_name="name"),

        GridSearch(dict(model_class=estimator.MLP,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="DNN")),

        GridSearch(dict(model_class=estimator.RandomForest,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="RF")),

        GridSearch(dict(model_class=estimator.RidgeClassifier,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="RC")),

        GridSearch(dict(model_class=estimator.SVM,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="SVM")),

    ]

# ------------------------- Physionet-Challenge Dataset Learning -------------------------
# iters = 200 < 150 (150 is better than 200), 100 < 50.
# {"dynamic vectors reconstruction loss": Grid(1.0, +++ 2.0 +++), "static reconstruction loss": 2.0, "prediction": Grid("binary cross entropy", +++ 2.0 +++)}
# Grid(30, 60, +++ 90 +++)
# "prediction": 1e+2 ~ 1e+1
# iters = +++ 150 +++, 50
# optimizer = Grid(optimizers.Adam(+++ learning_rate= 0.0003 +++, 1e-5))
# init_number_for_unobserved_entry = +++ 0.0 +++,
# optimizer = Grid(optimizers.Adam(learning_rate= 0.0003), +++ optimizers.Adamax(learning_rate= 0.0003) +++, optimizers.Nadam(learning_rate= 0.0003), optimizers.Adagrad(learning_rate= 0.0003))

# GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True),
#         fit = dict(
#             iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

#             predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(21 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             model_name_RNN_vectors= "GRU",
#             kwargs_RNN_vectors= {"units": 30, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}, # , "activation": "tanh"

#             static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
#         ),
#         predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SAE'
#         ), key_of_name= "name")

if dataset_kind == "challenge":
    experiment_settings_list = [
        GridSearch(dict(model_class=autoencoder.Autoencoder, init=dict(debug=1, verbose=1, whether_use_mask=True),
                        fit=dict(
            iters=30, factors_dict={"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer=optimizers.Adamax(learning_rate=0.0003), loss_kind_dict={"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "SquaredHinge"},

            predictor_structure_list=[["RandomFourierFeatures", dict(output_dim=400, scale=10.0, kernel_initializer="gaussian")], ["Dense", {"units": len(
                dataset_obj.prediction_labels_bag), "activation": "linear", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list=[["Dense", {"units": int(21 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": basic_activation,
                                                                                                                                                                                                                                              "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            model_name_RNN_vectors="GRU",
            kwargs_RNN_vectors={"units": 30, "activity_regularizer": basic_regularizer,
                                "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0},  # , "activation": "tanh"

            static_encoder_decoder_structure_dict=None, loss_factor_funct=(lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        ),
            predict=dict(perturbation_info_for_feature_importance={"center": 0., "proportion": 0.1}, swap_info_for_feature_importance={"proportion": 0.7}, feature_importance_calculate_prob_dict={"dynamic": 0.5, "static": 1.0}), fit_on="both", name='SAE'
        ), key_of_name="name"),


        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True),
        # fit = dict(
        #     iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(21 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     model_name_RNN_vectors= "GRU",
        #     kwargs_RNN_vectors= {"units": 30, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}, # , "activation": "tanh"

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SAE'
        # ), key_of_name= "name"),
        # # lambda x: x * np.array([[1., (3446 / 554) ** 2]]), (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.)

        # ## Reference Models
        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True),
        # fit = dict(
        #     iters = 70, factors_dict = {"dynamic vectors reconstruction loss": 0.0, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(21 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     model_name_RNN_vectors= "GRU",
        #     kwargs_RNN_vectors= {"units": 30, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}, # , "activation": "tanh"

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "train", name = 'BLSTM'
        # ), key_of_name= "name"),

        GridSearch(dict(model_class=estimator.RidgeClassifier,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="RC")),

        GridSearch(dict(model_class=estimator.RandomForest,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="RF")),

        GridSearch(dict(model_class=estimator.SVM,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="SVM")),

        GridSearch(dict(model_class=estimator.MLP,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="MLP")),
    ]

# ------------------------- chest-xray dataset -------------------------
# iters = 200 < 150 (150 is better than 200), 100 < 50.
# {"dynamic vectors reconstruction loss": Grid(1.0, +++ 2.0 +++), "static reconstruction loss": 2.0, "prediction": Grid("binary cross entropy", +++ 2.0 +++)}
# Grid(30, 60, +++ 90 +++)
if dataset_kind == "chestxray":
    # experiment_settings_list = [
    #     GridSearch(dict(model_class=autoencoder.Autoencoder, init=dict(debug=1, verbose=1, whether_use_mask=True),
    #                     fit=dict(
    #         iters=50, factors_dict={"dynamic reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": Grid(1e+1)}, optimizer=optimizers.Adam(learning_rate=1e-5), loss_kind_dict={"dynamic reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

    #         predictor_structure_list=[["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(30 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], [
    #             "Dense", {"units": int(10 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

    #         dynamic_decoder_structure_list=[["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha=0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(70 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], [
    #             "Dense", {"units": int(50 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

    #         model_name_RNN="LSTM",
    #         kwargs_RNN={"units": 90, "activity_regularizer": basic_regularizer,
    #                     "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"},

    #         static_encoder_decoder_structure_dict=None,
    #     ),
    #         predict=dict(perturbation_info_for_feature_importance={"center": 0., "proportion": k0.1}, swap_info_for_feature_importance={"proportion": 0.7}, feature_importance_calculate_prob_dict={"dynamic": 0.1, "static": 0.1}), fit_on="both", name='SAE'
    #     ), key_of_name="name"),

    #     # Reference Models
    # ]
    class_weight = Grid({0: 456/(2 * (278 + 68)),
                         1: ((456) / (2*(50)))})
    experiment_settings_list = [
        GridSearch(dict(model_class=estimator.CNN, init=dict(),
                        fit=dict(epochs=Grid(70),
                                 class_weight=class_weight),
                        predict=dict(), fit_on="train", name='CNN'), key_of_name="name")
        # GridSearch(dict(model_class=estimator.RidgeClassifier,
        #                 init=dict(), fit=dict(), predict=dict(), fit_on="train", name="RC")),

        # GridSearch(dict(model_class=estimator.RandomForest,
        #                 init=dict(), fit=dict(), predict=dict(), fit_on="train", name="RF")),

        # GridSearch(dict(model_class=estimator.SVM,
        #                 init=dict(), fit=dict(), predict=dict(), fit_on="train", name="SVM")),

        # GridSearch(dict(model_class=estimator.MLP,
        #                 init=dict(), fit=dict(), predict=dict(), fit_on="train", name="MLP")),


        # Reference Models
    ]

# ------------------------- Alzheimer's Disease Dataset Learning -------------------------
# Good hyperparameters
# loss_kind_dict["prediction"] : binary > 2.0
# optimizer = optimizers.Adam(learning_rate= 0.00001): learning_rate= 0.001 > learning_rate= 0.00001
# The larger iterations is better for SAE.
# GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True),
#         fit = dict(
#             iters = 150, factors_dict = {"dynamic vectors reconstruction loss": 0.5 * 1e-1, "static reconstruction loss": 1 * 1e-1, "prediction": 1e+2}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "binary cross entropy"},

#             predictor_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(33 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             dynamic_vectors_decoder_structure_list=
#             [["Dense", {"units": int(75 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer":  basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             model_name_RNN_vectors= "LSTM",
#             kwargs_RNN_vectors= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"},

#             static_encoder_decoder_structure_dict =
#             dict(encoder= [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 80, "activation": "tanh", "activity_regularizer":  basic_regularizer}]],
#             decoder = [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),
#         ),
#         predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.1, "static": 0.05}), fit_on = "both", name = 'SAE'
#         ), key_of_name= "name")
if dataset_kind == "alz":
    experiment_settings_list = [

        GridSearch(dict(model_class=autoencoder.RidgeClassifier, init=dict(debug=1, verbose=1, whether_use_mask=True),
                        fit=dict(
            iters=1400, factors_dict={"dynamic vectors reconstruction loss": 1, "static reconstruction loss": 1e+1, "prediction": 1e+2}, optimizer=optimizers.Adam(learning_rate=0.0003), loss_kind_dict={"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "binary cross entropy"},

            predictor_structure_list=[["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {
                "units": int(33 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list=[["Dense", {"units": int(75 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha=0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer,
                                                                                                                                                                                                                                            "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer":  basic_regularizer, "bias_regularizer": basic_regularizer}]],

            model_name_RNN_vectors="LSTM",
            kwargs_RNN_vectors={"units": 64, "activity_regularizer": basic_regularizer,
                                "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"},

            static_encoder_decoder_structure_dict=dict(encoder=[["Dense", {"units": 300, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 80, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ],
                                                       decoder=[["Dense", {"units": 300, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),
        ),
            predict=dict(perturbation_info_for_feature_importance={"center": 0., "proportion": 0.1}, swap_info_for_feature_importance={"proportion": 0.7}, feature_importance_calculate_prob_dict={"dynamic": 0.2, "static": 0.05}), fit_on="both", name='SAE'
        ), key_of_name="name"),

        dict(model_class=autoencoder.Autoencoder, init=dict(debug=1, verbose=1, whether_use_mask=True),
             fit=dict(
            iters=1400, factors_dict={"dynamic vectors reconstruction loss": 0, "static reconstruction loss": 0, "prediction": 1e+2}, optimizer=optimizers.Adam(learning_rate=0.0003), loss_kind_dict={"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "binary cross entropy"},

            predictor_structure_list=[["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {
                "units": int(33 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list=[["Dense", {"units": int(75 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha=0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer,
                                                                                                                                                                                                                                            "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer":  basic_regularizer, "bias_regularizer": basic_regularizer}]],

            model_name_RNN_vectors="LSTM",
            kwargs_RNN_vectors={"units": 64, "activity_regularizer": basic_regularizer,
                                "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"},

            static_encoder_decoder_structure_dict=dict(encoder=[["Dense", {"units": 300, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 80, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ],
                                                       decoder=[["Dense", {"units": 300, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),
        ),
            predict=dict(perturbation_info_for_feature_importance={"center": 0., "proportion": 0.1}, swap_info_for_feature_importance={"proportion": 0.7}, feature_importance_calculate_prob_dict={"dynamic": 0.2, "static": 0.05}), fit_on="train", name='BLSTM'
        ),

        GridSearch(dict(model_class=estimator.RandomForest,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="RF")),

        GridSearch(dict(model_class=estimator.MLP,
                        init=dict(), fit=dict(), predict=dict(), fit_on="train", name="DNN")),
    ]

# ------------------------- Toy Dataset Learning -------------------------
if dataset_kind == "toy":
    experiment_settings_list = [
        GridSearch(dict(model_class=autoencoder.Autoencoder, init=dict(debug=1, verbose=1, whether_use_mask=Grid(True, False)),
                        fit=dict(
            iters=10, factors_dict={"dynamic vectors reconstruction loss": 0.5 * 1e-1, "static reconstruction loss": 1 * 1e-1, "prediction": 1e-1}, optimizer=optimizers.Adam(learning_rate=0.001), loss_kind_dict={"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

            predictor_structure_list=[["Dense", {"units": int(12 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(6 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], [
                "Dense", {"units": int(2 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list=[["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha=0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(10 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {
                "units": int(7 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer":  basic_regularizer, "bias_regularizer": basic_regularizer}]],

            model_name_RNN_vectors="LSTM",
            kwargs_RNN_vectors={"units": 12, "activity_regularizer": basic_regularizer,
                                "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"},

            static_encoder_decoder_structure_dict=dict(encoder=[["Dense", {"units": 60, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 30, "activation": "tanh", "activity_regularizer":  basic_regularizer}]],
                                                       decoder=[["Dense", {"units": 60, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),
        ),
            predict=dict(perturbation_info_for_feature_importance={"center": 0., "proportion": 0.1}, swap_info_for_feature_importance={"proportion": 0.7}, feature_importance_calculate_prob_dict={"dynamic": 0.1, "static": 0.05}), fit_on="both", name='SAE'
        ), key_of_name="name"),
    ]

# ------------------------- Do Experiment and Plot the Results. -------------------------

# Conduct Experiments
experiment.set_experimental_result(dataset=dataset_obj, experiment_settings_list=experiment_settings_list,
                                   save_result=True, note=str(seeds), dataset_path=dataset_path)

# Plot experimental results.
utils.plot_experimental_results(experiment=experiment, dataset=dataset_obj,
                                num_loss_plotting_points=200, num_top_features=15, verbose=2)

print("FINISHED.")
