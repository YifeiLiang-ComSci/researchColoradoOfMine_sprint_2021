from keras import backend as K
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers.convolutional import Conv2D
import utilsforminds.numpy_array as numpy_array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import torch
import math


from tensorflow.keras import datasets, layers, models


class Estimator():
    name = "model name"

    def __init__(self):
        """This is required method."""
        pass

    def fit(self, dataset, indices_of_patients, x_train, y_train):
        """This method is only required for trainable estimators."""
        pass

    def predict(self, dataset, indices_of_patients, x_test):
        """This is required method."""

        return {"predicted_labels_stack": ["list of numbers predicted"]}


class RidgeClassifier():
    name = "Ridge Classifier"

    def __init__(self):
        """This is required method."""
        self.model = sklearn.linear_model.RidgeClassifier()

    def fit(self, x_train, y_train, **kwarg):
        """This method is only required for trainable estimators."""
        y_train_single_label, self.encode_list = numpy_array.inverse_one_hot_encode(
            y_train, return_encode_list=True)
        self.model.fit(x_train, y_train_single_label)

    def predict(self, x_test, **kwarg):
        """This is required method."""
        prediction_dict = {}
        decision_values = self.model.decision_function(x_test)
        predictions = []
        if len(decision_values.shape) == 2:  # multi classes
            for sample_idx in range(decision_values.shape[0]):
                prediction = decision_values[sample_idx]
                if prediction.max() > prediction.min():
                    prediction = (prediction - prediction.min()) / \
                        (prediction.max() - prediction.min())
                    # makes summaion one.
                    prediction = prediction / np.sum(prediction)
                else:
                    prediction = [1 / decision_values.shape[1]
                                  for i in range(decision_values.shape[1])]
                predictions.append(prediction)
        else:  # binary class
            if decision_values.max() > decision_values.min():
                decision_values = (decision_values - decision_values.min()) / \
                    (decision_values.max() - decision_values.min())
            else:
                decision_values = decision_values / decision_values.max()  # all ones
            for sample_idx in range(decision_values.shape[0]):
                predictions.append(
                    [decision_values[sample_idx], 1 - decision_values[sample_idx]])
        predictions = np.array(predictions)

        # predictions = self.model.predict(x_test)
        # prediction_dict["predicted_labels_stack"] = np.array([self.encode_list[label] for label in predictions])
        print(predictions)
        prediction_dict["predicted_labels_stack"] = predictions
        return prediction_dict


class RandomForest():
    name = "Random Forest"

    def __init__(self, max_depth=34):
        """This is required method."""
        self.model = sklearn.ensemble.RandomForestClassifier()

    def fit(self, dataset, x_train, y_train, **kwarg):
        """This method is only required for trainable estimators."""
        dataset
        self.model.fit(x_train, y_train)

    def predict(self, x_test, **kwarg):
        """This is required method."""
        prediction_dict = {}
        predict_proba = self.model.predict_proba(x_test)
        predictions = []
        for sample_idx in range(predict_proba[0].shape[0]):
            predictions.append([predict_proba[class_][sample_idx][1]
                                for class_ in range(len(predict_proba))])
        prediction_dict["predicted_labels_stack"] = np.array(predictions)
        # prediction_dict["predicted_labels_stack"] = self.model.predict_proba(x_test)[1]
        return prediction_dict


class SVM():
    name = "SVM"

    def __init__(self, _gamma='auto'):
        """This is required method."""
        self.model = sklearn.svm.SVC(gamma=_gamma)

    def fit(self, x_train, y_train, **kwarg):
        """This method is only required for trainable estimators."""
        y_train_single_label, self.encode_list = numpy_array.inverse_one_hot_encode(
            y_train, return_encode_list=True)
        self.model.fit(x_train, y_train_single_label)

    def predict(self, x_test, **kwarg):
        """This is required method."""
        prediction_dict = {}

        decision_values = self.model.decision_function(x_test)
        predictions = []
        if len(decision_values.shape) == 2:  # multi classes
            for sample_idx in range(decision_values.shape[0]):
                prediction = decision_values[sample_idx]
                if prediction.max() > prediction.min():
                    prediction = (prediction - prediction.min()) / \
                        (prediction.max() - prediction.min())
                    # makes summaion one.
                    prediction = prediction / np.sum(prediction)
                else:
                    prediction = [1 / decision_values.shape[1]
                                  for i in range(decision_values.shape[1])]
                predictions.append(prediction)
        else:  # binary class
            if decision_values.max() > decision_values.min():
                decision_values = (decision_values - decision_values.min()) / \
                    (decision_values.max() - decision_values.min())
            else:
                decision_values = decision_values / decision_values.max()  # all ones
            for sample_idx in range(decision_values.shape[0]):
                predictions.append(
                    [decision_values[sample_idx], 1 - decision_values[sample_idx]])
        predictions = np.array(predictions)

        # prediction_dict["predicted_labels_stack"] = self.model.predict_proba(x_test)[1]
        # prediction_dict["predicted_labels_stack"] = np.array([self.encode_list[label] for label in predictions])

        prediction_dict["predicted_labels_stack"] = predictions
        return prediction_dict


class MLP():
    name = "MLP"

    def __init__(self, hidden_layer_sizes=(150, 125, 100, 50, 25)):
        """This is required method."""
        self.model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes)

    def fit(self, x_train, y_train, **kwarg):
        """This method is only required for trainable estimators."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test, **kwarg):
        """This is required method."""
        prediction_dict = {}
        prediction_dict["predicted_labels_stack"] = self.model.predict(x_test)
        return prediction_dict


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("tanh")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    # return our model
    return model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class CNN():
    name = "CNN"

    def __init__(self):
        mlp = create_mlp(36)
        cnn = create_cnn(400, 400, 1)

        combinedInput = concatenate([mlp.output, cnn.output])
        x = Dense(4, activation="tanh")(combinedInput)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=[mlp.input, cnn.input], outputs=x)
        tf.keras.utils.plot_model(
            self.model, to_file="a.png", show_shapes=True, show_layer_names=True)
        pass

    def fit(self, dataset, x_train, indices_of_patients, y_train, epochs, class_weight):
        """This method is only required for trainable estimators."""
        features_name = ['sex_F',  'age', 'RT_PCR_positive_Unclear', 'RT_PCR_positive_Y',  'finding_Pneumonia', 'finding_Pneumonia/Aspiration', 'finding_Pneumonia/Bacterial', 'finding_Pneumonia/Bacterial/Chlamydophila', 'finding_Pneumonia/Bacterial/E.Coli', 'finding_Pneumonia/Bacterial/Klebsiella', 'finding_Pneumonia/Bacterial/Legionella', 'finding_Pneumonia/Bacterial/Mycoplasma', 'finding_Pneumonia/Bacterial/Nocardia', 'finding_Pneumonia/Bacterial/Staphylococcus/MRSA', 'finding_Pneumonia/Bacterial/Streptococcus', 'finding_Pneumonia/Fungal/Aspergillosis',
                         'finding_Pneumonia/Fungal/Pneumocystis', 'finding_Pneumonia/Lipoid', 'finding_Pneumonia/Viral/COVID-19', 'finding_Pneumonia/Viral/Herpes ', 'finding_Pneumonia/Viral/Influenza', 'finding_Pneumonia/Viral/Influenza/H1N1', 'finding_Pneumonia/Viral/MERS-CoV', 'finding_Pneumonia/Viral/SARS', 'finding_Pneumonia/Viral/Varicella', 'finding_Tuberculosis', 'finding_Unknown', 'finding_todo', 'intubated_N', 'intubated_Y', 'went_icu_N', 'went_icu_Y', 'needed_supplemental_O2_N', 'needed_supplemental_O2_Y', 'extubated_N', 'extubated_Y']

        all_static_data = []
        all_dynamic_data = []
        labels = []
        count_0 = 0
        count_1 = 1
        for i in indices_of_patients:
            currentPatient = []
            currentPatient_image = []
            run = True
            for key in features_name:
                if(math.isnan(dataset.dicts[i][key])):
                    currentPatient.append(-1)
                else:
                    currentPatient.append(dataset.dicts[i][key])
            if(run == False):
                continue
            all_static_data.append(currentPatient)
            all_dynamic_data.append(
                dataset.dicts[i]["RNN_2D"]["data"][-1][-1].reshape(400, 400))
            if(dataset.dicts[i]["outcome"] == 0):
                count_0 += 1
            else:
                count_1 += 1

            labels.append(dataset.dicts[i]["outcome"])
        print(count_0, count_1, "=============")
        x_train_static = np.array(all_static_data)
        x_train_static = x_train_static.reshape(
            x_train_static.shape[0], len(features_name))
        x_train_dynamic = np.array(all_dynamic_data)
        x_train_dynamic = x_train_dynamic.reshape(
            x_train_dynamic.shape[0], 400, 400, 1)
        # labels = [[1] if i[0] == 1 else [0] for i in y_train]
        labels = np.array(labels)
        labels = labels.reshape(labels.shape[0], 1)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,
                  nesterov=True, clipnorm=0.5)

        self.model.compile(loss='binary_crossentropy',
                           optimizer="adam", metrics=[f1_m, "accuracy"])

        self.model.fit([x_train_static, x_train_dynamic],
                       labels, epochs=epochs, verbose=2, class_weight=class_weight)

        pass

    def predict(self, x_test, dataset, indices_of_patients):
        features_name = ['sex_F',  'age', 'RT_PCR_positive_Unclear', 'RT_PCR_positive_Y',  'finding_Pneumonia', 'finding_Pneumonia/Aspiration', 'finding_Pneumonia/Bacterial', 'finding_Pneumonia/Bacterial/Chlamydophila', 'finding_Pneumonia/Bacterial/E.Coli', 'finding_Pneumonia/Bacterial/Klebsiella', 'finding_Pneumonia/Bacterial/Legionella', 'finding_Pneumonia/Bacterial/Mycoplasma', 'finding_Pneumonia/Bacterial/Nocardia', 'finding_Pneumonia/Bacterial/Staphylococcus/MRSA', 'finding_Pneumonia/Bacterial/Streptococcus', 'finding_Pneumonia/Fungal/Aspergillosis',
                         'finding_Pneumonia/Fungal/Pneumocystis', 'finding_Pneumonia/Lipoid', 'finding_Pneumonia/Viral/COVID-19', 'finding_Pneumonia/Viral/Herpes ', 'finding_Pneumonia/Viral/Influenza', 'finding_Pneumonia/Viral/Influenza/H1N1', 'finding_Pneumonia/Viral/MERS-CoV', 'finding_Pneumonia/Viral/SARS', 'finding_Pneumonia/Viral/Varicella', 'finding_Tuberculosis', 'finding_Unknown', 'finding_todo', 'intubated_N', 'intubated_Y', 'went_icu_N', 'went_icu_Y', 'needed_supplemental_O2_N', 'needed_supplemental_O2_Y', 'extubated_N', 'extubated_Y']

        all_static_data = []
        all_dynamic_data = []
        labels = []
        count_0 = 0
        count_1 = 1
        for i in indices_of_patients:
            currentPatient = []
            currentPatient_image = []
            run = True
            for key in features_name:
                if(math.isnan(dataset.dicts[i][key])):
                    currentPatient.append(-1)
                else:
                    currentPatient.append(dataset.dicts[i][key])
            if(run == False):
                continue
            all_static_data.append(currentPatient)
            all_dynamic_data.append(
                dataset.dicts[i]["RNN_2D"]["data"][-1][-1].reshape(400, 400))
            if(dataset.dicts[i]["outcome"] == 0):
                count_0 += 1
            else:
                count_1 += 1

            labels.append(dataset.dicts[i]["outcome"])
        print(count_0, count_1, "=============")

        x_test_static = np.array(all_static_data)
        x_test_static = x_test_static.reshape(
            x_test_static.shape[0], len(features_name))
        x_test_dynamic = np.array(all_dynamic_data)
        x_test_dynamic = x_test_dynamic.reshape(
            x_test_dynamic.shape[0], 400, 400, 1)
        # labels = [[1] if i[0] == 1 else [0] for i in y_test]
        labels = np.array(labels)
        labels = labels.reshape(labels.shape[0], 1)
        predict = self.model.predict([x_test_static, x_test_dynamic])
        prediction_dict = {}
        predict = [[1-i, i] for i in predict]
        prediction_dict["predicted_labels_stack"] = predict
        return prediction_dict
        """This is required method."""


features = ['sex_F',  'age', 'RT_PCR_positive_Unclear', 'RT_PCR_positive_Y',  'finding_Pneumonia', 'finding_Pneumonia/Aspiration', 'finding_Pneumonia/Bacterial', 'finding_Pneumonia/Bacterial/Chlamydophila', 'finding_Pneumonia/Bacterial/E.Coli', 'finding_Pneumonia/Bacterial/Klebsiella', 'finding_Pneumonia/Bacterial/Legionella', 'finding_Pneumonia/Bacterial/Mycoplasma', 'finding_Pneumonia/Bacterial/Nocardia', 'finding_Pneumonia/Bacterial/Staphylococcus/MRSA', 'finding_Pneumonia/Bacterial/Streptococcus', 'finding_Pneumonia/Fungal/Aspergillosis',
            'finding_Pneumonia/Fungal/Pneumocystis', 'finding_Pneumonia/Lipoid', 'finding_Pneumonia/Viral/COVID-19', 'finding_Pneumonia/Viral/Herpes ', 'finding_Pneumonia/Viral/Influenza', 'finding_Pneumonia/Viral/Influenza/H1N1', 'finding_Pneumonia/Viral/MERS-CoV', 'finding_Pneumonia/Viral/SARS', 'finding_Pneumonia/Viral/Varicella', 'finding_Tuberculosis', 'finding_Unknown', 'finding_todo', 'survival_N', 'survival_Y', 'intubated_N', 'intubated_Y', 'went_icu_N', 'went_icu_Y', 'needed_supplemental_O2_N', 'needed_supplemental_O2_Y', 'extubated_N', 'extubated_Y',  'outcome']


# class CNN():
#     name = "CNN"

#     def __init__(self):
#         mlp = create_mlp(39)
#         cnn = create_cnn(1024, 1051, 1)

#         combinedInput = concatenate([mlp.output])
#         x = Dense(4, activation="tanh")(combinedInput)
#         x = Dense(1, activation='sigmoid')(x)
#         self.model = Model(inputs=[mlp.input], outputs=x)
#         tf.keras.utils.plot_model(
#             self.model, to_file="b.png", show_shapes=True, show_layer_names=True)

#         pass

# def fit(self, dataset, indices_of_patients, x_train, y_train):
#     """This method is only required for trainable estimators."""
#     features_name = ['sex_F',  'age', 'RT_PCR_positive_Unclear', 'RT_PCR_positive_Y',  'finding_Pneumonia', 'finding_Pneumonia/Aspiration', 'finding_Pneumonia/Bacterial', 'finding_Pneumonia/Bacterial/Chlamydophila', 'finding_Pneumonia/Bacterial/E.Coli', 'finding_Pneumonia/Bacterial/Klebsiella', 'finding_Pneumonia/Bacterial/Legionella', 'finding_Pneumonia/Bacterial/Mycoplasma', 'finding_Pneumonia/Bacterial/Nocardia', 'finding_Pneumonia/Bacterial/Staphylococcus/MRSA', 'finding_Pneumonia/Bacterial/Streptococcus', 'finding_Pneumonia/Fungal/Aspergillosis',
#                      'finding_Pneumonia/Fungal/Pneumocystis', 'finding_Pneumonia/Lipoid', 'finding_Pneumonia/Viral/COVID-19', 'finding_Pneumonia/Viral/Herpes ', 'finding_Pneumonia/Viral/Influenza', 'finding_Pneumonia/Viral/Influenza/H1N1', 'finding_Pneumonia/Viral/MERS-CoV', 'finding_Pneumonia/Viral/SARS', 'finding_Pneumonia/Viral/Varicella', 'finding_Tuberculosis', 'finding_Unknown', 'finding_todo', 'survival_N', 'survival_Y', 'intubated_N', 'intubated_Y', 'went_icu_N', 'went_icu_Y', 'needed_supplemental_O2_N', 'needed_supplemental_O2_Y', 'extubated_N', 'extubated_Y',  'outcome']
#     all_static_data = []
#     all_dynamic_data = []
#      labels = []
#       num_test = 40
#        for i in indices_of_patients[1:1+num_test]:
#             currentPatient = []
#             currentPatient_image = []
#             run = True
#             for key in features_name:
#                 if(math.isnan(dataset.dicts[i][key])):
#                     run = False
#                     num_test -= 1
#                     break
#                 currentPatient.append(dataset.dicts[i][key])
#             if(run == False):
#                 continue
#             currentPatient_image.append(
#                 dataset.dicts[i]["RNN_2D"]["data"][-1].reshape(1051, 1024, 1))
#             all_static_data.append(currentPatient)
#             all_dynamic_data.append(currentPatient_image)
#             # labels.append(dataset.dicts[i]["predictor label"])
#             labels.append(dataset.dicts[i]["outcome"])
#         print(labels)
#         print("=======================================")
#         x_train_static = np.array(all_static_data)

#         label = np.array(labels).reshape(num_test, 1)
#         sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9,
#                   nesterov=True, clipnorm=0.5)
#         print(x_train_static)
#         print(label)
#         self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
#                            optimizer=sgd, metrics=['accuracy'])
#         self.model.fit([x_train_static],
#                        label, epochs=4, verbose=1)
#         print("pass")
#         pass

#     def predict(self, dataset, indices_of_patients, x_test):
#         features_name = ['sex_F',  'age', 'RT_PCR_positive_Unclear', 'RT_PCR_positive_Y',  'finding_Pneumonia', 'finding_Pneumonia/Aspiration', 'finding_Pneumonia/Bacterial', 'finding_Pneumonia/Bacterial/Chlamydophila', 'finding_Pneumonia/Bacterial/E.Coli', 'finding_Pneumonia/Bacterial/Klebsiella', 'finding_Pneumonia/Bacterial/Legionella', 'finding_Pneumonia/Bacterial/Mycoplasma', 'finding_Pneumonia/Bacterial/Nocardia', 'finding_Pneumonia/Bacterial/Staphylococcus/MRSA', 'finding_Pneumonia/Bacterial/Streptococcus', 'finding_Pneumonia/Fungal/Aspergillosis',
#                          'finding_Pneumonia/Fungal/Pneumocystis', 'finding_Pneumonia/Lipoid', 'finding_Pneumonia/Viral/COVID-19', 'finding_Pneumonia/Viral/Herpes ', 'finding_Pneumonia/Viral/Influenza', 'finding_Pneumonia/Viral/Influenza/H1N1', 'finding_Pneumonia/Viral/MERS-CoV', 'finding_Pneumonia/Viral/SARS', 'finding_Pneumonia/Viral/Varicella', 'finding_Tuberculosis', 'finding_Unknown', 'finding_todo', 'survival_N', 'survival_Y', 'intubated_N', 'intubated_Y', 'went_icu_N', 'went_icu_Y', 'needed_supplemental_O2_N', 'needed_supplemental_O2_Y', 'extubated_N', 'extubated_Y',  'outcome']
#         all_static_data = []
#         all_dynamic_data = []
#         labels = []

#         for i in indices_of_patients:
#             currentPatient = []
#             currentPatient_image = []
#             run = True
#             for key in features_name:
#                 if(math.isnan(dataset.dicts[i][key])):
#                     run = False

#                     break
#                 currentPatient.append(dataset.dicts[i][key])
#             if(run == False):
#                 continue
#             currentPatient_image.append(
#                 dataset.dicts[i]["RNN_2D"]["data"][-1].reshape(1051, 1024, 1))
#             all_static_data.append(currentPatient)
#             all_dynamic_data.append(currentPatient_image)
#             # labels.append(dataset.dicts[i]["predictor label"])
#             labels.append(dataset.dicts[i]["outcome"])

#         x_test_static = np.array(all_static_data)
#         x_test_dynamic = np.array(all_dynamic_data)
#         x_test_dynamic = x_test_dynamic.reshape(
#             x_test_dynamic.shape[0], 1051, 1024, 1)
#         labels = np.array(labels)
#         labels = labels.reshape(labels.shape[0], 1)
#         predict = self.model.predict([x_test_static])
#         prediction_dict = {}
#         print(predict)
#         print(x_test_static)
#         print(labels)

#         prediction_dict["predicted_labels_stack"] = predict
#         return prediction_dict
#         """This is required method."""
# Incrsae the number of parameter for the layers
# Remove the regularzation technieques
# Oversampleing and undersampling
# Change Loss Function
# Class weight https://keras.io/api/losses/
