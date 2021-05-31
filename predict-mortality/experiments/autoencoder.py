import numpy as np
from keras.layers import Dropout, BatchNormalization, Concatenate, RepeatVector, Dense, LSTM, Input, LeakyReLU, Lambda, GRU, SimpleRNN, Flatten, Conv2DTranspose, ConvLSTM2D, MaxPooling3D, TimeDistributed, Reshape, Subtract, Add, Multiply
from keras.models import Model
from keras import backend as K
from keras import regularizers, activations
from keras.utils import plot_model
from keras.losses import SquaredHinge
from tensorflow.keras import optimizers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

import tensorflow as tf

from utilsforminds.containers import merge_dictionaries
from utilsforminds.numpy_array import mask_prob

from tqdm import tqdm
from copy import deepcopy
from random import random
from random import sample

keras_functions_dict = {"Dense": Dense, "Dropout": Dropout, "BatchNormalization": BatchNormalization, "LSTM": LSTM, "GRU": GRU, "Flatten": Flatten, "Conv2DTranspose": Conv2DTranspose, "ConvLSTM2D": ConvLSTM2D, "MaxPooling3D": MaxPooling3D, "TimeDistributed": TimeDistributed, "Reshape": Reshape, "RandomFourierFeatures": RandomFourierFeatures}
# keras_optimizers_dict = {"Adam": optimizers.Adam} # (learning_rate= 0.001, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-7)
basic_regularizer = None

class Autoencoder():
    name = "Autoencoder"
    def __init__(self, debug = 0, verbose = 0, small_delta = 1e-7, whether_use_mask = True):
        """RNN Autoencoder Estimator"""

        self.debug = debug
        self.verbose = verbose
        self.small_delta = small_delta
        self.whether_use_mask = whether_use_mask
    
    def fit(self, dataset, indices_of_patients, model_name_RNN_vectors = "LSTM", kwargs_RNN_vectors = None, dynamic_vectors_decoder_structure_list = None, predictor_structure_list = None, static_encoder_decoder_structure_dict = None, factors_dict = None, optimizer = None, loss_kind_dict = None, iters = 10, shuffle = True, run_eagerly = False, x_train = None, y_train = None, loss_wrapper_funct = None, loss_factor_funct = None, kwargs_RNN_images = None, conv_encoder_decoder_structure_dict = None):
        """Train Autoencoder.
        
        Parameters
        ----------
        x_train, y_train: None.
            Dummy parameters, just leave them None, do not use.
        indices_of_patients : list of int
            Indices of participants to learn.
        """

        if self.whether_use_mask:
            record_length_factor = 2
            self.data_kind_key = "concat"
        else:
            record_length_factor = 1
            self.data_kind_key = "data"
        indices_of_patients_loc = deepcopy(indices_of_patients)
        print(f"Autoencoder fits on {len(indices_of_patients_loc)} samples.")
        output_dict = {"dynamic_vectors_decoder": None, "dynamic_images_decoder": None, "static_decoder": None, "predictor": None}
        concatenated_representations = []

        ## Set default arguments for each kwargs.
        if isinstance(kwargs_RNN_vectors, dict):
            kwargs_RNN_vectors_local = [merge_dictionaries([{"units": 64}, kwargs_RNN_vectors])]
        else:
            kwargs_RNN_vectors_local = []
            for kwargs_dict in kwargs_RNN_vectors:
                kwargs_RNN_vectors_local.append(merge_dictionaries([{"units": 64}, kwargs_dict]))
        # assert(("dropout" not in kwargs_RNN_vectors_local.keys() or kwargs_RNN_vectors_local["dropout"] == 0.) and ("recurrent_dropout" not in kwargs_RNN_vectors_local.keys() or kwargs_RNN_vectors_local["recurrent_dropout"] == 0.)) ## dropout largely degrade the performance, dropout may not work with predict_on_batch.
        factors_dict_local = merge_dictionaries([{"dynamic vectors reconstruction loss": 1.0, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 1.0, "prediction": 10.0}, factors_dict])
        loss_kind_dict = merge_dictionaries([{"dynamic vectors reconstruction loss": 2, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 2, "prediction": "binary cross entropy"}, loss_kind_dict]) ## binary cross entropy works for multi-class classification with one-hot encoded label.
        if optimizer is None: optimizer = optimizers.Adam()

        num_vectors_hidden_dynamic_features = kwargs_RNN_vectors_local[-1]["units"]

        ## --- --- Timeseries AE for vector inputs
        ## --- RNN Encoder (for vectors) structure
        vectors_RNN_input = Input(shape= (None, len(dataset.input_to_features_map["RNN_vectors"]) * record_length_factor), batch_size= 1, name= "vectors_RNN_input") ## shape = (batch_size = 1, time_steps = None, num_features), input shape of numpy array should be (1, time_steps, num_features)
        vectors_hidden_states_LSTM = vectors_RNN_input

        for kwargs_dict in kwargs_RNN_vectors_local:
            if model_name_RNN_vectors == "LSTM":
                vectors_hidden_states_LSTM, vectors_last_hidden_state, vectors_last_cell_state = LSTM(return_sequences=True, return_state=True, **kwargs_dict)(vectors_hidden_states_LSTM) ## vectors_last_hidden_state's shape: (1, num_hidden_dynamic_features), the input shape looks like (batch_size = 1, time_steps, units) # batch_input_shape= (1, None, num_features_LSTM_input)
            elif model_name_RNN_vectors == "GRU":
                vectors_hidden_states_LSTM, vectors_last_hidden_state = GRU(return_sequences=True, return_state=True, **kwargs_dict)(vectors_hidden_states_LSTM) ## vectors_last_hidden_state's shape: (1, num_hidden_dynamic_features), the input shape looks like (batch_size = 1, time_steps, units) # batch_input_shape= (1, None, num_features_LSTM_input)
            elif model_name_RNN_vectors == "SimpleRNN":
                vectors_hidden_states_LSTM, vectors_last_hidden_state = SimpleRNN(return_sequences=True, return_state=True, **kwargs_dict)(vectors_hidden_states_LSTM) ## vectors_last_hidden_state's shape: (1, num_hidden_dynamic_features), the input shape looks like (batch_size = 1, time_steps, units) # batch_input_shape= (1, None, num_features_LSTM_input)
            else:
                raise Exception(NotImplementedError)

        ## --- RNN Decoder (for vectors) structure
        arr_RE_DATE = Input(shape = (None, 1), batch_size= 1, name= "arr_RE_DATE") ## (batch_size= 1, time_steps, num_features = 1)

        ## Copy the last hidden representation.
        # copies_vectors_last_hidden_state = RepeatVector(n = K.shape(vectors_RNN_input)[1])(last_hidden_state) ## Output shape : 3D tensor of shape (num_samples = 1, n, features), Input shape : 2D tensor of shape (num_samples = 1, features).
        def repeat_vector(args): ## https://github.com/keras-team/keras/issues/7949
            layer_to_repeat = args[0]
            sequence_layer = args[1]
            return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)
        copies_vectors_last_hidden_state = Lambda(repeat_vector, output_shape = (None, num_vectors_hidden_dynamic_features))([vectors_last_hidden_state, vectors_RNN_input]) ## (None = 1, None = time_steps, num_hidden_dynamic_features)

        copies_vectors_last_hidden_state_RE_DATE_concatenated = Concatenate(axis = -1)([copies_vectors_last_hidden_state, arr_RE_DATE]) ## (1, time_steps, num_hidden_dynamic_features + 1).

        ### Decoder (for vectors) Reconstruction
        if dynamic_vectors_decoder_structure_list is None: ## Basic structure, if structure is not given.
            dynamic_vectors_decoder_structure_list = [["Dense", {"units": 128, "activation": LeakyReLU(alpha=0.01), "activity_regularizer": basic_regularizer}], ["Dense", {"units": 70, "activation": LeakyReLU(alpha=0.01), "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset.input_to_features_map["dynamic_vectors_decoder"]), "activation": LeakyReLU(alpha=0.01), "activity_regularizer":  basic_regularizer}]]
        assert(dynamic_vectors_decoder_structure_list[-1][0] == "Dense")
        assert(dynamic_vectors_decoder_structure_list[0][0] == "Dense")
        dynamic_vectors_decoder_structure_list[-1][1]["units"] = len(dataset.input_to_features_map["dynamic_vectors_decoder"]) ## output shape (1, time_steps, len(dataset.input_to_features_map["dynamic_vectors_decoder"])).

        output_dict["dynamic_vectors_decoder"] = copies_vectors_last_hidden_state_RE_DATE_concatenated ## https://keras.io/api/layers/core_layers/dense/
        for layer in dynamic_vectors_decoder_structure_list: ## output_dict["dynamic_vectors_decoder"] shape (1, time_steps, len(dataset.input_to_features_map["dynamic_vectors_decoder"])).
            if layer is not None: output_dict["dynamic_vectors_decoder"] = keras_functions_dict[layer[0]](**layer[1])(output_dict["dynamic_vectors_decoder"]) ## Output of final layer's shape: (batch_size = 1, time_steps, len(dataset.input_to_features_map["dynamic_vectors_decoder"])).

        ## --- --- Timeseries AE for 2D inputs
        if dataset.shape_2D_records is not None: ## Dataset contains images.
            ## Set default settings
            images_RNN_input = Input(shape= (None, dataset.shape_2D_records[0] * record_length_factor, dataset.shape_2D_records[1], 1), batch_size= 1, name= "images_RNN_input") ## shape = (batch_size = 1, time_steps = None, num_rows (multiplied twice if concat), num_cols, 1)
            # kwargs_RNN_images_local = merge_dictionaries([{"units": 64}, kwargs_RNN_images])
            output_dict["dynamic_images_decoder"] = images_RNN_input

            if conv_encoder_decoder_structure_dict is None: ## Default argument.
                conv_encoder_decoder_structure_dict = {}
                # conv_encoder_decoder_structure_dict["encoder"] = [["Conv2D", dict(filters= 32, kernel_size= (5, 5), stride= (2, 2), activation= "relu")], ["MaxPooling2D", dict(pool_size= (2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (5, 5), stride= (2, 2), activation= "relu")], ["MaxPooling2D", dict(pool_size= (2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (3, 3), stride= (1, 1), activation= "relu")], ["Flatten", dict()]] # ["Flatten", dict()]

                conv_encoder_decoder_structure_dict["encoder"] = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (2, 2), padding= "valid", activation= "tanh", return_sequences= True)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (2, 2), padding= "valid", activation= "tanh", return_sequences= True)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (2, 2), padding= "valid", activation= "tanh", return_sequences= False)], ["TimeDistributed", dict(layer= Flatten())], ["LSTM", dict(units= 64, return_sequences=True, return_state=True)]] ## For MaxPooling3D, pool_size = (1, 2, 2) means no pooling (1) along time steps, and (2, 2) pooling along width and height.

                conv_encoder_decoder_structure_dict["decoder"] = [["Dense", {"units": 100, "activation": lambda x: activations.relu(x, alpha = 0.1)}], ["Dense", {"units": 169, "activation": lambda x: activations.relu(x, alpha = 0.1)}], ["TimeDistributed", dict(layer = Reshape(target_shape= (13, 13, 1)))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (2, 2), padding= "valid", activation= "tanh"))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5))]] ## Conv2DTranspose, (batch_size (time_steps), rows, cols, channels) -> (batch_size, new_rows, new_cols, filters).
                # ["Reshape", dict(target_shape= (kwargs_RNN_images_local["units"] - kwargs_RNN_images_local["units"] // 2, kwargs_RNN_images_local["units"] // 2))]

            ## Image Records Encoding
            for layer_ in conv_encoder_decoder_structure_dict["encoder"]:
                if self.verbose >= 1: print(f"2D Encoding: {layer_[0]}, previous shape: {output_dict['dynamic_images_decoder'].shape}")
                output_dict["dynamic_images_decoder"] = keras_functions_dict[layer_[0]](**layer_[1])(output_dict["dynamic_images_decoder"])
            images_hidden_states_LSTM, images_last_hidden_state, images_last_cell_state = output_dict["dynamic_images_decoder"]

            ## Copy the last hidden representation.
            copies_images_last_hidden_state = Lambda(repeat_vector, output_shape = (None, conv_encoder_decoder_structure_dict["encoder"][-1][1]["units"]))([images_last_hidden_state, images_RNN_input]) ## (None = 1, None = time_steps, num_hidden_dynamic_features)

            copies_images_last_hidden_state_RE_DATE_concatenated = Concatenate(axis = -1)([copies_images_last_hidden_state, arr_RE_DATE]) ## (1 (deleted by [0]), time_steps, num_hidden_dynamic_features + 1).
            output_dict["dynamic_images_decoder"] = copies_images_last_hidden_state_RE_DATE_concatenated

            ## Image Records Decoding
            for layer_ in conv_encoder_decoder_structure_dict["decoder"]:
                if self.verbose >= 1: print(f"2D Decoding: {layer_[0]}, previous shape: {output_dict['dynamic_images_decoder'].shape}")
                if layer_[0] == "Conv2DTranspose-fit-to-image-shape":
                    strides= [0, 0]
                    kernel_size = layer_[1]["kernel_size"]
                    for axis in [0, 1]:
                        # kernel_size[axis] = dataset.shape_2D_records[axis] % (output_dict["dynamic_images_decoder"].shape[1 + axis] - 1)
                        # strides[axis] = dataset.shape_2D_records[axis] // (output_dict["dynamic_images_decoder"].shape[1 + axis] - 1)
                        strides[axis] = (dataset.shape_2D_records[axis] - kernel_size[axis]) // (output_dict["dynamic_images_decoder"].shape[2 + axis] - 1)
                        if (dataset.shape_2D_records[axis] - kernel_size[axis]) % (output_dict["dynamic_images_decoder"].shape[2 + axis] - 1) != 0: strides[axis] += 1
                    output_dict["dynamic_images_decoder"] = TimeDistributed(layer= Conv2DTranspose(filters= 1, strides = strides, padding= "valid", **layer_[1]))(output_dict["dynamic_images_decoder"]) ## Reconstruce the image with similar (slightly larger or equal) shape as input image.
                else:
                    output_dict["dynamic_images_decoder"] = keras_functions_dict[layer_[0]](**layer_[1])(output_dict["dynamic_images_decoder"])
            output_dict["dynamic_images_decoder"] = output_dict["dynamic_images_decoder"][:, :, :dataset.shape_2D_records[0], :dataset.shape_2D_records[1], :] ## Cut the remaining paddings to keep the same shape as input image.
            assert(output_dict["dynamic_images_decoder"].shape[2] == dataset.shape_2D_records[0] and output_dict["dynamic_images_decoder"].shape[3] == dataset.shape_2D_records[1]) ## Same shape as input image.

            concatenated_representations.append(images_last_hidden_state)

        ### Predictor structure
        if predictor_structure_list is None: ## Basic structure, if structure is not given.
            predictor_structure_list = [["Dense", {"units": 80, "activation": LeakyReLU(alpha=0.01), "activity_regularizer": basic_regularizer}], ["Dense", {"units": 40, "activation": LeakyReLU(alpha=0.01), "activity_regularizer":  basic_regularizer}], ["Dense", {"units": 1, "activation": "sigmoid", "activity_regularizer":  basic_regularizer}]]
        assert(predictor_structure_list[-1][0] == "Dense" and predictor_structure_list[-1][1]["units"] == len(dataset.prediction_labels_bag))
        # assert(predictor_structure_list[0][0] == "Dense")

        ## --- --- About static features.
        ## --- Static Encoding.
        ## Set default structure.
        def num_neurons(factor): ## number of neurons for default argument.
            return max(1, round(factor * len(dataset.input_to_features_map["static_encoder"])))
        if static_encoder_decoder_structure_dict is None: ## Default argument.
            static_encoder_decoder_structure_dict = {}
            static_encoder_decoder_structure_dict["encoder"] = [["Dense", {"units": num_neurons(0.6), "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": num_neurons(0.2), "activation": "tanh", "activity_regularizer":  basic_regularizer}],  ["Dense", {"units": num_neurons(0.1), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]
            static_encoder_decoder_structure_dict["decoder"] = [["Dense", {"units": num_neurons(0.2), "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": num_neurons(0.6), "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]
        assert(static_encoder_decoder_structure_dict["decoder"][-1][1]["units"] == len(dataset.input_to_features_map["static_encoder"])) ## For reconstruction.

        ## Start Static Encoding.
        ## Static info is concatenated to input of predictor.
        static_encoder_input = Input(shape = (len(dataset.input_to_features_map["static_encoder"]) * record_length_factor, ), batch_size= 1, name= "static_encoder_input")
        static_encoder_data = Input(shape = (len(dataset.input_to_features_map["static_encoder"]), ), batch_size= 1, name= "static_encoder_data")
        static_encoder_mask = Input(shape = (len(dataset.input_to_features_map["static_encoder"]), ), batch_size= 1, name= "static_encoder_mask")
        encoded_static_vector = static_encoder_input
        if len(dataset.input_to_features_map["static_encoder"]) > 0:
            for layer in static_encoder_decoder_structure_dict["encoder"]:
                encoded_static_vector = keras_functions_dict[layer[0]](**layer[1])(encoded_static_vector) ## output encoded_static_vector in shape (1, num_hidden_static_features)
            ## Static Decoding.
            output_dict["static_decoder"] = encoded_static_vector
            for layer in static_encoder_decoder_structure_dict["decoder"]:
                output_dict["static_decoder"] = keras_functions_dict[layer[0]](**layer[1])(output_dict["static_decoder"]) ## output encoded_static_vector in shape (num_hidden_features, )
            concatenated_representations.append(encoded_static_vector)

        static_raw = Input(shape = (len(dataset.input_to_features_map["raw"]) * record_length_factor, ), batch_size= 1, name= "static_raw")
        if len(dataset.input_to_features_map["raw"]) > 0:
            concatenated_representations.append(static_raw)

        ## Concatenate Encoded Static Vector and Raw Vector to input of Predictor.
        concatenated_representations.append(vectors_last_hidden_state)
        
        output_dict["predictor"] = Concatenate(axis = -1)(concatenated_representations) ## encoded_static_vector's shape: (1, num_hidden_static_features), final shape = (1, num_hidden_static_features + num_static_raw_features + num_hidden_dynamic_features). If static_encoder_input is zero shape (,0) then encoded_static_vector is also zero shape, but can be concatenated without syntax error.

        ## Build Predictor.
        for layer in predictor_structure_list:
            if layer is not None: output_dict["predictor"] = keras_functions_dict[layer[0]](**layer[1])(output_dict["predictor"]) ## Output of final layer's shape: (1, 1).
        
        ## --- --- Define model.
        ## --- Define inputs for labels.
        ## For Decoder.
        decoder_vectors_labels_data = Input(shape = (None, len(dataset.input_to_features_map["dynamic_vectors_decoder"])), batch_size= 1, name= "decoder_vectors_labels_data") ## (1, time steps, features).
        decoder_vectors_labels_mask = Input(shape = (None, len(dataset.input_to_features_map["dynamic_vectors_decoder"])), batch_size= 1, name= "decoder_vectors_labels_mask") ## (1, time steps, features).
        if dataset.shape_2D_records is not None:
            decoder_images_labels_data = Input(shape = (None, dataset.shape_2D_records[0], dataset.shape_2D_records[1], 1), batch_size= 1, name= "decoder_images_labels_data") ## (1, time steps, features).
            decoder_images_labels_mask = Input(shape = (None, dataset.shape_2D_records[0], dataset.shape_2D_records[1], 1), batch_size= 1, name= "decoder_images_labels_mask") ## (1, time steps, features).


        ## For Predictor.
        predictor_label = Input(shape = (len(dataset.prediction_labels_bag), ), batch_size= 1, name= "predictor_label") ## *** Assuming one-hot encoded label, shape = (1, number of classes).
        predictor_label_observability = Input(shape = (1, ), batch_size= 1, name= "predictor_label_observability") ## shape = (1, 1), {1., 0.}.

        inputs_list = [vectors_RNN_input, arr_RE_DATE, decoder_vectors_labels_data, decoder_vectors_labels_mask, predictor_label, predictor_label_observability, static_encoder_input, static_encoder_data, static_encoder_mask, static_raw]
        outputs_list = [vectors_last_hidden_state, output_dict["predictor"], output_dict["dynamic_vectors_decoder"]]
        if dataset.shape_2D_records is not None:
            inputs_list.append(decoder_images_labels_data)
            inputs_list.append(decoder_images_labels_mask)
            inputs_list.append(images_RNN_input)
            outputs_list.append(images_last_hidden_state)
        self.model = Model(inputs = inputs_list, outputs = outputs_list) ## Output = enriched representation vector, predicted target, reconstructed original representations matrix.
        self.model.trainable = True
        self.model.run_eagerly = run_eagerly

        ## Plot model, because eager tensor(output tensor calculated by non-Layer function, such as + * - /) cannot be plotted, this should be located before adding losses.
        plot_model(self.model, to_file= "./outputs/misc/model_diagram.png", show_shapes= True)

        ### Define losses.
        self.model.add_loss(self.dynamic_vectors_reconstruction_loss(decoder_vectors_labels_data = decoder_vectors_labels_data, decoder_vectors_labels_mask = decoder_vectors_labels_mask, decoder_vectors_labels_data_predicted = output_dict["dynamic_vectors_decoder"], factor = factors_dict_local["dynamic vectors reconstruction loss"], kind= loss_kind_dict["dynamic vectors reconstruction loss"]))
        self.model.add_metric(self.dynamic_vectors_reconstruction_loss(decoder_vectors_labels_data = decoder_vectors_labels_data, decoder_vectors_labels_mask = decoder_vectors_labels_mask, decoder_vectors_labels_data_predicted = output_dict["dynamic_vectors_decoder"], factor = factors_dict_local["dynamic vectors reconstruction loss"], kind= loss_kind_dict["dynamic vectors reconstruction loss"]), name = "Dynamic Vectors Reconstruction Error")
        if dataset.shape_2D_records is not None:
            self.model.add_loss(self.dynamic_images_reconstruction_loss(decoder_images_labels_data = decoder_images_labels_data, decoder_images_labels_mask = decoder_images_labels_mask, decoder_images_labels_data_predicted = output_dict["dynamic_images_decoder"], factor = factors_dict_local["dynamic images reconstruction loss"], kind= loss_kind_dict["dynamic images reconstruction loss"]))
            self.model.add_metric(self.dynamic_images_reconstruction_loss(decoder_images_labels_data = decoder_images_labels_data, decoder_images_labels_mask = decoder_images_labels_mask, decoder_images_labels_data_predicted = output_dict["dynamic_images_decoder"], factor = factors_dict_local["dynamic images reconstruction loss"], kind= loss_kind_dict["dynamic images reconstruction loss"]), name = "Dynamic Images Reconstruction Error")
        if len(dataset.input_to_features_map["static_encoder"]) > 0:
            self.model.add_loss(self.static_reconstruction_loss(origina_static_data = static_encoder_data, origina_static_mask = static_encoder_mask, predicted_static_data = output_dict["static_decoder"], factor = factors_dict_local["static reconstruction loss"], kind = loss_kind_dict["static reconstruction loss"]))
            self.model.add_metric(self.static_reconstruction_loss(origina_static_data = static_encoder_data, origina_static_mask = static_encoder_mask, predicted_static_data = output_dict["static_decoder"], factor = factors_dict_local["static reconstruction loss"], kind = loss_kind_dict["static reconstruction loss"]), name = "Static Reconstruction Error")
        self.model.add_loss(self.prediction_loss(predictor_label_predicted = output_dict["predictor"], predictor_label = predictor_label, predictor_label_observability = predictor_label_observability, factor = factors_dict_local["prediction"], kind= loss_kind_dict["prediction"], loss_wrapper_funct = loss_wrapper_funct, loss_factor_funct = loss_factor_funct))
        self.model.add_metric(self.prediction_loss(predictor_label_predicted = output_dict["predictor"], predictor_label = predictor_label, predictor_label_observability = predictor_label_observability, factor = factors_dict_local["prediction"], kind= loss_kind_dict["prediction"], loss_wrapper_funct = loss_wrapper_funct, loss_factor_funct = loss_factor_funct), name = "Prediction Error")

        ## Compile model.
        self.model.compile(optimizer= optimizer)
        if self.verbose >= 2:
            self.model.summary()

        ## Get dictionary of losses to check convergence.
        loss_dict = {name: [] for name in self.model.metrics_names}
        loss_dict["loss"] = []
        ## Train model.
        for it in tqdm(range(iters)):
            if shuffle: np.random.shuffle(indices_of_patients_loc) 
            for patient_idx in indices_of_patients_loc:
                input_dict = {"vectors_RNN_input": dataset.dicts[patient_idx]["RNN_vectors"][self.data_kind_key], "arr_RE_DATE": dataset.dicts[patient_idx]["RE_DATE"], "decoder_vectors_labels_data": dataset.dicts[patient_idx]['dynamic_vectors_decoder']["data"], "decoder_vectors_labels_mask": dataset.dicts[patient_idx]['dynamic_vectors_decoder']["mask"], "predictor_label": dataset.dicts[patient_idx]["predictor label"], "predictor_label_observability": dataset.dicts[patient_idx]["observability"], "static_encoder_input": dataset.dicts[patient_idx]["static_encoder"][self.data_kind_key], "static_encoder_data": dataset.dicts[patient_idx]["static_encoder"]["data"], "static_encoder_mask": dataset.dicts[patient_idx]["static_encoder"]["mask"], "static_raw": dataset.dicts[patient_idx]["raw"][self.data_kind_key]}
                if dataset.shape_2D_records is not None:
                    input_dict["images_RNN_input"] = dataset.dicts[patient_idx]["RNN_2D"][self.data_kind_key]
                    input_dict["decoder_images_labels_data"] = dataset.dicts[patient_idx]["RNN_2D"]["data"]
                    input_dict["decoder_images_labels_mask"] = dataset.dicts[patient_idx]["RNN_2D"]["mask"]
                metric_loss_dict_single_batch = self.model.train_on_batch(x = input_dict, y = {}, return_dict= True)
                for metric in metric_loss_dict_single_batch.keys():
                    if metric != "Prediction Error" or (metric == "Prediction Error" and dataset.dicts[patient_idx]["observability"][0][0] == 1.): ## For prediction loss, in training set, label is provided.
                        loss_dict[metric].append(metric_loss_dict_single_batch[metric])
            if self.verbose >= 1:
                print(metric_loss_dict_single_batch)
                # print(f"Decayed Learning Rate: {self.model.optimizer._decayed_lr('float32').numpy()}, Learning Rate: {float(self.model.optimizer.learning_rate)}")
        return loss_dict
    
    def predict(self, dataset, indices_of_patients, perturbation_info_for_feature_importance = None, swap_info_for_feature_importance = None, feature_importance_calculate_prob_dict = None, x_test = None):
        """Predict target label, reconstructed data.

        Parameters
        ----------
        dataset : Dataset
            Dataset object.
        perturbation_info_for_feature_importance : dict
            For example, perturbation_info_for_feature_importance = {"center": "mean" or 0., "proportion": 1.0}. If None, then do not plot feature importance by perturbation method.
        swap_info_for_feature_importance : dict
            For example, swap_info_for_feature_importance = {"proportion": 0.7 <= 1.0}.
        feature_importance_calculate_prob_dict : dict
            For example, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}.
        x_test : None
            Dummy variable, just leave it None, Do not use.
        
        Attributes
        ----------
        """

        ## Set default arguments
        feature_importance_calculate_prob_dict = merge_dictionaries([{"dynamic": 1.0, "static": 1.0}, feature_importance_calculate_prob_dict])

        indices_of_patients_loc = deepcopy(indices_of_patients)
        print(f"Autoencoder predicts on {len(indices_of_patients_loc)} samples.")
        dicts_input_to_keras_input_map = dict(RNN_vectors = "vectors_RNN_input", static_encoder= "static_encoder_input", raw= "static_raw")
        
        self.model.trainable = False

        ## Output: Set results containers
        enriched_vectors_stack = []
        predicted_labels_stack = []
        reconstructed_vectors_stack = []
        ## Output: Feature importance
        feature_importance_dict = {}
        if perturbation_info_for_feature_importance is not None: feature_importance_dict["perturbation"] = {group_name: {feature_name: [] for feature_name in dataset.groups_of_features_info[group_name].keys()} for group_name in dataset.groups_of_features_info.keys()}
        if swap_info_for_feature_importance is not None: feature_importance_dict["swap"] = {group_name: {feature_name: [] for feature_name in dataset.groups_of_features_info[group_name].keys()} for group_name in dataset.groups_of_features_info.keys()}

        for patient_idx in tqdm(indices_of_patients_loc): ## The particiapant's sequence of predicted_labels_stack is same as the sequence of indices_of_patients_loc.
            original_inputs_dict = deepcopy({"vectors_RNN_input": dataset.dicts[patient_idx]["RNN_vectors"][self.data_kind_key], "arr_RE_DATE": dataset.dicts[patient_idx]["RE_DATE"], "decoder_vectors_labels_data": dataset.dicts[patient_idx]['dynamic_vectors_decoder']["data"], "decoder_vectors_labels_mask": dataset.dicts[patient_idx]['dynamic_vectors_decoder']["mask"], "predictor_label": dataset.dicts[patient_idx]["predictor label"], "predictor_label_observability": dataset.dicts[patient_idx]["observability"], "static_encoder_input": dataset.dicts[patient_idx]["static_encoder"][self.data_kind_key], "static_encoder_data": dataset.dicts[patient_idx]["static_encoder"]["data"], "static_encoder_mask": dataset.dicts[patient_idx]["static_encoder"]["mask"], "static_raw": dataset.dicts[patient_idx]["raw"][self.data_kind_key]})
            if dataset.shape_2D_records is not None:
                original_inputs_dict["images_RNN_input"] = deepcopy(dataset.dicts[patient_idx]["RNN_2D"][self.data_kind_key])
                original_inputs_dict["decoder_images_labels_data"] = deepcopy(dataset.dicts[patient_idx]["RNN_2D"]["data"])
                original_inputs_dict["decoder_images_labels_mask"] = deepcopy(dataset.dicts[patient_idx]["RNN_2D"]["mask"])

            ## Normal Prediction for just Prediction
            # enriched_vector, predicted_label, reconstructed_vectors = self.model.predict_on_batch(x = original_inputs_dict) ## outputs_list = [enriched_vector, predicted_label, reconstructed_vectors, reconstructed_images]
            outputs_list = self.model.predict_on_batch(x = original_inputs_dict)
            enriched_vectors_stack.append(outputs_list[0][0])
            predicted_labels_stack.append(outputs_list[1][0]) ## predicted_label[0] = [0.01, 0.98, ...]
            reconstructed_vectors_stack.append(outputs_list[2][0])

            ### Calculate Feature Importance.
            whether_calculate_importance_static = random() < feature_importance_calculate_prob_dict["static"]
            whether_calculate_importance_dynamic = random() < feature_importance_calculate_prob_dict["dynamic"]
            for feature_group in dataset.groups_of_features_info.keys():
                for feature_name in dataset.groups_of_features_info[feature_group].keys():
                    ## Set variable of dict for convenience.
                    feature_info_dict = dataset.groups_of_features_info[feature_group][feature_name] ## {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": [], "mean": None, "std": None}
                    feature_idx = feature_info_dict["idx"]
                    keras_input = dicts_input_to_keras_input_map[feature_info_dict["input"]]
                    dicts_input = feature_info_dict["input"]

                    feature_importance_changed_input = {} ## Changed input: for "perturbation", "swap" each.
                    if feature_info_dict["input"] in ["static_encoder", "raw"] and whether_calculate_importance_static and dataset.dicts[patient_idx][feature_info_dict["input"]]["mask"][0][feature_idx] == 1.: ## For static input features, and change dataset.dicts[patient_idx]["static_features_data"], only in case observed.
                        ## Prepare the changed inputs for each method, as the input is the only difference.
                        if perturbation_info_for_feature_importance is not None: ## "perturbation" input
                            if perturbation_info_for_feature_importance["center"] is "mean": perturbation_loc = feature_info_dict["mean"]
                            else: perturbation_loc = perturbation_info_for_feature_importance["center"]
                            perturbation_scalar = perturbation_info_for_feature_importance["proportion"] * np.random.normal(loc = perturbation_loc, scale= feature_info_dict["std"], size = 1)[0] ## PERTURBATION: gaussian: scalar.
                            feature_importance_changed_input["perturbation"] = deepcopy(dataset.dicts[patient_idx][dicts_input][self.data_kind_key])
                            feature_importance_changed_input["perturbation"][0][feature_idx] += perturbation_scalar ## Input after perturbation.
                        if swap_info_for_feature_importance is not None: ## "swap" input
                            feature_importance_changed_input["swap"] = deepcopy(dataset.dicts[patient_idx][dicts_input][self.data_kind_key])
                            feature_importance_changed_input["swap"][0][feature_idx] = sample(feature_info_dict["observed_numbers"], k= 1)[0]
                        ## Calculate the changes in the prediction, with the changed input for each method.
                        for method in feature_importance_changed_input.keys():
                            outputs_changed_list = self.model.predict_on_batch(x = merge_dictionaries([original_inputs_dict, {keras_input: feature_importance_changed_input[method]}])) ## Output after input changes, merge_dictionaries uses shallow copy so does not change original_dict.
                            changes_on_prediction = np.sum(np.abs((outputs_list[1] - outputs_changed_list[1])[0])) ## [0] for batch dim.
                            feature_importance_dict[method][feature_group][feature_name].append(changes_on_prediction)

                    elif feature_info_dict["input"] in ["RNN_vectors"] and whether_calculate_importance_dynamic and np.sum(dataset.dicts[patient_idx][feature_info_dict["input"]]["mask"][0, :, feature_idx]) >= 1.: ## Find features in RNN input, and change dataset.dicts[patient_idx][self.key_name_LSTM_input], only in case observed.
                        num_timesteps = dataset.dicts[patient_idx][dicts_input]["data"].shape[1] ## self.key_name_LSTM_input == "LSTM inputs data and mask concatenated" or "LSTM inputs data"
                        ## Prepare the changed inputs for each method, as the input is the only difference.
                        if perturbation_info_for_feature_importance is not None: ## "perturbation" input
                            if perturbation_info_for_feature_importance["center"] is "mean": perturbation_loc = feature_info_dict["mean"]
                            else: perturbation_loc = perturbation_info_for_feature_importance["center"]
                            perturbation_time_steps = perturbation_info_for_feature_importance["proportion"] * np.random.normal(loc = perturbation_loc, scale= feature_info_dict["std"], size = (num_timesteps, )) * dataset.dicts[patient_idx][dicts_input]["mask"][0, :, feature_idx] ## PERTURBATION: gaussian: (time_steps, ) * mask: (time_steps, )
                            feature_importance_changed_input["perturbation"] = deepcopy(dataset.dicts[patient_idx][dicts_input][self.data_kind_key]) ## self.key_name_LSTM_input == "LSTM inputs data and mask concatenated" or "LSTM inputs data"
                            feature_importance_changed_input["perturbation"][0, :, feature_idx] += perturbation_time_steps ## Input after perturbation.
                        if swap_info_for_feature_importance is not None: ## "swap" input
                            mask_to_swap = mask_prob(shape= (num_timesteps, ), p= swap_info_for_feature_importance["proportion"]) * dataset.dicts[patient_idx][dicts_input]["mask"][0, :, feature_idx] ## vector in shape (time_steps, ), 1 if entry to swap, 0 if entry to keep the original.
                            observed_numbers_pool = feature_info_dict["observed_numbers"]
                            sample_indices = np.random.choice(len(observed_numbers_pool), (num_timesteps, ), replace=True) ## Randomly choose the indices.
                            sampled_record = np.array([observed_numbers_pool[sample_indices[t]] for t in range(num_timesteps)]) ## Fully populated sampled numbers.
                            feature_importance_changed_input["swap"] = deepcopy(dataset.dicts[patient_idx][dicts_input][self.data_kind_key]) ## self.key_name_LSTM_input == "LSTM inputs data and mask concatenated" or "LSTM inputs data"
                            feature_importance_changed_input["swap"][0, :, feature_idx] = sampled_record * mask_to_swap + feature_importance_changed_input["swap"][0, :, feature_idx] * (1. - mask_to_swap)
                        ## Calculate the changes in the prediction, with the changed input for each method.
                        for method in feature_importance_changed_input.keys():
                            outputs_changed_list = self.model.predict_on_batch(x = merge_dictionaries([original_inputs_dict, {keras_input: feature_importance_changed_input[method]}])) ## Output after input changes, merge_dictionaries uses shallow copy so does not change original_dict.
                            changes_on_prediction = np.sum(np.abs((outputs_list[1] - outputs_changed_list[1])[0])) / np.sum(dataset.dicts[patient_idx][dicts_input]["mask"][0, :, feature_idx])
                            feature_importance_dict[method][feature_group][feature_name].append(changes_on_prediction)

        self.enriched_vectors_stack = np.array(enriched_vectors_stack)
        self.predicted_labels_stack = np.array(predicted_labels_stack)
        self.reconstructed_vectors_stack = reconstructed_vectors_stack ## cannot stack to numpy array, because the arrays have different shapes due to the different time steps.

        return {"enriched_vectors_stack": self.enriched_vectors_stack, "predicted_labels_stack": self.predicted_labels_stack, "reconstructed_vectors_stack": self.reconstructed_vectors_stack, "feature_importance_dict": feature_importance_dict}
    
    def clear_model(self):
        """Delete self.model to save it. Without deleting model, strange error occurs when saving because of RepeatVector layer."""

        K.clear_session()
        del self.model
    
    def dynamic_vectors_reconstruction_loss(self, decoder_vectors_labels_data, decoder_vectors_labels_mask, decoder_vectors_labels_data_predicted, factor = 1.0, kind = 0.5):
        """
        
        Parameters
        ----------
        decoder_vectors_labels_data_predicted : tensor
            It's shape is (1, time_steps, num_features_decoder_labels).
        """

        if isinstance(kind, (int, float, complex)) and not isinstance(kind, bool):
            p = kind
            loss = K.sum(decoder_vectors_labels_mask * K.abs(decoder_vectors_labels_data - decoder_vectors_labels_data_predicted) ** p) / (K.sum(decoder_vectors_labels_mask) + self.small_delta)
        else:
            raise Exception(NotImplementedError)

        return factor * loss
    
    def dynamic_images_reconstruction_loss(self, decoder_images_labels_data, decoder_images_labels_mask, decoder_images_labels_data_predicted, factor = 1.0, kind = 0.5):
        """
        
        Parameters
        ----------
        decoder_images_labels_data_predicted : tensor
            It's shape is (1, time_steps, rows, columns, channels).
        """

        if isinstance(kind, (int, float, complex)) and not isinstance(kind, bool):
            p = kind
            loss = K.sum(decoder_images_labels_mask * K.abs(decoder_images_labels_data - decoder_images_labels_data_predicted) ** p) / (K.sum(decoder_images_labels_mask) + self.small_delta)
        else:
            raise Exception(NotImplementedError)

        return factor * loss
    
    def static_reconstruction_loss(self, origina_static_data, origina_static_mask, predicted_static_data, factor = 1.0, kind = 0.5):
        """
        
        Parameters
        ----------
        origina_static_data, origina_static_mask, predicted_static_data : tensor
            It's shape is (1, num_features_static).
        """

        if isinstance(kind, (int, float, complex)) and not isinstance(kind, bool):
            p = kind
            loss = K.sum(origina_static_mask * K.abs(origina_static_data - predicted_static_data) ** p) / (K.sum(origina_static_mask) + self.small_delta)
        else:
            raise Exception(NotImplementedError)

        return factor * loss

    def prediction_loss(self, predictor_label_predicted, predictor_label, predictor_label_observability, factor = 1.0, kind = "binary cross entropy", loss_wrapper_funct = None, loss_factor_funct = None):
        """
        
        Parameters
        ----------
        predictor_label_observability : Tensor in shape (1, 1)
        """

        if loss_wrapper_funct is None: loss_wrapper_funct = lambda x: x
        if loss_factor_funct is None: loss_factor_funct = lambda **kwargs: 1.

        if kind == "binary cross entropy":
            loss = - 1.0 * K.sum(loss_wrapper_funct((predictor_label * K.log(predictor_label_predicted + self.small_delta) + (1 - predictor_label) * K.log(1 - predictor_label_predicted + self.small_delta)))) ## log(smaller than 1) is negative, log(larger than 1) is positive, because of the self.small_delta log(something) can be positive or negative both.
        elif kind == "SquaredHinge":
            loss = SquaredHinge()(predictor_label * 2. - 1., predictor_label_predicted * 2. - 1.)
        elif isinstance(kind, (int, float, complex)) and not isinstance(kind, bool): ## p is number, p-norm.
            p = kind
            loss = K.sum(loss_wrapper_funct(K.abs(predictor_label - predictor_label_predicted) ** p))
        else:
            raise Exception(NotImplementedError)
        
        loss = loss_factor_funct(predictor_label_predicted = predictor_label_predicted, predictor_label = predictor_label, predictor_label_observability = predictor_label_observability, factor = factor, kind = kind) * loss

        return factor * loss * predictor_label_observability[0][0]