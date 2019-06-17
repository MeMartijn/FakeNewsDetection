import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Reshape, Conv1D, Flatten
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from hypopt import GridSearch

class Classifiers:
    '''Interface for using pre-defined classifiers'''
    @staticmethod
    def get_bilstm_score(X_train, X_test, X_validation, y_train, y_test, y_validation, reshape = False):
        # Rearrange data types
        params = locals().copy()
        inputs = {
            dataset: np.array(params[dataset])
            for dataset in params.keys()
        }

        for dataset in inputs.keys():
            if dataset[0:1] == 'X' and reshape:
                # Reshape datasets from 2D to 3D
                inputs[dataset] = np.reshape(
                    inputs[dataset], (inputs[dataset].shape[0], inputs[dataset].shape[1], 1))
            elif dataset[0:1] == 'y':
                inputs[dataset] = np_utils.to_categorical(
                    np.array(inputs[dataset]), 3)

        # Set model parameters
        epochs = 5
        batch_size = 64
        input_shape = X_train.shape

        # Create the model
        model = Sequential()
        model.add(Bidirectional(LSTM(64, input_shape=input_shape)))
        model.add(Dropout(0.8))
        model.add(Dense(3, activation='softmax'))
        model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

        # Fit the training set over the model and correct on the validation set
        model.fit(inputs['X_train'], inputs['y_train'],
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(inputs['X_validation'], inputs['y_validation']))

        # Get score over the test set
        score, acc = model.evaluate(inputs['X_test'], inputs['y_test'])

        print(acc)
        return acc

    @staticmethod
    def get_cnn_score(X_train, X_test, X_validation, y_train, y_test, y_validation, reshape = False):
        # Rearrange data types
        params = locals().copy()
        inputs = {
            dataset: np.array(params[dataset])
            for dataset in params.keys()
        }

        # Reshape datasets
        for dataset in inputs.keys():
            if dataset[0:1] == 'X':
                if reshape:
                    inputs[dataset] = np.reshape(
                        inputs[dataset], (inputs[dataset].shape[0], inputs[dataset].shape[1], 1))

            elif dataset[0:1] == 'y':
                inputs[dataset] = np_utils.to_categorical(
                    np.array(inputs[dataset]), 3)

        # Set model parameters
        epochs = 5
        batch_size = 64
        input_shape = inputs['X_train'].shape

        # Create the model
        model = Sequential()
        model.add(Conv1D(128, kernel_size=2, activation='relu', input_shape=(
            input_shape[1], input_shape[2]), data_format='channels_first'))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(Conv1D(128, kernel_size=4, activation='relu'))
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(3, activation='softmax'))
        model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

        # Fit the training set over the model and correct on the validation set
        model.fit(inputs['X_train'], inputs['y_train'],
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(inputs['X_validation'], inputs['y_validation']))

        # Get score over the test set
        score, acc = model.evaluate(inputs['X_test'], inputs['y_test'])

        print(acc)
        return acc

    @staticmethod
    def get_svm_score(X_train, X_test, X_validation, y_train, y_test, y_validation, penalty = 'l2'):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gs = GridSearch(model = LogisticRegression(penalty = penalty), param_grid = param_grid)
        gs.fit(X_train, y_train, X_validation, y_validation)

        return gs.score(X_test, y_test)

    @staticmethod
    def get_logres_score(X_train, X_test, X_validation, y_train, y_test, y_validation, penalty = 'l2'):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gs = GridSearch(model = LogisticRegression(penalty = penalty), param_grid = param_grid)
        gs.fit(X_train, y_train, X_validation, y_validation)

        return gs.score(X_test, y_test)
    
    @staticmethod
    def get_gradientboosting_score(X_train, X_test, X_validation, y_train, y_test, y_validation):
        param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01]}
        gs = GridSearch(model = GradientBoostingClassifier(), param_grid = param_grid)
        gs.fit(X_train, y_train, X_validation, y_validation)

        return gs.score(X_test, y_test)
