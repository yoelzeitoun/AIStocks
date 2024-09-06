import os
import numpy as np
from keras.src.layers import LayerNormalization, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Dropout, \
    LSTM, GRU, InputLayer, Attention, GlobalAveragePooling1D, Add, BatchNormalization
from keras.src.optimizers import Adam, Nadam
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.models import Sequential
from matplotlib import pyplot as plt
from tcn import TCN
import random
import tensorflow as tf
from plotter import PredictionVisualizer

# Model Creation and Training
def set_seed(seed_value=45):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

from enum import Enum

class Model(Enum):
    LSTM = "LSTM"
    MLP = "MLP"
    CNN = "CNN"
    RNN = "RNN"
    GRU = "GRU"
    CONV1D = "CONV1D"
    CONV2D = "CONV2D"
    TRANSFORMER = "TRANSFORMER"
    BERT = "BERT"
    GPT = "GPT"
    XGBOOST = "XGBOOST"
    RANDOM_FOREST = "RANDOM_FOREST"
    DECISION_TREE = "DECISION_TREE"
    SVM = "SVM"
    NAIVE_BAYES = "NAIVE_BAYES"
    KNN = "KNN"
    LINEAR_REGRESSION = "LINEAR_REGRESSION"
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
    AUTOENCODER = "AUTOENCODER"
    GAN = "GAN"
    TCN = "TCN"

class ModelBuilder:
    def __init__(self, data, model_type, plot=False):
        set_seed()
        self.data = data
        self.input_shape = (data.x_train.shape[1], data.x_train.shape[2])
        self.model_type = model_type
        self.model = self.create_model()
        self.compiled_model = self.compile_model(self.model)
        self.history = self.fit_model(self.compiled_model, data.x_train, data.y_train, data.x_val, data.y_val)
        self.test_loss, self.test_mae = self.evaluate_model(self.model, data.x_test, data.y_test)
        if plot:
            self.plot_history()
        PredictionVisualizer(self, data, plot)

    # Set random seeds for reproducibility

    def create_model(self):
        if self.model_type == Model.MLP:
            return self._create_mlp_model()
        elif self.model_type == Model.LSTM:
            return self._create_lstm_model()
        elif self.model_type == Model.CONV1D:
            return self._create_conv1d_model()
        elif self.model_type == Model.CONV2D:
            return self._create_conv2d_model()
        elif self.model_type == Model.GRU:
            return self._create_gru_model()
        elif self.model_type == Model.TCN:
            return self._create_tcn_model()
        else:
            raise ValueError(f"Model type {self.model_type} is not supported.")

    def _create_mlp_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        return model

    def _create_lstm_model(self):
        model = Sequential()
        model.add(LSTM(512, activation='tanh', input_shape=self.input_shape, return_sequences=True))
        # model.add(Dropout(0.2))
        model.add(LSTM(256, activation='tanh', return_sequences=True))
        # model.add(Dropout(0.2))
        model.add(LSTM(512, activation='tanh'))
        # model.add(Dropout(0.2))
        model.add(Dense(1))
        return model

    def _create_gru_model(self):
        model = Sequential()
        model.add(GRU(512, activation='tanh', input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(256, activation='tanh', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(512, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        return model

    def _create_conv1d_model(self):
        model = Sequential()
        # First Conv1D Layer with 'same' padding to preserve the input dimensions
        model.add(Conv1D(filters=512, kernel_size=2, activation='relu', input_shape=self.input_shape, padding='same'))
        # Reduce pool size to 1 to avoid shrinking the dimension to zero
        model.add(MaxPooling1D(pool_size=1))
        # Second Conv1D Layer with 'same' padding to prevent negative dimensions
        model.add(Conv1D(filters=512, kernel_size=2, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1))
        return model

    def _create_conv2d_model(self):
        model = Sequential()
        model.add(Conv2D(filters=256, kernel_size=(1, 2), activation='relu', input_shape=(self.data.x_train.shape[1], self.data.x_train.shape[2], 1)))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Conv2D(filters=128, kernel_size=(1, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1))
        return model

    def _create_tcn_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))

        # First TCN Layer
        model.add(TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=[1, 2],
                      padding='causal', use_skip_connections=True, return_sequences=True))
        model.add(LayerNormalization())
        model.add(Dropout(0.2))

        # Second TCN Layer
        model.add(TCN(nb_filters=64, kernel_size=2, nb_stacks=1, dilations=[1, 2],
                      padding='causal', use_skip_connections=True, return_sequences=False))
        model.add(LayerNormalization())
        model.add(Dropout(0.2))

        # Dense Layer for Prediction
        model.add(Dense(1))

        return model

    def compile_model(self, model):
        model.compile(optimizer=Nadam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
        return model

    def fit_model(self, model, x_train, y_train, x_val, y_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00000001, verbose=1)

        history = model.fit(
            x_train,
            y_train,
            epochs=1000,
            batch_size=32,
            validation_data=(x_val, y_val),
            callbacks=[reduce_lr, early_stopping],
            verbose=1,
            shuffle=False
        )
        return history

    def plot_history(self):
        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # MAE plot
        if 'mae' in self.history.history:
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['mae'], label='Train MAE')
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Absolute Error')
            plt.legend()

        plt.tight_layout()
        plt.show()


    def evaluate_model(self, model, x_test, y_test):
        test_loss, test_mae = model.evaluate(x_test, y_test)
        return test_loss, test_mae
