from tensorflow.keras import models, layers

from dataset import generator


def fully_connected_network(n_features):
    return models.Sequential([
        layers.Dense(
            100,
            input_shape=generator[0][0].shape[1:],
            activation='relu'
        ),
        layers.Flatten(),
        layers.Dense(n_features, activation='linear')
    ])


def LSTM_network(n_features):
    return models.Sequential([
        layers.LSTM(
            50,
            input_shape=generator[0][0].shape[1:],
            activation='relu'
        ),
        layers.Dense(10, activation='relu'),
        layers.Dense(n_features)
    ])


def one_dimensional_convolution(n_features):
    return models.Sequential([
        layers.Conv1D(
            64,
            4,
            input_shape=generator[0][0].shape[1:],
            activation='relu'
        ),
        layers.Conv1D(
            64,
            4,
            activation='relu'
        ),
        layers.MaxPooling1D(),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dense(n_features, activation='linear')
    ])
