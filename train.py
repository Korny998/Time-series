from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from constants import EPOCHS, NUM_FEATURES
from dataset import load_series, make_datasets

from graphs_example import get_pred, show_corr, show_predict, history_plot
from models import (
    fully_connected_network,
    LSTM_network,
    one_dimensional_convolution)


URL = 'https://storage.yandexcloud.net/academy.ai/AAPL.csv'

MODELS = [
    ('Fully Connected Network', fully_connected_network),
    ('LSTM(50)', LSTM_network),
    (
        'One-dimensional convolution',
        one_dimensional_convolution
    )
]


def train_model(model_builder, input_shape, train_gen, val_gen):
    model = model_builder(input_shape, NUM_FEATURES)
    model.compile(
        optimizer=Adam(),
        loss=MeanSquaredError(),
        metrics=['mae']
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
    )
    return model, history


def evaluate_model(model, history, title, x_test, y_test, scaler):
    history_plot(history, title)
    y_pred, y_true = get_pred(model, x_test, y_test, scaler)
    show_predict(y_pred, y_true, title=f'{title}. True vs Pred')
    show_corr(y_pred, y_true, title=f'{title}. Correlation')


if __name__ == '__main__':
    data = load_series(URL)
    train_gen, val_gen, x_test, y_test, scaler = make_datasets(data)

    input_shape = train_gen[0][0].shape[1:]

    for title, model_builder in MODELS:
        model, history = train_model(
            model_builder,
            input_shape,
            train_gen,
            val_gen
        )
        evaluate_model(model, history, title, x_test, y_test, scaler)
