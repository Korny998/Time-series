import numpy as np
import matplotlib.pyplot as plt


def history_plot(history, title):
    fig = plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Error in the training set')
    plt.plot(history.history['val_loss'], label='Error on the test set')
    plt.title(f'{title}. Training schedule')

    fig.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('Epoch')
    plt.ylabel('Mean error')
    plt.legend()
    plt.show()


def correlate(a, b):
    return np.corrcoef(a, b)[0, 1]


def show_predict(y_pred, y_true, title=''):
    fig = plt.figure(figsize=(14, 7))
    plt.plot(y_pred[1:], label='Forecast')
    plt.plot(y_true[:-1], label='Basic')
    plt.title(title)

    fig.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('Date (relative to the start of sampling)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def get_pred(
        model,
        x_test, y_test,
        y_scaler
):
    y_pred_unscaled = y_scaler.inverse_transform(
        model.predict(x_test, verbose=0)
    )
    y_test_unscaled = y_scaler.inverse_transform(y_test)
    return y_pred_unscaled, y_test_unscaled


def show_corr(y_pred, y_true, title='', break_step=30):
    y_len = y_true.shape[0]
    steps = range(1, np.min([y_len+1, break_step+1]))
    cross_corr = [
        correlate(
            y_true[:-step, 0], y_pred[step:, 0]
        ) for step in steps
    ]
    auto_corr = [
        correlate(
            y_true[:-step, 0], y_true[step:, 0]
        ) for step in steps
    ]
    plt.plot(steps, cross_corr, label='Forecast')
    plt.plot(steps, auto_corr, label='Reference')
    plt.title(title)
    plt.xticks(steps)
    plt.xlabel('Displacement steps')
    plt.ylabel('Correlation coefficient')
    plt.legend()
    plt.show()
