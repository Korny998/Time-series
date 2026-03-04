import io

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from constants import BATCH_SIZE, TEST_START, TRAIN_END, WINDOW_SIZE


def load_series(url):
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_csv(
        io.StringIO(r.text),
        index_col="Date",
        usecols=["Adj Close", "Date"],
        parse_dates=["Date"],
    )


def make_datasets(data):
    train = data[:TRAIN_END]
    val = data[TRAIN_END:TEST_START]
    test = data[TEST_START:]

    scaler = MinMaxScaler()
    scaler.fit(train)

    train_s = scaler.transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)

    val_input = np.vstack([train_s[-WINDOW_SIZE:], val_s])
    test_input = np.vstack([val_s[-WINDOW_SIZE:], test_s])

    train_gen = TimeseriesGenerator(
        train_s,
        train_s,
        length=WINDOW_SIZE,
        batch_size=BATCH_SIZE
    )
    val_gen = TimeseriesGenerator(
        val_input,
        val_input,
        length=WINDOW_SIZE,
        batch_size=BATCH_SIZE
    )

    test_gen = TimeseriesGenerator(
        test_input,
        test_input,
        length=WINDOW_SIZE,
        batch_size=len(test_input)
    )
    x_test, y_test = test_gen[0]

    return train_gen, val_gen, x_test, y_test, scaler
