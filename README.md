# Neural Networks for Time-Series Forecasting (AAPL Stock Price)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A simple deep learning project demonstrating how different neural network architectures can be used for **time-series forecasting**.

The project predicts the **Apple (AAPL) adjusted closing price** using historical data and compares three neural network architectures:

- Fully Connected Neural Network
- LSTM (Long Short-Term Memory)
- 1D Convolutional Neural Network

The goal is to illustrate how deep learning models can capture **temporal patterns in financial time-series data**.

---

# 📁 Project Structure

```bash
├── constants.py        # Global configuration and hyperparameters
├── dataset.py          # Data loading, scaling, and sequence generation
├── models.py           # Neural network architectures
├── train.py            # Training and evaluation pipeline
├── graphs_example.py   # Visualization utilities
├── venv/               # Virtual environment (not tracked)
└── README.md           # Project documentation
```

---

# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
- [Training Pipeline](#training-pipeline)
- [Visualization](#visualization)
- [License](#license)

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Korny998/Time-series.git
cd time-series-neural-networks
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Usage

To train all models and visualize the results, run:

```bash
python train.py
```

The script will automatically:

1. Download the **AAPL historical dataset**
2. Normalize the data using **MinMaxScaler**
3. Create time-series windows
4. Train multiple neural network architectures
5. Evaluate model predictions
6. Plot training history and prediction results

---

# Dataset

The dataset contains historical **Apple stock prices**.

Source:

```
https://storage.yandexcloud.net/academy.ai/AAPL.csv
```

Features used:

- **Adj Close** — adjusted closing price

The dataset is loaded dynamically using `requests` and processed with `pandas`.

---

# Data Preprocessing

The preprocessing pipeline performs the following steps:

### 1. Train / Validation / Test Split

The dataset is split chronologically:

- Training data: before `TRAIN_END`
- Validation data: between `TRAIN_END` and `TEST_START`
- Test data: after `TEST_START`

This preserves the **temporal order** of the time series.

---

### 2. Data Normalization

The training data is normalized using:

```
MinMaxScaler
```

The same scaler is applied to validation and test sets.

---

### 3. Time-Series Window Generation

Training samples are created using **sliding windows** with:

```
WINDOW_SIZE = 14
```

Each training example uses **14 previous time steps** to predict the next value.

This is implemented using:

```
TimeseriesGenerator (Keras)
```

---

# Model Architectures

The project compares three different neural network architectures.

---

## 1️⃣ Fully Connected Neural Network

Architecture:

```
Dense(100) → ReLU
Flatten
Dense(1)
```

Purpose:

- Baseline model
- Learns simple nonlinear relationships

---

## 2️⃣ LSTM Network

Architecture:

```
LSTM(50)
Dense(10) → ReLU
Dense(1)
```

Purpose:

- Captures **long-term temporal dependencies**
- Widely used for time-series forecasting

---

## 3️⃣ 1D Convolutional Neural Network

Architecture:

```
Conv1D(64, kernel_size=4) → ReLU
Conv1D(64, kernel_size=4) → ReLU
MaxPooling1D
Flatten
Dense(50) → ReLU
Dense(1)
```

Purpose:

- Learns **local temporal patterns**
- Often faster than recurrent models

---

# Training Pipeline

The training process follows these steps:

1. Load and preprocess the dataset
2. Generate time-series windows
3. Build each model architecture
4. Compile models with:

```
Adam optimizer
Mean Squared Error loss
Mean Absolute Error metric
```

5. Train models for:

```
EPOCHS = 20
```

6. Evaluate predictions on the test set

---

# Visualization

The `graphs_example.py` module generates several plots.

### Training Curves

Shows:

- Training loss
- Validation loss

These help detect:

- overfitting
- convergence issues
- training stability

---

### Forecast vs True Values

Plots the model's predicted time series against the actual values.

---

### Correlation Analysis

Computes cross-correlation between:

- predicted values
- actual values

This helps evaluate **temporal prediction quality**.

---

# Example Workflow

```
Load data → Scale data → Generate sequences →
Train models → Evaluate predictions → Plot results
```

---

# License

This project is released under the **MIT License**.

You are free to use, modify, and distribute the code for educational or commercial purposes.