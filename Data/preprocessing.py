import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + horizon - 1])
    return np.array(X), np.array(y)

def prepare_data(prices, window_size, horizon):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = create_sequences(scaled, window_size, horizon)

    n = len(X)
    train, val = int(0.6*n), int(0.8*n)

    return (
        X[:train], y[:train],
        X[train:val], y[train:val],
        X[val:], y[val:],
        scaler
    )
