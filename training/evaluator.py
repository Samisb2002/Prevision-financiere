import numpy as np

def evaluate(model, X_test, y_test, scaler):
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    acc = 100 - (mae / y_true.mean() * 100)

    return y_true, y_pred, {'mae': mae, 'rmse': rmse, 'accuracy': acc}
