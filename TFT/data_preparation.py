import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def prepare_data(file_path, window_size=60):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    target_cols = ['target_1d', 'target_5d', 'target_20d']
    forbidden = target_cols + ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    feature_cols = [c for c in df.columns if c.lower() not in forbidden]
    
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    train_x = scaler_x.fit_transform(train_df[feature_cols])
    train_y = scaler_y.fit_transform(train_df[target_cols])
    
    test_x = scaler_x.transform(test_df[feature_cols])
    test_y = scaler_y.transform(test_df[target_cols])
    
    def create_sequences(x, y, seq_length):
        X_s, y_s = [], []
        for i in range(len(x) - seq_length):
            X_s.append(x[i : i + seq_length])
            y_s.append(y[i + seq_length])
        return np.array(X_s), np.array(y_s)
    
    X_train, y_train = create_sequences(train_x, train_y, window_size)
    X_test, y_test = create_sequences(test_x, test_y, window_size)
    
    return X_train, y_train, X_test, y_test, scaler_y, feature_cols