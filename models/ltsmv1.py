import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# ============================================
# 1. TÉLÉCHARGER LES DONNÉES
# ============================================

def get_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    return df


# ============================================
# 2. CRÉER LES SÉQUENCES
# ============================================

def create_sequences(data, window_size, horizon):
    """
    data: array normalisé
    window_size: nombre de jours en entrée
    horizon: nombre de jours à prédire
    """
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + horizon - 1])
    return np.array(X), np.array(y)


# ============================================
# 3. CONSTRUIRE LE MODÈLE
# ============================================

def build_model(window_size, n_features=1):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ============================================
# 4. PIPELINE COMPLET
# ============================================

def run_prediction(symbol='AAPL', start='2010-01-01', end='2025-01-01',
                   window_size=60, horizon=1, epochs=50, batch_size=32):
    
    # --- Données ---
    print("1. Téléchargement des données...")
    df = get_data(symbol, start, end)
    prices = df['close'].values.reshape(-1, 1)
    
    # --- Normalisation ---
    print("2. Normalisation...")
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    
    # --- Séquences ---
    print("3. Création des séquences...")
    X, y = create_sequences(prices_scaled, window_size, horizon)
    
    # --- Split (60/20/20) ---
    print("4. Split des données...")
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # --- Modèle ---
    print("5. Construction du modèle...")
    model = build_model(window_size)
    
    # --- Entraînement ---
    print("6. Entraînement...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # --- Prédiction ---
    print("7. Prédiction...")
    y_pred_scaled = model.predict(X_test)
    
    # --- Dénormalisation ---
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    
    # --- Métriques ---
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    accuracy = 100 - (mae / np.mean(y_true) * 100)
    
    print("\n" + "=" * 40)
    print("RÉSULTATS")
    print("=" * 40)
    print(f"MAE:      {mae:.2f}")
    print(f"RMSE:     {rmse:.2f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # --- Visualisation ---
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label='Réel', linewidth=2)
    plt.plot(y_pred, label='Prédit', linewidth=2, alpha=0.8)
    plt.title(f'{symbol} - Prédiction LSTM (Horizon: {horizon} jour(s))')
    plt.xlabel('Jours')
    plt.ylabel('Prix')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    return model, scaler, history, {'mae': mae, 'rmse': rmse, 'accuracy': accuracy}


# ============================================
# EXÉCUTION
# ============================================

if __name__ == "__main__":

    results = {}
    horizons=[1, 5, 20]
    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"HORIZON: {horizon} JOUR(S)")
        print(f"{'='*50}")
        
        model, scaler, history, metrics = run_prediction(
            symbol='^GSPC',
            start='2018-01-01',
            end='2024-01-01',
            window_size=60,
            horizon=horizon,
            epochs=50,
            batch_size=32
        )
        
        results[horizon] = metrics
    
    # Résumé
    print("\n" + "=" * 50)
    print("RÉSUMÉ MULTI-HORIZONS")
    print("=" * 50)
    print(f"{'Horizon':<10} {'MAE':<12} {'RMSE':<12} {'Accuracy':<12}")
    print("-" * 46)
    for h, m in results.items():
        print(f"{h} jour(s)   {m['mae']:<12.2f} {m['rmse']:<12.2f} {m['accuracy']:<12.2f}%")
    


