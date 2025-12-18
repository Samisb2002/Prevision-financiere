import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
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
# 2. POSITIONAL ENCODING (pour Transformer)
# ============================================

class PositionalEncoding(Layer):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(seq_len, d_model)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, seq_len, d_model):
        angle_rads = self.get_angles(
            np.arange(seq_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


# ============================================
# 3. INFORMER ENCODER LAYER
# ============================================

class InformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(InformerEncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, x, training):
        # Multi-head attention
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


# ============================================
# 4. CRÉER LES SÉQUENCES
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
# 5. CONSTRUIRE LE MODÈLE INFORMER
# ============================================

def build_informer_model(window_size, d_model=64, num_heads=4, dff=256, 
                         num_encoder_layers=2, dropout_rate=0.1):
    """
    Construit un modèle Informer simplifié pour la prédiction de séries temporelles
    
    Args:
        window_size: taille de la fenêtre d'entrée
        d_model: dimension du modèle
        num_heads: nombre de têtes d'attention
        dff: dimension du feed-forward network
        num_encoder_layers: nombre de couches d'encodeur
        dropout_rate: taux de dropout
    """
    # Input
    inputs = Input(shape=(window_size, 1))
    
    # Projection linéaire pour augmenter la dimension
    x = Dense(d_model)(inputs)
    
    # Positional Encoding
    pos_encoding = PositionalEncoding(window_size, d_model)
    x = pos_encoding(x)
    
    # Encoder Layers
    for _ in range(num_encoder_layers):
        encoder_layer = InformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
        x = encoder_layer(x, training=True)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dropout final
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


# ============================================
# 6. PIPELINE COMPLET
# ============================================

def run_prediction(symbol='AAPL', start='2010-01-01', end='2025-01-01',
                   window_size=60, horizon=1, epochs=50, batch_size=32,
                   d_model=64, num_heads=4, num_encoder_layers=2):
    
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
    print("5. Construction du modèle Informer...")
    model = build_informer_model(
        window_size=window_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers
    )
    
    print(f"   Paramètres: d_model={d_model}, num_heads={num_heads}, layers={num_encoder_layers}")
    
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
    plt.title(f'{symbol} - Prédiction INFORMER (Horizon: {horizon} jour(s))')
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
    horizons = [1, 5, 20]
    
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
            batch_size=32,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2
        )
        
        results[horizon] = metrics
    
    # Résumé
    print("\n" + "=" * 50)
    print("RÉSUMÉ MULTI-HORIZONS (INFORMER)")
    print("=" * 50)
    print(f"{'Horizon':<10} {'MAE':<12} {'RMSE':<12} {'Accuracy':<12}")
    print("-" * 46)
    for h, m in results.items():
        print(f"{h} jour(s)   {m['mae']:<12.2f} {m['rmse']:<12.2f} {m['accuracy']:<12.2f}%")