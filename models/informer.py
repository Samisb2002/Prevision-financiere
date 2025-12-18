import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt




# ============================================
# POSITIONAL ENCODING (pour Transformer)
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
# INFORMER ENCODER LAYER
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
# CONSTRUIRE LE MODÈLE INFORMER
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
