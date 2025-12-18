import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    """
    Composant clé du TFT : Filtre les entrées via une Gated Linear Unit (GLU).
    Permet au modèle de ne laisser passer que les signaux financiers pertinents.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(input_size, output_size)
        self.elu = nn.ELU()
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Chemin de transformation
        h = self.elu(self.lin1(x))
        h = self.dropout(self.lin2(h))
        # Mécanisme de porte (Gate)
        g = self.sigmoid(self.gate(x))
        # Connexion résiduelle + Normalisation
        return self.layer_norm(x + g * h)

class MultiHorizonTFT(nn.Module):
    """
    Architecture Temporal Fusion Transformer (TFT) optimisée.
    Prédit simultanément 3 horizons (1j, 5j, 20j) avec 3 quantiles.
    """
    def __init__(self, n_features, hidden_size=128, n_heads=8):
        super(MultiHorizonTFT, self).__init__()
        
        # 1. Projection des features d'entrée
        self.input_projection = nn.Linear(n_features, hidden_size)
        
        # 2. Variable Selection Network (VSN) simplifié via GRN
        self.vsn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size)
        
        # 3. Attention Temporelle (Multi-Head Attention)
        # Permet de focuser sur des jours spécifiques dans la fenêtre de 60 jours
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)
        self.ln_attn = nn.LayerNorm(hidden_size)
        
        # 4. Tête de sortie Quantile (3 horizons * 3 quantiles = 9)
        self.quantile_output = nn.Linear(hidden_size, 9)

    def forward(self, x):
        # x shape attendue: (Batch, 60, n_features)
        
        # Projection et filtrage
        x = self.input_projection(x)
        x = self.vsn(x)
        
        # Calcul de l'attention sur la séquence temporelle
        attn_out, _ = self.attn(x, x, x)
        x = self.ln_attn(x + attn_out)
        
        # On extrait uniquement le dernier vecteur (le présent) pour prédire le futur
        last_step = x[:, -1, :]
        
        # Génération des 9 sorties et remodelage en (Batch, Horizon, Quantile)
        out = self.quantile_output(last_step)
        return out.view(-1, 3, 3)

# --- SECTION DE TEST (POUR LE TERMINAL) ---
if __name__ == "__main__":
    print("\n--- Test de l'architecture Multi-Horizon TFT ---")
    
    # Simulation de paramètres
    batch_size = 32
    seq_len = 60
    n_features = 14
    
    model = MultiHorizonTFT(n_features=n_features)
    sample_input = torch.randn(batch_size, seq_len, n_features)
    
    print(f"Entrée : {sample_input.shape} (Batch, Jours, Features)")
    
    # Forward pass de test
    output = model(sample_input)
    
    print(f"Sortie : {output.shape} (Batch, Horizons, Quantiles)")
    
    if output.shape == (batch_size, 3, 3):
        print("✅ Architecture validée : Les dimensions sont parfaites.\n")
    else:
        print("❌ Erreur de dimensions dans la sortie du modèle.")