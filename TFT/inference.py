import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from TFT import MultiHorizonTFT
from data_preparation import prepare_data


def visualize_results():
    """
    Charge le modèle entraîné et affiche les prédictions
    avec UN graphe séparé pour chaque horizon (1j, 5j, 20j).
    """

    # 1. Chargement et préparation des données
    print("⏳ Préparation des données de test...")
    X_train, y_train, X_test, y_test, scaler_y, f_cols = prepare_data(
        "C:/Users/samis/Desktop/Prevision-financiere/TFT/sp500_features.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Chargement du modèle
    model = MultiHorizonTFT(n_features=len(f_cols)).to(device)

    model_path = "results/tft_multi.pth"
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Le fichier {model_path} est introuvable. Lancez train.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Prédictions
    print(f"🔮 Génération des prédictions sur {device}...")
    with torch.no_grad():
        test_input = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(test_input).cpu().numpy()  # Shape: [N, 3, 3]

    # 4. Visualisation : UN graphe par horizon
    horizons_labels = ["Horizon 1 Jour", "Horizon 5 Jours", "Horizon 20 Jours"]
    os.makedirs("results", exist_ok=True)

    for h in range(3):
        plt.figure(figsize=(15, 5))

        # Dénormalisation des valeurs réelles
        actual = scaler_y.inverse_transform(y_test)[:, h]

        # Dénormalisation des quantiles
        p10 = scaler_y.inverse_transform(
            np.repeat(preds[:, h, 0:1], 3, axis=1)
        )[:, h]

        p50 = scaler_y.inverse_transform(
            np.repeat(preds[:, h, 1:2], 3, axis=1)
        )[:, h]

        p90 = scaler_y.inverse_transform(
            np.repeat(preds[:, h, 2:3], 3, axis=1)
        )[:, h]

        # Tracé
        plt.plot(
            actual,
            label="Réalité ",
            color="black",
            alpha=0.3,
            lw=1
        )

        plt.plot(
            p50,
            label="Prédiction  (TFT)",
            color="blue",
            lw=1.5
        )

        plt.fill_between(
            range(len(actual)),
            p10,
            p90,
            color="blue",
            alpha=0.15,
        )

        plt.title(horizons_labels[h], fontsize=14, fontweight="bold")
        plt.xlabel("Échantillons de Test (Temps)")
        plt.ylabel("Rendements")
        plt.grid(True, alpha=0.2)
        plt.legend(loc="upper left")

        # Sauvegarde
        filename = f"results/predictions_horizon_{h+1}.png"
        plt.tight_layout()
        plt.savefig(filename)
        print(f"✅ Graphique sauvegardé : {filename}")

        plt.show()


# Point d'entrée
if __name__ == "__main__":
    visualize_results()
