import sys
from pathlib import Path

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent))

from Data.data_loader import load_data
from Data.preprocessing import prepare_data

from models.lstm import build_lstm
from models.informer import build_informer_model

from training.trainer import train
from training.evaluator import evaluate

from utils.visualization import plot_prediction

# ==================================================
# CONFIGURATION GLOBALE
# ==================================================

CONFIG = {
    "symbol": "^GSPC",
    "start": "2010-01-01",
    "end": "2025-01-01",
    "window_size": 60,
    "epochs": 50,
    "batch_size": 32,
    "horizons": [1, 5, 20]
}

# ==================================================
# REGISTRE DES MODÈLES
# ==================================================

MODELS = {
    "LSTM": lambda window: build_lstm(window),
    "INFORMER": lambda window: build_informer_model(
        window_size=window,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2
    )
}

# ==================================================
# MAIN
# ==================================================

def run_experiment():

    print("\n🔥 Chargement des données...")
    df = load_data(
        CONFIG["symbol"],
        CONFIG["start"],
        CONFIG["end"]
    )
    prices = df.values

    results = {}

    for model_name, model_builder in MODELS.items():
        print(f"\n{'='*60}")
        print(f"🚀 MODÈLE : {model_name}")
        print(f"{'='*60}")

        results[model_name] = {}

        for horizon in CONFIG["horizons"]:
            print(f"\n🔹 Horizon : {horizon} jour(s)")

            # --- Préparation données ---
            Xtr, ytr, Xv, yv, Xt, yt, scaler = prepare_data(
                prices,
                CONFIG["window_size"],
                horizon
            )

            # --- Modèle ---
            model = model_builder(CONFIG["window_size"])

            # --- Entraînement ---
            train(
                model,
                Xtr, ytr,
                Xv, yv,
                epochs=CONFIG["epochs"],
                batch_size=CONFIG["batch_size"]
            )

            # --- Évaluation ---
            y_true, y_pred, metrics = evaluate(
                model,
                Xt,
                yt,
                scaler
            )

            # --- Stockage ---
            results[model_name][horizon] = metrics

            # --- Visualisation ---
            plot_prediction(
                y_true,
                y_pred,
                title=f"{model_name} — Horizon {horizon} jour(s)"
            )

            print(
                f"MAE={metrics['mae']:.2f} | "
                f"RMSE={metrics['rmse']:.2f} | "
                f"Accuracy={metrics['accuracy']:.2f}%"
            )

    # ==================================================
    # RÉSUMÉ FINAL
    # ==================================================

    print("\n" + "="*70)
    print("📊 RÉSUMÉ FINAL")
    print("="*70)

    for model_name, horizons in results.items():
        print(f"\n🔷 {model_name}")
        for h, m in horizons.items():
            print(
                f"  Horizon {h:>2}j → "
                f"MAE={m['mae']:.2f} | "
                f"RMSE={m['rmse']:.2f} | "
                f"Acc={m['accuracy']:.2f}%"
            )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    run_experiment()