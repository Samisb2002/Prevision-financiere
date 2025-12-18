import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
from TFT import MultiHorizonTFT
from data_preparation import prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quantile_loss(preds, targets, quantiles):
    total_loss = 0
    for h in range(3): 
        for i, q in enumerate(quantiles):
            error = targets[:, h] - preds[:, h, i]
            total_loss += torch.max((q - 1) * error, q * error).mean()
    return total_loss

def run_training():
    X_train, y_train, X_test, y_test, scaler_y, f_cols = prepare_data("C:/Users/samis/Desktop/Prevision-financiere/TFT/sp500_features.csv")
    
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32)
    ), batch_size=64, shuffle=True)
    
    model = MultiHorizonTFT(n_features=len(f_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    quantiles = torch.tensor([0.1, 0.5, 0.9]).to(DEVICE)
    
    print(f"\n🚀 Entraînement sur {DEVICE} | {len(f_cols)} features")
    
    model.train()
    for epoch in range(100):
        epoch_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Époque {epoch+1}/100")
            for bx, by in tepoch:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                
                optimizer.zero_grad()
                out = model(bx)
                loss = quantile_loss(out, by, quantiles)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=f"{loss.item():.4f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/tft_multi.pth")
    print("\n✅ Modèle sauvegardé : results/tft_multi.pth")
    return model, X_test, y_test, scaler_y

if __name__ == "__main__":
    run_training()