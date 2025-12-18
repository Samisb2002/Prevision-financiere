import pandas as pd
import numpy as np

# =====================================
# 0. CHARGEMENT DES DONNÉES
# =====================================

df = pd.read_csv("Data/Apple.csv", parse_dates=['Date'])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# =====================================
# 1. RETURNS & MOMENTUM
# =====================================

df["Return_1"] = df["Close"].pct_change()
df["LogReturn_1"] = np.log(df["Close"]).diff()

for n in [5, 10, 20]:
    df[f"Momentum_{n}"] = df["Close"].pct_change(n)

# =====================================
# 2. RSI (Relative Strength Index)
# =====================================

def compute_RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    avg_gain = up.rolling(period).mean()
    avg_loss = down.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI_14"] = compute_RSI(df["Close"], 14)

# =====================================
# 3. MACD
# =====================================

df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# =====================================
# 4. BOLLINGER BANDS
# =====================================

window = 20
df["BB_Mid"] = df["Close"].rolling(window).mean()
df["BB_Std"] = df["Close"].rolling(window).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

# =====================================
# 5. VOLATILITÉ
# =====================================

# True Range pour ATR
df["H-L"] = df["High"] - df["Low"]
df["H-PC"] = (df["High"] - df["Close"].shift()).abs()
df["L-PC"] = (df["Low"] - df["Close"].shift()).abs()

df["TrueRange"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
df["ATR_14"] = df["TrueRange"].rolling(14).mean()

# Volatilité des returns
df["Volatility_20"] = df["Return_1"].rolling(20).std()

# =====================================
# 6. VOLUME INDICATORS
# =====================================

df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
df["Volume_Normalized"] = df["Volume"] / df["Volume_SMA20"]

# OBV (On-Balance Volume)
obv = [0]
for i in range(1, len(df)):
    if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
        obv.append(obv[-1] + df["Volume"].iloc[i])
    elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
        obv.append(obv[-1] - df["Volume"].iloc[i])
    else:
        obv.append(obv[-1])

df["OBV"] = obv

# =====================================
# 7. SAUVEGARDE FINALE
# =====================================

df.to_csv("Data/Applefeatures.csv")
print("🎉 Tous les indicateurs ont été calculés dans sp500_features.csv")
