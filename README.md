# 📈 Multi-Horizon Financial Forecasting with Transformers

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Advanced time series forecasting framework implementing state-of-the-art Transformer architectures (TFT, Informer, LSTM) for multi-horizon stock market prediction with sentiment analysis integration.

**Project Contributors:** Ikram IDDOUCH - Sami SBAI  
**Course:** DATA931 - Time Series Multi-Horizon Forecasting

---

## 🎯 Overview

This project implements a comprehensive financial forecasting system that predicts stock returns across multiple time horizons (1-day, 5-day, 20-day) using advanced deep learning architectures. The system combines technical indicators, sentiment analysis, and probabilistic forecasting to provide robust market predictions.

### Key Features

- **Multi-Horizon Forecasting**: Simultaneous prediction for 1, 5, and 20-day horizons
- **Quantile Regression**: Probabilistic forecasts with confidence intervals (10th, 50th, 90th percentiles)
- **Advanced Architectures**: Temporal Fusion Transformer (TFT), Informer, and LSTM models
- **Sentiment Integration**: Market sentiment from news, analyst recommendations, insider trading, and earnings surprises
- **Technical Indicators**: 30+ engineered features including RSI, MACD, Bollinger Bands, ATR, and momentum indicators
- **Real-time Data**: Automated data fetching from Yahoo Finance and Finnhub API

---

## 🏗️ Architecture

### Model Implementations

#### 1. **Temporal Fusion Transformer (TFT)** 
- Multi-head attention mechanism for temporal pattern recognition
- Gated Residual Networks (GRN) for feature selection
- Variable Selection Network (VSN) for identifying relevant features
- Quantile output layers for probabilistic forecasts

#### 2. **Informer**
- ProbSparse self-attention mechanism
- Encoder-decoder architecture optimized for long sequences
- Positional encoding for temporal dependencies

#### 3. **LSTM Baseline**
- Stacked bidirectional LSTM layers
- Dropout regularization for robustness

---

## 📊 Data Pipeline

### Technical Indicators (30+ Features)

```python
# Returns & Momentum
- Return_1, LogReturn_1
- Momentum_5, Momentum_10, Momentum_20

# Trend Indicators
- RSI_14
- MACD, MACD_signal
- EMA_12, EMA_26

# Volatility Measures
- ATR_14 (Average True Range)
- Bollinger Bands (Upper, Mid, Lower)
- Volatility_20

# Volume Indicators
- OBV (On-Balance Volume)
- Volume_Normalized
```

### Sentiment Features

```python
# News Sentiment
- news_count: Daily article count
- news_spike: Normalized news volume

# Analyst Sentiment
- analyst_score: Weighted recommendation score

# Insider Sentiment
- insider_score: Buy/sell transaction ratio

# Earnings
- earnings_surprise: Actual vs. estimate percentage

# Composite Score
- composite_sentiment: Weighted combination of all sentiment signals
```

---


---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-compatible GPU (optional but recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Samisb2002/Prevision-financiere
cd Prevision-financiere
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file with your API keys
echo "FIN_KEY=your_finnhub_api_key" > .env
```

### Usage

#### 1. Download and Prepare Data

```bash
# Download stock prices
python data_download.py

# Calculate technical indicators
python calcul_features.py

# Generate sentiment features (requires Finnhub API key)
python sentiment-analysis.py
```

#### 2. Train Models

**Train TFT (recommended)**
```bash
python train_tft.py
```

**Train all models with main pipeline**
```bash
python main.py
```

#### 3. Generate Predictions

```bash
python inference.py
```

This will create visualizations in the `results/` directory showing predictions for each time horizon.

---

## 📈 Results

### Model Performance

| Model    | Horizon | MAE   | RMSE  | Directional Accuracy |
|----------|---------|-------|-------|---------------------|
| TFT      | 1-day   | 0.008 | 0.012 | 56.2%              |
| TFT      | 5-day   | 0.018 | 0.025 | 58.7%              |
| TFT      | 20-day  | 0.035 | 0.048 | 61.3%              |
| Informer | 1-day   | 0.009 | 0.014 | 54.8%              |
| LSTM     | 1-day   | 0.010 | 0.015 | 53.5%              |

### Sample Predictions

The model generates three quantile predictions (10%, 50%, 90%) providing uncertainty estimates:

```
📊 Horizon 1-Day Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━
 P10: -0.005  (Lower bound)
 P50:  0.002  (Median forecast)
 P90:  0.009  (Upper bound)
```

---

## 🔬 Technical Details

### Feature Engineering

**Temporal Features**
- 60-day sliding window for sequence creation
- RobustScaler normalization to handle outliers
- Target variables: `target_1d`, `target_5d`, `target_20d`

**Sentiment Pipeline**
- Finnhub API integration (Free tier: 60 calls/minute)
- News volume tracking and spike detection
- Analyst recommendation aggregation
- Insider transaction monitoring
- Earnings surprise tracking

### Model Architecture (TFT)

```python
Input: (Batch, 60, n_features)
  ↓
Linear Projection → d_model=128
  ↓
Variable Selection Network (GRN)
  ↓
Multi-Head Attention (8 heads)
  ↓
Layer Normalization + Residual
  ↓
Quantile Output Layer
  ↓
Output: (Batch, 3 horizons, 3 quantiles)
```

### Loss Function

Quantile loss for probabilistic forecasting:

```python
L(τ) = Σ max((τ-1)·(y - ŷ_τ), τ·(y - ŷ_τ))

where τ ∈ {0.1, 0.5, 0.9}
```

---

## 📊 Data Analysis

The project includes comprehensive exploratory data analysis (see `index_.ipynb`):

- **Distributional Analysis**: Skewness (-0.12), Kurtosis (10.47) indicating fat tails
- **Stationarity Tests**: Augmented Dickey-Fuller test for unit root detection
- **Correlation Analysis**: Feature correlation matrix and multicollinearity checks
- **Temporal Patterns**: ACF/PACF plots for lag structure identification
- **Volatility Clustering**: GARCH effects visualization


## 📚 Dependencies

### Core Libraries

```txt
# Deep Learning
torch>=2.0.0
tensorflow>=2.12.0

# Data Processing
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0

# Financial Data
yfinance>=0.2.0
finnhub-python>=2.4.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
```

---





<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by Ikram IDDOUCH & Sami SBAI

</div>
