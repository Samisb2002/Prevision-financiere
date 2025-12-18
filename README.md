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

#### 1. **Temporal Fusion Transformer (TFT)** ⭐ Main Model
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

## 📁 Project Structure

```
├── Data/
│   ├── Apple.csv                    # Raw stock price data
│   ├── Applefeatures.csv           # Computed technical indicators
│   ├── SP500_features.csv          # S&P 500 features dataset
│   └── sentiment_features.csv      # Sentiment analysis results
│
├── models/
│   ├── TFT.py                      # Temporal Fusion Transformer implementation
│   ├── informer.py                 # Informer model architecture
│   └── lstm.py                     # LSTM baseline model
│
├── data/
│   ├── data_download.py            # Yahoo Finance data fetcher
│   ├── data_loader.py              # Data loading utilities
│   ├── data_preparation.py         # Feature engineering & preprocessing
│   ├── calcul_features.py          # Technical indicator computation
│   └── preprocessing.py            # Sequence creation & scaling
│
├── training/
│   ├── train_tft.py                # TFT training pipeline
│   ├── trainer.py                  # Generic training loop
│   └── evaluator.py                # Model evaluation metrics
│
├── sentiment/
│   └── sentiment-analysis.py       # Finnhub API integration
│
├── inference/
│   ├── inference.py                # Model prediction & visualization
│   └── visualization.py            # Plotting utilities
│
├── main.py                         # Main execution script
├── index_.ipynb                    # Exploratory data analysis notebook
└── requirements.txt                # Python dependencies
```

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
git clone https://github.com/yourusername/financial-forecasting-transformers.git
cd financial-forecasting-transformers
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

### Key Statistics (S&P 500: 2000-2025)

```
Period: 2000-01-03 to 2025-12-08 (6,523 days)
Total Return: +370.48%
Average Daily Return: 0.0312%
Volatility (σ): 1.22%
Maximum Drawdown: -57.8% (2009 Financial Crisis)
```

---

## 🛠️ Configuration

### Hyperparameters (TFT)

```python
# Model Architecture
d_model = 128              # Hidden dimension
n_heads = 8                # Attention heads
dropout = 0.1              # Dropout rate

# Training
batch_size = 64
learning_rate = 0.0005
epochs = 100
window_size = 60           # Input sequence length

# Data Split
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
```

### Modifying Configurations

Edit the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "symbol": "^GSPC",           # S&P 500 index
    "start": "2010-01-01",
    "end": "2025-01-01",
    "window_size": 60,
    "epochs": 50,
    "batch_size": 32,
    "horizons": [1, 5, 20]       # Prediction horizons
}
```

---

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

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{iddouch2025multihorizon,
  title={Multi-Horizon Financial Forecasting with Transformers},
  author={Iddouch, Ikram and Sbai, Sami},
  year={2025},
  publisher={GitHub},
  howpublished={\\url{https://github.com/yourusername/financial-forecasting-transformers}}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Temporal Fusion Transformer**: Based on [Lim et al. (2021)](https://arxiv.org/abs/1912.09363)
- **Informer**: Based on [Zhou et al. (2021)](https://arxiv.org/abs/2012.07436)
- **Data Sources**: Yahoo Finance, Finnhub API
- **Course Instructor**: DATA931 - Time Series Analysis

---

## 📧 Contact

For questions or collaborations:

- **Ikram IDDOUCH** - [email@example.com](mailto:email@example.com)
- **Sami SBAI** - [email@example.com](mailto:email@example.com)

**Project Link**: [https://github.com/yourusername/financial-forecasting-transformers](https://github.com/yourusername/financial-forecasting-transformers)

---

## 🔮 Future Work

- [ ] Add transformer attention visualization
- [ ] Implement ensemble methods
- [ ] Extend to multi-asset portfolio optimization
- [ ] Real-time prediction API
- [ ] Integration with trading simulators
- [ ] Add explainability features (SHAP values)
- [ ] Support for cryptocurrency data
- [ ] Backtesting framework with transaction costs

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by Ikram IDDOUCH & Sami SBAI

</div>
