import yfinance as yf

# Télécharger l'historique journalier du S&P 500 depuis 2015
sp500 = yf.download("AAPL", start="2010-01-01", interval="1d")

# Enregistrer les données en CSV
sp500.to_csv("Apple.csv")

