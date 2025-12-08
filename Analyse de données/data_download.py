import yfinance as yf

# Télécharger l'historique journalier du S&P 500 depuis 2015
sp500 = yf.download("^GSPC", start="2000-01-01", interval="1d")

# Enregistrer les données en CSV
sp500.to_csv("SP500_2000.csv")

