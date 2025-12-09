import finnhub
import pandas as pd
import numpy as np
import yfinance as yf
import time


class FinnhubSentiment:
    """
    Classe pour collecter les données de sentiment depuis Finnhub
    Compatible avec le plan GRATUIT
    """
    
    def __init__(self, api_key):
        self.client = finnhub.Client(api_key=api_key)
    
    def _safe_call(self, func, *args, **kwargs):
        """Appel API avec gestion du rate limit et des erreurs"""
        time.sleep(1.2)  # 60 calls/min max
        try:
            return func(*args, **kwargs)
        except finnhub.FinnhubAPIException as e:
            print(f"API non disponible : {e}")
            return None
        except Exception as e:
            print(f"Erreur: {e}")
            return None
    
    def get_company_news(self, symbol, start_date, end_date):
        """
        Récupère les articles de presse
        On peut calculer un sentiment basique à partir du volume de news
        """
        data = self._safe_call(
            self.client.company_news,
            symbol,
            _from=start_date,
            to=end_date
        )
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df['date'] = df['datetime'].dt.date
        
        return df
    
    def get_analyst_sentiment(self, symbol):
        """
        Récupère les recommandations analystes
        """
        data = self._safe_call(self.client.recommendation_trends, symbol)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['period'] = pd.to_datetime(df['period'])
        
        df['total'] = df['strongBuy'] + df['buy'] + df['hold'] + df['sell'] + df['strongSell']
        df['analyst_score'] = (
            2*df['strongBuy'] + df['buy'] - df['sell'] - 2*df['strongSell']
        ) / df['total']
        
        return df[['period', 'analyst_score', 'total']]
    
    def get_insider_sentiment(self, symbol):
        """
        Récupère les transactions insiders
        """
        data = self._safe_call(self.client.stock_insider_transactions, symbol)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return df
        
        df['date'] = pd.to_datetime(df['transactionDate'])
        df['is_buy'] = df['transactionCode'].isin(['P', 'A']).astype(int)
        df['is_sell'] = (df['transactionCode'] == 'S').astype(int)
        
        return df[['date', 'name', 'share', 'is_buy', 'is_sell']]
    
    def get_earnings_surprises(self, symbol):
        """
        Récupère les surprises aux résultats
        """
        data = self._safe_call(self.client.company_earnings, symbol, limit=12)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['period'] = pd.to_datetime(df['period'])
        
        return df[['period', 'actual', 'estimate', 'surprisePercent']]
    
    def create_features(self, symbol, price_df, start_date=None, end_date=None):
        """
        Crée toutes les features de sentiment (plan gratuit)
        """
        features = pd.DataFrame(index=price_df.index)
        
        # Déterminer les dates
        if start_date is None:
            start_date = price_df.index[0].strftime('%Y-%m-%d')
        if end_date is None:
            end_date = price_df.index[-1].strftime('%Y-%m-%d')
        
        print("  Récupération des news...")
        # 1. News (volume et fréquence)
        news_df = self.get_company_news(symbol, start_date, end_date)
        if not news_df.empty:
            # Compter les articles par jour
            news_count = news_df.groupby('date').size().reset_index(name='news_count')
            news_count['date'] = pd.to_datetime(news_count['date'])
            news_count = news_count.set_index('date')
            
            features['news_count'] = news_count['news_count'].reindex(
                price_df.index, fill_value=0
            )
            features['news_count_ma5'] = features['news_count'].rolling(5).mean()
            features['news_spike'] = (
                features['news_count'] / (features['news_count_ma5'] + 0.1)
            )
        else:
            features['news_count'] = 0
            features['news_count_ma5'] = 0
            features['news_spike'] = 0
        
        print("  Récupération des recommandations analystes...")
        # 2. Analyst sentiment
        analyst_df = self.get_analyst_sentiment(symbol)
        if not analyst_df.empty:
            analyst_df = analyst_df.set_index('period')
            features['analyst_score'] = analyst_df['analyst_score'].reindex(
                price_df.index, method='ffill'
            )
            features['analyst_score'] = features['analyst_score'].fillna(0)
        else:
            features['analyst_score'] = 0
        
        print("  Récupération des transactions insiders...")
        # 3. Insider sentiment
        insider_df = self.get_insider_sentiment(symbol)
        if not insider_df.empty:
            insider_df['month'] = insider_df['date'].dt.to_period('M').dt.start_time
            monthly = insider_df.groupby('month').agg({'is_buy': 'sum', 'is_sell': 'sum'})
            monthly['insider_score'] = (monthly['is_buy'] - monthly['is_sell']) / (
                monthly['is_buy'] + monthly['is_sell'] + 1
            )
            features['insider_score'] = monthly['insider_score'].reindex(
                price_df.index, method='ffill'
            ).fillna(0)
        else:
            features['insider_score'] = 0
        
        print("  Récupération des earnings surprises...")
        # 4. Earnings surprises
        earnings_df = self.get_earnings_surprises(symbol)
        if not earnings_df.empty:
            earnings_df = earnings_df.set_index('period')
            features['earnings_surprise'] = earnings_df['surprisePercent'].reindex(
                price_df.index, method='ffill'
            ).fillna(0)
        else:
            features['earnings_surprise'] = 0
        
        # 5. Score composite (sans news_sentiment)
        features['composite_sentiment'] = (
            0.4 * features['analyst_score'] / 2 +      # Normaliser [-2,2] -> [-1,1]
            0.3 * features['insider_score'] +
            0.2 * features['earnings_surprise'] / 10 + # Normaliser
            0.1 * np.tanh(features['news_spike'] - 1)  # News spike normalisé
        )
        features.to_csv("Data/sentiment_features.csv")
        return features


def download_prices(symbol, start, end):
    """Télécharge les prix avec correction du format"""
    df = yf.download(symbol, start=start, end=end)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    
    return df


# ===========================================
# UTILISATION
# ===========================================

if __name__ == "__main__":
    
    # 1. Configuration
    API_KEY = "d4ru51pr01qi8t3ir8g0d4ru51pr01qi8t3ir8gg"  # Gratuit sur finnhub.io
    SYMBOL = "^GSPC"
    START = "2000-01-03"
    END = "2025-12-08"
    
    # 2. Récupérer les prix
    print("Récupération des prix...")
    price_df = download_prices(SYMBOL, START, END)
    print(f"  {len(price_df)} jours récupérés")
    
    # 3. Créer les features de sentiment
    print("\nCréation des features de sentiment...")
    sentiment = FinnhubSentiment(API_KEY)
    sentiment_features = sentiment.create_features(SYMBOL, price_df, START, END)
    
    # 4. Afficher les résultats
    print("\n--- Features créées ---")
    print(sentiment_features.columns.tolist())
    
    print("\n--- Aperçu ---")
    print(sentiment_features.tail(10))
    
    print("\n--- Statistiques ---")
    print(sentiment_features.describe())