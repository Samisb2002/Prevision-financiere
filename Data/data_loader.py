import yfinance as yf
import pandas as pd

def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    return df[['close']]
