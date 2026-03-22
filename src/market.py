"""
Module de recuperation des donnees de marche via yfinance
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class MarketDataFetcher:
    def __init__(self):
        self.default_tickers = {
            'SP500': '^GSPC',
            'VIX': '^VIX',
            'TREASURY_10Y': '^TNX',
            'DXY': 'DX-Y.NYB',
            'GOLD': 'GC=F'
        }

    def fetch_data(self, tickers=None, days_back=30):
        """Recupere les donnees de marche"""
        if tickers is None:
            tickers = self.default_tickers

        start = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')

        all_data = {}
        for name, symbol in tickers.items():
            print(f"  Recuperation {name} ({symbol})...")
            try:
                data = yf.download(symbol, start=start, end=end, progress=False)
                if not data.empty:
                    all_data[name] = data
                    print(f"    {len(data)} jours de donnees")
                else:
                    print(f"    Aucune donnee")
            except Exception as e:
                print(f"    [WARN] Erreur: {e}")

        return all_data

    def get_daily_returns(self, market_data):
        """Calcule les rendements journaliers"""
        returns = pd.DataFrame()

        for name, data in market_data.items():
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            returns[f'{name}_close'] = close
            returns[f'{name}_return'] = close.pct_change()

        returns.index = pd.to_datetime(returns.index)
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        returns['date'] = returns.index.date

        return returns

    def save_data(self, returns_df, filename=None):
        """Sauvegarde les donnees"""
        if filename is None:
            filename = f"data/market_data/market_{datetime.now().strftime('%Y%m%d')}.csv"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        returns_df.to_csv(filename)
        print(f"[SAVE] Sauvegarde: {filename}")
        return filename
