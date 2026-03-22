"""
Collecteur de news financieres via GNews (gratuit, pas de cle API)
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from gnews import GNews


class NewsScraper:
    """Collecte des headlines financieres via Google News"""

    QUERIES = [
        "stock market",
        "S&P 500",
        "Federal Reserve interest rates",
        "Wall Street economy",
        "inflation GDP jobs report",
    ]

    def fetch_headlines(self, days_back=25):
        """Recupere des headlines sur les N derniers jours"""
        end = datetime.now()
        start = end - timedelta(days=days_back)

        gn = GNews(
            language='en',
            country='US',
            start_date=(start.year, start.month, start.day),
            end_date=(end.year, end.month, end.day),
            max_results=100,
        )

        all_articles = []

        for query in self.QUERIES:
            print(f"  Recherche: '{query}'...")
            try:
                results = gn.get_news(query)
                for art in results:
                    pub_date = self._parse_date(art.get('published date', ''))
                    all_articles.append({
                        'title': art.get('title', ''),
                        'source': art.get('publisher', {}).get('title', 'Unknown'),
                        'date': pub_date,
                        'url': art.get('url', ''),
                    })
            except Exception as e:
                print(f"  [WARN] Erreur pour '{query}': {e}")

        df = pd.DataFrame(all_articles)

        if not df.empty:
            df = df.drop_duplicates(subset='title')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date').reset_index(drop=True)

        print(f"\nTotal: {len(df)} articles uniques sur {df['date'].nunique()} jours")
        return df

    def _parse_date(self, date_str):
        """Parse la date GNews en date"""
        try:
            dt = datetime.strptime(date_str.strip(), "%a, %d %b %Y %H:%M:%S %Z")
            return dt.date()
        except Exception:
            try:
                dt = pd.to_datetime(date_str)
                return dt.date()
            except Exception:
                return None

    def save_headlines(self, df, path="data/headlines.csv"):
        """Sauvegarde les headlines"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[SAVE] {len(df)} articles sauvegardes -> {path}")
