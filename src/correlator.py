"""
Module de correlation entre sentiment et donnees de marche
"""
import pandas as pd
import numpy as np
from scipy import stats


class SentimentMarketCorrelator:
    def __init__(self):
        self.merged_data = None

    def merge_data(self, sentiment_daily, market_returns):
        """Fusionne sentiment journalier et donnees de marche"""
        sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])
        market_returns['date'] = pd.to_datetime(market_returns['date'])

        self.merged_data = pd.merge(
            sentiment_daily,
            market_returns,
            on='date',
            how='inner'
        )

        print(f"Donnees fusionnees: {len(self.merged_data)} jours en commun")
        return self.merged_data

    def compute_correlations(self, df=None):
        """Calcule les correlations entre sentiment et marches"""
        if df is None:
            df = self.merged_data

        if df is None or df.empty:
            print("[WARN] Pas de donnees fusionnees")
            return None

        return_cols = [c for c in df.columns if '_return' in c]
        sentiment_cols = ['mean_score', 'positive_pct', 'negative_pct']

        results = []
        for s_col in sentiment_cols:
            if s_col not in df.columns:
                continue
            for r_col in return_cols:
                if r_col not in df.columns:
                    continue

                valid = df[[s_col, r_col]].dropna()
                if len(valid) < 5:
                    continue

                corr, p_value = stats.pearsonr(valid[s_col], valid[r_col])
                spearman, sp_pvalue = stats.spearmanr(valid[s_col], valid[r_col])

                results.append({
                    'sentiment_metric': s_col,
                    'market_metric': r_col,
                    'pearson_corr': round(corr, 4),
                    'pearson_pvalue': round(p_value, 4),
                    'spearman_corr': round(spearman, 4),
                    'spearman_pvalue': round(sp_pvalue, 4),
                    'n_observations': len(valid),
                    'significant': p_value < 0.05
                })

        results_df = pd.DataFrame(results)
        self._print_correlations(results_df)
        return results_df

    def compute_lagged_correlations(self, df=None, max_lag=5):
        """Correlations avec decalage temporel (le sentiment predit-il le marche ?)"""
        if df is None:
            df = self.merged_data

        if df is None or df.empty:
            return None

        df = df.sort_values('date').reset_index(drop=True)
        return_cols = [c for c in df.columns if '_return' in c]

        results = []
        for lag in range(1, max_lag + 1):
            for r_col in return_cols:
                if r_col not in df.columns:
                    continue

                valid = pd.DataFrame({
                    'sentiment': df['mean_score'],
                    'future_return': df[r_col].shift(-lag)
                }).dropna()

                if len(valid) < 5:
                    continue

                corr, p_value = stats.pearsonr(valid['sentiment'], valid['future_return'])

                results.append({
                    'lag_days': lag,
                    'market_metric': r_col,
                    'correlation': round(corr, 4),
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05
                })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            print(f"\n{'='*50}")
            print("CORRELATIONS AVEC DECALAGE (sentiment -> marche futur)")
            print(f"{'='*50}")
            sig = results_df[results_df['significant']]
            if not sig.empty:
                print(sig.to_string(index=False))
            else:
                print("Aucune correlation significative trouvee")

        return results_df

    def _print_correlations(self, df):
        """Affiche les correlations"""
        print(f"\n{'='*50}")
        print("CORRELATIONS SENTIMENT / MARCHE")
        print(f"{'='*50}")
        if df.empty:
            print("Aucun resultat")
            return
        for _, row in df.iterrows():
            sig = "[SIG]" if row['significant'] else "[NS]"
            print(f"  {sig} {row['sentiment_metric']} x {row['market_metric']}: "
                  f"r={row['pearson_corr']:.3f} (p={row['pearson_pvalue']:.3f})")
