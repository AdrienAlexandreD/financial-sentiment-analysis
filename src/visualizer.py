"""
Module de visualisation des resultats
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class ResultVisualizer:
    def __init__(self, output_dir="data/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_sentiment_timeline(self, daily_sentiment):
        """Graphique de l'evolution du sentiment dans le temps"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        dates = pd.to_datetime(daily_sentiment['date'])

        # Score moyen
        ax1 = axes[0]
        colors = ['green' if x > 0 else 'red' for x in daily_sentiment['mean_score']]
        ax1.bar(dates, daily_sentiment['mean_score'], color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Score Sentiment Moyen')
        ax1.set_title('Evolution du Sentiment Financier (FinBERT)')

        # Repartition
        ax2 = axes[1]
        ax2.stackplot(dates,
                       daily_sentiment['positive_pct'] * 100,
                       daily_sentiment['neutral_pct'] * 100,
                       daily_sentiment['negative_pct'] * 100,
                       labels=['Positif', 'Neutre', 'Negatif'],
                       colors=['#2ecc71', '#95a5a6', '#e74c3c'],
                       alpha=0.8)
        ax2.set_ylabel('Repartition (%)')
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Date')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'sentiment_timeline.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVE] {path}")

    def plot_sentiment_vs_market(self, merged_data):
        """Graphique sentiment vs rendement du marche"""
        if merged_data is None or merged_data.empty:
            print("[WARN] Pas de donnees a afficher")
            return

        return_cols = [c for c in merged_data.columns if '_return' in c]
        n_plots = len(return_cols)

        if n_plots == 0:
            print("[WARN] Pas de colonnes de rendement trouvees")
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for i, col in enumerate(return_cols):
            ax = axes[i]
            valid = merged_data[['mean_score', col]].dropna()

            ax.scatter(valid['mean_score'], valid[col], alpha=0.6, color='steelblue')

            # Ligne de tendance
            if len(valid) > 2:
                z = np.polyfit(valid['mean_score'], valid[col], 1)
                p = np.poly1d(z)
                x_range = np.linspace(valid['mean_score'].min(), valid['mean_score'].max(), 100)
                ax.plot(x_range, p(x_range), "r--", alpha=0.8)

            name = col.replace('_return', '')
            ax.set_xlabel('Score Sentiment')
            ax.set_ylabel('Rendement')
            ax.set_title(f'Sentiment vs {name}')

        plt.suptitle('Correlation Sentiment / Marche', fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'sentiment_vs_market.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVE] {path}")

    def plot_correlation_heatmap(self, correlation_results):
        """Heatmap des correlations"""
        if correlation_results is None or correlation_results.empty:
            print("[WARN] Pas de resultats de correlation")
            return

        pivot = correlation_results.pivot_table(
            index='sentiment_metric',
            columns='market_metric',
            values='pearson_corr'
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c.replace('_return', '') for c in pivot.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Valeurs dans les cases
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            color='black', fontweight='bold')

        plt.colorbar(im, label='Correlation de Pearson')
        plt.title('Heatmap des Correlations Sentiment / Marche')
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVE] {path}")

    def generate_report(self, daily_sentiment, correlation_results, lagged_results, merged_data):
        """Genere un rapport texte complet"""
        report = []
        report.append("=" * 60)
        report.append("RAPPORT D'ANALYSE - SENTIMENT & MARCHES FINANCIERS")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)

        if daily_sentiment is not None:
            report.append("\n--- DONNEES ANALYSEES ---")
            report.append(f"Periode: {daily_sentiment['date'].min()} -> {daily_sentiment['date'].max()}")
            report.append(f"Jours couverts: {len(daily_sentiment)}")
            report.append(f"Score sentiment moyen: {daily_sentiment['mean_score'].mean():.4f}")

        if correlation_results is not None and not correlation_results.empty:
            report.append("\n--- CORRELATIONS SIGNIFICATIVES ---")
            sig = correlation_results[correlation_results['significant']]
            if not sig.empty:
                for _, row in sig.iterrows():
                    report.append(f"  {row['sentiment_metric']} x {row['market_metric']}: r={row['pearson_corr']:.3f}")
            else:
                report.append("  Aucune correlation significative (p < 0.05)")

        if lagged_results is not None and not lagged_results.empty:
            report.append("\n--- POUVOIR PREDICTIF (correlations decalees) ---")
            sig_lag = lagged_results[lagged_results['significant']]
            if not sig_lag.empty:
                for _, row in sig_lag.iterrows():
                    report.append(f"  Lag {row['lag_days']}j -> {row['market_metric']}: r={row['correlation']:.3f}")
            else:
                report.append("  Aucun signal predictif significatif detecte")

        report_text = '\n'.join(report)
        print(report_text)

        path = os.path.join(self.output_dir, 'rapport.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n[SAVE] Rapport sauvegarde: {path}")
        return report_text
