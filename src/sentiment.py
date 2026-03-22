"""
Module d'analyse de sentiment des headlines avec FinBERT
"""
import os
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """Charge le modele FinBERT specialise finance"""
        print("Chargement du modele FinBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        print("Modele charge.")

    def analyze_headline(self, headline):
        """Analyse le sentiment d'une seule headline"""
        if not headline or not isinstance(headline, str):
            return {'label': 'neutral', 'confidence': 0.0, 'numeric_score': 0.0}

        headline = headline[:512]

        try:
            result = self.nlp(headline)[0]
            label = result['label']
            confidence = result['score']

            if label == 'positive':
                numeric_score = confidence
            elif label == 'negative':
                numeric_score = -confidence
            else:
                numeric_score = 0.0

            return {
                'label': label,
                'confidence': confidence,
                'numeric_score': numeric_score
            }
        except Exception as e:
            print(f"[WARN] Erreur analyse: {e}")
            return {'label': 'neutral', 'confidence': 0.0, 'numeric_score': 0.0}

    def analyze_dataframe(self, df):
        """Analyse toutes les headlines d'un DataFrame"""
        print(f"Analyse de {len(df)} headlines...")

        results = []
        for i, row in df.iterrows():
            result = self.analyze_headline(row['title'])
            results.append(result)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(df)} analysees...")

        results_df = pd.DataFrame(results)
        df_analyzed = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        print("Analyse terminee.")
        self._print_summary(df_analyzed)

        return df_analyzed

    def get_daily_sentiment(self, df):
        """Calcule le sentiment moyen par jour"""
        daily = df.groupby('date').agg(
            mean_score=('numeric_score', 'mean'),
            median_score=('numeric_score', 'median'),
            num_articles=('numeric_score', 'count'),
            positive_pct=('label', lambda x: (x == 'positive').mean()),
            negative_pct=('label', lambda x: (x == 'negative').mean()),
            neutral_pct=('label', lambda x: (x == 'neutral').mean())
        ).reset_index()

        return daily

    def _print_summary(self, df):
        """Affiche un resume de l'analyse"""
        print(f"\n{'='*50}")
        print("RESUME SENTIMENT")
        print(f"{'='*50}")
        print(f"Positif:  {(df['label'] == 'positive').sum()} ({(df['label'] == 'positive').mean()*100:.1f}%)")
        print(f"Negatif:  {(df['label'] == 'negative').sum()} ({(df['label'] == 'negative').mean()*100:.1f}%)")
        print(f"Neutre:   {(df['label'] == 'neutral').sum()} ({(df['label'] == 'neutral').mean()*100:.1f}%)")
        print(f"Score moyen: {df['numeric_score'].mean():.4f}")
        print(f"{'='*50}")

    def save_results(self, df, filename=None):
        """Sauvegarde les resultats"""
        if filename is None:
            filename = f"data/sentiment_scores/sentiment_{datetime.now().strftime('%Y%m%d')}.csv"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"[SAVE] Sauvegarde: {filename}")
        return filename
