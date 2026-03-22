"""
Pipeline principal - Analyse Sentiment & Marches Financiers
"""
from src.scraper import NewsScraper
from src.sentiment import SentimentAnalyzer
from src.market import MarketDataFetcher
from src.correlator import SentimentMarketCorrelator
from src.visualizer import ResultVisualizer

def main():
    print("=" * 60)
    print("LANCEMENT DU PIPELINE D'ANALYSE")
    print("=" * 60)

    # 1. Scraping
    print("\n[ETAPE 1] Collecte des articles...")
    scraper = NewsScraper()
    articles = scraper.fetch_headlines(days_back=25)
    scraper.save_headlines(articles)

    # 2. Sentiment
    print("\n[ETAPE 2] Analyse de sentiment...")
    analyzer = SentimentAnalyzer()
    scored = analyzer.analyze_dataframe(articles)
    daily = analyzer.get_daily_sentiment(scored)
    analyzer.save_results(scored)

    # 3. Marche
    print("\n[ETAPE 3] Donnees de marche...")
    market = MarketDataFetcher()
    market_data = market.fetch_data(days_back=30)
    returns = market.get_daily_returns(market_data)
    market.save_data(returns)

    # 4. Correlation
    print("\n[ETAPE 4] Correlation sentiment/marche...")
    correlator = SentimentMarketCorrelator()
    merged = correlator.merge_data(daily, returns)
    results = correlator.compute_correlations(merged)
    lagged = correlator.compute_lagged_correlations(merged)

    # 5. Visualisation
    print("\n[ETAPE 5] Generation des graphiques...")
    viz = ResultVisualizer()
    viz.plot_sentiment_timeline(merged)
    viz.plot_sentiment_vs_market(merged)
    viz.plot_correlation_heatmap(results)
    viz.generate_report(daily, results, lagged, merged)

    print("\n" + "=" * 60)
    print("ANALYSE TERMINEE")
    print("=" * 60)

if __name__ == "__main__":
    main()
