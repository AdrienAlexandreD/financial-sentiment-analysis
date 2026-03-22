# Economic Sentiment Analyzer

Analyse automatisee de la correlation entre le sentiment des actualites financieres et l'evolution des marches boursiers.

---

## Objectif

Ce projet cherche a repondre a une question simple :
**Le ton des actualites financieres a-t-il un lien mesurable avec les mouvements du marche ?**

Le pipeline collecte automatiquement des articles financiers, evalue leur sentiment grace a un modele d'intelligence artificielle specialise en finance, puis mesure statistiquement la correlation avec les donnees boursieres reelles.

---

## Fonctionnement

Le projet s'execute en 5 etapes sequentielles :

### 1. Collecte des articles
- Recuperation automatique des headlines financieres via **Google News**
- Mots-cles cibles : *stock market, economy, inflation, Fed, earnings...*
- Fenetre de collecte configurable (25 jours par defaut)

### 2. Analyse de sentiment
- Chaque headline est evaluee par **FinBERT**, un modele de NLP pre-entraine specifiquement sur des textes financiers
- Chaque article recoit un score entre **-1** (tres negatif) et **+1** (tres positif)
- Les scores sont ensuite agreges par jour (moyenne, ecart-type, volume)

### 3. Donnees de marche
- Recuperation des cours historiques via **Yahoo Finance** (librairie `yfinance`)

## Indices suivis

| Indicateur | Ticker | Description |
|---|---|---|
| S&P 500 | `^GSPC` | 500 plus grandes entreprises americaines |
| VIX | `^VIX` | Indice de volatilite (« indice de la peur ») |
| Treasury 10Y | `^TNX` | Taux des obligations d'Etat US a 10 ans |
| Dollar (DXY) | `DX-Y.NYB` | Force du dollar face aux autres devises |
| Or (Gold) | `GC=F` | Valeur refuge en periode d'incertitude |

- Calcul des rendements journaliers (variation en %)

### 4. Correlation statistique
- Fusion des donnees sentiment et marche par date
- Calcul des correlations de **Pearson** et **Spearman**
- Test de significativite statistique (p-value < 0.05)
- Analyse avec decalage temporel (le sentiment d'aujourd'hui predit-il le marche de demain ?)

### 5. Visualisation et rapport
- Generation de graphiques : evolution temporelle, scatter plots, heatmap des correlations
- Export d'un rapport texte synthetisant les resultats

---

## Structure du projet

economic-sentiment/
├── main.py                  # Point d'entree - lance le pipeline complet
├── requirements.txt         # Dependances Python
├── src/
│   ├── scraper.py           # Collecte des articles (Google News)
│   ├── sentiment.py         # Analyse de sentiment (FinBERT)
│   ├── market.py            # Donnees boursieres (Yahoo Finance)
│   ├── correlator.py        # Calculs statistiques
│   └── visualizer.py        # Graphiques et rapport
├── data/                    # Donnees generees (CSV)
└── README.md


---

## Installation


# Creer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Installer les dependances
pip install -r requirements.txt


python main.py
