#!/usr/bin/env python3
"""
Analyze Guardian financial news articles for S&P 500 company mentions and sentiment.
This script processes articles and identifies sentiment (-1, 0, 1) for each company mentioned.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import sentiment analysis libraries
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, using simple sentiment analysis")

# S&P 500 companies - focusing on major companies that are frequently mentioned in news
# This is a curated list of the most commonly mentioned S&P 500 companies
SP500_COMPANIES = {
    # Tech Giants (FAANG+)
    'Apple': ['Apple', 'AAPL', 'iPhone', 'iPad', 'Mac', 'Apple Inc'],
    'Microsoft': ['Microsoft', 'MSFT', 'Windows', 'Azure', 'Office 365'],
    'Google': ['Google', 'Alphabet', 'GOOGL', 'GOOG', 'YouTube', 'Android'],
    'Amazon': ['Amazon', 'AMZN', 'AWS', 'Amazon Web Services', 'Alexa'],
    'Meta': ['Meta', 'Facebook', 'META', 'Instagram', 'WhatsApp', 'Zuckerberg'],
    'Tesla': ['Tesla', 'TSLA', 'Elon Musk Tesla'],
    'Nvidia': ['Nvidia', 'NVDA'],
    'Netflix': ['Netflix', 'NFLX'],

    # Financial Services
    'JPMorgan': ['JPMorgan', 'JP Morgan', 'JPM', 'Chase'],
    'Bank of America': ['Bank of America', 'BofA', 'BAC'],
    'Wells Fargo': ['Wells Fargo', 'WFC'],
    'Goldman Sachs': ['Goldman Sachs', 'Goldman', 'GS'],
    'Morgan Stanley': ['Morgan Stanley', 'MS'],
    'Citigroup': ['Citigroup', 'Citi', 'Citibank', 'C'],
    'American Express': ['American Express', 'Amex', 'AXP'],
    'Visa': ['Visa', 'V'],
    'Mastercard': ['Mastercard', 'MA'],
    'PayPal': ['PayPal', 'PYPL'],

    # Healthcare & Pharma
    'Johnson & Johnson': ['Johnson & Johnson', 'J&J', 'JNJ'],
    'Pfizer': ['Pfizer', 'PFE'],
    'Moderna': ['Moderna', 'MRNA'],
    'UnitedHealth': ['UnitedHealth', 'UNH', 'United Healthcare'],
    'Eli Lilly': ['Eli Lilly', 'Lilly', 'LLY'],

    # Retail & Consumer
    'Walmart': ['Walmart', 'WMT', 'Wal-Mart'],
    'Target': ['Target', 'TGT'],
    'Home Depot': ['Home Depot', 'HD'],
    'Nike': ['Nike', 'NKE'],
    'Starbucks': ['Starbucks', 'SBUX'],
    'McDonald\'s': ['McDonald', 'McDonalds', 'MCD'],
    'Coca-Cola': ['Coca-Cola', 'Coke', 'KO', 'Coca Cola'],
    'PepsiCo': ['PepsiCo', 'Pepsi', 'PEP'],

    # Automotive
    'Ford': ['Ford', 'F', 'Ford Motor'],
    'General Motors': ['General Motors', 'GM'],

    # Energy
    'ExxonMobil': ['ExxonMobil', 'Exxon', 'Mobil', 'XOM'],
    'Chevron': ['Chevron', 'CVX'],

    # Telecom & Media
    'AT&T': ['AT&T', 'T'],
    'Verizon': ['Verizon', 'VZ'],
    'Comcast': ['Comcast', 'CMCSA'],
    'Disney': ['Disney', 'DIS', 'Walt Disney'],

    # Aerospace & Defense
    'Boeing': ['Boeing', 'BA'],

    # Other Major Companies
    'Berkshire Hathaway': ['Berkshire Hathaway', 'Berkshire', 'Warren Buffett'],
    'Intel': ['Intel', 'INTC'],
    'IBM': ['IBM', 'International Business Machines'],
    'Oracle': ['Oracle', 'ORCL'],
    'Salesforce': ['Salesforce', 'CRM'],
    'Adobe': ['Adobe', 'ADBE'],
    'Cisco': ['Cisco', 'CSCO'],
    'Qualcomm': ['Qualcomm', 'QCOM'],
    'AMD': ['AMD', 'Advanced Micro Devices'],
    'OpenAI': ['OpenAI', 'Open AI', 'ChatGPT maker', 'Sam Altman OpenAI'],  # Not in S&P 500 but often mentioned
    'Anthropic': ['Anthropic', 'Claude'],  # Not in S&P 500 but often mentioned
    'DeepSeek': ['DeepSeek'],  # Chinese company, not in S&P 500 but in news
}


class SentimentAnalyzer:
    """Analyze sentiment of text mentions - optimized for speed"""

    def __init__(self):
        # Always use keyword-based sentiment for speed
        self.use_transformers = False
        print("Using fast keyword-based sentiment analysis")
        self._setup_keyword_sentiment()

    def _setup_keyword_sentiment(self):
        """Setup comprehensive keyword-based sentiment analysis"""
        # Expanded keyword lists for better accuracy
        self.positive_words = {
            'profit', 'gain', 'growth', 'success', 'surge', 'rise', 'boost',
            'improve', 'improved', 'strong', 'stronger', 'better', 'positive', 'advance', 'win', 'beat',
            'exceed', 'exceeded', 'outperform', 'outperformed', 'record', 'high', 'higher', 'rally', 'soar', 'jump',
            'innovation', 'innovative', 'breakthrough', 'achievement', 'milestone', 'up', 'increase',
            'increased', 'increasing', 'expand', 'expansion', 'growing', 'climbed', 'climbing',
            'recover', 'recovery', 'recovering', 'rebound', 'rebounding', 'bullish', 'optimistic',
            'confident', 'confidence', 'leading', 'leader', 'top', 'best', 'excellent', 'outstanding',
            'impressive', 'strong demand', 'revenue growth', 'profit surge'
        }
        self.negative_words = {
            'loss', 'losses', 'decline', 'declining', 'fall', 'falling', 'fell', 'drop', 'dropped', 'dropping',
            'crash', 'crashed', 'fail', 'failed', 'failing', 'failure', 'weak', 'weaker', 'poor', 'worst',
            'negative', 'concern', 'concerned', 'concerning', 'worry', 'worried', 'worrying', 'risk', 'risky',
            'threat', 'threatened', 'plunge', 'plunged', 'tumble', 'tumbled', 'slump', 'slumped',
            'lawsuit', 'sued', 'scandal', 'fraud', 'fraudulent', 'breach', 'breached', 'hack', 'hacked',
            'layoff', 'layoffs', 'cut', 'cuts', 'cutting', 'crisis', 'problem', 'problems', 'issue', 'issues',
            'struggle', 'struggling', 'miss', 'missed', 'underperform', 'underperformed', 'down', 'decrease',
            'decreased', 'decreasing', 'shrink', 'shrinking', 'bearish', 'pessimistic', 'volatile', 'volatility',
            'uncertain', 'uncertainty', 'downturn', 'recession', 'bankrupt', 'bankruptcy'
        }

    def analyze_context(self, text: str, company: str) -> int:
        """
        Analyze sentiment of text around company mention
        Returns: -1 (negative), 0 (neutral), 1 (positive)
        """
        if not text:
            return 0

        # Extract context around company mention (500 chars before and after)
        context = self._extract_context(text, company, window=500)
        return self._keyword_sentiment(context)

    def _extract_context(self, text: str, company: str, window: int = 500) -> str:
        """Extract text around company mention"""
        # Find company mention
        pattern = re.compile(re.escape(company), re.IGNORECASE)
        match = pattern.search(text)

        if not match:
            return text[:1000]  # Return beginning if not found

        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        return text[start:end]

    def _keyword_sentiment(self, text: str) -> int:
        """Fast keyword-based sentiment analysis"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)

        # Calculate sentiment score with better thresholds
        total = pos_count + neg_count
        if total == 0:
            return 0

        # Calculate sentiment ratio
        if pos_count > neg_count * 1.3:
            return 1
        elif neg_count > pos_count * 1.3:
            return -1
        else:
            return 0


def find_company_mentions(text: str, companies_dict: Dict[str, List[str]]) -> List[str]:
    """
    Find which S&P 500 companies are mentioned in the text
    Returns list of company names
    """
    if not text or pd.isna(text):
        return []

    text_lower = text.lower()
    mentioned_companies = []

    for company_name, variations in companies_dict.items():
        for variation in variations:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(variation.lower()) + r'\b'
            if re.search(pattern, text_lower):
                mentioned_companies.append(company_name)
                break  # Found this company, move to next

    return mentioned_companies


def process_articles(csv_path: str, output_path: str):
    """
    Process all articles and generate sentiment analysis output
    """
    print(f"Loading articles from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} articles")

    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()

    # Results storage
    results = []

    print("\nProcessing articles...")
    import time
    start_time = time.time()

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            elapsed = time.time() - start_time
            rate = (idx / elapsed) if elapsed > 0 else 0
            remaining = (len(df) - idx) / rate if rate > 0 else 0
            print(f"Progress: {idx}/{len(df)} articles ({100*idx/len(df):.1f}%) | "
                  f"Rate: {rate:.0f} articles/sec | "
                  f"Est. time remaining: {remaining/60:.1f} min | "
                  f"Found: {len(results)} mentions")

        article_id = idx
        url = row['url']
        title = row['title']
        body = str(row['body']) if pd.notna(row['body']) else ''
        summary = str(row['summary']) if pd.notna(row['summary']) else ''

        # Combine title, summary, and body for analysis
        full_text = f"{title}. {summary}. {body}"

        # Find company mentions
        mentioned_companies = find_company_mentions(full_text, SP500_COMPANIES)

        if mentioned_companies:
            # Analyze sentiment for each company
            for company in mentioned_companies:
                sentiment_score = analyzer.analyze_context(full_text, company)

                results.append({
                    'article_id': article_id,
                    'url': url,
                    'title': title,
                    'company': company,
                    'sentiment_score': sentiment_score
                })

    elapsed_total = time.time() - start_time
    print(f"\nProcessing complete in {elapsed_total/60:.1f} minutes!")
    print(f"Found {len(results)} company mentions across articles.")

    # Create output DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total articles analyzed: {len(df)}")
    print(f"Articles with S&P 500 mentions: {len(results_df['article_id'].unique())}")
    print(f"Total company mentions: {len(results_df)}")
    print(f"\nUnique companies mentioned: {results_df['company'].nunique()}")
    print(f"\nTop 10 most mentioned companies:")
    print(results_df['company'].value_counts().head(10))
    print(f"\nSentiment distribution:")
    print(f"Positive (1): {len(results_df[results_df['sentiment_score'] == 1])}")
    print(f"Neutral (0): {len(results_df[results_df['sentiment_score'] == 0])}")
    print(f"Negative (-1): {len(results_df[results_df['sentiment_score'] == -1])}")
    print("="*70)

    return results_df


if __name__ == "__main__":
    # File paths
    input_csv = "/Users/rddddddd/Documents/ML-Capstone-Project-006/data/guardian_financial_news_master.csv"
    output_csv = "/Users/rddddddd/Documents/ML-Capstone-Project-006/data/sp500_sentiment_analysis.csv"

    print("="*70)
    print("S&P 500 SENTIMENT ANALYSIS - GUARDIAN FINANCIAL NEWS")
    print("="*70)

    # Process articles
    results = process_articles(input_csv, output_csv)

    print("\nDone! Check the output file for detailed results.")
