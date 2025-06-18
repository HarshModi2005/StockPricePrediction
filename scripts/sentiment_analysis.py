#!/usr/bin/env python3
"""
Advanced Sentiment Analysis Pipeline for Stock Prediction
========================================================

This module provides comprehensive sentiment analysis capabilities including:
- News data fetching from multiple sources
- Multi-model sentiment analysis (VADER, TextBlob, FinBERT)
- Social media sentiment tracking
- Market sentiment indicators
- Integration with stock prediction models

Author: Stock Prediction Team
Date: 2024
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import time

import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Advanced sentiment analysis for financial news and social media
    """
    
    def __init__(self, newsapi_key: Optional[str] = None):
        """
        Initialize sentiment analyzer
        
        Args:
            newsapi_key: NewsAPI key for fetching news data
        """
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize FinBERT if available
        self.finbert_available = False
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer
            )
            self.finbert_available = True
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT not available: {e}")
            
        # Financial keywords for enhanced analysis
        self.positive_financial_words = [
            'profit', 'gain', 'growth', 'increase', 'rise', 'bull', 'bullish',
            'upgrade', 'outperform', 'beat', 'exceed', 'strong', 'robust',
            'positive', 'optimistic', 'rally', 'surge', 'soar', 'breakthrough'
        ]
        
        self.negative_financial_words = [
            'loss', 'decline', 'decrease', 'fall', 'bear', 'bearish',
            'downgrade', 'underperform', 'miss', 'weak', 'poor', 'negative',
            'pessimistic', 'crash', 'plunge', 'dive', 'concern', 'risk'
        ]
    
    def fetch_news_data(self, 
                       symbol: str, 
                       days_back: int = 7,
                       sources: List[str] = None) -> List[Dict]:
        """
        Fetch news data for a given stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            days_back: Number of days to look back for news
            sources: List of news sources to search
            
        Returns:
            List of news articles with metadata
        """
        news_articles = []
        
        # Default sources
        if sources is None:
            sources = [
                'reuters', 'bloomberg', 'cnbc', 'marketwatch', 'yahoo-finance',
                'financial-times', 'wall-street-journal', 'business-insider'
            ]
        
        if not self.newsapi_key:
            logger.warning("NewsAPI key not provided, using alternative methods")
            return self._fetch_alternative_news(symbol, days_back)
        
        try:
            from newsapi import NewsApiClient
            newsapi = NewsApiClient(api_key=self.newsapi_key)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for news articles
            query = f"{symbol} OR {self._get_company_name(symbol)}"
            
            articles = newsapi.get_everything(
                q=query,
                sources=','.join(sources),
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            for article in articles.get('articles', []):
                news_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'symbol': symbol
                })
                
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return self._fetch_alternative_news(symbol, days_back)
        
        return news_articles
    
    def _fetch_alternative_news(self, symbol: str, days_back: int) -> List[Dict]:
        """
        Alternative method to fetch news using Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            news_articles = []
            for item in news[:20]:  # Limit to 20 articles
                news_articles.append({
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'content': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'source': item.get('publisher', ''),
                    'published_at': datetime.fromtimestamp(
                        item.get('providerPublishTime', 0)
                    ).isoformat(),
                    'symbol': symbol
                })
            
            return news_articles
            
        except Exception as e:
            logger.error(f"Error fetching alternative news: {e}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """
        Get company name from stock symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('longName', symbol)
        except:
            return symbol
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using multiple methods
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores from different analyzers
        """
        if not text or not isinstance(text, str):
            return {
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 0.0,
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'finbert_score': 0.0,
                'finbert_label': 'neutral',
                'financial_sentiment': 0.0
            }
        
        results = {}
        
        # VADER Sentiment Analysis
        try:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results.update({
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu']
            })
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            results.update({
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 0.0
            })
        
        # TextBlob Sentiment Analysis
        try:
            blob = TextBlob(text)
            results.update({
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            })
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {e}")
            results.update({
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            })
        
        # FinBERT Analysis (if available)
        if self.finbert_available:
            try:
                # Truncate text to avoid token limit
                truncated_text = text[:512]
                finbert_result = self.finbert_pipeline(truncated_text)[0]
                
                # Convert label to score
                label = finbert_result['label'].lower()
                score = finbert_result['score']
                
                if label == 'positive':
                    finbert_score = score
                elif label == 'negative':
                    finbert_score = -score
                else:
                    finbert_score = 0.0
                
                results.update({
                    'finbert_score': finbert_score,
                    'finbert_label': label
                })
            except Exception as e:
                logger.error(f"FinBERT analysis failed: {e}")
                results.update({
                    'finbert_score': 0.0,
                    'finbert_label': 'neutral'
                })
        else:
            results.update({
                'finbert_score': 0.0,
                'finbert_label': 'neutral'
            })
        
        # Financial-specific sentiment
        results['financial_sentiment'] = self._calculate_financial_sentiment(text)
        
        return results
    
    def _calculate_financial_sentiment(self, text: str) -> float:
        """
        Calculate sentiment based on financial keywords
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_financial_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_financial_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        return positive_ratio - negative_ratio
    
    def get_aggregated_sentiment(self, 
                               symbol: str, 
                               days_back: int = 7,
                               weight_by_recency: bool = True) -> Dict[str, float]:
        """
        Get aggregated sentiment scores for a stock symbol
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            weight_by_recency: Whether to weight more recent news more heavily
            
        Returns:
            Dictionary with aggregated sentiment scores
        """
        # Fetch news data
        news_articles = self.fetch_news_data(symbol, days_back)
        
        if not news_articles:
            logger.warning(f"No news articles found for {symbol}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'news_volume': 0,
                'vader_sentiment': 0.0,
                'textblob_sentiment': 0.0,
                'finbert_sentiment': 0.0,
                'financial_sentiment': 0.0
            }
        
        # Analyze sentiment for each article
        sentiment_results = []
        for article in news_articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment_scores = self.analyze_text_sentiment(text)
            sentiment_scores['published_at'] = article.get('published_at', '')
            sentiment_results.append(sentiment_scores)
        
        sentiment_df = pd.DataFrame(sentiment_results)
        
        if sentiment_df.empty:
            return {
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'news_volume': 0,
                'vader_sentiment': 0.0,
                'textblob_sentiment': 0.0,
                'finbert_sentiment': 0.0,
                'financial_sentiment': 0.0
            }
        
        # Calculate weights based on recency
        if weight_by_recency and 'published_at' in sentiment_df.columns:
            sentiment_df['published_at'] = pd.to_datetime(sentiment_df['published_at'], errors='coerce')
            now = datetime.now()
            sentiment_df['hours_ago'] = (
                now - sentiment_df['published_at']
            ).dt.total_seconds() / 3600
            
            # Exponential decay weight (more recent = higher weight)
            sentiment_df['weight'] = np.exp(-sentiment_df['hours_ago'] / 24)
        else:
            sentiment_df['weight'] = 1.0
        
        # Calculate weighted averages
        total_weight = sentiment_df['weight'].sum()
        
        if total_weight == 0:
            return {
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'news_volume': len(news_articles),
                'vader_sentiment': 0.0,
                'textblob_sentiment': 0.0,
                'finbert_sentiment': 0.0,
                'financial_sentiment': 0.0
            }
        
        # Weighted sentiment scores
        vader_sentiment = (
            sentiment_df['vader_compound'] * sentiment_df['weight']
        ).sum() / total_weight
        
        textblob_sentiment = (
            sentiment_df['textblob_polarity'] * sentiment_df['weight']
        ).sum() / total_weight
        
        finbert_sentiment = (
            sentiment_df['finbert_score'] * sentiment_df['weight']
        ).sum() / total_weight
        
        financial_sentiment = (
            sentiment_df['financial_sentiment'] * sentiment_df['weight']
        ).sum() / total_weight
        
        # Overall sentiment (average of all methods)
        overall_sentiment = np.mean([
            vader_sentiment, textblob_sentiment, 
            finbert_sentiment, financial_sentiment
        ])
        
        # Sentiment strength (standard deviation indicates uncertainty)
        sentiment_strength = np.std([
            vader_sentiment, textblob_sentiment, 
            finbert_sentiment, financial_sentiment
        ])
        
        return {
            'overall_sentiment': float(overall_sentiment),
            'sentiment_strength': float(sentiment_strength),
            'news_volume': len(news_articles),
            'vader_sentiment': float(vader_sentiment),
            'textblob_sentiment': float(textblob_sentiment),
            'finbert_sentiment': float(finbert_sentiment),
            'financial_sentiment': float(financial_sentiment)
        }

def main():
    """
    Main function for testing sentiment analysis
    """
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test with a sample stock
    symbol = "AAPL"
    print(f"Analyzing sentiment for {symbol}...")
    
    # Get aggregated sentiment
    sentiment_data = analyzer.get_aggregated_sentiment(symbol, days_back=7)
    
    print("\nSentiment Analysis Results:")
    print("=" * 40)
    for key, value in sentiment_data.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 