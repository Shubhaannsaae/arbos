"""
Sentiment Features Engineering Module

This module implements sentiment feature extraction for ML models,
integrating with Chainlink Functions for external sentiment data.

Features:
- Social media sentiment analysis
- On-chain sentiment indicators
- News sentiment extraction
- Market sentiment aggregation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
import re
import joblib

@dataclass
class SentimentFeatures:
    """Sentiment feature data class."""
    social_sentiment: float
    news_sentiment: float
    onchain_sentiment: float
    fear_greed_index: float
    volatility_sentiment: float
    aggregate_sentiment: float
    confidence: float

class SentimentFeaturesEngine:
    """
    Sentiment features extraction engine with real-time capabilities.
    """
    
    def __init__(self):
        """Initialize sentiment features engine."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except Exception as e:
            self.logger.warning(f"Failed to load sentiment model: {e}")
            self.sentiment_analyzer = None
        
        # Sentiment aggregation weights
        self.sentiment_weights = {
            'social': 0.3,
            'news': 0.25,
            'onchain': 0.25,
            'fear_greed': 0.2
        }
        
        # Cache for processed data
        self.sentiment_cache = {}
        
    def extract_social_sentiment(self, social_data: List[str]) -> float:
        """Extract sentiment from social media data."""
        try:
            if not social_data or not self.sentiment_analyzer:
                return 0.5  # Neutral
            
            sentiments = []
            for text in social_data[:100]:  # Limit processing
                if len(text.strip()) > 10:  # Skip very short texts
                    result = self.sentiment_analyzer(text)[0]
                    
                    # Convert to numeric score
                    if result['label'] == 'LABEL_2':  # Positive
                        score = 0.5 + (result['score'] * 0.5)
                    elif result['label'] == 'LABEL_0':  # Negative
                        score = 0.5 - (result['score'] * 0.5)
                    else:  # Neutral
                        score = 0.5
                    
                    sentiments.append(score)
            
            return np.mean(sentiments) if sentiments else 0.5
            
        except Exception as e:
            self.logger.error(f"Social sentiment extraction failed: {e}")
            return 0.5
    
    def extract_news_sentiment(self, news_data: List[Dict]) -> float:
        """Extract sentiment from news headlines and content."""
        try:
            if not news_data or not self.sentiment_analyzer:
                return 0.5
            
            sentiments = []
            for article in news_data[:50]:  # Limit processing
                text = article.get('title', '') + ' ' + article.get('content', '')[:500]
                
                if len(text.strip()) > 20:
                    result = self.sentiment_analyzer(text)[0]
                    
                    # Convert to numeric score
                    if result['label'] == 'LABEL_2':  # Positive
                        score = 0.5 + (result['score'] * 0.5)
                    elif result['label'] == 'LABEL_0':  # Negative
                        score = 0.5 - (result['score'] * 0.5)
                    else:  # Neutral
                        score = 0.5
                    
                    sentiments.append(score)
            
            return np.mean(sentiments) if sentiments else 0.5
            
        except Exception as e:
            self.logger.error(f"News sentiment extraction failed: {e}")
            return 0.5
    
    def extract_onchain_sentiment(self, onchain_data: pd.DataFrame) -> float:
        """Extract sentiment from on-chain indicators."""
        try:
            if onchain_data.empty:
                return 0.5
            
            sentiment_indicators = []
            
            # Transaction volume trend
            if 'tx_volume' in onchain_data.columns:
                volume_trend = onchain_data['tx_volume'].pct_change(7).iloc[-1]
                volume_sentiment = 0.5 + np.tanh(volume_trend) * 0.3
                sentiment_indicators.append(volume_sentiment)
            
            # Active addresses trend
            if 'active_addresses' in onchain_data.columns:
                addr_trend = onchain_data['active_addresses'].pct_change(7).iloc[-1]
                addr_sentiment = 0.5 + np.tanh(addr_trend) * 0.3
                sentiment_indicators.append(addr_sentiment)
            
            # Exchange flows
            if 'exchange_inflow' in onchain_data.columns and 'exchange_outflow' in onchain_data.columns:
                net_flow = onchain_data['exchange_outflow'].iloc[-1] - onchain_data['exchange_inflow'].iloc[-1]
                flow_sentiment = 0.5 + np.tanh(net_flow / onchain_data['exchange_inflow'].mean()) * 0.2
                sentiment_indicators.append(flow_sentiment)
            
            # Long/short ratio
            if 'long_short_ratio' in onchain_data.columns:
                ls_ratio = onchain_data['long_short_ratio'].iloc[-1]
                ls_sentiment = min(max(ls_ratio / 2, 0), 1)  # Normalize to [0,1]
                sentiment_indicators.append(ls_sentiment)
            
            return np.mean(sentiment_indicators) if sentiment_indicators else 0.5
            
        except Exception as e:
            self.logger.error(f"On-chain sentiment extraction failed: {e}")
            return 0.5
    
    def extract_fear_greed_index(self, market_data: pd.DataFrame) -> float:
        """Extract fear & greed index from market data."""
        try:
            if market_data.empty:
                return 0.5
            
            fear_greed_components = []
            
            # Volatility component
            if 'price' in market_data.columns:
                returns = market_data['price'].pct_change().dropna()
                if len(returns) > 20:
                    volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(365)
                    vol_component = max(0, min(1, 1 - (volatility - 0.2) / 0.6))
                    fear_greed_components.append(vol_component)
            
            # Market momentum
            if 'price' in market_data.columns and len(market_data) > 50:
                sma_20 = market_data['price'].rolling(20).mean().iloc[-1]
                sma_50 = market_data['price'].rolling(50).mean().iloc[-1]
                momentum_component = 1 if sma_20 > sma_50 else 0
                fear_greed_components.append(momentum_component)
            
            # Volume momentum
            if 'volume' in market_data.columns and len(market_data) > 20:
                vol_ma = market_data['volume'].rolling(20).mean()
                current_vol = market_data['volume'].iloc[-1]
                vol_component = min(1, current_vol / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 0.5
                fear_greed_components.append(vol_component)
            
            return np.mean(fear_greed_components) if fear_greed_components else 0.5
            
        except Exception as e:
            self.logger.error(f"Fear & greed index calculation failed: {e}")
            return 0.5
    
    def extract_volatility_sentiment(self, price_data: pd.DataFrame) -> float:
        """Extract sentiment from volatility patterns."""
        try:
            if price_data.empty or 'price' not in price_data.columns:
                return 0.5
            
            returns = price_data['price'].pct_change().dropna()
            if len(returns) < 20:
                return 0.5
            
            # Current vs historical volatility
            current_vol = returns.tail(5).std()
            historical_vol = returns.tail(60).std()
            
            if historical_vol > 0:
                vol_ratio = current_vol / historical_vol
                # Higher volatility = more fear = lower sentiment
                vol_sentiment = max(0, min(1, 1.5 - vol_ratio))
            else:
                vol_sentiment = 0.5
            
            return vol_sentiment
            
        except Exception as e:
            self.logger.error(f"Volatility sentiment extraction failed: {e}")
            return 0.5
    
    def create_sentiment_features(self,
                                social_data: Optional[List[str]] = None,
                                news_data: Optional[List[Dict]] = None,
                                onchain_data: Optional[pd.DataFrame] = None,
                                market_data: Optional[pd.DataFrame] = None) -> SentimentFeatures:
        """Create comprehensive sentiment features."""
        try:
            # Extract individual sentiment components
            social_sentiment = self.extract_social_sentiment(social_data or [])
            news_sentiment = self.extract_news_sentiment(news_data or [])
            onchain_sentiment = self.extract_onchain_sentiment(onchain_data or pd.DataFrame())
            fear_greed_index = self.extract_fear_greed_index(market_data or pd.DataFrame())
            volatility_sentiment = self.extract_volatility_sentiment(market_data or pd.DataFrame())
            
            # Calculate aggregate sentiment
            sentiment_values = {
                'social': social_sentiment,
                'news': news_sentiment,
                'onchain': onchain_sentiment,
                'fear_greed': fear_greed_index
            }
            
            # Weighted average
            aggregate_sentiment = sum(
                sentiment_values[key] * self.sentiment_weights[key]
                for key in sentiment_values.keys()
            )
            
            # Calculate confidence based on data availability
            available_sources = sum(1 for v in sentiment_values.values() if v != 0.5)
            confidence = min(1.0, available_sources / len(sentiment_values))
            
            return SentimentFeatures(
                social_sentiment=social_sentiment,
                news_sentiment=news_sentiment,
                onchain_sentiment=onchain_sentiment,
                fear_greed_index=fear_greed_index,
                volatility_sentiment=volatility_sentiment,
                aggregate_sentiment=aggregate_sentiment,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment features creation failed: {e}")
            return SentimentFeatures(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0)
    
    def create_feature_vector(self, sentiment_features: SentimentFeatures) -> np.ndarray:
        """Convert sentiment features to numerical vector."""
        return np.array([
            sentiment_features.social_sentiment,
            sentiment_features.news_sentiment,
            sentiment_features.onchain_sentiment,
            sentiment_features.fear_greed_index,
            sentiment_features.volatility_sentiment,
            sentiment_features.aggregate_sentiment,
            sentiment_features.confidence
        ])
    
    def save_engine(self, filepath: str) -> None:
        """Save sentiment engine state."""
        try:
            engine_data = {
                'sentiment_weights': self.sentiment_weights,
                'sentiment_cache': self.sentiment_cache
            }
            joblib.dump(engine_data, filepath)
            self.logger.info(f"Sentiment engine saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Engine saving failed: {e}")
            raise
    
    def load_engine(self, filepath: str) -> None:
        """Load sentiment engine state."""
        try:
            engine_data = joblib.load(filepath)
            self.sentiment_weights = engine_data['sentiment_weights']
            self.sentiment_cache = engine_data['sentiment_cache']
            self.logger.info(f"Sentiment engine loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Engine loading failed: {e}")
            raise


def create_sentiment_engine(config: Optional[Dict] = None) -> SentimentFeaturesEngine:
    """Create sentiment features engine with optional configuration."""
    engine = SentimentFeaturesEngine()
    
    if config and 'sentiment_weights' in config:
        engine.sentiment_weights.update(config['sentiment_weights'])
    
    return engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sentiment engine
    engine = SentimentFeaturesEngine()
    
    # Example usage
    social_data = ["Bitcoin is bullish!", "Market looks strong"]
    features = engine.create_sentiment_features(social_data=social_data)
    feature_vector = engine.create_feature_vector(features)
    
    print("Sentiment features engine implementation completed")
