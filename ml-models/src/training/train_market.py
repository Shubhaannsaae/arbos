"""
Market Model Training Pipeline

Production training pipeline for sentiment analysis, volatility prediction, and trend analysis.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.market.sentiment_analysis import SentimentAnalysisModel
from models.market.volatility_prediction import VolatilityPredictionModel  
from models.market.trend_analysis import TrendAnalysisModel
from data.loaders.market_loader import MarketLoader
from data.features.sentiment_features import SentimentFeaturesEngine
from utils.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketTrainingPipeline:
    """Production market model training pipeline."""
    
    def __init__(self, config: Dict):
        """Initialize training pipeline."""
        self.config = config
        self.models = {}
        self.sentiment_engine = SentimentFeaturesEngine()
        self.evaluator = ModelEvaluator()
        
    async def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for training."""
        logger.info("Loading market data...")
        
        # Initialize market loader
        market_loader = MarketLoader(
            api_keys=self.config.get('api_keys', {}),
            cache_ttl=300
        )
        
        # Load data for major assets
        symbols = self.config.get('symbols', ['BTC', 'ETH', 'LINK', 'AVAX'])
        market_data = {}
        
        try:
            # Load crypto data
            crypto_data = await market_loader.get_multiple_assets(symbols, 'crypto')
            for symbol, data in crypto_data.items():
                if data:
                    df = market_loader.to_dataframe(data)
                    market_data[symbol] = df
                    logger.info(f"Loaded {len(df)} records for {symbol}")
        
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
        
        return market_data
    
    def generate_sentiment_data(self) -> Dict:
        """Generate synthetic sentiment data for training."""
        logger.info("Generating sentiment data...")
        
        # Generate synthetic social media posts
        sentiment_keywords = {
            'bullish': ['moon', 'bullish', 'buy', 'pump', 'green', 'up', 'rocket'],
            'bearish': ['dump', 'crash', 'bear', 'sell', 'red', 'down', 'fall'],
            'neutral': ['hold', 'wait', 'maybe', 'unsure', 'stable', 'sideways']
        }
        
        social_posts = []
        for _ in range(1000):
            sentiment = np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.4, 0.3, 0.3])
            keywords = sentiment_keywords[sentiment]
            post = f"Bitcoin is {np.random.choice(keywords)} today"
            social_posts.append(post)
        
        # Generate synthetic news headlines
        news_data = []
        for _ in range(200):
            sentiment = np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.35, 0.35, 0.3])
            if sentiment == 'bullish':
                headline = "Crypto market shows strong momentum"
            elif sentiment == 'bearish':
                headline = "Market faces regulatory concerns"  
            else:
                headline = "Crypto market remains stable"
            
            news_data.append({
                'title': headline,
                'content': headline + " with additional market analysis.",
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 168))
            })
        
        return {
            'social_data': social_posts,
            'news_data': news_data
        }
    
    def create_sentiment_features(self, sentiment_data: Dict, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create sentiment features."""
        logger.info("Creating sentiment features...")
        
        # Get sentiment features for each symbol
        sentiment_features_list = []
        
        for symbol, df in market_data.items():
            if len(df) < 10:
                continue
            
            # Create sentiment features
            sentiment_features = self.sentiment_engine.create_sentiment_features(
                social_data=sentiment_data['social_data'],
                news_data=sentiment_data['news_data'],
                market_data=df
            )
            
            # Convert to feature vector
            feature_vector = self.sentiment_engine.create_feature_vector(sentiment_features)
            
            sentiment_features_list.append({
                'symbol': symbol,
                'social_sentiment': feature_vector[0],
                'news_sentiment': feature_vector[1], 
                'onchain_sentiment': feature_vector[2],
                'fear_greed_index': feature_vector[3],
                'volatility_sentiment': feature_vector[4],
                'aggregate_sentiment': feature_vector[5],
                'confidence': feature_vector[6]
            })
        
        return pd.DataFrame(sentiment_features_list)
    
    def train_sentiment_analysis(self, sentiment_features: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Train sentiment analysis model."""
        logger.info("Training sentiment analysis model...")
        
        if sentiment_features.empty:
            return {}
        
        # Create training data
        training_data = []
        labels = []
        
        for _, row in sentiment_features.iterrows():
            symbol = row['symbol']
            if symbol in market_data:
                df = market_data[symbol]
                if len(df) > 0:
                    # Use price change as sentiment label
                    price_change = df['close'].pct_change().iloc[-1]
                    sentiment_label = 1 if price_change > 0 else 0
                    
                    # Features
                    features = {
                        'social_sentiment': row['social_sentiment'],
                        'news_sentiment': row['news_sentiment'],
                        'aggregate_sentiment': row['aggregate_sentiment']
                    }
                    
                    training_data.append(features)
                    labels.append(sentiment_label)
        
        if len(training_data) < 4:
            logger.warning("Insufficient data for sentiment analysis training")
            return {}
        
        # Convert to DataFrame
        features_df = pd.DataFrame(training_data)
        labels_series = pd.Series(labels)
        
        # Train model
        sentiment_model = SentimentAnalysisModel()
        
        # Prepare data for training (simplified)
        onchain_data = pd.DataFrame({
            'tx_volume': np.random.exponential(1000, 100),
            'active_addresses': np.random.poisson(5000, 100)
        })
        
        sentiment_model.train(features_df, onchain_data)
        
        # Evaluate
        predictions = []
        for _, row in features_df.iterrows():
            pred = sentiment_model.predict(features_df.iloc[:1], 'BTC')
            predictions.append(pred[0].sentiment_score)
        
        # Convert predictions to binary
        binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
        
        metrics = {
            'accuracy': accuracy_score(labels, binary_predictions),
            'avg_sentiment_score': np.mean(predictions)
        }
        
        self.models['sentiment_analysis'] = sentiment_model
        logger.info(f"Sentiment analysis - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_volatility_prediction(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Train volatility prediction model."""
        logger.info("Training volatility prediction model...")
        
        volatility_models = {}
        metrics = {}
        
        for symbol, df in market_data.items():
            if len(df) < 100:
                continue
            
            # Prepare data for LSTM
            returns = df['close'].pct_change().dropna()
            if len(returns) < 50:
                continue
            
            # Train volatility model
            vol_model = VolatilityPredictionModel(sequence_length=30)
            
            # Create sequence data
            X, y = vol_model.prepare_features(df)
            
            if len(X) > 0:
                # Train LSTM
                vol_model.train_lstm(X, y)
                
                # Test prediction
                forecast = vol_model.predict(df, symbol, horizon=1)
                
                metrics[symbol] = {
                    'realized_vol': forecast.realized_vol,
                    'predicted_vol': forecast.predicted_vol,
                    'confidence': forecast.confidence
                }
                
                volatility_models[symbol] = vol_model
                logger.info(f"Volatility model for {symbol} - Predicted: {forecast.predicted_vol:.4f}")
        
        if volatility_models:
            self.models['volatility_prediction'] = volatility_models
        
        return metrics
    
    def train_trend_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Train trend analysis model."""
        logger.info("Training trend analysis model...")
        
        trend_models = {}
        metrics = {}
        
        for symbol, df in market_data.items():
            if len(df) < 100:
                continue
            
            # Train trend model
            trend_model = TrendAnalysisModel()
            
            # Prepare features
            features_df = trend_model.prepare_features(df)
            
            if len(features_df) > 50:
                # Train model
                trend_model.train(features_df)
                
                # Test prediction
                trend_signal = trend_model.predict(features_df, symbol)
                
                # Evaluate trend prediction accuracy
                actual_trend = features_df['trend_label'].iloc[-10:].value_counts().index[0]
                predicted_trend = trend_signal.trend_label
                
                accuracy = 1.0 if actual_trend == predicted_trend else 0.0
                
                metrics[symbol] = {
                    'trend_prediction': predicted_trend,
                    'confidence': trend_signal.confidence,
                    'accuracy': accuracy
                }
                
                trend_models[symbol] = trend_model
                logger.info(f"Trend model for {symbol} - Trend: {predicted_trend}")
        
        if trend_models:
            self.models['trend_analysis'] = trend_models
        
        return metrics
    
    def save_models(self, output_dir: str):
        """Save trained models."""
        logger.info(f"Saving models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if isinstance(model, dict):
                # Multiple models per symbol
                for symbol, symbol_model in model.items():
                    filename = os.path.join(output_dir, f"{model_name}_{symbol}.joblib")
                    symbol_model.save_model(filename)
            else:
                # Single model
                filename = os.path.join(output_dir, f"{model_name}.joblib")
                model.save_model(filename)
        
        # Save sentiment engine
        sentiment_filename = os.path.join(output_dir, "sentiment_engine.joblib")
        self.sentiment_engine.save_engine(sentiment_filename)
    
    async def run_training(self) -> Dict:
        """Run complete training pipeline."""
        logger.info("Starting market model training pipeline...")
        
        try:
            # Load data
            market_data = await self.load_market_data()
            if not market_data:
                raise ValueError("No market data loaded")
            
            # Generate sentiment data
            sentiment_data = self.generate_sentiment_data()
            sentiment_features = self.create_sentiment_features(sentiment_data, market_data)
            
            # Train models
            sentiment_metrics = self.train_sentiment_analysis(sentiment_features, market_data)
            volatility_metrics = self.train_volatility_prediction(market_data)
            trend_metrics = self.train_trend_analysis(market_data)
            
            # Save models
            self.save_models(self.config.get('model_output_dir', './models'))
            
            # Compile results
            results = {
                'sentiment_metrics': sentiment_metrics,
                'volatility_metrics': volatility_metrics,
                'trend_metrics': trend_metrics,
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Market training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Market training pipeline failed: {e}")
            raise


async def main():
    """Main training function."""
    config = {
        'api_keys': {},  # Add API keys for market data
        'symbols': ['BTC', 'ETH', 'LINK'],
        'model_output_dir': './trained_models'
    }
    
    pipeline = MarketTrainingPipeline(config)
    results = await pipeline.run_training()
    
    print("Market Training Results:")
    print(f"Models trained: {list(pipeline.models.keys())}")
    print(f"Timestamp: {results['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())
