"""
Arbitrage Model Training Pipeline

Production training pipeline for arbitrage detection and profitability models
with Chainlink Data Feeds integration.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.arbitrage.price_prediction import PricePredictionModel
from models.arbitrage.opportunity_detection import ArbitrageOpportunityDetector
from models.arbitrage.profitability_model import ProfitabilityModel
from data.loaders.chainlink_loader import ChainlinkLoader
from data.loaders.dex_loader import DEXLoader
from data.preprocessing.price_data import PriceDataPreprocessor
from utils.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArbitrageTrainingPipeline:
    """Production arbitrage model training pipeline."""
    
    def __init__(self, config: Dict):
        """Initialize training pipeline."""
        self.config = config
        self.models = {}
        self.preprocessors = {}
        self.evaluator = ModelEvaluator()
        
    async def load_training_data(self) -> Dict[str, pd.DataFrame]:
        """Load training data from multiple sources."""
        logger.info("Loading training data...")
        
        # Initialize loaders
        chainlink_loader = ChainlinkLoader(self.config['rpc_urls'])
        
        # Load price data for major trading pairs
        symbols = self.config.get('symbols', ['BTC', 'ETH', 'LINK', 'AVAX'])
        price_data = {}
        
        for symbol in symbols:
            try:
                # Get recent price history
                end_time = datetime.now()
                start_time = end_time - timedelta(days=self.config.get('training_days', 90))
                
                prices = await chainlink_loader.get_historical_prices(
                    symbol, start_time, end_time
                )
                
                if prices:
                    df = chainlink_loader.to_dataframe(prices)
                    price_data[symbol] = df
                    logger.info(f"Loaded {len(df)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        return price_data
    
    def preprocess_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess price data for training."""
        logger.info("Preprocessing data...")
        
        preprocessor = PriceDataPreprocessor()
        processed_data = {}
        
        for symbol, df in price_data.items():
            try:
                # Clean and normalize data
                processed_df, _, _ = preprocessor.process_price_data(df, symbol)
                processed_data[symbol] = processed_df
                
            except Exception as e:
                logger.error(f"Preprocessing failed for {symbol}: {e}")
        
        self.preprocessors['price'] = preprocessor
        return processed_data
    
    def create_arbitrage_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features for arbitrage detection."""
        feature_dfs = []
        
        for symbol, df in data.items():
            if len(df) < 50:  # Skip insufficient data
                continue
                
            # Price-based features
            df_features = df.copy()
            df_features['symbol'] = symbol
            df_features['returns'] = df_features['price'].pct_change()
            df_features['volatility'] = df_features['returns'].rolling(20).std()
            df_features['momentum'] = df_features['price'].pct_change(10)
            
            # Cross-asset features
            for other_symbol, other_df in data.items():
                if other_symbol != symbol and len(other_df) >= len(df):
                    # Price ratio
                    aligned_other = other_df['price'].reindex(df.index, method='ffill')
                    df_features[f'{symbol}_{other_symbol}_ratio'] = df_features['price'] / aligned_other
                    
                    # Correlation
                    corr = df_features['returns'].rolling(30).corr(
                        other_df['price'].pct_change().reindex(df.index, method='ffill')
                    )
                    df_features[f'{symbol}_{other_symbol}_corr'] = corr
            
            feature_dfs.append(df_features)
        
        if not feature_dfs:
            return pd.DataFrame()
        
        # Combine all features
        combined_df = pd.concat(feature_dfs, axis=0, ignore_index=True)
        return combined_df.fillna(0)
    
    def train_price_prediction(self, features_df: pd.DataFrame) -> Dict:
        """Train price prediction models."""
        logger.info("Training price prediction models...")
        
        price_models = {}
        metrics = {}
        
        for symbol in features_df['symbol'].unique():
            symbol_data = features_df[features_df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 100:
                continue
            
            # Prepare features and targets
            feature_cols = [col for col in symbol_data.columns 
                          if col not in ['symbol', 'timestamp', 'price']]
            X = symbol_data[feature_cols]
            y = symbol_data['price'].shift(-1).dropna()  # Next period price
            X = X.iloc[:-1]  # Align with shifted target
            
            # Train-test split with time series
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            model = PricePredictionModel()
            model.train(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            metrics[symbol] = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            price_models[symbol] = model
            logger.info(f"Price model for {symbol} - MAE: {metrics[symbol]['mae']:.4f}")
        
        self.models['price_prediction'] = price_models
        return metrics
    
    def train_opportunity_detection(self, features_df: pd.DataFrame) -> Dict:
        """Train arbitrage opportunity detection."""
        logger.info("Training opportunity detection...")
        
        # Create opportunity labels (simplified)
        opportunities = []
        for symbol in features_df['symbol'].unique():
            symbol_data = features_df[features_df['symbol'] == symbol].copy()
            
            # Define opportunity as significant price deviation from moving average
            ma_20 = symbol_data['price'].rolling(20).mean()
            deviation = abs(symbol_data['price'] - ma_20) / ma_20
            symbol_data['opportunity'] = (deviation > 0.02).astype(int)
            
            opportunities.append(symbol_data)
        
        if not opportunities:
            return {}
        
        combined_data = pd.concat(opportunities, ignore_index=True)
        
        # Prepare features
        feature_cols = [col for col in combined_data.columns 
                       if col not in ['symbol', 'timestamp', 'price', 'opportunity']]
        X = combined_data[feature_cols]
        y = combined_data['opportunity']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        detector = ArbitrageOpportunityDetector()
        detector.train(X_train, y_train)
        
        # Evaluate
        y_pred = detector.predict(X_test)
        metrics = self.evaluator.evaluate_classification(y_test, y_pred)
        
        self.models['opportunity_detection'] = detector
        logger.info(f"Opportunity detection - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_profitability_model(self, features_df: pd.DataFrame) -> Dict:
        """Train profitability estimation model."""
        logger.info("Training profitability model...")
        
        # Create profitability targets (simplified)
        profit_data = []
        for symbol in features_df['symbol'].unique():
            symbol_data = features_df[features_df['symbol'] == symbol].copy()
            
            # Calculate forward returns as profit proxy
            returns = symbol_data['price'].pct_change(5).shift(-5)  # 5-period forward return
            symbol_data['profitability'] = returns.fillna(0)
            
            profit_data.append(symbol_data)
        
        if not profit_data:
            return {}
        
        combined_data = pd.concat(profit_data, ignore_index=True)
        
        # Prepare features
        feature_cols = [col for col in combined_data.columns 
                       if col not in ['symbol', 'timestamp', 'price', 'profitability']]
        X = combined_data[feature_cols]
        y = combined_data['profitability']
        
        # Remove rows with NaN targets
        mask = ~y.isna()
        X, y = X[mask], y[mask]
        
        if len(X) < 100:
            return {}
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        profit_model = ProfitabilityModel()
        profit_model.train(X_train, y_train)
        
        # Evaluate
        y_pred = profit_model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.models['profitability'] = profit_model
        logger.info(f"Profitability model - MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def save_models(self, output_dir: str):
        """Save trained models."""
        logger.info(f"Saving models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if isinstance(model, dict):
                # Multiple models (e.g., price prediction per symbol)
                for symbol, symbol_model in model.items():
                    filename = os.path.join(output_dir, f"{model_name}_{symbol}.joblib")
                    symbol_model.save_model(filename)
            else:
                # Single model
                filename = os.path.join(output_dir, f"{model_name}.joblib")
                model.save_model(filename)
        
        # Save preprocessors
        for prep_name, preprocessor in self.preprocessors.items():
            filename = os.path.join(output_dir, f"preprocessor_{prep_name}.joblib")
            preprocessor.save_preprocessor(filename)
    
    async def run_training(self) -> Dict:
        """Run complete training pipeline."""
        logger.info("Starting arbitrage model training pipeline...")
        
        try:
            # Load data
            price_data = await self.load_training_data()
            if not price_data:
                raise ValueError("No training data loaded")
            
            # Preprocess
            processed_data = self.preprocess_data(price_data)
            
            # Create features
            features_df = self.create_arbitrage_features(processed_data)
            if features_df.empty:
                raise ValueError("No features created")
            
            # Train models
            price_metrics = self.train_price_prediction(features_df)
            opportunity_metrics = self.train_opportunity_detection(features_df)
            profitability_metrics = self.train_profitability_model(features_df)
            
            # Save models
            self.save_models(self.config.get('model_output_dir', './models'))
            
            # Compile results
            results = {
                'price_prediction_metrics': price_metrics,
                'opportunity_detection_metrics': opportunity_metrics,
                'profitability_metrics': profitability_metrics,
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


async def main():
    """Main training function."""
    config = {
        'rpc_urls': {'ethereum': 'https://eth.llamarpc.com'},
        'symbols': ['BTC', 'ETH', 'LINK'],
        'training_days': 30,
        'model_output_dir': './trained_models'
    }
    
    pipeline = ArbitrageTrainingPipeline(config)
    results = await pipeline.run_training()
    
    print("Training Results:")
    print(f"Models trained: {list(pipeline.models.keys())}")
    print(f"Timestamp: {results['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())
