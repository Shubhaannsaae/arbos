"""
Portfolio Model Training Pipeline

Production training pipeline for portfolio optimization and risk assessment models.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.portfolio.risk_assessment import RiskAssessmentModel
from models.portfolio.allocation_optimizer import AllocationOptimizer
from models.portfolio.rebalancing_strategy import RebalancingStrategy
from data.loaders.market_loader import MarketLoader
from data.features.market_features import MarketFeaturesEngine
from utils.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioTrainingPipeline:
    """Production portfolio model training pipeline."""
    
    def __init__(self, config: Dict):
        """Initialize training pipeline."""
        self.config = config
        self.models = {}
        self.features_engine = MarketFeaturesEngine()
        self.evaluator = ModelEvaluator()
        
    async def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load multi-asset market data."""
        logger.info("Loading market data...")
        
        # Initialize market loader
        market_loader = MarketLoader(
            api_keys=self.config.get('api_keys', {}),
            cache_ttl=300
        )
        
        # Load data for portfolio assets
        assets = self.config.get('assets', ['BTC', 'ETH', 'LINK', 'AVAX'])
        market_data = {}
        
        try:
            # Load crypto data
            crypto_data = await market_loader.get_multiple_assets(assets, 'crypto')
            for symbol, data in crypto_data.items():
                if data:
                    df = market_loader.to_dataframe(data)
                    market_data[symbol] = df
                    logger.info(f"Loaded {len(df)} records for {symbol}")
        
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
        
        return market_data
    
    def create_portfolio_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create portfolio-specific features."""
        logger.info("Creating portfolio features...")
        
        # Extract comprehensive market features
        feature_matrix = self.features_engine.create_comprehensive_feature_matrix(
            market_data, lookback_period=252
        )
        
        if feature_matrix.empty:
            return pd.DataFrame()
        
        # Add portfolio-specific features
        returns_data = {}
        for symbol, df in market_data.items():
            if len(df) > 0 and 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            
            # Portfolio volatility
            cov_matrix = returns_df.cov()
            portfolio_vol = np.sqrt(np.diag(cov_matrix)).mean()
            feature_matrix['portfolio_volatility'] = portfolio_vol
            
            # Sharpe ratio
            excess_returns = returns_df.mean() - 0.02/252  # Risk-free rate
            sharpe_ratios = excess_returns / returns_df.std()
            feature_matrix['avg_sharpe_ratio'] = sharpe_ratios.mean()
            
            # Maximum drawdown
            cumulative_returns = (1 + returns_df).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            feature_matrix['max_drawdown'] = abs(drawdowns.min().mean())
        
        return feature_matrix.fillna(0)
    
    def create_risk_targets(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create risk assessment targets."""
        risk_targets = []
        
        for symbol, df in market_data.items():
            if len(df) < 50:
                continue
            
            returns = df['close'].pct_change().dropna()
            
            # Calculate rolling risk metrics
            for i in range(30, len(returns)):
                window_returns = returns.iloc[i-30:i]
                
                risk_score = min(1.0, abs(window_returns.std() * np.sqrt(252)))  # Annualized vol
                
                risk_targets.append({
                    'timestamp': returns.index[i],
                    'symbol': symbol,
                    'risk_score': risk_score,
                    'volatility': window_returns.std(),
                    'var_95': window_returns.quantile(0.05),
                    'max_drawdown': self._calculate_max_drawdown(window_returns)
                })
        
        return pd.DataFrame(risk_targets)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a return series."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return abs(drawdown.min())
    
    def train_risk_assessment(self, features_df: pd.DataFrame, risk_targets: pd.DataFrame) -> Dict:
        """Train risk assessment model."""
        logger.info("Training risk assessment model...")
        
        if features_df.empty or risk_targets.empty:
            return {}
        
        # Align features with targets
        feature_cols = [col for col in features_df.columns if col not in ['timestamp']]
        
        # Create training data
        X = features_df[feature_cols]
        y = risk_targets['risk_score']
        
        # Ensure same length
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        
        if len(X) < 100:
            logger.warning("Insufficient data for risk assessment training")
            return {}
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        risk_model = RiskAssessmentModel()
        risk_model.train(X_train, y_train)
        
        # Evaluate
        y_pred = risk_model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': abs(y_test - y_pred).mean(),
            'accuracy': accuracy_score(y_test > 0.5, y_pred > 0.5)
        }
        
        self.models['risk_assessment'] = risk_model
        logger.info(f"Risk assessment - MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def train_allocation_optimizer(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Train allocation optimization model."""
        logger.info("Training allocation optimizer...")
        
        # Prepare return data
        returns_data = {}
        for symbol, df in market_data.items():
            if len(df) > 50:
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            logger.warning("Insufficient assets for allocation optimization")
            return {}
        
        # Train optimizer
        optimizer = AllocationOptimizer()
        
        # Use market data for training (simplified)
        returns_df = pd.DataFrame(returns_data)
        aligned_returns = returns_df.dropna()
        
        if len(aligned_returns) < 100:
            return {}
        
        # Train on historical data
        train_size = int(len(aligned_returns) * 0.8)
        train_returns = aligned_returns.iloc[:train_size]
        test_returns = aligned_returns.iloc[train_size:]
        
        # Fit optimizer
        optimizer.fit(train_returns)
        
        # Test optimization
        optimal_weights = optimizer.optimize(target_return=0.1, risk_tolerance=0.5)
        
        # Calculate performance metrics
        test_portfolio_return = (test_returns * optimal_weights).sum(axis=1)
        portfolio_vol = test_portfolio_return.std() * np.sqrt(252)
        portfolio_return = test_portfolio_return.mean() * 252
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        metrics = {
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'weights': optimal_weights.to_dict()
        }
        
        self.models['allocation_optimizer'] = optimizer
        logger.info(f"Allocation optimizer - Sharpe: {sharpe_ratio:.4f}")
        
        return metrics
    
    def train_rebalancing_strategy(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Train rebalancing strategy model."""
        logger.info("Training rebalancing strategy...")
        
        # Prepare data for rebalancing
        returns_data = {}
        for symbol, df in market_data.items():
            if len(df) > 50:
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            return {}
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 100:
            return {}
        
        # Train rebalancing strategy
        rebalancer = RebalancingStrategy()
        
        # Use rolling window for training
        window_size = 60
        performance_metrics = []
        
        for i in range(window_size, len(returns_df) - 30):
            train_window = returns_df.iloc[i-window_size:i]
            test_window = returns_df.iloc[i:i+30]
            
            # Train on window
            rebalancer.fit(train_window)
            
            # Generate rebalancing signals
            signals = rebalancer.generate_rebalancing_signals(test_window)
            
            # Calculate performance (simplified)
            if len(signals) > 0:
                avg_return = test_window.mean().mean()
                performance_metrics.append(avg_return)
        
        metrics = {
            'avg_performance': np.mean(performance_metrics) if performance_metrics else 0,
            'performance_std': np.std(performance_metrics) if performance_metrics else 0,
            'total_signals': len(performance_metrics)
        }
        
        self.models['rebalancing_strategy'] = rebalancer
        logger.info(f"Rebalancing strategy - Avg performance: {metrics['avg_performance']:.6f}")
        
        return metrics
    
    def save_models(self, output_dir: str):
        """Save trained models."""
        logger.info(f"Saving models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = os.path.join(output_dir, f"{model_name}.joblib")
            model.save_model(filename)
        
        # Save features engine
        features_filename = os.path.join(output_dir, "features_engine.joblib")
        self.features_engine.save_features_engine(features_filename)
    
    async def run_training(self) -> Dict:
        """Run complete training pipeline."""
        logger.info("Starting portfolio model training pipeline...")
        
        try:
            # Load data
            market_data = await self.load_market_data()
            if not market_data:
                raise ValueError("No market data loaded")
            
            # Create features
            features_df = self.create_portfolio_features(market_data)
            risk_targets = self.create_risk_targets(market_data)
            
            # Train models
            risk_metrics = self.train_risk_assessment(features_df, risk_targets)
            allocation_metrics = self.train_allocation_optimizer(market_data)
            rebalancing_metrics = self.train_rebalancing_strategy(market_data)
            
            # Save models
            self.save_models(self.config.get('model_output_dir', './models'))
            
            # Compile results
            results = {
                'risk_assessment_metrics': risk_metrics,
                'allocation_metrics': allocation_metrics,
                'rebalancing_metrics': rebalancing_metrics,
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Portfolio training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Portfolio training pipeline failed: {e}")
            raise


async def main():
    """Main training function."""
    config = {
        'api_keys': {},  # Add API keys for market data
        'assets': ['BTC', 'ETH', 'LINK', 'AVAX'],
        'model_output_dir': './trained_models'
    }
    
    pipeline = PortfolioTrainingPipeline(config)
    results = await pipeline.run_training()
    
    print("Portfolio Training Results:")
    print(f"Models trained: {list(pipeline.models.keys())}")
    print(f"Timestamp: {results['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())
