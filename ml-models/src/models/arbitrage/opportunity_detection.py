"""
Arbitrage Opportunity Detection Model

This module implements ML models for detecting profitable arbitrage opportunities
across DEXs using price differentials, liquidity analysis, and market conditions.

Features:
- Real-time opportunity detection using Chainlink Data Feeds
- Multi-DEX price comparison and analysis
- Liquidity depth assessment
- Profitability scoring with gas cost consideration
- Cross-chain arbitrage opportunity identification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import joblib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from web3 import Web3

@dataclass
class ArbitrageOpportunity:
    """Data class for arbitrage opportunity."""
    token_pair: str
    source_dex: str
    target_dex: str
    source_price: float
    target_price: float
    price_difference: float
    profit_percentage: float
    liquidity_score: float
    gas_cost_estimate: float
    confidence_score: float
    risk_score: float
    execution_time_estimate: float
    chain_id: int
    timestamp: datetime
    
class OpportunityDetectionModel:
    """
    Advanced ML model for detecting and scoring arbitrage opportunities
    across multiple DEXs and blockchain networks.
    """
    
    def __init__(self, 
                 min_profit_threshold: float = 0.5,
                 min_liquidity_threshold: float = 10000,
                 max_gas_cost_percentage: float = 50):
        """
        Initialize the opportunity detection model.
        
        Args:
            min_profit_threshold: Minimum profit percentage to consider
            min_liquidity_threshold: Minimum liquidity in USD
            max_gas_cost_percentage: Maximum gas cost as percentage of profit
        """
        self.min_profit_threshold = min_profit_threshold
        self.min_liquidity_threshold = min_liquidity_threshold
        self.max_gas_cost_percentage = max_gas_cost_percentage
        
        # Model components
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.opportunity_classifier = None
        self.profitability_scorer = None
        self.risk_assessor = None
        
        # Feature scalers
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model metadata
        self.feature_columns = []
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # DEX information
        self.dex_info = {
            'uniswap_v2': {'fee': 0.003, 'gas_multiplier': 1.0},
            'uniswap_v3': {'fee': 0.0005, 'gas_multiplier': 1.2},
            'sushiswap': {'fee': 0.003, 'gas_multiplier': 1.0},
            'curve': {'fee': 0.0004, 'gas_multiplier': 0.8},
            'balancer': {'fee': 0.001, 'gas_multiplier': 1.1},
            'pancakeswap': {'fee': 0.0025, 'gas_multiplier': 0.7}
        }
        
    def prepare_features(self, opportunities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for opportunity detection model.
        
        Args:
            opportunities_df: DataFrame with opportunity data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = opportunities_df.copy()
            
            # Price differential features
            df['price_ratio'] = df['target_price'] / df['source_price']
            df['absolute_diff'] = abs(df['target_price'] - df['source_price'])
            df['relative_diff'] = df['absolute_diff'] / df['source_price']
            
            # Liquidity features
            df['liquidity_ratio'] = df['target_liquidity'] / df['source_liquidity']
            df['min_liquidity'] = np.minimum(df['source_liquidity'], df['target_liquidity'])
            df['avg_liquidity'] = (df['source_liquidity'] + df['target_liquidity']) / 2
            df['liquidity_imbalance'] = abs(df['source_liquidity'] - df['target_liquidity']) / df['avg_liquidity']
            
            # Volume features
            df['volume_ratio'] = df['target_volume'] / df['source_volume']
            df['min_volume'] = np.minimum(df['source_volume'], df['target_volume'])
            df['avg_volume'] = (df['source_volume'] + df['target_volume']) / 2
            
            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # DEX-specific features
            df['source_fee'] = df['source_dex'].map(lambda x: self.dex_info.get(x, {}).get('fee', 0.003))
            df['target_fee'] = df['target_dex'].map(lambda x: self.dex_info.get(x, {}).get('fee', 0.003))
            df['total_fees'] = df['source_fee'] + df['target_fee']
            
            # Gas cost features
            df['gas_price_percentile'] = df.groupby('chain_id')['gas_price'].rank(pct=True)
            df['gas_cost_ratio'] = df['gas_cost_estimate'] / df['potential_profit']
            
            # Market condition features
            df['volatility_score'] = self._calculate_volatility_score(df)
            df['market_momentum'] = self._calculate_market_momentum(df)
            
            # Cross-chain features
            df['is_cross_chain'] = (df['source_chain'] != df['target_chain']).astype(int)
            df['chain_fee_diff'] = df['target_chain_fee'] - df['source_chain_fee']
            
            # Profitability features
            df['profit_after_fees'] = df['potential_profit'] - (df['potential_profit'] * df['total_fees'])
            df['profit_after_gas'] = df['profit_after_fees'] - df['gas_cost_estimate']
            df['net_profit_percentage'] = df['profit_after_gas'] / df['trade_amount'] * 100
            
            # Risk features
            df['slippage_risk'] = self._calculate_slippage_risk(df)
            df['frontrun_risk'] = self._calculate_frontrun_risk(df)
            df['execution_risk'] = self._calculate_execution_risk(df)
            
            # Competition features
            df['opportunity_age'] = (datetime.now() - pd.to_datetime(df['discovered_at'])).dt.total_seconds()
            df['similar_opportunities'] = df.groupby(['token_pair', 'hour'])['token_pair'].transform('count')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility score for each opportunity."""
        try:
            # Use price history if available, otherwise use price difference as proxy
            if 'price_volatility' in df.columns:
                return df['price_volatility']
            else:
                return df['relative_diff'] * 10  # Scale factor
                
        except Exception as e:
            self.logger.error(f"Volatility score calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_market_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market momentum indicator."""
        try:
            # Use volume and price change as momentum proxy
            momentum = (df['avg_volume'] / df['avg_volume'].rolling(10).mean()) * df['relative_diff']
            return momentum.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Market momentum calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_slippage_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate slippage risk based on liquidity and trade size."""
        try:
            # Higher trade amount relative to liquidity increases slippage risk
            slippage_risk = df['trade_amount'] / df['min_liquidity']
            return np.clip(slippage_risk * 100, 0, 100)  # Scale to 0-100
            
        except Exception as e:
            self.logger.error(f"Slippage risk calculation failed: {str(e)}")
            return pd.Series(50, index=df.index)  # Default medium risk
    
    def _calculate_frontrun_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate frontrunning risk based on gas price and profit margin."""
        try:
            # Higher profit margins and gas prices increase frontrun risk
            frontrun_risk = (df['net_profit_percentage'] * df['gas_price_percentile']) / 10
            return np.clip(frontrun_risk, 0, 100)
            
        except Exception as e:
            self.logger.error(f"Frontrun risk calculation failed: {str(e)}")
            return pd.Series(30, index=df.index)  # Default low-medium risk
    
    def _calculate_execution_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate execution risk based on various factors."""
        try:
            # Combine multiple risk factors
            execution_risk = (
                df['slippage_risk'] * 0.4 +
                df['frontrun_risk'] * 0.3 +
                df['gas_cost_ratio'] * 20 +  # Scale gas cost ratio
                df['opportunity_age'] / 60 * 10  # Time decay factor
            )
            return np.clip(execution_risk, 0, 100)
            
        except Exception as e:
            self.logger.error(f"Execution risk calculation failed: {str(e)}")
            return pd.Series(50, index=df.index)
    
    def create_training_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create training labels based on profitability and risk criteria.
        
        Args:
            df: DataFrame with opportunity features
            
        Returns:
            Series with binary labels (1 for profitable opportunity, 0 otherwise)
        """
        try:
            conditions = [
                df['net_profit_percentage'] >= self.min_profit_threshold,
                df['min_liquidity'] >= self.min_liquidity_threshold,
                df['gas_cost_ratio'] <= (self.max_gas_cost_percentage / 100),
                df['execution_risk'] <= 70,  # Risk threshold
                df['slippage_risk'] <= 50   # Slippage threshold
            ]
            
            # All conditions must be met for positive label
            labels = np.all(conditions, axis=0).astype(int)
            
            return pd.Series(labels, index=df.index)
            
        except Exception as e:
            self.logger.error(f"Label creation failed: {str(e)}")
            raise
    
    def train(self, 
              training_data: pd.DataFrame,
              validation_split: float = 0.2) -> Dict[str, any]:
        """
        Train the opportunity detection models.
        
        Args:
            training_data: Historical opportunity data
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        try:
            self.logger.info("Starting opportunity detection model training...")
            
            # Prepare features
            features_df = self.prepare_features(training_data)
            
            # Create labels
            labels = self.create_training_labels(features_df)
            
            # Select feature columns
            feature_cols = [col for col in features_df.columns 
                          if col not in ['timestamp', 'discovered_at', 'token_pair', 
                                       'source_dex', 'target_dex', 'source_chain', 'target_chain']]
            
            self.feature_columns = feature_cols
            X = features_df[feature_cols].fillna(0)
            y = labels
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            training_metrics = {}
            
            # Train anomaly detector for outlier opportunities
            self.logger.info("Training anomaly detector...")
            self.anomaly_detector.fit(X_train_scaled)
            
            # Train opportunity classifier
            self.logger.info("Training opportunity classifier...")
            self.opportunity_classifier = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            
            self.opportunity_classifier.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Evaluate classifier
            y_pred = self.opportunity_classifier.predict(X_val_scaled)
            training_metrics['classifier'] = {
                'accuracy': (y_pred == y_val).mean(),
                'classification_report': classification_report(y_val, y_pred, output_dict=True)
            }
            
            # Train profitability scorer
            self.logger.info("Training profitability scorer...")
            profit_target = features_df['net_profit_percentage']
            
            self.profitability_scorer = lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                random_state=42
            )
            
            X_train_profit, X_val_profit, y_train_profit, y_val_profit = train_test_split(
                X_train_scaled, profit_target[X_train.index], 
                test_size=validation_split, random_state=42
            )
            
            self.profitability_scorer.fit(
                X_train_profit, y_train_profit,
                eval_set=[(X_val_profit, y_val_profit)],
                callbacks=[lgb.early_stopping(50)]
            )
            
            # Evaluate profitability scorer
            profit_pred = self.profitability_scorer.predict(X_val_profit)
            training_metrics['profitability_scorer'] = {
                'mae': np.mean(np.abs(y_val_profit - profit_pred)),
                'rmse': np.sqrt(np.mean((y_val_profit - profit_pred) ** 2)),
                'r2': np.corrcoef(y_val_profit, profit_pred)[0, 1] ** 2
            }
            
            # Train risk assessor
            self.logger.info("Training risk assessor...")
            risk_target = features_df['execution_risk']
            
            self.risk_assessor = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                random_state=42
            )
            
            # Bin risk scores for classification
            risk_bins = pd.cut(risk_target, bins=[0, 30, 60, 100], labels=['low', 'medium', 'high'])
            risk_encoded = self.label_encoder.fit_transform(risk_bins.dropna())
            
            X_risk = X[risk_bins.notna()]
            X_train_risk, X_val_risk, y_train_risk, y_val_risk = train_test_split(
                X_risk, risk_encoded, test_size=validation_split, random_state=42
            )
            
            X_train_risk_scaled = self.scaler.transform(X_train_risk)
            X_val_risk_scaled = self.scaler.transform(X_val_risk)
            
            self.risk_assessor.fit(X_train_risk_scaled, y_train_risk)
            
            # Evaluate risk assessor
            risk_pred = self.risk_assessor.predict(X_val_risk_scaled)
            training_metrics['risk_assessor'] = {
                'accuracy': (risk_pred == y_val_risk).mean(),
                'classification_report': classification_report(y_val_risk, risk_pred, output_dict=True)
            }
            
            self.last_trained = datetime.now()
            self.logger.info("Opportunity detection model training completed")
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def detect_opportunities(self, 
                           current_data: pd.DataFrame,
                           return_top_k: int = 10) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities from current market data.
        
        Args:
            current_data: Current market data across DEXs
            return_top_k: Number of top opportunities to return
            
        Returns:
            List of detected arbitrage opportunities
        """
        try:
            # Prepare features
            features_df = self.prepare_features(current_data)
            
            # Select feature columns
            X = features_df[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Detect anomalies (unusual opportunities)
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
            
            # Classify opportunities
            opportunity_probs = self.opportunity_classifier.predict_proba(X_scaled)[:, 1]
            
            # Score profitability
            profitability_scores = self.profitability_scorer.predict(X_scaled)
            
            # Assess risk
            risk_scores = self.risk_assessor.predict_proba(X_scaled)
            risk_categories = self.risk_assessor.predict(X_scaled)
            
            # Combine scores
            combined_scores = (
                opportunity_probs * 0.4 +
                np.clip(profitability_scores / 100, 0, 1) * 0.3 +
                (1 - risk_scores[:, -1]) * 0.2 +  # Invert high risk probability
                np.clip((anomaly_scores + 1) / 2, 0, 1) * 0.1  # Normalize anomaly scores
            )
            
            # Filter and rank opportunities
            valid_mask = (
                opportunity_probs >= 0.5,
                profitability_scores >= self.min_profit_threshold,
                features_df['min_liquidity'] >= self.min_liquidity_threshold
            )
            valid_indices = np.all(valid_mask, axis=0)
            
            if not np.any(valid_indices):
                return []
            
            # Get top opportunities
            valid_scores = combined_scores[valid_indices]
            valid_features = features_df[valid_indices]
            
            top_indices = np.argsort(valid_scores)[-return_top_k:][::-1]
            
            opportunities = []
            for idx in top_indices:
                row = valid_features.iloc[idx]
                
                opportunity = ArbitrageOpportunity(
                    token_pair=row['token_pair'],
                    source_dex=row['source_dex'],
                    target_dex=row['target_dex'],
                    source_price=row['source_price'],
                    target_price=row['target_price'],
                    price_difference=row['absolute_diff'],
                    profit_percentage=profitability_scores[valid_indices][idx],
                    liquidity_score=row['min_liquidity'],
                    gas_cost_estimate=row['gas_cost_estimate'],
                    confidence_score=valid_scores[idx],
                    risk_score=risk_scores[valid_indices][idx][-1] * 100,  # High risk probability
                    execution_time_estimate=row.get('execution_time_estimate', 30.0),
                    chain_id=row['chain_id'],
                    timestamp=datetime.now()
                )
                
                opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Opportunity detection failed: {str(e)}")
            return []
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        try:
            model_data = {
                'anomaly_detector': self.anomaly_detector,
                'opportunity_classifier': self.opportunity_classifier,
                'profitability_scorer': self.profitability_scorer,
                'risk_assessor': self.risk_assessor,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'min_profit_threshold': self.min_profit_threshold,
                'min_liquidity_threshold': self.min_liquidity_threshold,
                'max_gas_cost_percentage': self.max_gas_cost_percentage,
                'dex_info': self.dex_info,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        try:
            model_data = joblib.load(filepath)
            
            self.anomaly_detector = model_data['anomaly_detector']
            self.opportunity_classifier = model_data['opportunity_classifier']
            self.profitability_scorer = model_data['profitability_scorer']
            self.risk_assessor = model_data['risk_assessor']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.min_profit_threshold = model_data['min_profit_threshold']
            self.min_liquidity_threshold = model_data['min_liquidity_threshold']
            self.max_gas_cost_percentage = model_data['max_gas_cost_percentage']
            self.dex_info = model_data['dex_info']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from trained models."""
        try:
            importance = {}
            
            if self.opportunity_classifier is not None:
                importance['opportunity_classifier'] = self.opportunity_classifier.feature_importances_
            
            if self.profitability_scorer is not None:
                importance['profitability_scorer'] = self.profitability_scorer.feature_importances_
            
            if self.risk_assessor is not None:
                importance['risk_assessor'] = self.risk_assessor.feature_importances_
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {str(e)}")
            return {}


def create_opportunity_detection_pipeline(config: Dict) -> OpportunityDetectionModel:
    """
    Create an opportunity detection pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured OpportunityDetectionModel instance
    """
    return OpportunityDetectionModel(
        min_profit_threshold=config.get('min_profit_threshold', 0.5),
        min_liquidity_threshold=config.get('min_liquidity_threshold', 10000),
        max_gas_cost_percentage=config.get('max_gas_cost_percentage', 50)
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = OpportunityDetectionModel()
    
    training_data = load_historical_opportunities()
    
    # Train model
    training_metrics = model.train(training_data)
    
    # Detect opportunities
    current_data = load_current_market_data()
    opportunities = model.detect_opportunities(current_data)
    
    # Save model
    model.save_model('opportunity_detection_model.pkl')
    
    print("Opportunity detection model implementation completed")
