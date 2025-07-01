"""
Security Model Training Pipeline

Production training pipeline for fraud detection, anomaly detection, and risk scoring.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.security.fraud_detection import FraudDetectionModel
from models.security.anomaly_detection import AnomalyDetectionModel
from models.security.risk_scorer import RiskScoringModel
from data.loaders.chainlink_loader import ChainlinkLoader
from utils.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityTrainingPipeline:
    """Production security model training pipeline."""
    
    def __init__(self, config: Dict):
        """Initialize training pipeline."""
        self.config = config
        self.models = {}
        self.evaluator = ModelEvaluator()
        
    async def generate_transaction_data(self) -> pd.DataFrame:
        """Generate transaction data for security training."""
        logger.info("Generating transaction data...")
        
        # Initialize Chainlink loader
        chainlink_loader = ChainlinkLoader(self.config['rpc_urls'])
        
        # Get price data for context
        symbols = ['BTC', 'ETH', 'LINK']
        price_context = {}
        
        for symbol in symbols:
            try:
                price = await chainlink_loader.get_latest_price(symbol)
                if price:
                    price_context[symbol] = price.price
            except Exception as e:
                logger.warning(f"Could not get price for {symbol}: {e}")
        
        # Generate synthetic transaction data (in production, use real transaction logs)
        np.random.seed(42)
        n_transactions = self.config.get('n_transactions', 10000)
        
        transactions = []
        for i in range(n_transactions):
            # Create realistic transaction features
            tx = {
                'transaction_hash': f'0x{i:064x}',
                'from_address': f'0x{np.random.randint(0, 16**40):040x}',
                'to_address': f'0x{np.random.randint(0, 16**40):040x}',
                'value': np.random.exponential(0.1),  # Exponential distribution for values
                'gas_used': np.random.randint(21000, 500000),
                'gas_price': np.random.exponential(20),
                'block_number': 18000000 + i,
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 10080)),
                'contract_interaction': np.random.choice([True, False], p=[0.3, 0.7]),
                
                # Derived features
                'tx_count_from': np.random.poisson(50),
                'tx_count_to': np.random.poisson(30),
                'value_percentile': np.random.uniform(0, 1),
                'gas_efficiency': np.random.exponential(0.001),
                'time_since_prev_tx': np.random.exponential(3600),
                'unique_counterparties': np.random.poisson(20),
                'address_age_days': np.random.exponential(365),
                'round_number': np.random.choice([0, 1], p=[0.9, 0.1]),
                
                # Network features
                'degree_centrality': np.random.beta(2, 5),
                'betweenness_centrality': np.random.beta(1, 10),
                'community_id': np.random.randint(0, 100),
                
                # Volume and timing features  
                'is_night': np.random.choice([0, 1], p=[0.7, 0.3]),
                'is_weekend': np.random.choice([0, 1], p=[0.7, 0.3]),
                'tx_in_last_minute': np.random.poisson(2),
            }
            
            transactions.append(tx)
        
        return pd.DataFrame(transactions)
    
    def create_fraud_labels(self, transactions_df: pd.DataFrame) -> pd.Series:
        """Create fraud labels based on transaction patterns."""
        fraud_indicators = (
            (transactions_df['value'] > transactions_df['value'].quantile(0.99)) |  # Very high value
            (transactions_df['round_number'] == 1) |  # Round numbers
            (transactions_df['tx_in_last_minute'] > 10) |  # Burst activity
            (transactions_df['gas_price'] > transactions_df['gas_price'].quantile(0.95)) |  # High gas
            (transactions_df['time_since_prev_tx'] < 60)  # Very frequent
        )
        
        # Add some noise to make it realistic
        noise = np.random.choice([0, 1], size=len(transactions_df), p=[0.95, 0.05])
        fraud_labels = (fraud_indicators.astype(int) | noise).astype(int)
        
        return fraud_labels
    
    def create_anomaly_data(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for anomaly detection."""
        # Select numerical features for anomaly detection
        anomaly_features = [
            'value', 'gas_used', 'gas_price', 'tx_count_from', 'tx_count_to',
            'gas_efficiency', 'time_since_prev_tx', 'unique_counterparties',
            'address_age_days', 'degree_centrality', 'betweenness_centrality'
        ]
        
        return transactions_df[anomaly_features].fillna(0)
    
    def create_risk_features(self, transactions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create features for risk scoring by component."""
        # Group features by risk component
        risk_features = {
            'transaction_risk': transactions_df[[
                'value', 'gas_used', 'gas_price', 'gas_efficiency', 'round_number'
            ]].fillna(0),
            
            'behavioral_risk': transactions_df[[
                'tx_count_from', 'tx_count_to', 'time_since_prev_tx',
                'unique_counterparties', 'address_age_days'
            ]].fillna(0),
            
            'network_risk': transactions_df[[
                'degree_centrality', 'betweenness_centrality', 'community_id'
            ]].fillna(0),
            
            'temporal_risk': transactions_df[[
                'is_night', 'is_weekend', 'tx_in_last_minute'
            ]].fillna(0)
        }
        
        return risk_features
    
    def train_fraud_detection(self, transactions_df: pd.DataFrame) -> Dict:
        """Train fraud detection model."""
        logger.info("Training fraud detection model...")
        
        # Create features and labels
        feature_cols = [col for col in transactions_df.columns 
                       if col not in ['transaction_hash', 'from_address', 'to_address', 'timestamp']]
        
        X = transactions_df[feature_cols]
        y = self.create_fraud_labels(transactions_df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        fraud_model = FraudDetectionModel()
        training_metrics = fraud_model.train_fraud_detection_models(
            X_train, y_train, validation_split=0.2
        )
        
        # Evaluate on test set
        y_pred_proba = fraud_model.transaction_classifier.predict_proba(
            fraud_model.transaction_scaler.transform(X_test)
        )[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        test_metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.models['fraud_detection'] = fraud_model
        logger.info(f"Fraud detection - AUC: {test_metrics['auc_roc']:.4f}")
        
        return {**training_metrics, 'test_metrics': test_metrics}
    
    def train_anomaly_detection(self, transactions_df: pd.DataFrame) -> Dict:
        """Train anomaly detection model."""
        logger.info("Training anomaly detection model...")
        
        # Prepare anomaly detection data
        anomaly_data = self.create_anomaly_data(transactions_df)
        
        # Train model (unsupervised)
        anomaly_model = AnomalyDetectionModel()
        training_metrics = anomaly_model.train_anomaly_detectors(
            anomaly_data, validation_split=0.2
        )
        
        # Test anomaly detection
        test_data = anomaly_data.sample(frac=0.2, random_state=42)
        anomalies = anomaly_model.detect_anomalies(test_data)
        
        test_metrics = {
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / len(test_data),
            'avg_anomaly_score': np.mean([a.anomaly_score for a in anomalies]) if anomalies else 0
        }
        
        self.models['anomaly_detection'] = anomaly_model
        logger.info(f"Anomaly detection - Rate: {test_metrics['anomaly_rate']:.4f}")
        
        return {**training_metrics, 'test_metrics': test_metrics}
    
    def train_risk_scoring(self, transactions_df: pd.DataFrame) -> Dict:
        """Train risk scoring model."""
        logger.info("Training risk scoring model...")
        
        # Prepare risk features
        risk_features = self.create_risk_features(transactions_df)
        
        # Create risk labels (composite score)
        risk_labels = {}
        for component, features in risk_features.items():
            # Simple risk scoring based on feature percentiles
            risk_score = features.apply(lambda x: x.rank(pct=True)).mean(axis=1) * 100
            risk_labels[component] = risk_score
        
        # Train model
        risk_model = RiskScoringModel()
        training_metrics = risk_model.train_risk_scoring_models(
            risk_features, risk_labels, validation_split=0.2
        )
        
        # Test risk scoring
        test_idx = transactions_df.sample(frac=0.2, random_state=42).index
        test_entity = transactions_df.loc[test_idx.min()]
        
        risk_score = risk_model.calculate_risk_score(test_entity, 'address')
        
        test_metrics = {
            'test_risk_score': risk_score.overall_risk_score,
            'test_risk_category': risk_score.risk_category,
            'test_confidence': risk_score.confidence
        }
        
        self.models['risk_scoring'] = risk_model
        logger.info(f"Risk scoring - Test score: {risk_score.overall_risk_score:.2f}")
        
        return {**training_metrics, 'test_metrics': test_metrics}
    
    def save_models(self, output_dir: str):
        """Save trained models."""
        logger.info(f"Saving models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = os.path.join(output_dir, f"{model_name}.joblib")
            model.save_model(filename)
    
    async def run_training(self) -> Dict:
        """Run complete training pipeline."""
        logger.info("Starting security model training pipeline...")
        
        try:
            # Generate training data
            transactions_df = await self.generate_transaction_data()
            logger.info(f"Generated {len(transactions_df)} transactions for training")
            
            # Train models
            fraud_metrics = self.train_fraud_detection(transactions_df)
            anomaly_metrics = self.train_anomaly_detection(transactions_df)
            risk_metrics = self.train_risk_scoring(transactions_df)
            
            # Save models
            self.save_models(self.config.get('model_output_dir', './models'))
            
            # Compile results
            results = {
                'fraud_detection_metrics': fraud_metrics,
                'anomaly_detection_metrics': anomaly_metrics,
                'risk_scoring_metrics': risk_metrics,
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Security training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Security training pipeline failed: {e}")
            raise


async def main():
    """Main training function."""
    config = {
        'rpc_urls': {'ethereum': 'https://eth.llamarpc.com'},
        'n_transactions': 5000,
        'model_output_dir': './trained_models'
    }
    
    pipeline = SecurityTrainingPipeline(config)
    results = await pipeline.run_training()
    
    print("Security Training Results:")
    print(f"Models trained: {list(pipeline.models.keys())}")
    print(f"Timestamp: {results['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())
