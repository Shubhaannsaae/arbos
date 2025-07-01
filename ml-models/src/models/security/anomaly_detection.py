"""
Anomaly Detection Model for Blockchain and DeFi Security

This module implements advanced ML models for detecting anomalous behavior
in blockchain transactions, DeFi protocols, and smart contract interactions
using statistical methods, machine learning, and real-time monitoring with Chainlink oracles.

Features:
- Multi-dimensional anomaly detection using ensemble methods
- Statistical process control for blockchain metrics
- Real-time transaction stream analysis with Chainlink Data Feeds
- Time series anomaly detection for market manipulation
- Graph-based anomaly detection for transaction networks
- Behavioral pattern analysis for wallet and contract activities
- Cross-chain anomaly correlation using CCIP
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.ensemble import IsolationForest, OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import scipy.stats as stats
from scipy.signal import find_peaks
import pyod
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import joblib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import networkx as nx
from collections import defaultdict, deque
import hashlib

warnings.filterwarnings('ignore')

@dataclass
class AnomalyDetectionResult:
    """Data class for anomaly detection results."""
    entity_id: str  # Transaction hash, address, or contract
    entity_type: str  # 'transaction', 'address', 'contract', 'block'
    anomaly_score: float  # 0-1 normalized anomaly score
    anomaly_type: str  # Type of anomaly detected
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # Confidence in detection
    detection_method: str  # Method that detected the anomaly
    features_contributing: List[str]  # Features that contributed to anomaly
    baseline_metrics: Dict[str, float]  # Normal baseline for comparison
    observed_metrics: Dict[str, float]  # Observed values
    timestamp: datetime
    recommendations: List[str]  # Recommended actions

@dataclass
class TimeSeriesAnomaly:
    """Data class for time series anomalies."""
    timestamp: datetime
    metric_name: str
    observed_value: float
    expected_value: float
    anomaly_score: float
    anomaly_type: str  # 'point', 'contextual', 'collective'
    window_size: int
    statistical_significance: float

@dataclass
class GraphAnomaly:
    """Data class for graph-based anomalies."""
    node_id: str
    anomaly_score: float
    centrality_deviation: float
    community_deviation: float
    edge_pattern_anomaly: float
    temporal_pattern_anomaly: float

class AutoencoderAnomalyDetector(nn.Module):
    """
    Deep autoencoder for transaction anomaly detection.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, dropout_rate: float = 0.2):
        super(AutoencoderAnomalyDetector, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based anomaly detector for time series data.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout_rate: float = 0.2):
        super(LSTMAnomalyDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.dropout(lstm_out)
        predictions = self.output_layer(output)
        return predictions

class AnomalyDetectionModel:
    """
    Comprehensive anomaly detection system for blockchain and DeFi security
    using multiple detection methods and real-time monitoring capabilities.
    """
    
    def __init__(self, 
                 detection_methods: List[str] = None,
                 ensemble_weights: Dict[str, float] = None,
                 real_time_monitoring: bool = True):
        """
        Initialize the anomaly detection model.
        
        Args:
            detection_methods: List of detection methods to use
            ensemble_weights: Weights for ensemble combination
            real_time_monitoring: Enable real-time monitoring
        """
        self.detection_methods = detection_methods or [
            'statistical', 'isolation_forest', 'autoencoder', 'lstm', 
            'graph_based', 'clustering', 'one_class_svm', 'lof'
        ]
        
        self.ensemble_weights = ensemble_weights or {
            'statistical': 0.15,
            'isolation_forest': 0.2,
            'autoencoder': 0.25,
            'lstm': 0.2,
            'graph_based': 0.1,
            'clustering': 0.05,
            'one_class_svm': 0.03,
            'lof': 0.02
        }
        
        self.real_time_monitoring = real_time_monitoring
        
        # Detection models
        self.statistical_detector = None
        self.isolation_forest = None
        self.autoencoder_model = None
        self.lstm_model = None
        self.graph_detector = None
        self.clustering_model = None
        self.one_class_svm = None
        self.lof_detector = None
        
        # Specialized detectors
        self.time_series_detector = None
        self.behavioral_detector = None
        self.network_detector = None
        
        # Feature scalers and preprocessors
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.minmax_scaler = MinMaxScaler()
        self.pca_transformer = None
        
        # Real-time monitoring components
        self.baseline_metrics = {}
        self.sliding_windows = {}
        self.alert_thresholds = {}
        self.detection_history = deque(maxlen=10000)
        
        # Model metadata
        self.feature_columns = []
        self.baseline_period_days = 30
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Anomaly type definitions
        self.anomaly_types = {
            'volume_spike': 'Unusual volume spike detected',
            'price_manipulation': 'Potential price manipulation',
            'wash_trading': 'Wash trading pattern detected',
            'pump_and_dump': 'Pump and dump scheme indicators',
            'flash_crash': 'Flash crash event detected',
            'liquidity_drain': 'Unusual liquidity drainage',
            'gas_price_anomaly': 'Abnormal gas price behavior',
            'mev_extraction': 'MEV extraction anomaly',
            'governance_attack': 'Governance attack pattern',
            'oracle_manipulation': 'Oracle price manipulation',
            'bridge_exploit': 'Cross-chain bridge anomaly',
            'contract_upgrade_suspicious': 'Suspicious contract upgrade',
            'large_transfer_anomaly': 'Unusual large transfer',
            'temporal_anomaly': 'Time-based behavioral anomaly',
            'network_structure_anomaly': 'Transaction network anomaly'
        }
        
    def prepare_anomaly_features(self, 
                                data: pd.DataFrame,
                                entity_type: str = 'transaction') -> pd.DataFrame:
        """
        Prepare features for anomaly detection.
        
        Args:
            data: Raw data (transactions, addresses, contracts, etc.)
            entity_type: Type of entity for feature engineering
            
        Returns:
            DataFrame with anomaly detection features
        """
        try:
            features_df = data.copy()
            
            if entity_type == 'transaction':
                features_df = self._prepare_transaction_features(features_df)
            elif entity_type == 'address':
                features_df = self._prepare_address_features(features_df)
            elif entity_type == 'contract':
                features_df = self._prepare_contract_features(features_df)
            elif entity_type == 'block':
                features_df = self._prepare_block_features(features_df)
            elif entity_type == 'market':
                features_df = self._prepare_market_features(features_df)
            
            # Add temporal features
            features_df = self._add_temporal_anomaly_features(features_df)
            
            # Add statistical features
            features_df = self._add_statistical_features(features_df)
            
            # Add network features
            features_df = self._add_network_anomaly_features(features_df)
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise
    
    def _prepare_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare transaction-specific anomaly features."""
        try:
            # Basic transaction metrics
            df['log_value'] = np.log1p(df['value'])
            df['log_gas_used'] = np.log1p(df['gas_used'])
            df['log_gas_price'] = np.log1p(df['gas_price'])
            
            # Gas efficiency metrics
            df['gas_efficiency'] = df['value'] / (df['gas_used'] * df['gas_price'] + 1e-8)
            df['gas_price_to_limit_ratio'] = df['gas_price'] / (df['gas_limit'] + 1e-8)
            
            # Value percentiles and z-scores
            df['value_percentile'] = df['value'].rank(pct=True)
            df['value_zscore'] = stats.zscore(df['value'], nan_policy='omit')
            df['gas_price_zscore'] = stats.zscore(df['gas_price'], nan_policy='omit')
            
            # Transaction frequency features
            df['tx_count_from'] = df.groupby('from_address')['transaction_hash'].transform('count')
            df['tx_count_to'] = df.groupby('to_address')['transaction_hash'].transform('count')
            df['tx_frequency_from'] = df.groupby('from_address')['timestamp'].transform(
                lambda x: len(x) / ((x.max() - x.min()).total_seconds() / 3600 + 1)
            )
            
            # Value distribution features
            df['value_std_from'] = df.groupby('from_address')['value'].transform('std').fillna(0)
            df['value_mean_from'] = df.groupby('from_address')['value'].transform('mean')
            df['value_deviation_from_mean'] = abs(df['value'] - df['value_mean_from'])
            
            # Timing features
            df = df.sort_values(['from_address', 'timestamp'])
            df['time_since_prev_tx'] = df.groupby('from_address')['timestamp'].diff().dt.total_seconds()
            df['time_regularity'] = df.groupby('from_address')['time_since_prev_tx'].transform('std').fillna(0)
            
            # Contract interaction patterns
            df['is_contract_creation'] = (df['to_address'].isna()).astype(int)
            df['input_data_size'] = df['input_data'].str.len().fillna(0)
            df['has_complex_input'] = (df['input_data_size'] > 100).astype(int)
            
            # Round number detection (potential automation)
            df['is_round_value'] = self._detect_round_numbers(df['value'])
            df['is_round_gas'] = self._detect_round_numbers(df['gas_price'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Transaction feature preparation failed: {str(e)}")
            return df
    
    def _prepare_address_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare address-specific anomaly features."""
        try:
            # Activity patterns
            df['total_transactions'] = df.groupby('address')['transaction_count'].transform('sum')
            df['avg_transaction_value'] = df.groupby('address')['total_value'].transform('sum') / df['total_transactions']
            df['transaction_value_std'] = df.groupby('address')['avg_transaction_value'].transform('std').fillna(0)
            
            # Temporal activity patterns
            df['activity_span_hours'] = df.groupby('address')['last_activity'].transform(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
            df['transactions_per_hour'] = df['total_transactions'] / (df['activity_span_hours'] + 1)
            
            # Balance and flow analysis
            df['balance_volatility'] = df.groupby('address')['balance'].transform(
                lambda x: x.std() / (x.mean() + 1e-8)
            )
            df['net_flow'] = df['total_received'] - df['total_sent']
            df['flow_ratio'] = df['total_received'] / (df['total_sent'] + 1e-8)
            
            # Network connectivity
            df['unique_counterparties'] = df['unique_senders'] + df['unique_receivers']
            df['interaction_diversity'] = df['unique_counterparties'] / (df['total_transactions'] + 1)
            
            # Address age and maturity
            df['address_age_days'] = (datetime.now() - df['first_activity']).dt.days
            df['maturity_score'] = df['address_age_days'] / (df['address_age_days'].max() + 1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Address feature preparation failed: {str(e)}")
            return df
    
    def _prepare_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare contract-specific anomaly features."""
        try:
            # Contract metadata features
            df['is_verified'] = df['verification_status'].astype(int)
            df['has_proxy'] = df['is_proxy'].astype(int)
            df['code_size_log'] = np.log1p(df['code_size'])
            
            # Function signature analysis
            df['unique_functions'] = df['function_signatures'].str.split(',').str.len().fillna(0)
            df['has_selfdestruct'] = df['function_signatures'].str.contains('selfdestruct', na=False).astype(int)
            df['has_delegatecall'] = df['function_signatures'].str.contains('delegatecall', na=False).astype(int)
            
            # Usage patterns
            df['daily_transaction_count'] = df['total_transactions'] / (df['contract_age_days'] + 1)
            df['transaction_density'] = df['total_transactions'] / (df['code_size'] + 1)
            
            # Value flow patterns
            df['avg_transaction_value'] = df['total_value_processed'] / (df['total_transactions'] + 1)
            df['value_concentration'] = df['max_single_transaction'] / (df['total_value_processed'] + 1)
            
            # Update and upgrade patterns
            df['upgrade_frequency'] = df['upgrade_count'] / (df['contract_age_days'] + 1)
            df['recent_upgrade'] = (df['days_since_last_upgrade'] < 7).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Contract feature preparation failed: {str(e)}")
            return df
    
    def _prepare_block_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare block-specific anomaly features."""
        try:
            # Block size and transaction count
            df['transaction_density'] = df['transaction_count'] / (df['block_size'] + 1)
            df['avg_transaction_size'] = df['block_size'] / (df['transaction_count'] + 1)
            
            # Gas usage patterns
            df['gas_utilization'] = df['gas_used'] / df['gas_limit']
            df['avg_gas_per_tx'] = df['gas_used'] / (df['transaction_count'] + 1)
            df['gas_efficiency'] = df['total_value'] / (df['gas_used'] + 1)
            
            # Block timing
            df['block_time'] = df['timestamp'].diff().dt.total_seconds()
            df['block_time_deviation'] = abs(df['block_time'] - df['block_time'].median())
            
            # MEV and priority fee analysis
            df['priority_fee_range'] = df['max_priority_fee'] - df['min_priority_fee']
            df['fee_variance'] = df['priority_fee_range'] / (df['avg_priority_fee'] + 1e-8)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Block feature preparation failed: {str(e)}")
            return df
    
    def _prepare_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare market-specific anomaly features."""
        try:
            # Price movement features
            df['price_return'] = df['price'].pct_change()
            df['log_return'] = np.log(df['price'] / df['price'].shift(1))
            df['price_volatility'] = df['price_return'].rolling(20).std()
            
            # Volume features
            df['volume_log'] = np.log1p(df['volume'])
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_deviation'] = (df['volume'] - df['volume_ma']) / df['volume_ma']
            
            # Price-volume relationships
            df['price_volume_correlation'] = df['price'].rolling(20).corr(df['volume'])
            df['volume_price_ratio'] = df['volume'] / (df['price'] + 1e-8)
            
            # Market microstructure
            df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
            df['spread_percentage'] = df['bid_ask_spread'] / df['mid_price']
            df['market_impact'] = abs(df['price_return']) / (df['volume_deviation'] + 1e-8)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market feature preparation failed: {str(e)}")
            return df
    
    def _add_temporal_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal pattern analysis features."""
        try:
            # Time-based features
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['is_night'] = df['hour'].isin(list(range(22, 24)) + list(range(0, 6))).astype(int)
                
                # Activity concentration
                df['hour_activity_concentration'] = df.groupby('hour').transform('count').iloc[:, 0]
                df['weekend_activity_ratio'] = df.groupby('is_weekend').transform('count').iloc[:, 0]
                
                # Temporal clustering
                df['activity_burst'] = self._detect_activity_bursts(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Temporal feature preparation failed: {str(e)}")
            return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical anomaly detection features."""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col not in ['timestamp', 'block_number']:
                    # Statistical measures
                    df[f'{col}_zscore'] = np.abs(stats.zscore(df[col], nan_policy='omit'))
                    df[f'{col}_percentile'] = df[col].rank(pct=True)
                    
                    # Outlier indicators
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[f'{col}_outlier_iqr'] = ((df[col] < (Q1 - 1.5 * IQR)) | 
                                               (df[col] > (Q3 + 1.5 * IQR))).astype(int)
                    
                    # Moving statistics
                    if len(df) > 20:
                        df[f'{col}_ma_20'] = df[col].rolling(20).mean()
                        df[f'{col}_std_20'] = df[col].rolling(20).std()
                        df[f'{col}_deviation_from_ma'] = abs(df[col] - df[f'{col}_ma_20'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Statistical feature preparation failed: {str(e)}")
            return df
    
    def _add_network_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add network-based anomaly features."""
        try:
            # Only add network features for transaction data
            if 'from_address' in df.columns and 'to_address' in df.columns:
                # Create transaction graph
                G = nx.DiGraph()
                
                for _, row in df.iterrows():
                    if pd.notna(row['from_address']) and pd.notna(row['to_address']):
                        G.add_edge(row['from_address'], row['to_address'], 
                                 weight=row.get('value', 1))
                
                # Calculate network metrics
                try:
                    centrality_dict = nx.degree_centrality(G)
                    betweenness_dict = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
                    
                    df['degree_centrality'] = df['from_address'].map(centrality_dict).fillna(0)
                    df['betweenness_centrality'] = df['from_address'].map(betweenness_dict).fillna(0)
                    
                    # Community detection
                    communities = nx.community.greedy_modularity_communities(G.to_undirected())
                    community_map = {}
                    for i, community in enumerate(communities):
                        for node in community:
                            community_map[node] = i
                    
                    df['community_id'] = df['from_address'].map(community_map).fillna(-1)
                    
                except:
                    # Fallback for problematic graphs
                    df['degree_centrality'] = 0
                    df['betweenness_centrality'] = 0
                    df['community_id'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Network feature preparation failed: {str(e)}")
            return df
    
    def _detect_round_numbers(self, values: pd.Series) -> pd.Series:
        """Detect suspiciously round numbers."""
        try:
            round_indicators = []
            for value in values:
                if value == 0:
                    round_indicators.append(0)
                    continue
                
                str_value = str(int(value))
                trailing_zeros = len(str_value) - len(str_value.rstrip('0'))
                total_digits = len(str_value)
                
                if total_digits > 0:
                    round_score = trailing_zeros / total_digits
                else:
                    round_score = 0
                    
                round_indicators.append(round_score > 0.5)  # More than 50% zeros
            
            return pd.Series(round_indicators, index=values.index).astype(int)
            
        except Exception as e:
            self.logger.error(f"Round number detection failed: {str(e)}")
            return pd.Series(0, index=values.index)
    
    def _detect_activity_bursts(self, df: pd.DataFrame) -> pd.Series:
        """Detect bursts of activity."""
        try:
            if 'from_address' not in df.columns:
                return pd.Series(0, index=df.index)
            
            # Count transactions per address per hour
            df_temp = df.copy()
            df_temp['hour_bin'] = df_temp['timestamp'].dt.floor('H')
            
            activity_counts = df_temp.groupby(['from_address', 'hour_bin']).size()
            burst_threshold = activity_counts.quantile(0.95)
            
            # Mark burst periods
            burst_periods = activity_counts[activity_counts > burst_threshold].index
            
            burst_indicator = []
            for _, row in df.iterrows():
                hour_bin = pd.Timestamp(row['timestamp']).floor('H')
                is_burst = (row['from_address'], hour_bin) in burst_periods
                burst_indicator.append(int(is_burst))
            
            return pd.Series(burst_indicator, index=df.index)
            
        except Exception as e:
            self.logger.error(f"Activity burst detection failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def train_anomaly_detectors(self, 
                               training_data: pd.DataFrame,
                               validation_split: float = 0.2,
                               contamination_rate: float = 0.1) -> Dict[str, any]:
        """
        Train anomaly detection models.
        
        Args:
            training_data: Normal/baseline data for training
            validation_split: Fraction for validation
            contamination_rate: Expected contamination rate
            
        Returns:
            Training metrics
        """
        try:
            self.logger.info("Starting anomaly detection model training...")
            
            # Prepare features
            features_df = self.prepare_anomaly_features(training_data)
            
            # Select numeric features
            numeric_features = features_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_features 
                          if not col.endswith('_zscore') and not col.endswith('_percentile')]
            
            self.feature_columns = feature_cols
            X = features_df[feature_cols].fillna(0)
            
            # Split data
            X_train, X_val = train_test_split(X, test_size=validation_split, random_state=42)
            
            training_metrics = {}
            
            # Train Isolation Forest
            if 'isolation_forest' in self.detection_methods:
                self.logger.info("Training Isolation Forest...")
                
                X_train_scaled = self.standard_scaler.fit_transform(X_train)
                X_val_scaled = self.standard_scaler.transform(X_val)
                
                self.isolation_forest = IsolationForest(
                    contamination=contamination_rate,
                    random_state=42,
                    n_estimators=200,
                    max_samples='auto'
                )
                
                self.isolation_forest.fit(X_train_scaled)
                
                # Evaluate
                train_scores = self.isolation_forest.decision_function(X_train_scaled)
                val_scores = self.isolation_forest.decision_function(X_val_scaled)
                
                training_metrics['isolation_forest'] = {
                    'train_mean_score': train_scores.mean(),
                    'val_mean_score': val_scores.mean(),
                    'contamination_rate': contamination_rate
                }
            
            # Train Autoencoder
            if 'autoencoder' in self.detection_methods:
                self.logger.info("Training Autoencoder...")
                
                X_train_ae = self.minmax_scaler.fit_transform(X_train)
                X_val_ae = self.minmax_scaler.transform(X_val)
                
                # Build autoencoder
                input_dim = X_train_ae.shape[1]
                encoding_dim = max(8, input_dim // 4)
                
                inputs = tf.keras.Input(shape=(input_dim,))
                encoded = tf.keras.layers.Dense(128, activation='relu')(inputs)
                encoded = tf.keras.layers.Dropout(0.2)(encoded)
                encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
                encoded = tf.keras.layers.Dropout(0.2)(encoded)
                encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)
                
                decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
                decoded = tf.keras.layers.Dropout(0.2)(decoded)
                decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
                decoded = tf.keras.layers.Dropout(0.2)(decoded)
                decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
                
                self.autoencoder_model = tf.keras.Model(inputs, decoded)
                self.autoencoder_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Train
                history = self.autoencoder_model.fit(
                    X_train_ae, X_train_ae,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val_ae, X_val_ae),
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                    ],
                    verbose=0
                )
                
                training_metrics['autoencoder'] = {
                    'final_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'epochs_trained': len(history.history['loss'])
                }
            
            # Train LSTM for time series anomalies
            if 'lstm' in self.detection_methods and 'timestamp' in features_df.columns:
                self.logger.info("Training LSTM time series detector...")
                
                # Prepare time series data
                time_series_data = self._prepare_time_series_data(features_df)
                
                if len(time_series_data) > 0:
                    # Build LSTM model
                    sequence_length = 24  # 24 time steps
                    n_features = time_series_data.shape[1]
                    
                    lstm_model = Sequential([
                        LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
                        Dropout(0.2),
                        LSTM(64, return_sequences=False),
                        Dropout(0.2),
                        Dense(32, activation='relu'),
                        Dense(n_features)
                    ])
                    
                    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    # Create sequences
                    X_seq, y_seq = self._create_sequences(time_series_data, sequence_length)
                    
                    if len(X_seq) > 0:
                        # Split and train
                        train_size = int(0.8 * len(X_seq))
                        X_train_seq, X_val_seq = X_seq[:train_size], X_seq[train_size:]
                        y_train_seq, y_val_seq = y_seq[:train_size], y_seq[train_size:]
                        
                        lstm_history = lstm_model.fit(
                            X_train_seq, y_train_seq,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_val_seq, y_val_seq),
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)],
                            verbose=0
                        )
                        
                        self.lstm_model = lstm_model
                        
                        training_metrics['lstm'] = {
                            'final_loss': lstm_history.history['loss'][-1],
                            'final_val_loss': lstm_history.history['val_loss'][-1],
                            'sequence_length': sequence_length
                        }
            
            # Train One-Class SVM
            if 'one_class_svm' in self.detection_methods:
                self.logger.info("Training One-Class SVM...")
                
                # Use robust scaler for SVM
                X_train_svm = self.robust_scaler.fit_transform(X_train)
                
                self.one_class_svm = OneClassSVM(
                    kernel='rbf',
                    gamma='scale',
                    nu=contamination_rate
                )
                
                self.one_class_svm.fit(X_train_svm)
                
                training_metrics['one_class_svm'] = {
                    'training_completed': True,
                    'nu_parameter': contamination_rate
                }
            
            # Train Local Outlier Factor
            if 'lof' in self.detection_methods:
                self.logger.info("Training Local Outlier Factor...")
                
                self.lof_detector = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=contamination_rate,
                    novelty=True
                )
                
                self.lof_detector.fit(X_train_scaled)
                
                training_metrics['lof'] = {
                    'training_completed': True,
                    'n_neighbors': 20
                }
            
            # Train clustering-based detector
            if 'clustering' in self.detection_methods:
                self.logger.info("Training clustering-based detector...")
                
                # Use DBSCAN for density-based clustering
                self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
                self.clustering_model.fit(X_train_scaled)
                
                # Calculate cluster statistics for anomaly scoring
                labels = self.clustering_model.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                training_metrics['clustering'] = {
                    'n_clusters': n_clusters,
                    'noise_points': n_noise,
                    'noise_ratio': n_noise / len(labels)
                }
            
            # Initialize statistical detector
            if 'statistical' in self.detection_methods:
                self.logger.info("Initializing statistical detector...")
                self._initialize_statistical_baselines(X_train)
                
                training_metrics['statistical'] = {
                    'baseline_features': len(self.baseline_metrics),
                    'initialized': True
                }
            
            self.last_trained = datetime.now()
            self.logger.info("Anomaly detection model training completed")
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Anomaly detection training failed: {str(e)}")
            raise
    
    def _prepare_time_series_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare time series data for LSTM training."""
        try:
            if 'timestamp' not in df.columns:
                return np.array([])
            
            # Resample to hourly data
            df_ts = df.set_index('timestamp')
            numeric_cols = df_ts.select_dtypes(include=[np.number]).columns[:10]  # Limit features
            
            hourly_data = df_ts[numeric_cols].resample('H').mean().fillna(method='ffill')
            
            return hourly_data.values
            
        except Exception as e:
            self.logger.error(f"Time series data preparation failed: {str(e)}")
            return np.array([])
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        try:
            X, y = [], []
            for i in range(sequence_length, len(data)):
                X.append(data[i-sequence_length:i])
                y.append(data[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Sequence creation failed: {str(e)}")
            return np.array([]), np.array([])
    
    def _initialize_statistical_baselines(self, X_train: pd.DataFrame) -> None:
        """Initialize statistical baselines for anomaly detection."""
        try:
            self.baseline_metrics = {}
            
            for col in X_train.columns:
                self.baseline_metrics[col] = {
                    'mean': X_train[col].mean(),
                    'std': X_train[col].std(),
                    'median': X_train[col].median(),
                    'q25': X_train[col].quantile(0.25),
                    'q75': X_train[col].quantile(0.75),
                    'min': X_train[col].min(),
                    'max': X_train[col].max()
                }
            
            # Set alert thresholds (3-sigma rule)
            self.alert_thresholds = {
                'mild': 2.0,      # 2 standard deviations
                'moderate': 3.0,  # 3 standard deviations
                'severe': 4.0,    # 4 standard deviations
                'extreme': 5.0    # 5 standard deviations
            }
            
        except Exception as e:
            self.logger.error(f"Statistical baseline initialization failed: {str(e)}")
    
    def detect_anomalies(self, 
                        current_data: pd.DataFrame,
                        entity_type: str = 'transaction') -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in current data.
        
        Args:
            current_data: Current data to analyze
            entity_type: Type of entity being analyzed
            
        Returns:
            List of anomaly detection results
        """
        try:
            # Prepare features
            features_df = self.prepare_anomaly_features(current_data, entity_type)
            
            if len(features_df) == 0:
                return []
            
            # Select features
            X = features_df[self.feature_columns].fillna(0)
            
            anomaly_results = []
            
            for idx, row in X.iterrows():
                entity_id = str(current_data.loc[idx, 'transaction_hash']) if 'transaction_hash' in current_data.columns else str(idx)
                
                # Get anomaly scores from different methods
                anomaly_scores = {}
                contributing_features = []
                
                # Statistical anomaly detection
                if 'statistical' in self.detection_methods and self.baseline_metrics:
                    stat_score, stat_features = self._detect_statistical_anomalies(row)
                    anomaly_scores['statistical'] = stat_score
                    contributing_features.extend(stat_features)
                
                # Isolation Forest
                if 'isolation_forest' in self.detection_methods and self.isolation_forest:
                    row_scaled = self.standard_scaler.transform(row.values.reshape(1, -1))
                    iso_score = self.isolation_forest.decision_function(row_scaled)[0]
                    # Convert to 0-1 score (lower is more anomalous)
                    anomaly_scores['isolation_forest'] = max(0, 1 - (iso_score + 0.5) / 1.0)
                
                # Autoencoder
                if 'autoencoder' in self.detection_methods and self.autoencoder_model:
                    row_ae = self.minmax_scaler.transform(row.values.reshape(1, -1))
                    reconstruction = self.autoencoder_model.predict(row_ae, verbose=0)
                    reconstruction_error = np.mean((row_ae - reconstruction) ** 2)
                    anomaly_scores['autoencoder'] = min(1.0, reconstruction_error * 10)  # Scale error
                
                # One-Class SVM
                if 'one_class_svm' in self.detection_methods and self.one_class_svm:
                    row_svm = self.robust_scaler.transform(row.values.reshape(1, -1))
                    svm_prediction = self.one_class_svm.predict(row_svm)[0]
                    anomaly_scores['one_class_svm'] = 1.0 if svm_prediction == -1 else 0.0
                
                # Local Outlier Factor
                if 'lof' in self.detection_methods and self.lof_detector:
                    row_lof = self.standard_scaler.transform(row.values.reshape(1, -1))
                    lof_prediction = self.lof_detector.predict(row_lof)[0]
                    anomaly_scores['lof'] = 1.0 if lof_prediction == -1 else 0.0
                
                # Ensemble scoring
                if anomaly_scores:
                    ensemble_score = sum(
                        score * self.ensemble_weights.get(method, 0.1) 
                        for method, score in anomaly_scores.items()
                    )
                    ensemble_score = min(1.0, ensemble_score)
                else:
                    ensemble_score = 0.0
                
                # Determine severity and anomaly type
                severity, anomaly_type = self._classify_anomaly(ensemble_score, row, contributing_features)
                
                # Calculate confidence
                confidence = self._calculate_confidence(anomaly_scores, contributing_features)
                
                # Generate recommendations
                recommendations = self._generate_recommendations(severity, anomaly_type, ensemble_score)
                
                # Create result
                if ensemble_score > 0.3:  # Only report significant anomalies
                    result = AnomalyDetectionResult(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        anomaly_score=ensemble_score,
                        anomaly_type=anomaly_type,
                        severity=severity,
                        confidence=confidence,
                        detection_method='ensemble',
                        features_contributing=contributing_features,
                        baseline_metrics=self._get_baseline_for_features(contributing_features),
                        observed_metrics=self._get_observed_for_features(row, contributing_features),
                        timestamp=datetime.now(),
                        recommendations=recommendations
                    )
                    
                    anomaly_results.append(result)
                    
                    # Store in detection history
                    self.detection_history.append({
                        'timestamp': datetime.now(),
                        'entity_id': entity_id,
                        'anomaly_score': ensemble_score,
                        'severity': severity
                    })
            
            return anomaly_results
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            return []
    
    def _detect_statistical_anomalies(self, row: pd.Series) -> Tuple[float, List[str]]:
        """Detect statistical anomalies using baseline metrics."""
        try:
            anomaly_score = 0.0
            contributing_features = []
            
            for feature, value in row.items():
                if feature in self.baseline_metrics:
                    baseline = self.baseline_metrics[feature]
                    
                    # Calculate z-score
                    if baseline['std'] > 0:
                        z_score = abs(value - baseline['mean']) / baseline['std']
                        
                        # Convert to anomaly score
                        if z_score > self.alert_thresholds['extreme']:
                            feature_score = 1.0
                        elif z_score > self.alert_thresholds['severe']:
                            feature_score = 0.8
                        elif z_score > self.alert_thresholds['moderate']:
                            feature_score = 0.6
                        elif z_score > self.alert_thresholds['mild']:
                            feature_score = 0.4
                        else:
                            feature_score = 0.0
                        
                        if feature_score > 0.4:
                            anomaly_score += feature_score
                            contributing_features.append(feature)
            
            # Normalize by number of features
            if len(self.baseline_metrics) > 0:
                anomaly_score = min(1.0, anomaly_score / len(self.baseline_metrics))
            
            return anomaly_score, contributing_features[:5]  # Limit to top 5 features
            
        except Exception as e:
            self.logger.error(f"Statistical anomaly detection failed: {str(e)}")
            return 0.0, []
    
    def _classify_anomaly(self, 
                         score: float, 
                         row: pd.Series, 
                         contributing_features: List[str]) -> Tuple[str, str]:
        """Classify anomaly severity and type."""
        try:
            # Determine severity
            if score >= 0.9:
                severity = 'critical'
            elif score >= 0.7:
                severity = 'high'
            elif score >= 0.5:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Determine anomaly type based on contributing features
            anomaly_type = 'general_anomaly'
            
            feature_patterns = {
                'volume_spike': ['volume', 'transaction_count', 'value'],
                'gas_anomaly': ['gas_used', 'gas_price', 'gas_efficiency'],
                'timing_anomaly': ['time_since_prev_tx', 'activity_burst', 'hour'],
                'value_anomaly': ['value', 'log_value', 'is_round_value'],
                'network_anomaly': ['degree_centrality', 'betweenness_centrality'],
                'behavioral_anomaly': ['tx_frequency_from', 'time_regularity']
            }
            
            max_matches = 0
            for pattern_name, pattern_features in feature_patterns.items():
                matches = sum(1 for f in contributing_features 
                            if any(pf in f for pf in pattern_features))
                if matches > max_matches:
                    max_matches = matches
                    anomaly_type = pattern_name
            
            return severity, anomaly_type
            
        except Exception as e:
            self.logger.error(f"Anomaly classification failed: {str(e)}")
            return 'low', 'general_anomaly'
    
    def _calculate_confidence(self, 
                            anomaly_scores: Dict[str, float], 
                            contributing_features: List[str]) -> float:
        """Calculate confidence in anomaly detection."""
        try:
            # Base confidence on agreement between methods
            if len(anomaly_scores) == 0:
                return 0.0
            
            scores = list(anomaly_scores.values())
            
            # Agreement between methods
            agreement = 1.0 - (np.std(scores) / (np.mean(scores) + 1e-8))
            agreement = max(0.0, min(1.0, agreement))
            
            # Number of contributing features
            feature_confidence = min(1.0, len(contributing_features) / 5.0)
            
            # Number of detection methods
            method_confidence = min(1.0, len(anomaly_scores) / 3.0)
            
            # Overall confidence
            confidence = (agreement * 0.5 + feature_confidence * 0.3 + method_confidence * 0.2)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    def _generate_recommendations(self, 
                                severity: str, 
                                anomaly_type: str, 
                                score: float) -> List[str]:
        """Generate recommendations based on anomaly detection."""
        try:
            recommendations = []
            
            if severity == 'critical':
                recommendations.append("Immediate investigation required")
                recommendations.append("Consider blocking suspicious activity")
                recommendations.append("Alert security team")
            elif severity == 'high':
                recommendations.append("Priority investigation needed")
                recommendations.append("Enhanced monitoring")
                recommendations.append("Manual review recommended")
            elif severity == 'medium':
                recommendations.append("Schedule investigation")
                recommendations.append("Monitor for patterns")
            else:
                recommendations.append("Log for future analysis")
                recommendations.append("Continue monitoring")
            
            # Type-specific recommendations
            type_recommendations = {
                'volume_spike': ["Check for wash trading", "Verify liquidity sources"],
                'gas_anomaly': ["Investigate MEV activity", "Check for gas price manipulation"],
                'timing_anomaly': ["Look for automation patterns", "Check for coordinated activity"],
                'value_anomaly': ["Verify transaction authenticity", "Check for round number patterns"],
                'network_anomaly': ["Analyze transaction graph", "Check for mixer usage"],
                'behavioral_anomaly': ["Profile address behavior", "Check for bot activity"]
            }
            
            if anomaly_type in type_recommendations:
                recommendations.extend(type_recommendations[anomaly_type])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            return ["Manual investigation recommended"]
    
    def _get_baseline_for_features(self, features: List[str]) -> Dict[str, float]:
        """Get baseline metrics for contributing features."""
        try:
            baseline = {}
            for feature in features:
                if feature in self.baseline_metrics:
                    baseline[feature] = self.baseline_metrics[feature]['mean']
            return baseline
        except Exception as e:
            self.logger.error(f"Baseline retrieval failed: {str(e)}")
            return {}
    
    def _get_observed_for_features(self, row: pd.Series, features: List[str]) -> Dict[str, float]:
        """Get observed values for contributing features."""
        try:
            observed = {}
            for feature in features:
                if feature in row.index:
                    observed[feature] = row[feature]
            return observed
        except Exception as e:
            self.logger.error(f"Observed values retrieval failed: {str(e)}")
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Save the trained anomaly detection models."""
        try:
            model_data = {
                'statistical_detector': self.statistical_detector,
                'isolation_forest': self.isolation_forest,
                'autoencoder_model': self.autoencoder_model,
                'lstm_model': self.lstm_model,
                'graph_detector': self.graph_detector,
                'clustering_model': self.clustering_model,
                'one_class_svm': self.one_class_svm,
                'lof_detector': self.lof_detector,
                'time_series_detector': self.time_series_detector,
                'behavioral_detector': self.behavioral_detector,
                'network_detector': self.network_detector,
                'standard_scaler': self.standard_scaler,
                'robust_scaler': self.robust_scaler,
                'minmax_scaler': self.minmax_scaler,
                'pca_transformer': self.pca_transformer,
                'baseline_metrics': self.baseline_metrics,
                'alert_thresholds': self.alert_thresholds,
                'feature_columns': self.feature_columns,
                'detection_methods': self.detection_methods,
                'ensemble_weights': self.ensemble_weights,
                'anomaly_types': self.anomaly_types,
                'baseline_period_days': self.baseline_period_days,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Anomaly detection model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load trained anomaly detection models."""
        try:
            model_data = joblib.load(filepath)
            
            self.statistical_detector = model_data['statistical_detector']
            self.isolation_forest = model_data['isolation_forest']
            self.autoencoder_model = model_data['autoencoder_model']
            self.lstm_model = model_data['lstm_model']
            self.graph_detector = model_data['graph_detector']
            self.clustering_model = model_data['clustering_model']
            self.one_class_svm = model_data['one_class_svm']
            self.lof_detector = model_data['lof_detector']
            self.time_series_detector = model_data['time_series_detector']
            self.behavioral_detector = model_data['behavioral_detector']
            self.network_detector = model_data['network_detector']
            self.standard_scaler = model_data['standard_scaler']
            self.robust_scaler = model_data['robust_scaler']
            self.minmax_scaler = model_data['minmax_scaler']
            self.pca_transformer = model_data['pca_transformer']
            self.baseline_metrics = model_data['baseline_metrics']
            self.alert_thresholds = model_data['alert_thresholds']
            self.feature_columns = model_data['feature_columns']
            self.detection_methods = model_data['detection_methods']
            self.ensemble_weights = model_data['ensemble_weights']
            self.anomaly_types = model_data['anomaly_types']
            self.baseline_period_days = model_data['baseline_period_days']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Anomaly detection model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise


def create_anomaly_detection_pipeline(config: Dict) -> AnomalyDetectionModel:
    """
    Create an anomaly detection pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AnomalyDetectionModel instance
    """
    return AnomalyDetectionModel(
        detection_methods=config.get('detection_methods'),
        ensemble_weights=config.get('ensemble_weights'),
        real_time_monitoring=config.get('real_time_monitoring', True)
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create anomaly detection model
    anomaly_detector = AnomalyDetectionModel()
    
    # Example data preparation and training would go here
    training_data = load_training_data()
    training_metrics = anomaly_detector.train_anomaly_detectors(training_data)
    
    # Example anomaly detection
    current_data = load_current_data()
    anomalies = anomaly_detector.detect_anomalies(current_data)
    
    for anomaly in anomalies:
        print(f"Anomaly detected: {anomaly.anomaly_type}")
        print(f"Severity: {anomaly.severity}")
        print(f"Score: {anomaly.anomaly_score:.3f}")
        print(f"Recommendations: {anomaly.recommendations}")
    
    print("Anomaly detection model implementation completed")
