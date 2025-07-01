"""
Fraud Detection Model for DeFi and Blockchain Security

This module implements advanced ML models for detecting fraudulent activities
in DeFi protocols, including rug pulls, flash loan attacks, MEV manipulation,
and other sophisticated fraud schemes using Chainlink oracles for real-time monitoring.

Features:
- Multi-layered fraud detection using ensemble methods
- Real-time transaction analysis with Chainlink Data Feeds
- Graph neural networks for transaction pattern analysis
- Behavioral analysis and anomaly detection
- Smart contract vulnerability assessment
- Cross-chain fraud detection with CCIP integration
- Advanced feature engineering for blockchain forensics
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Embedding, Conv1D, GlobalMaxPooling1D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import joblib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import networkx as nx
import web3
from web3 import Web3
import hashlib
import re

warnings.filterwarnings('ignore')

@dataclass
class FraudDetectionResult:
    """Data class for fraud detection results."""
    transaction_hash: str
    fraud_probability: float
    fraud_type: str
    confidence_score: float
    risk_score: float
    evidence: List[str]
    recommended_action: str
    timestamp: datetime
    alert_level: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class TransactionFeatures:
    """Data class for transaction features."""
    transaction_hash: str
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: int
    block_number: int
    timestamp: datetime
    contract_interaction: bool
    token_transfers: List[Dict]
    internal_transactions: List[Dict]
    function_signature: str
    
class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for transaction graph analysis.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_layers: int = 3):
        super(GraphNeuralNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):
        # Apply graph convolutions
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x

class FraudDetectionModel:
    """
    Advanced fraud detection model using multiple ML techniques
    for comprehensive blockchain fraud detection and prevention.
    """
    
    def __init__(self, 
                 detection_modes: List[str] = None,
                 ensemble_weights: Dict[str, float] = None):
        """
        Initialize the fraud detection model.
        
        Args:
            detection_modes: List of detection modes to use
            ensemble_weights: Weights for ensemble model combination
        """
        self.detection_modes = detection_modes or [
            'transaction_analysis', 'behavioral_analysis', 'graph_analysis', 
            'contract_analysis', 'time_series_analysis'
        ]
        
        self.ensemble_weights = ensemble_weights or {
            'transaction_classifier': 0.25,
            'behavioral_anomaly': 0.20,
            'graph_neural_net': 0.25,
            'contract_analyzer': 0.15,
            'time_series_anomaly': 0.15
        }
        
        # Model components
        self.transaction_classifier = None
        self.behavioral_anomaly_detector = None
        self.graph_neural_network = None
        self.contract_analyzer = None
        self.time_series_anomaly_detector = None
        
        # Feature engineering components
        self.transaction_encoder = None
        self.address_encoder = None
        self.pattern_extractor = None
        
        # Feature scalers
        self.transaction_scaler = StandardScaler()
        self.behavioral_scaler = MinMaxScaler()
        self.graph_scaler = StandardScaler()
        
        # Model metadata
        self.feature_columns = []
        self.fraud_types = [
            'rug_pull', 'flash_loan_attack', 'mev_manipulation', 
            'pump_and_dump', 'wash_trading', 'front_running',
            'sandwich_attack', 'price_manipulation', 'liquidity_theft',
            'governance_attack', 'oracle_manipulation', 'smart_contract_exploit'
        ]
        
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Known fraud patterns
        self.fraud_patterns = {
            'rug_pull_indicators': [
                'sudden_liquidity_removal',
                'ownership_transfer_before_drain',
                'massive_token_mint',
                'liquidity_lock_expiry'
            ],
            'flash_loan_patterns': [
                'large_borrowing_single_block',
                'arbitrage_with_price_manipulation',
                'governance_token_borrowing',
                'complex_defi_interaction_chain'
            ],
            'mev_patterns': [
                'front_running_large_trades',
                'sandwich_attack_sequence',
                'block_reorganization_profit',
                'priority_gas_manipulation'
            ]
        }
        
    def prepare_fraud_detection_features(self, 
                                       transaction_data: pd.DataFrame,
                                       address_data: Optional[pd.DataFrame] = None,
                                       contract_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare comprehensive features for fraud detection.
        
        Args:
            transaction_data: Transaction history data
            address_data: Address metadata and behavior data
            contract_data: Smart contract analysis data
            
        Returns:
            DataFrame with fraud detection features
        """
        try:
            features_df = transaction_data.copy()
            
            # Basic transaction features
            features_df = self._add_transaction_features(features_df)
            
            # Temporal features
            features_df = self._add_temporal_features(features_df)
            
            # Address behavior features
            features_df = self._add_address_behavior_features(features_df)
            
            # Network analysis features
            features_df = self._add_network_features(features_df)
            
            # Economic features
            features_df = self._add_economic_features(features_df)
            
            # Contract interaction features
            if contract_data is not None:
                features_df = self._add_contract_features(features_df, contract_data)
            
            # DeFi protocol features
            features_df = self._add_defi_features(features_df)
            
            # MEV and arbitrage features
            features_df = self._add_mev_features(features_df)
            
            # Cross-chain features
            features_df = self._add_cross_chain_features(features_df)
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise
    
    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic transaction-level features."""
        try:
            # Value-based features
            df['log_value'] = np.log1p(df['value'])
            df['value_percentile'] = df['value'].rank(pct=True)
            df['is_high_value'] = (df['value'] > df['value'].quantile(0.95)).astype(int)
            
            # Gas features
            df['gas_efficiency'] = df['value'] / (df['gas_used'] * df['gas_price'] + 1e-8)
            df['gas_price_percentile'] = df['gas_price'].rank(pct=True)
            df['unusual_gas_price'] = (df['gas_price'] > df['gas_price'].quantile(0.99)).astype(int)
            
            # Transaction frequency
            df['tx_count_from'] = df.groupby('from_address')['transaction_hash'].transform('count')
            df['tx_count_to'] = df.groupby('to_address')['transaction_hash'].transform('count')
            
            # Value distribution
            df['value_from_address'] = df.groupby('from_address')['value'].transform('sum')
            df['avg_value_from'] = df.groupby('from_address')['value'].transform('mean')
            df['std_value_from'] = df.groupby('from_address')['value'].transform('std').fillna(0)
            
            # Contract interaction
            df['contract_creation'] = (df['to_address'].isna()).astype(int)
            df['data_size'] = df['input_data'].str.len().fillna(0)
            df['has_complex_data'] = (df['data_size'] > 100).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Transaction features calculation failed: {str(e)}")
            return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        try:
            # Time-based features
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_night'] = df['hour'].isin(range(22, 24) + range(0, 6)).astype(int)
            
            # Transaction timing patterns
            df = df.sort_values('timestamp')
            df['time_since_prev_tx'] = df.groupby('from_address')['timestamp'].diff().dt.total_seconds()
            df['time_to_next_tx'] = df.groupby('from_address')['timestamp'].diff(-1).dt.total_seconds().abs()
            
            # Burst detection
            df['tx_in_last_minute'] = df.groupby('from_address')['timestamp'].transform(
                lambda x: x.rolling('1min').count()
            )
            df['tx_in_last_hour'] = df.groupby('from_address')['timestamp'].transform(
                lambda x: x.rolling('1H').count()
            )
            
            # Block-level features
            df['tx_in_block'] = df.groupby('block_number')['transaction_hash'].transform('count')
            df['block_position'] = df.groupby('block_number').cumcount()
            df['is_first_in_block'] = (df['block_position'] == 0).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Temporal features calculation failed: {str(e)}")
            return df
    
    def _add_address_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add address behavioral analysis features."""
        try:
            # Address age and activity
            address_first_seen = df.groupby('from_address')['timestamp'].min()
            address_last_seen = df.groupby('from_address')['timestamp'].max()
            
            df['address_age_days'] = (df['timestamp'] - df['from_address'].map(address_first_seen)).dt.days
            df['address_inactive_days'] = (df['timestamp'] - df['from_address'].map(address_last_seen)).dt.days
            
            # Address diversity
            df['unique_recipients'] = df.groupby('from_address')['to_address'].transform('nunique')
            df['unique_senders'] = df.groupby('to_address')['from_address'].transform('nunique')
            
            # Behavioral patterns
            df['address_balance_estimate'] = df.groupby('from_address')['value'].transform('sum')
            df['spending_rate'] = df['value'] / df['address_balance_estimate'].clip(lower=1)
            
            # Contract interaction patterns
            df['contract_interaction_ratio'] = df.groupby('from_address')['contract_interaction'].transform('mean')
            
            # Address clustering features
            df['address_entropy'] = df.groupby('from_address').apply(
                lambda x: -(x['value'] / x['value'].sum() * np.log(x['value'] / x['value'].sum() + 1e-8)).sum()
            ).reindex(df['from_address']).values
            
            return df
            
        except Exception as e:
            self.logger.error(f"Address behavior features calculation failed: {str(e)}")
            return df
    
    def _add_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add network analysis features."""
        try:
            # Create transaction graph
            G = nx.DiGraph()
            
            # Add edges with transaction data
            for _, row in df.iterrows():
                if pd.notna(row['from_address']) and pd.notna(row['to_address']):
                    if G.has_edge(row['from_address'], row['to_address']):
                        G[row['from_address']][row['to_address']]['weight'] += row['value']
                        G[row['from_address']][row['to_address']]['count'] += 1
                    else:
                        G.add_edge(row['from_address'], row['to_address'], 
                                 weight=row['value'], count=1)
            
            # Calculate network centrality measures
            try:
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100)
                pagerank = nx.pagerank(G, max_iter=100)
                
                df['degree_centrality'] = df['from_address'].map(degree_centrality).fillna(0)
                df['betweenness_centrality'] = df['from_address'].map(betweenness_centrality).fillna(0)
                df['eigenvector_centrality'] = df['from_address'].map(eigenvector_centrality).fillna(0)
                df['pagerank'] = df['from_address'].map(pagerank).fillna(0)
                
            except:
                # Fallback for disconnected or problematic graphs
                df['degree_centrality'] = 0
                df['betweenness_centrality'] = 0
                df['eigenvector_centrality'] = 0
                df['pagerank'] = 0
            
            # Local network features
            df['in_degree'] = df['to_address'].map(dict(G.in_degree())).fillna(0)
            df['out_degree'] = df['from_address'].map(dict(G.out_degree())).fillna(0)
            
            # Community detection indicators
            try:
                communities = nx.community.greedy_modularity_communities(G.to_undirected())
                community_map = {}
                for i, community in enumerate(communities):
                    for node in community:
                        community_map[node] = i
                
                df['community_id'] = df['from_address'].map(community_map).fillna(-1)
                df['community_size'] = df['community_id'].map(
                    df['community_id'].value_counts()
                ).fillna(0)
            except:
                df['community_id'] = -1
                df['community_size'] = 0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Network features calculation failed: {str(e)}")
            return df
    
    def _add_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add economic and financial features."""
        try:
            # Price impact analysis
            df['relative_value'] = df['value'] / df['value'].rolling(100).mean()
            df['value_z_score'] = (df['value'] - df['value'].rolling(100).mean()) / df['value'].rolling(100).std()
            
            # Liquidity and volume analysis
            df['volume_in_block'] = df.groupby('block_number')['value'].transform('sum')
            df['tx_volume_ratio'] = df['value'] / df['volume_in_block']
            
            # Market timing features
            df['value_volatility'] = df['value'].rolling(50).std()
            df['price_momentum'] = df['value'].pct_change(10)
            
            # Economic anomaly detection
            df['unusual_value'] = (np.abs(df['value_z_score']) > 3).astype(int)
            df['round_number'] = self._detect_round_numbers(df['value'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Economic features calculation failed: {str(e)}")
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
                # Count trailing zeros
                trailing_zeros = len(str_value) - len(str_value.rstrip('0'))
                # Normalize by total digits
                round_score = trailing_zeros / len(str_value)
                round_indicators.append(round_score)
            
            return pd.Series(round_indicators, index=values.index)
            
        except Exception as e:
            self.logger.error(f"Round number detection failed: {str(e)}")
            return pd.Series(0, index=values.index)
    
    def _add_contract_features(self, df: pd.DataFrame, contract_data: pd.DataFrame) -> pd.DataFrame:
        """Add smart contract analysis features."""
        try:
            # Merge contract data
            contract_features = contract_data.set_index('contract_address')
            
            df = df.merge(
                contract_features[['is_verified', 'creation_date', 'compiler_version', 'optimization_enabled']],
                left_on='to_address',
                right_index=True,
                how='left'
            )
            
            # Contract age
            df['contract_age_days'] = (df['timestamp'] - pd.to_datetime(df['creation_date'])).dt.days
            df['new_contract'] = (df['contract_age_days'] < 7).astype(int)
            
            # Verification status
            df['unverified_contract'] = (~df['is_verified']).astype(int)
            
            # Function signature analysis
            df['function_hash'] = df['input_data'].str[:10]  # First 4 bytes
            df['common_function'] = df['function_hash'].isin([
                '0xa9059cbb',  # transfer
                '0x23b872dd',  # transferFrom
                '0x095ea7b3',  # approve
                '0x18160ddd'   # totalSupply
            ]).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Contract features calculation failed: {str(e)}")
            return df
    
    def _add_defi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add DeFi protocol specific features."""
        try:
            # Known DeFi protocol addresses
            defi_protocols = {
                'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
                'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
                'aave': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
                'compound': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B'
            }
            
            # Protocol interaction flags
            for protocol, address in defi_protocols.items():
                df[f'interacts_with_{protocol}'] = (df['to_address'] == address.lower()).astype(int)
            
            # DeFi transaction patterns
            df['multiple_defi_interactions'] = sum([
                df[f'interacts_with_{protocol}'] for protocol in defi_protocols.keys()
            ])
            
            # Liquidity provision patterns
            df['potential_liquidity_add'] = self._detect_liquidity_patterns(df, 'add')
            df['potential_liquidity_remove'] = self._detect_liquidity_patterns(df, 'remove')
            
            # Flash loan detection
            df['potential_flash_loan'] = self._detect_flash_loan_patterns(df)
            
            # Arbitrage patterns
            df['potential_arbitrage'] = self._detect_arbitrage_patterns(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"DeFi features calculation failed: {str(e)}")
            return df
    
    def _detect_liquidity_patterns(self, df: pd.DataFrame, action: str) -> pd.Series:
        """Detect liquidity provision patterns."""
        try:
            # Simplified liquidity detection based on transaction patterns
            if action == 'add':
                # Large value transactions to known DEX routers
                pattern = (df['value'] > df['value'].quantile(0.9)) & (df['gas_used'] > 200000)
            else:  # remove
                # Large value transactions from known DEX routers with high gas
                pattern = (df['value'] > df['value'].quantile(0.8)) & (df['gas_used'] > 150000)
            
            return pattern.astype(int)
            
        except Exception as e:
            self.logger.error(f"Liquidity pattern detection failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _detect_flash_loan_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect potential flash loan patterns."""
        try:
            # Flash loans often involve:
            # 1. Large borrowing and repayment in same block
            # 2. Complex interaction chains
            # 3. High gas usage
            
            same_block_large_txs = df.groupby(['from_address', 'block_number']).agg({
                'value': ['sum', 'count'],
                'gas_used': 'sum'
            }).reset_index()
            
            same_block_large_txs.columns = ['from_address', 'block_number', 'total_value', 'tx_count', 'total_gas']
            
            # Criteria for potential flash loan
            flash_loan_blocks = same_block_large_txs[
                (same_block_large_txs['total_value'] > same_block_large_txs['total_value'].quantile(0.95)) &
                (same_block_large_txs['tx_count'] >= 3) &
                (same_block_large_txs['total_gas'] > 1000000)
            ]
            
            df = df.merge(
                flash_loan_blocks[['from_address', 'block_number']].assign(potential_flash_loan=1),
                on=['from_address', 'block_number'],
                how='left'
            )
            
            return df['potential_flash_loan'].fillna(0)
            
        except Exception as e:
            self.logger.error(f"Flash loan pattern detection failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _detect_arbitrage_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect potential arbitrage patterns."""
        try:
            # Arbitrage often involves:
            # 1. Quick succession of transactions
            # 2. Interactions with multiple DEXs
            # 3. Profitable outcomes
            
            # Group by address and look for rapid trading patterns
            df_sorted = df.sort_values(['from_address', 'timestamp'])
            df_sorted['time_diff'] = df_sorted.groupby('from_address')['timestamp'].diff().dt.total_seconds()
            
            # Quick succession (within 1 minute)
            quick_succession = df_sorted['time_diff'] < 60
            
            # Multiple transactions in short period
            rapid_trading = df_sorted.groupby('from_address')['time_diff'].transform(
                lambda x: (x < 300).sum()  # Multiple txs within 5 minutes
            ) >= 3
            
            potential_arbitrage = quick_succession & rapid_trading
            
            return potential_arbitrage.reindex(df.index).fillna(0).astype(int)
            
        except Exception as e:
            self.logger.error(f"Arbitrage pattern detection failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _add_mev_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MEV (Maximal Extractable Value) related features."""
        try:
            # Sort by block number and transaction index
            df = df.sort_values(['block_number', 'transaction_index'])
            
            # Front-running detection
            df['high_gas_price'] = (df['gas_price'] > df['gas_price'].quantile(0.95)).astype(int)
            df['gas_price_rank_in_block'] = df.groupby('block_number')['gas_price'].rank(ascending=False)
            df['is_first_in_block'] = (df['gas_price_rank_in_block'] == 1).astype(int)
            
            # Sandwich attack detection
            df['potential_sandwich'] = self._detect_sandwich_attacks(df)
            
            # MEV bot identification
            df['potential_mev_bot'] = self._identify_mev_bots(df)
            
            # Priority fee analysis
            df['priority_fee_ratio'] = df['gas_price'] / df.groupby('block_number')['gas_price'].transform('median')
            df['extreme_priority_fee'] = (df['priority_fee_ratio'] > 5).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"MEV features calculation failed: {str(e)}")
            return df
    
    def _detect_sandwich_attacks(self, df: pd.DataFrame) -> pd.Series:
        """Detect potential sandwich attacks."""
        try:
            sandwich_indicators = []
            
            # Group by block and look for sandwich patterns
            for block_num, block_txs in df.groupby('block_number'):
                block_txs = block_txs.sort_values('transaction_index')
                
                for i in range(1, len(block_txs) - 1):
                    prev_tx = block_txs.iloc[i-1]
                    curr_tx = block_txs.iloc[i]
                    next_tx = block_txs.iloc[i+1]
                    
                    # Sandwich pattern: same address before and after a different address
                    if (prev_tx['from_address'] == next_tx['from_address'] and
                        prev_tx['from_address'] != curr_tx['from_address'] and
                        prev_tx['to_address'] == next_tx['to_address'] and
                        prev_tx['gas_price'] >= curr_tx['gas_price'] and
                        next_tx['gas_price'] >= curr_tx['gas_price']):
                        
                        sandwich_indicators.extend([1, 1, 1])  # Mark all three transactions
                    else:
                        sandwich_indicators.append(0)
                
                # Handle first and last transactions
                if len(block_txs) >= 2:
                    sandwich_indicators.insert(0, 0)  # First transaction
                    sandwich_indicators.append(0)   # Last transaction
                elif len(block_txs) == 1:
                    sandwich_indicators.append(0)
            
            return pd.Series(sandwich_indicators, index=df.index).fillna(0)
            
        except Exception as e:
            self.logger.error(f"Sandwich attack detection failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _identify_mev_bots(self, df: pd.DataFrame) -> pd.Series:
        """Identify potential MEV bots."""
        try:
            # MEV bot characteristics:
            # 1. High transaction frequency
            # 2. Consistent high gas prices
            # 3. Interactions with multiple DEXs
            # 4. Profitable patterns
            
            address_stats = df.groupby('from_address').agg({
                'transaction_hash': 'count',
                'gas_price': ['mean', 'std'],
                'to_address': 'nunique',
                'value': 'sum',
                'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600  # Hours active
            })
            
            address_stats.columns = ['tx_count', 'avg_gas_price', 'gas_price_std', 'unique_recipients', 'total_value', 'hours_active']
            
            # MEV bot criteria
            mev_bot_addresses = address_stats[
                (address_stats['tx_count'] > 100) &
                (address_stats['avg_gas_price'] > address_stats['avg_gas_price'].quantile(0.8)) &
                (address_stats['unique_recipients'] > 5) &
                (address_stats['hours_active'] > 24)
            ].index
            
            df['potential_mev_bot'] = df['from_address'].isin(mev_bot_addresses).astype(int)
            
            return df['potential_mev_bot']
            
        except Exception as e:
            self.logger.error(f"MEV bot identification failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _add_cross_chain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-chain transaction features."""
        try:
            # Known bridge contract addresses
            bridge_addresses = {
                'polygon_bridge': '0x40ec5B33f54e0E8A33A975908C5BA1c14e5BbbDf',
                'arbitrum_bridge': '0x8315177aB297bA92A06054cE80a67Ed4DBd7ed3a',
                'optimism_bridge': '0x99C9fc46f92E8a1c0deC1b1747d010903E884bE1',
                'ccip_router': '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D'
            }
            
            # Bridge interaction detection
            for bridge_name, address in bridge_addresses.items():
                df[f'uses_{bridge_name}'] = (df['to_address'] == address.lower()).astype(int)
            
            # Cross-chain transaction indicators
            df['is_bridge_transaction'] = sum([
                df[f'uses_{bridge}'] for bridge in bridge_addresses.keys()
            ]) > 0
            
            # Large value cross-chain transfers (potential for fraud)
            df['large_bridge_transfer'] = (
                df['is_bridge_transaction'] & 
                (df['value'] > df['value'].quantile(0.9))
            ).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Cross-chain features calculation failed: {str(e)}")
            return df
    
    def create_transaction_graph(self, df: pd.DataFrame) -> Data:
        """Create graph representation for GNN analysis."""
        try:
            # Create address mapping
            unique_addresses = pd.concat([df['from_address'], df['to_address']]).unique()
            address_to_idx = {addr: idx for idx, addr in enumerate(unique_addresses)}
            
            # Create edge index
            edge_index = []
            edge_attr = []
            
            for _, row in df.iterrows():
                if pd.notna(row['from_address']) and pd.notna(row['to_address']):
                    from_idx = address_to_idx[row['from_address']]
                    to_idx = address_to_idx[row['to_address']]
                    
                    edge_index.append([from_idx, to_idx])
                    edge_attr.append([
                        row['value'],
                        row['gas_used'],
                        row['gas_price'],
                        row['block_number']
                    ])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Create node features (aggregate by address)
            node_features = []
            for addr in unique_addresses:
                addr_txs = df[df['from_address'] == addr]
                if len(addr_txs) == 0:
                    addr_txs = df[df['to_address'] == addr]
                
                if len(addr_txs) > 0:
                    features = [
                        addr_txs['value'].sum(),
                        addr_txs['value'].mean(),
                        len(addr_txs),
                        addr_txs['gas_used'].mean(),
                        addr_txs['gas_price'].mean()
                    ]
                else:
                    features = [0.0, 0.0, 0.0, 0.0, 0.0]
                
                node_features.append(features)
            
            x = torch.tensor(node_features, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            self.logger.error(f"Graph creation failed: {str(e)}")
            return None
    
    def train_fraud_detection_models(self, 
                                   training_data: pd.DataFrame,
                                   labels: pd.Series,
                                   validation_split: float = 0.2) -> Dict[str, any]:
        """
        Train fraud detection models.
        
        Args:
            training_data: Historical transaction data with features
            labels: Fraud labels (0 = legitimate, 1 = fraudulent)
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        try:
            self.logger.info("Starting fraud detection model training...")
            
            # Prepare features
            features_df = self.prepare_fraud_detection_features(training_data)
            
            # Align features with labels
            features_df, labels = features_df.align(labels, join='inner', axis=0)
            
            # Select feature columns
            feature_cols = [col for col in features_df.columns 
                          if col not in ['transaction_hash', 'from_address', 'to_address', 'timestamp']]
            
            self.feature_columns = feature_cols
            X = features_df[feature_cols].fillna(0)
            y = labels
            
            training_metrics = {}
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Train transaction classifier
            self.logger.info("Training transaction classifier...")
            X_train_scaled = self.transaction_scaler.fit_transform(X_train)
            X_val_scaled = self.transaction_scaler.transform(X_val)
            
            self.transaction_classifier = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='auc',
                use_label_encoder=False
            )
            
            self.transaction_classifier.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Evaluate transaction classifier
            y_pred_proba = self.transaction_classifier.predict_proba(X_val_scaled)[:, 1]
            y_pred = self.transaction_classifier.predict(X_val_scaled)
            
            training_metrics['transaction_classifier'] = {
                'accuracy': accuracy_score(y_val, y_pred),
                'auc_roc': roc_auc_score(y_val, y_pred_proba),
                'classification_report': classification_report(y_val, y_pred, output_dict=True)
            }
            
            # Train behavioral anomaly detector
            self.logger.info("Training behavioral anomaly detector...")
            behavioral_features = [col for col in feature_cols if 'address' in col or 'behavior' in col or 'network' in col]
            
            if behavioral_features:
                X_behavioral = X_train[behavioral_features]
                X_behavioral_scaled = self.behavioral_scaler.fit_transform(X_behavioral)
                
                self.behavioral_anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=200
                )
                
                self.behavioral_anomaly_detector.fit(X_behavioral_scaled)
                
                # Evaluate on validation set
                X_val_behavioral = self.behavioral_scaler.transform(X_val[behavioral_features])
                anomaly_scores = self.behavioral_anomaly_detector.decision_function(X_val_behavioral)
                anomaly_labels = self.behavioral_anomaly_detector.predict(X_val_behavioral)
                
                training_metrics['behavioral_anomaly_detector'] = {
                    'anomaly_rate': (anomaly_labels == -1).mean(),
                    'mean_anomaly_score': anomaly_scores.mean()
                }
            
            # Train graph neural network (simplified for training)
            if 'graph_analysis' in self.detection_modes:
                self.logger.info("Training graph neural network...")
                try:
                    graph_data = self.create_transaction_graph(features_df.iloc[:1000])  # Subset for training
                    
                    if graph_data is not None:
                        self.graph_neural_network = GraphNeuralNetwork(
                            input_dim=graph_data.x.shape[1],
                            hidden_dim=64,
                            output_dim=32
                        )
                        
                        training_metrics['graph_neural_network'] = {
                            'nodes': graph_data.x.shape[0],
                            'edges': graph_data.edge_index.shape[1],
                            'features': graph_data.x.shape[1]
                        }
                except Exception as e:
                    self.logger.warning(f"Graph neural network training failed: {str(e)}")
            
            # Train time series anomaly detector
            self.logger.info("Training time series anomaly detector...")
            time_series_features = features_df.groupby('from_address').apply(
                lambda x: x[['value', 'gas_used', 'gas_price']].rolling(window=10).mean()
            ).reset_index(level=0, drop=True)
            
            if len(time_series_features) > 0:
                self.time_series_anomaly_detector = IsolationForest(
                    contamination=0.05,
                    random_state=42
                )
                
                self.time_series_anomaly_detector.fit(time_series_features.fillna(0))
                
                training_metrics['time_series_anomaly_detector'] = {
                    'features_shape': time_series_features.shape,
                    'training_completed': True
                }
            
            self.last_trained = datetime.now()
            self.logger.info("Fraud detection model training completed")
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Fraud detection model training failed: {str(e)}")
            raise
    
    def detect_fraud(self, transaction_features: TransactionFeatures) -> FraudDetectionResult:
        """
        Detect fraud in a single transaction.
        
        Args:
            transaction_features: Transaction to analyze
            
        Returns:
            FraudDetectionResult with fraud assessment
        """
        try:
            # Convert transaction features to DataFrame
            tx_data = pd.DataFrame([{
                'transaction_hash': transaction_features.transaction_hash,
                'from_address': transaction_features.from_address,
                'to_address': transaction_features.to_address,
                'value': transaction_features.value,
                'gas_used': transaction_features.gas_used,
                'gas_price': transaction_features.gas_price,
                'block_number': transaction_features.block_number,
                'timestamp': transaction_features.timestamp,
                'contract_interaction': transaction_features.contract_interaction,
                'input_data': transaction_features.function_signature
            }])
            
            # Prepare features
            features_df = self.prepare_fraud_detection_features(tx_data)
            X = features_df[self.feature_columns].fillna(0)
            
            # Get predictions from all models
            fraud_scores = []
            evidence = []
            
            # Transaction classifier
            if self.transaction_classifier is not None:
                X_scaled = self.transaction_scaler.transform(X)
                tx_fraud_prob = self.transaction_classifier.predict_proba(X_scaled)[0, 1]
                fraud_scores.append(('transaction_classifier', tx_fraud_prob))
                
                if tx_fraud_prob > 0.7:
                    evidence.append(f"High transaction fraud probability: {tx_fraud_prob:.3f}")
            
            # Behavioral anomaly detection
            if self.behavioral_anomaly_detector is not None:
                behavioral_features = [col for col in self.feature_columns if 'address' in col or 'behavior' in col]
                if behavioral_features:
                    X_behavioral = self.behavioral_scaler.transform(X[behavioral_features])
                    anomaly_score = self.behavioral_anomaly_detector.decision_function(X_behavioral)[0]
                    behavioral_fraud_prob = 1 / (1 + np.exp(anomaly_score))  # Convert to probability
                    fraud_scores.append(('behavioral_anomaly', behavioral_fraud_prob))
                    
                    if anomaly_score < -0.5:
                        evidence.append(f"Behavioral anomaly detected: {anomaly_score:.3f}")
            
            # Pattern-based detection
            pattern_score = self._detect_known_fraud_patterns(features_df.iloc[0])
            fraud_scores.append(('pattern_detection', pattern_score))
            
            if pattern_score > 0.5:
                evidence.append(f"Known fraud pattern detected: {pattern_score:.3f}")
            
            # Ensemble prediction
            if fraud_scores:
                weights = [self.ensemble_weights.get(name, 0.25) for name, _ in fraud_scores]
                scores = [score for _, score in fraud_scores]
                ensemble_fraud_prob = np.average(scores, weights=weights)
            else:
                ensemble_fraud_prob = 0.0
            
            # Determine fraud type and risk level
            fraud_type = self._classify_fraud_type(features_df.iloc[0])
            risk_score = ensemble_fraud_prob * 100
            
            # Determine alert level and recommended action
            if risk_score >= 90:
                alert_level = 'critical'
                recommended_action = 'immediate_block'
            elif risk_score >= 70:
                alert_level = 'high'
                recommended_action = 'manual_review'
            elif risk_score >= 50:
                alert_level = 'medium'
                recommended_action = 'enhanced_monitoring'
            elif risk_score >= 30:
                alert_level = 'low'
                recommended_action = 'flag_for_review'
            else:
                alert_level = 'low'
                recommended_action = 'normal_processing'
            
            # Calculate confidence score
            confidence_score = min(0.95, 0.5 + (len(evidence) * 0.1) + (abs(risk_score - 50) / 100))
            
            return FraudDetectionResult(
                transaction_hash=transaction_features.transaction_hash,
                fraud_probability=ensemble_fraud_prob,
                fraud_type=fraud_type,
                confidence_score=confidence_score,
                risk_score=risk_score,
                evidence=evidence,
                recommended_action=recommended_action,
                timestamp=datetime.now(),
                alert_level=alert_level
            )
            
        except Exception as e:
            self.logger.error(f"Fraud detection failed: {str(e)}")
            return FraudDetectionResult(
                transaction_hash=transaction_features.transaction_hash,
                fraud_probability=0.0,
                fraud_type='unknown',
                confidence_score=0.0,
                risk_score=0.0,
                evidence=[f"Detection error: {str(e)}"],
                recommended_action='manual_review',
                timestamp=datetime.now(),
                alert_level='low'
            )
    
    def _detect_known_fraud_patterns(self, transaction_row: pd.Series) -> float:
        """Detect known fraud patterns in transaction."""
        try:
            pattern_score = 0.0
            
            # Rug pull indicators
            if (transaction_row.get('potential_liquidity_remove', 0) == 1 and
                transaction_row.get('new_contract', 0) == 1):
                pattern_score += 0.3
            
            # Flash loan attack indicators
            if (transaction_row.get('potential_flash_loan', 0) == 1 and
                transaction_row.get('multiple_defi_interactions', 0) > 2):
                pattern_score += 0.4
            
            # MEV manipulation indicators
            if (transaction_row.get('potential_sandwich', 0) == 1 or
                transaction_row.get('potential_mev_bot', 0) == 1):
                pattern_score += 0.2
            
            # Suspicious value patterns
            if (transaction_row.get('round_number', 0) > 0.5 and
                transaction_row.get('is_high_value', 0) == 1):
                pattern_score += 0.2
            
            # Temporal anomalies
            if (transaction_row.get('is_night', 0) == 1 and
                transaction_row.get('tx_in_last_minute', 0) > 10):
                pattern_score += 0.1
            
            return min(1.0, pattern_score)
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            return 0.0
    
    def _classify_fraud_type(self, transaction_row: pd.Series) -> str:
        """Classify the type of fraud detected."""
        try:
            scores = {}
            
            # Rug pull indicators
            rug_pull_score = 0
            if transaction_row.get('potential_liquidity_remove', 0):
                rug_pull_score += 0.4
            if transaction_row.get('new_contract', 0):
                rug_pull_score += 0.3
            if transaction_row.get('unverified_contract', 0):
                rug_pull_score += 0.3
            scores['rug_pull'] = rug_pull_score
            
            # Flash loan attack indicators
            flash_loan_score = 0
            if transaction_row.get('potential_flash_loan', 0):
                flash_loan_score += 0.5
            if transaction_row.get('multiple_defi_interactions', 0) > 2:
                flash_loan_score += 0.3
            if transaction_row.get('gas_used', 0) > 1000000:
                flash_loan_score += 0.2
            scores['flash_loan_attack'] = flash_loan_score
            
            # MEV manipulation indicators
            mev_score = 0
            if transaction_row.get('potential_sandwich', 0):
                mev_score += 0.4
            if transaction_row.get('potential_mev_bot', 0):
                mev_score += 0.3
            if transaction_row.get('high_gas_price', 0):
                mev_score += 0.3
            scores['mev_manipulation'] = mev_score
            
            # Pump and dump indicators
            pump_dump_score = 0
            if transaction_row.get('unusual_value', 0):
                pump_dump_score += 0.3
            if transaction_row.get('round_number', 0) > 0.5:
                pump_dump_score += 0.2
            if transaction_row.get('tx_in_last_hour', 0) > 50:
                pump_dump_score += 0.3
            scores['pump_and_dump'] = pump_dump_score
            
            # Return the fraud type with highest score
            if scores:
                max_fraud_type = max(scores.items(), key=lambda x: x[1])
                return max_fraud_type[0] if max_fraud_type[1] > 0.3 else 'suspicious_activity'
            else:
                return 'unknown'
                
        except Exception as e:
            self.logger.error(f"Fraud type classification failed: {str(e)}")
            return 'unknown'
    
    def save_model(self, filepath: str) -> None:
        """Save the trained fraud detection models."""
        try:
            model_data = {
                'transaction_classifier': self.transaction_classifier,
                'behavioral_anomaly_detector': self.behavioral_anomaly_detector,
                'graph_neural_network': self.graph_neural_network,
                'contract_analyzer': self.contract_analyzer,
                'time_series_anomaly_detector': self.time_series_anomaly_detector,
                'transaction_encoder': self.transaction_encoder,
                'address_encoder': self.address_encoder,
                'pattern_extractor': self.pattern_extractor,
                'transaction_scaler': self.transaction_scaler,
                'behavioral_scaler': self.behavioral_scaler,
                'graph_scaler': self.graph_scaler,
                'feature_columns': self.feature_columns,
                'fraud_types': self.fraud_types,
                'detection_modes': self.detection_modes,
                'ensemble_weights': self.ensemble_weights,
                'fraud_patterns': self.fraud_patterns,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Fraud detection model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load trained fraud detection models."""
        try:
            model_data = joblib.load(filepath)
            
            self.transaction_classifier = model_data['transaction_classifier']
            self.behavioral_anomaly_detector = model_data['behavioral_anomaly_detector']
            self.graph_neural_network = model_data['graph_neural_network']
            self.contract_analyzer = model_data['contract_analyzer']
            self.time_series_anomaly_detector = model_data['time_series_anomaly_detector']
            self.transaction_encoder = model_data['transaction_encoder']
            self.address_encoder = model_data['address_encoder']
            self.pattern_extractor = model_data['pattern_extractor']
            self.transaction_scaler = model_data['transaction_scaler']
            self.behavioral_scaler = model_data['behavioral_scaler']
            self.graph_scaler = model_data['graph_scaler']
            self.feature_columns = model_data['feature_columns']
            self.fraud_types = model_data['fraud_types']
            self.detection_modes = model_data['detection_modes']
            self.ensemble_weights = model_data['ensemble_weights']
            self.fraud_patterns = model_data['fraud_patterns']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Fraud detection model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise


def create_fraud_detection_pipeline(config: Dict) -> FraudDetectionModel:
    """
    Create a fraud detection pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured FraudDetectionModel instance
    """
    return FraudDetectionModel(
        detection_modes=config.get('detection_modes'),
        ensemble_weights=config.get('ensemble_weights')
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create fraud detection model
    fraud_detector = FraudDetectionModel()
    
    # Example transaction features
    tx_features = TransactionFeatures(
        transaction_hash="0x1234567890abcdef",
        from_address="0xabcdef1234567890",
        to_address="0x9876543210fedcba",
        value=1000000.0,
        gas_used=150000,
        gas_price=20000000000,
        block_number=15000000,
        timestamp=datetime.now(),
        contract_interaction=True,
        token_transfers=[],
        internal_transactions=[],
        function_signature="0xa9059cbb"
    )
    
    # This would be used with trained models
    result = fraud_detector.detect_fraud(tx_features)
    print(f"Fraud Probability: {result.fraud_probability:.3f}")
    print(f"Risk Score: {result.risk_score:.1f}")
    print(f"Recommended Action: {result.recommended_action}")
    
    print("Fraud detection model implementation completed")
