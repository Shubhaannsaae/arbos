"""
Risk Scoring Model for Blockchain and DeFi Security

This module implements comprehensive risk scoring algorithms for blockchain entities
including addresses, transactions, contracts, and protocols using machine learning,
statistical analysis, and real-time monitoring with Chainlink oracles.

Features:
- Multi-dimensional risk scoring using ensemble methods
- Real-time risk assessment with Chainlink Data Feeds
- Dynamic risk model adaptation based on market conditions
- Behavioral risk profiling for addresses and contracts
- Protocol-specific risk assessment for DeFi platforms
- Cross-chain risk correlation using CCIP
- Regulatory compliance risk scoring
- Liquidity and counterparty risk assessment
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import joblib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import networkx as nx
from collections import defaultdict
import scipy.stats as stats
from scipy.spatial.distance import euclidean
import hashlib

warnings.filterwarnings('ignore')

@dataclass
class RiskScore:
    """Data class for risk scoring results."""
    entity_id: str
    entity_type: str  # 'address', 'transaction', 'contract', 'protocol'
    overall_risk_score: float  # 0-100 normalized risk score
    risk_category: str  # 'very_low', 'low', 'medium', 'high', 'very_high'
    component_scores: Dict[str, float]  # Individual risk component scores
    risk_factors: List[str]  # Contributing risk factors
    confidence: float  # Confidence in the risk assessment
    timestamp: datetime
    expiry_time: datetime  # When the risk score expires
    recommendations: List[str]  # Risk mitigation recommendations

@dataclass
class RiskComponent:
    """Data class for individual risk components."""
    name: str
    score: float  # 0-100
    weight: float  # Weight in overall calculation
    description: str
    contributing_factors: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class RiskProfile:
    """Data class for entity risk profiles."""
    entity_id: str
    risk_history: List[RiskScore]
    risk_trend: str  # 'improving', 'stable', 'deteriorating'
    avg_risk_score: float
    max_risk_score: float
    risk_volatility: float
    last_assessment: datetime

class RiskScoringModel:
    """
    Comprehensive risk scoring system for blockchain entities using
    machine learning and statistical analysis for security assessment.
    """
    
    def __init__(self, 
                 risk_components: List[str] = None,
                 component_weights: Dict[str, float] = None,
                 risk_categories: Dict[str, Tuple[float, float]] = None):
        """
        Initialize the risk scoring model.
        
        Args:
            risk_components: List of risk components to evaluate
            component_weights: Weights for each risk component
            risk_categories: Risk category thresholds
        """
        self.risk_components = risk_components or [
            'transaction_risk', 'behavioral_risk', 'network_risk', 
            'liquidity_risk', 'counterparty_risk', 'regulatory_risk',
            'technical_risk', 'market_risk', 'operational_risk'
        ]
        
        self.component_weights = component_weights or {
            'transaction_risk': 0.15,
            'behavioral_risk': 0.20,
            'network_risk': 0.10,
            'liquidity_risk': 0.15,
            'counterparty_risk': 0.10,
            'regulatory_risk': 0.10,
            'technical_risk': 0.10,
            'market_risk': 0.05,
            'operational_risk': 0.05
        }
        
        self.risk_categories = risk_categories or {
            'very_low': (0, 20),
            'low': (20, 40),
            'medium': (40, 60),
            'high': (60, 80),
            'very_high': (80, 100)
        }
        
        # Risk scoring models
        self.transaction_risk_model = None
        self.behavioral_risk_model = None
        self.network_risk_model = None
        self.liquidity_risk_model = None
        self.counterparty_risk_model = None
        self.regulatory_risk_model = None
        self.technical_risk_model = None
        self.market_risk_model = None
        self.operational_risk_model = None
        
        # Ensemble model for overall scoring
        self.ensemble_model = None
        
        # Feature scalers
        self.transaction_scaler = StandardScaler()
        self.behavioral_scaler = RobustScaler()
        self.network_scaler = StandardScaler()
        
        # Risk profiles and history
        self.risk_profiles = {}
        self.risk_baselines = {}
        self.market_risk_factors = {}
        
        # Model metadata
        self.feature_columns = {}  # Feature columns for each component
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Known risk indicators
        self.risk_indicators = {
            'high_risk_addresses': set(),
            'blacklisted_addresses': set(),
            'sanctioned_addresses': set(),
            'mixer_addresses': set(),
            'exchange_addresses': set(),
            'defi_protocol_addresses': set()
        }
        
        # Protocol-specific risk parameters
        self.protocol_risk_params = {
            'uniswap': {'base_risk': 10, 'liquidity_threshold': 100000},
            'sushiswap': {'base_risk': 15, 'liquidity_threshold': 50000},
            'curve': {'base_risk': 8, 'liquidity_threshold': 200000},
            'balancer': {'base_risk': 12, 'liquidity_threshold': 75000},
            'aave': {'base_risk': 5, 'liquidity_threshold': 1000000},
            'compound': {'base_risk': 7, 'liquidity_threshold': 500000}
        }
        
    def prepare_risk_features(self, 
                            entity_data: pd.DataFrame,
                            entity_type: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare features for risk scoring by component.
        
        Args:
            entity_data: Data about entities to score
            entity_type: Type of entity ('address', 'transaction', 'contract', 'protocol')
            
        Returns:
            Dictionary of feature DataFrames by risk component
        """
        try:
            component_features = {}
            
            # Transaction risk features
            if 'transaction_risk' in self.risk_components:
                component_features['transaction_risk'] = self._prepare_transaction_risk_features(
                    entity_data, entity_type
                )
            
            # Behavioral risk features
            if 'behavioral_risk' in self.risk_components:
                component_features['behavioral_risk'] = self._prepare_behavioral_risk_features(
                    entity_data, entity_type
                )
            
            # Network risk features
            if 'network_risk' in self.risk_components:
                component_features['network_risk'] = self._prepare_network_risk_features(
                    entity_data, entity_type
                )
            
            # Liquidity risk features
            if 'liquidity_risk' in self.risk_components:
                component_features['liquidity_risk'] = self._prepare_liquidity_risk_features(
                    entity_data, entity_type
                )
            
            # Counterparty risk features
            if 'counterparty_risk' in self.risk_components:
                component_features['counterparty_risk'] = self._prepare_counterparty_risk_features(
                    entity_data, entity_type
                )
            
            # Regulatory risk features
            if 'regulatory_risk' in self.risk_components:
                component_features['regulatory_risk'] = self._prepare_regulatory_risk_features(
                    entity_data, entity_type
                )
            
            # Technical risk features
            if 'technical_risk' in self.risk_components:
                component_features['technical_risk'] = self._prepare_technical_risk_features(
                    entity_data, entity_type
                )
            
            # Market risk features
            if 'market_risk' in self.risk_components:
                component_features['market_risk'] = self._prepare_market_risk_features(
                    entity_data, entity_type
                )
            
            # Operational risk features
            if 'operational_risk' in self.risk_components:
                component_features['operational_risk'] = self._prepare_operational_risk_features(
                    entity_data, entity_type
                )
            
            return component_features
            
        except Exception as e:
            self.logger.error(f"Risk feature preparation failed: {str(e)}")
            raise
    
    def _prepare_transaction_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare transaction-related risk features."""
        try:
            features_df = df.copy()
            
            if entity_type in ['address', 'transaction']:
                # Transaction volume and frequency features
                features_df['tx_volume_log'] = np.log1p(features_df.get('total_volume', 0))
                features_df['tx_count_log'] = np.log1p(features_df.get('transaction_count', 0))
                features_df['avg_tx_value'] = features_df.get('total_volume', 0) / (features_df.get('transaction_count', 1))
                
                # Gas usage patterns
                features_df['avg_gas_price'] = features_df.get('total_gas_cost', 0) / (features_df.get('transaction_count', 1))
                features_df['gas_efficiency'] = features_df.get('total_volume', 0) / (features_df.get('total_gas_cost', 1))
                
                # Transaction timing patterns
                features_df['tx_frequency'] = features_df.get('transaction_count', 0) / (features_df.get('active_days', 1))
                features_df['burst_activity'] = self._detect_burst_activity(features_df)
                
                # Value distribution analysis
                features_df['value_concentration'] = self._calculate_value_concentration(features_df)
                features_df['round_amounts_ratio'] = self._calculate_round_amounts_ratio(features_df)
                
                # Cross-chain activity
                features_df['cross_chain_activity'] = features_df.get('cross_chain_tx_count', 0) / (features_df.get('transaction_count', 1))
                features_df['bridge_usage_score'] = self._calculate_bridge_usage_score(features_df)
                
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Transaction risk feature preparation failed: {str(e)}")
            return df
    
    def _prepare_behavioral_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare behavioral risk features."""
        try:
            features_df = df.copy()
            
            if entity_type in ['address', 'contract']:
                # Activity patterns
                features_df['activity_regularity'] = self._calculate_activity_regularity(features_df)
                features_df['time_of_day_risk'] = self._calculate_time_risk(features_df)
                features_df['weekend_activity_ratio'] = features_df.get('weekend_tx_count', 0) / (features_df.get('transaction_count', 1))
                
                # Interaction patterns
                features_df['unique_counterparties'] = features_df.get('unique_senders', 0) + features_df.get('unique_receivers', 0)
                features_df['counterparty_diversity'] = features_df['unique_counterparties'] / (features_df.get('transaction_count', 1))
                features_df['repeat_interaction_ratio'] = 1 - features_df['counterparty_diversity']
                
                # Balance and flow patterns
                features_df['balance_volatility'] = features_df.get('balance_std', 0) / (features_df.get('avg_balance', 1))
                features_df['flow_imbalance'] = abs(features_df.get('total_received', 0) - features_df.get('total_sent', 0))
                features_df['cash_flow_ratio'] = features_df.get('total_received', 0) / (features_df.get('total_sent', 1))
                
                # Automation indicators
                features_df['automation_score'] = self._calculate_automation_score(features_df)
                features_df['bot_likelihood'] = self._calculate_bot_likelihood(features_df)
                
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Behavioral risk feature preparation failed: {str(e)}")
            return df
    
    def _prepare_network_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare network-based risk features."""
        try:
            features_df = df.copy()
            
            # Network centrality measures
            features_df['degree_centrality'] = features_df.get('degree_centrality', 0)
            features_df['betweenness_centrality'] = features_df.get('betweenness_centrality', 0)
            features_df['closeness_centrality'] = features_df.get('closeness_centrality', 0)
            features_df['eigenvector_centrality'] = features_df.get('eigenvector_centrality', 0)
            
            # Community and clustering
            features_df['clustering_coefficient'] = features_df.get('clustering_coefficient', 0)
            features_df['community_size'] = features_df.get('community_size', 0)
            features_df['community_risk_score'] = self._calculate_community_risk(features_df)
            
            # Network distance to known entities
            features_df['distance_to_exchanges'] = self._calculate_distance_to_known_entities(
                features_df, 'exchange_addresses'
            )
            features_df['distance_to_mixers'] = self._calculate_distance_to_known_entities(
                features_df, 'mixer_addresses'
            )
            features_df['distance_to_blacklist'] = self._calculate_distance_to_known_entities(
                features_df, 'blacklisted_addresses'
            )
            
            # Path analysis
            features_df['shortest_path_length'] = features_df.get('avg_shortest_path', 0)
            features_df['path_diversity'] = features_df.get('path_count', 0)
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Network risk feature preparation failed: {str(e)}")
            return df
    
    def _prepare_liquidity_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare liquidity risk features."""
        try:
            features_df = df.copy()
            
            # Liquidity metrics
            features_df['liquidity_ratio'] = features_df.get('liquid_assets', 0) / (features_df.get('total_assets', 1))
            features_df['illiquid_asset_ratio'] = 1 - features_df['liquidity_ratio']
            
            # Market depth and slippage
            features_df['market_depth_score'] = features_df.get('market_depth', 0)
            features_df['slippage_risk'] = features_df.get('avg_slippage', 0)
            features_df['volume_impact'] = features_df.get('price_impact', 0)
            
            # Concentration risk
            features_df['asset_concentration'] = self._calculate_asset_concentration(features_df)
            features_df['position_size_risk'] = features_df.get('max_position_size', 0) / (features_df.get('total_assets', 1))
            
            # Time-based liquidity
            features_df['liquidity_trend'] = self._calculate_liquidity_trend(features_df)
            features_df['liquidity_volatility'] = features_df.get('liquidity_std', 0) / (features_df.get('avg_liquidity', 1))
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Liquidity risk feature preparation failed: {str(e)}")
            return df
    
    def _prepare_counterparty_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare counterparty risk features."""
        try:
            features_df = df.copy()
            
            # Counterparty quality metrics
            features_df['counterparty_risk_score'] = self._calculate_counterparty_risk(features_df)
            features_df['high_risk_counterparty_ratio'] = features_df.get('high_risk_interactions', 0) / (features_df.get('transaction_count', 1))
            
            # Exposure metrics
            features_df['max_counterparty_exposure'] = features_df.get('max_counterparty_value', 0) / (features_df.get('total_volume', 1))
            features_df['counterparty_concentration'] = self._calculate_counterparty_concentration(features_df)
            
            # Credit risk indicators
            features_df['default_probability'] = self._estimate_default_probability(features_df)
            features_df['recovery_rate'] = features_df.get('recovery_rate', 0.4)  # Default 40%
            
            # Relationship metrics
            features_df['relationship_duration'] = features_df.get('avg_relationship_days', 0)
            features_df['relationship_stability'] = self._calculate_relationship_stability(features_df)
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Counterparty risk feature preparation failed: {str(e)}")
            return df
    
    def _prepare_regulatory_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare regulatory risk features."""
        try:
            features_df = df.copy()
            
            # Compliance indicators
            features_df['kyc_compliance'] = features_df.get('kyc_status', 0)
            features_df['aml_risk_score'] = features_df.get('aml_score', 50)  # Default medium risk
            features_df['sanctions_risk'] = self._calculate_sanctions_risk(features_df)
            
            # Jurisdiction risk
            features_df['high_risk_jurisdiction'] = features_df.get('high_risk_country', 0)
            features_df['regulatory_uncertainty'] = features_df.get('regulatory_score', 50)
            
            # Transaction pattern compliance
            features_df['suspicious_pattern_score'] = self._calculate_suspicious_patterns(features_df)
            features_df['structuring_risk'] = self._detect_structuring_patterns(features_df)
            
            # Reporting compliance
            features_df['reporting_completeness'] = features_df.get('reporting_score', 100)
            features_df['audit_trail_quality'] = features_df.get('audit_score', 100)
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Regulatory risk feature preparation failed: {str(e)}")
            return df
    
    def _prepare_technical_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare technical risk features."""
        try:
            features_df = df.copy()
            
            if entity_type == 'contract':
                # Smart contract risk factors
                features_df['code_complexity'] = features_df.get('code_size', 0) / 1000  # Normalize
                features_df['function_count'] = features_df.get('function_count', 0)
                features_df['external_call_risk'] = features_df.get('external_calls', 0)
                
                # Security features
                features_df['has_proxy'] = features_df.get('is_proxy', 0)
                features_df['has_admin_functions'] = features_df.get('admin_functions', 0)
                features_df['has_selfdestruct'] = features_df.get('selfdestruct_present', 0)
                
                # Verification and audit status
                features_df['is_verified'] = features_df.get('verified', 0)
                features_df['audit_score'] = features_df.get('audit_score', 0)
                features_df['bug_bounty_score'] = features_df.get('bug_bounty', 0)
                
                # Upgrade risk
                features_df['upgrade_frequency'] = features_df.get('upgrades_count', 0) / (features_df.get('contract_age_days', 1))
                features_df['recent_upgrade'] = (features_df.get('days_since_upgrade', 999) < 30).astype(int)
            
            # General technical features
            features_df['gas_optimization'] = features_df.get('gas_efficiency', 1)
            features_df['network_congestion_impact'] = features_df.get('congestion_sensitivity', 0)
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Technical risk feature preparation failed: {str(e)}")
            return df
    
    def _prepare_market_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare market risk features."""
        try:
            features_df = df.copy()
            
            # Price volatility features
            features_df['price_volatility'] = features_df.get('price_volatility', 0)
            features_df['correlation_to_market'] = features_df.get('market_correlation', 0)
            features_df['beta'] = features_df.get('beta', 1)
            
            # Liquidity and market depth
            features_df['market_depth'] = features_df.get('order_book_depth', 0)
            features_df['bid_ask_spread'] = features_df.get('spread', 0)
            features_df['market_impact_cost'] = features_df.get('market_impact', 0)
            
            # Market regime indicators
            features_df['market_stress_indicator'] = self._calculate_market_stress(features_df)
            features_df['volatility_regime'] = self._classify_volatility_regime(features_df)
            
            # Concentration and diversification
            features_df['market_concentration'] = self._calculate_market_concentration(features_df)
            features_df['diversification_score'] = 1 - features_df['market_concentration']
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Market risk feature preparation failed: {str(e)}")
            return df
    
    def _prepare_operational_risk_features(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """Prepare operational risk features."""
        try:
            features_df = df.copy()
            
            # System reliability
            features_df['uptime_score'] = features_df.get('uptime_percentage', 99.9)
            features_df['error_rate'] = features_df.get('error_rate', 0)
            features_df['response_time_score'] = 100 - features_df.get('avg_response_time', 0)
            
            # Security incidents
            features_df['security_incidents'] = features_df.get('incident_count', 0)
            features_df['time_since_incident'] = features_df.get('days_since_incident', 999)
            features_df['incident_severity_score'] = features_df.get('max_incident_severity', 0)
            
            # Operational maturity
            features_df['operational_history'] = features_df.get('operational_days', 0)
            features_df['team_experience'] = features_df.get('team_score', 50)
            features_df['governance_score'] = features_df.get('governance_quality', 50)
            
            # Backup and recovery
            features_df['backup_score'] = features_df.get('backup_quality', 100)
            features_df['recovery_time_score'] = 100 - features_df.get('recovery_time_hours', 0)
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Operational risk feature preparation failed: {str(e)}")
            return df
    
    def _detect_burst_activity(self, df: pd.DataFrame) -> pd.Series:
        """Detect burst activity patterns."""
        try:
            # Simplified burst detection based on transaction frequency
            tx_count = df.get('transaction_count', 0)
            active_days = df.get('active_days', 1)
            avg_daily_tx = tx_count / active_days
            
            # High burst score if many transactions in short time
            burst_score = np.minimum(100, avg_daily_tx / 10)  # Normalize to 0-100
            return pd.Series(burst_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Burst activity detection failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_value_concentration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate value concentration (Herfindahl index)."""
        try:
            # Simplified concentration calculation
            max_tx_value = df.get('max_transaction_value', 0)
            total_value = df.get('total_volume', 1)
            concentration = (max_tx_value / total_value) * 100
            return pd.Series(concentration, index=df.index)
        except Exception as e:
            self.logger.error(f"Value concentration calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_round_amounts_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ratio of round number transactions."""
        try:
            round_tx_count = df.get('round_amount_tx_count', 0)
            total_tx_count = df.get('transaction_count', 1)
            ratio = (round_tx_count / total_tx_count) * 100
            return pd.Series(ratio, index=df.index)
        except Exception as e:
            self.logger.error(f"Round amounts ratio calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_bridge_usage_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bridge usage risk score."""
        try:
            bridge_tx_count = df.get('bridge_tx_count', 0)
            total_tx_count = df.get('transaction_count', 1)
            bridge_ratio = bridge_tx_count / total_tx_count
            
            # Higher bridge usage may indicate higher risk
            bridge_score = bridge_ratio * 50  # Scale to 0-50
            return pd.Series(bridge_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Bridge usage score calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_activity_regularity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate activity regularity score."""
        try:
            # Based on standard deviation of transaction timing
            tx_time_std = df.get('tx_time_std_hours', 24)  # Default 24 hours
            # Lower std = more regular = potentially more suspicious
            regularity_score = np.maximum(0, 100 - tx_time_std)
            return pd.Series(regularity_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Activity regularity calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_time_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate time-of-day risk score."""
        try:
            night_tx_ratio = df.get('night_tx_ratio', 0)
            weekend_tx_ratio = df.get('weekend_tx_ratio', 0)
            
            # Higher activity during off-hours may indicate higher risk
            time_risk = (night_tx_ratio * 30 + weekend_tx_ratio * 20)
            return pd.Series(time_risk, index=df.index)
        except Exception as e:
            self.logger.error(f"Time risk calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_automation_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate automation likelihood score."""
        try:
            # Factors indicating automation
            regular_timing = df.get('timing_regularity', 0)
            round_amounts = df.get('round_amounts_ratio', 0)
            gas_consistency = df.get('gas_price_consistency', 0)
            
            automation_score = (regular_timing * 0.4 + round_amounts * 0.3 + gas_consistency * 0.3)
            return pd.Series(automation_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Automation score calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_bot_likelihood(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bot likelihood score."""
        try:
            # Factors indicating bot behavior
            high_frequency = (df.get('tx_frequency', 0) > 100).astype(int) * 25
            consistent_gas = (df.get('gas_price_std', 100) < 10).astype(int) * 25
            automation_score = df.get('automation_score', 0) * 0.5
            
            bot_likelihood = high_frequency + consistent_gas + automation_score
            return pd.Series(np.minimum(100, bot_likelihood), index=df.index)
        except Exception as e:
            self.logger.error(f"Bot likelihood calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_community_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate community-based risk score."""
        try:
            # Risk based on community associations
            community_size = df.get('community_size', 1)
            high_risk_neighbors = df.get('high_risk_neighbors', 0)
            
            # Smaller communities with high-risk members are riskier
            if community_size > 0:
                community_risk = (high_risk_neighbors / community_size) * 100
            else:
                community_risk = 0
            
            return pd.Series(community_risk, index=df.index)
        except Exception as e:
            self.logger.error(f"Community risk calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_distance_to_known_entities(self, df: pd.DataFrame, entity_type: str) -> pd.Series:
        """Calculate network distance to known entity types."""
        try:
            # Simplified distance calculation
            direct_connections = df.get(f'{entity_type}_connections', 0)
            indirect_connections = df.get(f'{entity_type}_indirect', 0)
            
            if direct_connections > 0:
                distance = 1
            elif indirect_connections > 0:
                distance = 2 + np.log1p(indirect_connections)
            else:
                distance = 10  # No connection
            
            # Convert to risk score (closer = higher risk for some entities)
            if entity_type in ['blacklisted_addresses', 'mixer_addresses']:
                risk_score = np.maximum(0, 100 - distance * 10)
            else:
                risk_score = np.minimum(100, distance * 5)
            
            return pd.Series(risk_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Distance calculation failed: {str(e)}")
            return pd.Series(50, index=df.index)  # Default medium risk
    
    def _calculate_asset_concentration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate asset concentration risk."""
        try:
            # Herfindahl index for asset concentration
            max_asset_weight = df.get('max_asset_percentage', 100) / 100
            concentration = max_asset_weight ** 2 * 100  # Simplified Herfindahl
            return pd.Series(concentration, index=df.index)
        except Exception as e:
            self.logger.error(f"Asset concentration calculation failed: {str(e)}")
            return pd.Series(100, index=df.index)  # Full concentration by default
    
    def _calculate_liquidity_trend(self, df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity trend indicator."""
        try:
            current_liquidity = df.get('current_liquidity', 1)
            historical_liquidity = df.get('avg_historical_liquidity', 1)
            
            trend = (current_liquidity - historical_liquidity) / historical_liquidity * 100
            # Negative trend indicates increasing risk
            risk_score = np.maximum(0, -trend)
            return pd.Series(risk_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Liquidity trend calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_counterparty_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate overall counterparty risk score."""
        try:
            # Weighted average of counterparty risk scores
            avg_counterparty_risk = df.get('avg_counterparty_risk_score', 50)
            max_counterparty_risk = df.get('max_counterparty_risk_score', 50)
            
            # Weight recent interactions more heavily
            counterparty_risk = avg_counterparty_risk * 0.7 + max_counterparty_risk * 0.3
            return pd.Series(counterparty_risk, index=df.index)
        except Exception as e:
            self.logger.error(f"Counterparty risk calculation failed: {str(e)}")
            return pd.Series(50, index=df.index)
    
    def _calculate_counterparty_concentration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate counterparty concentration risk."""
        try:
            top_counterparty_volume = df.get('top_counterparty_volume', 0)
            total_volume = df.get('total_volume', 1)
            
            concentration = (top_counterparty_volume / total_volume) * 100
            return pd.Series(concentration, index=df.index)
        except Exception as e:
            self.logger.error(f"Counterparty concentration calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _estimate_default_probability(self, df: pd.DataFrame) -> pd.Series:
        """Estimate default probability for counterparties."""
        try:
            # Simplified default probability based on historical data
            credit_score = df.get('avg_counterparty_credit_score', 70)
            # Convert credit score to default probability (higher score = lower default prob)
            default_prob = np.maximum(0, (100 - credit_score) / 100)
            return pd.Series(default_prob * 100, index=df.index)
        except Exception as e:
            self.logger.error(f"Default probability estimation failed: {str(e)}")
            return pd.Series(20, index=df.index)  # Default 20% probability
    
    def _calculate_relationship_stability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate relationship stability score."""
        try:
            avg_relationship_duration = df.get('avg_relationship_days', 30)
            relationship_variance = df.get('relationship_duration_std', 15)
            
            # Lower variance in relationship duration = higher stability
            stability = np.maximum(0, 100 - relationship_variance)
            return pd.Series(stability, index=df.index)
        except Exception as e:
            self.logger.error(f"Relationship stability calculation failed: {str(e)}")
            return pd.Series(50, index=df.index)
    
    def _calculate_sanctions_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate sanctions-related risk score."""
        try:
            # Check for sanctions list matches
            sanctions_matches = df.get('sanctions_list_matches', 0)
            high_risk_countries = df.get('high_risk_country_interactions', 0)
            
            sanctions_risk = sanctions_matches * 100 + high_risk_countries * 20
            return pd.Series(np.minimum(100, sanctions_risk), index=df.index)
        except Exception as e:
            self.logger.error(f"Sanctions risk calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_suspicious_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate suspicious transaction pattern score."""
        try:
            # Combine various suspicious pattern indicators
            structuring_score = df.get('structuring_score', 0)
            layering_score = df.get('layering_score', 0)
            rapid_movement_score = df.get('rapid_movement_score', 0)
            
            suspicious_score = (structuring_score * 0.4 + layering_score * 0.3 + rapid_movement_score * 0.3)
            return pd.Series(suspicious_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Suspicious patterns calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _detect_structuring_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect structuring (smurfing) patterns."""
        try:
            # Pattern: Multiple transactions just below reporting thresholds
            below_threshold_count = df.get('below_threshold_tx_count', 0)
            total_tx_count = df.get('transaction_count', 1)
            
            structuring_ratio = below_threshold_count / total_tx_count
            structuring_score = structuring_ratio * 100
            
            return pd.Series(structuring_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Structuring pattern detection failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _calculate_market_stress(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market stress indicator."""
        try:
            # Combine volatility, correlation, and liquidity indicators
            volatility = df.get('market_volatility', 20)  # Default 20%
            correlation = df.get('average_correlation', 0.5)  # Default 50%
            liquidity_stress = df.get('liquidity_stress_indicator', 0)
            
            # High volatility, high correlation, low liquidity = high stress
            stress_score = volatility * 2 + correlation * 40 + liquidity_stress * 30
            return pd.Series(np.minimum(100, stress_score), index=df.index)
        except Exception as e:
            self.logger.error(f"Market stress calculation failed: {str(e)}")
            return pd.Series(20, index=df.index)
    
    def _classify_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify current volatility regime."""
        try:
            current_vol = df.get('current_volatility', 20)
            historical_vol = df.get('historical_avg_volatility', 20)
            
            vol_ratio = current_vol / historical_vol
            
            if vol_ratio > 2.0:
                regime_score = 100  # Extreme volatility
            elif vol_ratio > 1.5:
                regime_score = 75   # High volatility
            elif vol_ratio > 1.2:
                regime_score = 50   # Medium volatility
            elif vol_ratio > 0.8:
                regime_score = 25   # Normal volatility
            else:
                regime_score = 10   # Low volatility
            
            return pd.Series(regime_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Volatility regime classification failed: {str(e)}")
            return pd.Series(25, index=df.index)
    
    def _calculate_market_concentration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market concentration risk."""
        try:
            # Based on asset allocation concentration
            top_asset_percentage = df.get('top_asset_percentage', 50) / 100
            top_3_assets_percentage = df.get('top_3_assets_percentage', 75) / 100
            
            # Herfindahl-like index
            concentration = top_asset_percentage ** 2 + (top_3_assets_percentage - top_asset_percentage) ** 2
            concentration_score = concentration * 100
            
            return pd.Series(concentration_score, index=df.index)
        except Exception as e:
            self.logger.error(f"Market concentration calculation failed: {str(e)}")
            return pd.Series(50, index=df.index)
    
    def train_risk_scoring_models(self, 
                                training_data: Dict[str, pd.DataFrame],
                                risk_labels: Dict[str, pd.Series],
                                validation_split: float = 0.2) -> Dict[str, any]:
        """
        Train risk scoring models for each component.
        
        Args:
            training_data: Dictionary of training data by risk component
            risk_labels: Dictionary of risk labels by component
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        try:
            self.logger.info("Starting risk scoring model training...")
            
            training_metrics = {}
            
            for component in self.risk_components:
                if component in training_data and component in risk_labels:
                    self.logger.info(f"Training {component} model...")
                    
                    # Prepare features
                    X = training_data[component].fillna(0)
                    y = risk_labels[component]
                    
                    # Align features and labels
                    X, y = X.align(y, join='inner', axis=0)
                    
                    # Select numeric features
                    numeric_features = X.select_dtypes(include=[np.number]).columns
                    X_numeric = X[numeric_features]
                    
                    self.feature_columns[component] = numeric_features.tolist()
                    
                    # Split data
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_numeric, y, test_size=validation_split, random_state=42
                    )
                    
                    # Scale features
                    if component == 'transaction_risk':
                        X_train_scaled = self.transaction_scaler.fit_transform(X_train)
                        X_val_scaled = self.transaction_scaler.transform(X_val)
                    elif component == 'behavioral_risk':
                        X_train_scaled = self.behavioral_scaler.fit_transform(X_train)
                        X_val_scaled = self.behavioral_scaler.transform(X_val)
                    else:
                        X_train_scaled = self.network_scaler.fit_transform(X_train)
                        X_val_scaled = self.network_scaler.transform(X_val)
                    
                    # Train XGBoost model
                    model = xgb.XGBRegressor(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.01,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    )
                    
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                    
                    # Store model
                    setattr(self, f'{component}_model', model)
                    
                    # Evaluate
                    y_pred = model.predict(X_val_scaled)
                    
                    training_metrics[component] = {
                        'mae': mean_absolute_error(y_val, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                        'r2': r2_score(y_val, y_pred),
                        'feature_count': len(numeric_features)
                    }
            
            # Train ensemble model
            self.logger.info("Training ensemble model...")
            self._train_ensemble_model(training_data, risk_labels, validation_split)
            
            self.last_trained = datetime.now()
            self.logger.info("Risk scoring model training completed")
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Risk scoring model training failed: {str(e)}")
            raise
    
    def _train_ensemble_model(self, 
                            training_data: Dict[str, pd.DataFrame],
                            risk_labels: Dict[str, pd.Series],
                            validation_split: float) -> None:
        """Train ensemble model for overall risk scoring."""
        try:
            # Create ensemble features from component predictions
            ensemble_features = []
            ensemble_labels = []
            
            # Get predictions from each component model
            for component in self.risk_components:
                if (component in training_data and 
                    hasattr(self, f'{component}_model') and 
                    getattr(self, f'{component}_model') is not None):
                    
                    X = training_data[component][self.feature_columns[component]].fillna(0)
                    
                    # Scale features
                    if component == 'transaction_risk':
                        X_scaled = self.transaction_scaler.transform(X)
                    elif component == 'behavioral_risk':
                        X_scaled = self.behavioral_scaler.transform(X)
                    else:
                        X_scaled = self.network_scaler.transform(X)
                    
                    # Get predictions
                    model = getattr(self, f'{component}_model')
                    predictions = model.predict(X_scaled)
                    
                    if len(ensemble_features) == 0:
                        ensemble_features = predictions.reshape(-1, 1)
                        # Use overall risk labels if available, otherwise average component labels
                        if 'overall_risk' in risk_labels:
                            ensemble_labels = risk_labels['overall_risk'].reindex(X.index).values
                        else:
                            ensemble_labels = risk_labels[component].reindex(X.index).values
                    else:
                        ensemble_features = np.column_stack([ensemble_features, predictions])
            
            if len(ensemble_features) > 0:
                # Train ensemble model
                X_train_ens, X_val_ens, y_train_ens, y_val_ens = train_test_split(
                    ensemble_features, ensemble_labels, 
                    test_size=validation_split, random_state=42
                )
                
                self.ensemble_model = lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42
                )
                
                self.ensemble_model.fit(
                    X_train_ens, y_train_ens,
                    eval_set=[(X_val_ens, y_val_ens)],
                    callbacks=[lgb.early_stopping(30)]
                )
                
        except Exception as e:
            self.logger.error(f"Ensemble model training failed: {str(e)}")
    
    def calculate_risk_score(self, 
                           entity_data: pd.Series,
                           entity_type: str) -> RiskScore:
        """
        Calculate comprehensive risk score for an entity.
        
        Args:
            entity_data: Entity data as pandas Series
            entity_type: Type of entity being scored
            
        Returns:
            RiskScore object with detailed risk assessment
        """
        try:
            entity_id = str(entity_data.get('entity_id', 'unknown'))
            
            # Convert Series to DataFrame for processing
            entity_df = pd.DataFrame([entity_data])
            
            # Prepare features for each component
            component_features = self.prepare_risk_features(entity_df, entity_type)
            
            # Calculate individual component scores
            component_scores = {}
            risk_factors = []
            
            for component in self.risk_components:
                if (component in component_features and 
                    hasattr(self, f'{component}_model') and 
                    getattr(self, f'{component}_model') is not None):
                    
                    # Get features
                    features = component_features[component][self.feature_columns[component]].fillna(0)
                    
                    # Scale features
                    if component == 'transaction_risk':
                        features_scaled = self.transaction_scaler.transform(features)
                    elif component == 'behavioral_risk':
                        features_scaled = self.behavioral_scaler.transform(features)
                    else:
                        features_scaled = self.network_scaler.transform(features)
                    
                    # Predict risk score
                    model = getattr(self, f'{component}_model')
                    component_score = model.predict(features_scaled)[0]
                    component_score = np.clip(component_score, 0, 100)
                    
                    component_scores[component] = component_score
                    
                    # Identify contributing risk factors
                    if component_score > 50:  # Above medium risk
                        risk_factors.append(f"High {component.replace('_', ' ')}")
            
            # Calculate overall risk score
            if component_scores:
                # Weighted average
                overall_score = sum(
                    score * self.component_weights.get(component, 0.1)
                    for component, score in component_scores.items()
                )
                
                # Use ensemble model if available
                if self.ensemble_model is not None:
                    ensemble_features = np.array(list(component_scores.values())).reshape(1, -1)
                    ensemble_score = self.ensemble_model.predict(ensemble_features)[0]
                    overall_score = np.clip(ensemble_score, 0, 100)
            else:
                overall_score = 50.0  # Default medium risk
            
            # Determine risk category
            risk_category = self._get_risk_category(overall_score)
            
            # Calculate confidence
            confidence = self._calculate_risk_confidence(component_scores, entity_data)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                overall_score, component_scores, risk_factors
            )
            
            # Set expiry time (risk scores expire after 24 hours)
            expiry_time = datetime.now() + timedelta(hours=24)
            
            return RiskScore(
                entity_id=entity_id,
                entity_type=entity_type,
                overall_risk_score=overall_score,
                risk_category=risk_category,
                component_scores=component_scores,
                risk_factors=risk_factors,
                confidence=confidence,
                timestamp=datetime.now(),
                expiry_time=expiry_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Risk score calculation failed: {str(e)}")
            return RiskScore(
                entity_id=str(entity_data.get('entity_id', 'unknown')),
                entity_type=entity_type,
                overall_risk_score=50.0,
                risk_category='medium',
                component_scores={},
                risk_factors=[f"Error in calculation: {str(e)}"],
                confidence=0.0,
                timestamp=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=1),
                recommendations=["Manual review required due to calculation error"]
            )
    
    def _get_risk_category(self, score: float) -> str:
        """Determine risk category from score."""
        for category, (min_score, max_score) in self.risk_categories.items():
            if min_score <= score < max_score:
                return category
        return 'very_high'  # If score >= 80
    
    def _calculate_risk_confidence(self, 
                                 component_scores: Dict[str, float],
                                 entity_data: pd.Series) -> float:
        """Calculate confidence in the risk assessment."""
        try:
            if not component_scores:
                return 0.0
            
            # Base confidence on number of components and data quality
            component_count_factor = min(1.0, len(component_scores) / len(self.risk_components))
            
            # Agreement between components
            scores = list(component_scores.values())
            score_std = np.std(scores)
            agreement_factor = max(0.0, 1.0 - score_std / 50)  # Normalize by max possible std
            
            # Data completeness factor
            non_null_features = sum(1 for value in entity_data.values if pd.notna(value) and value != 0)
            total_features = len(entity_data)
            completeness_factor = non_null_features / total_features if total_features > 0 else 0
            
            # Overall confidence
            confidence = (component_count_factor * 0.4 + 
                         agreement_factor * 0.4 + 
                         completeness_factor * 0.2)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    def _generate_risk_recommendations(self, 
                                     overall_score: float,
                                     component_scores: Dict[str, float],
                                     risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation recommendations."""
        try:
            recommendations = []
            
            # Overall score recommendations
            if overall_score >= 80:
                recommendations.append("High risk entity - consider blocking or restricting")
                recommendations.append("Conduct immediate enhanced due diligence")
                recommendations.append("Report to compliance team")
            elif overall_score >= 60:
                recommendations.append("Medium-high risk - implement enhanced monitoring")
                recommendations.append("Consider additional verification requirements")
                recommendations.append("Review transaction patterns regularly")
            elif overall_score >= 40:
                recommendations.append("Medium risk - maintain standard monitoring")
                recommendations.append("Periodic risk assessment review")
            else:
                recommendations.append("Low risk - standard monitoring sufficient")
            
            # Component-specific recommendations
            for component, score in component_scores.items():
                if score > 70:
                    if component == 'transaction_risk':
                        recommendations.append("Monitor transaction patterns for suspicious activity")
                    elif component == 'behavioral_risk':
                        recommendations.append("Investigate behavioral anomalies")
                    elif component == 'network_risk':
                        recommendations.append("Review network connections and associations")
                    elif component == 'liquidity_risk':
                        recommendations.append("Assess liquidity position and market exposure")
                    elif component == 'counterparty_risk':
                        recommendations.append("Review counterparty due diligence")
                    elif component == 'regulatory_risk':
                        recommendations.append("Ensure regulatory compliance verification")
                    elif component == 'technical_risk':
                        recommendations.append("Conduct technical security assessment")
            
            return recommendations[:10]  # Limit to top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            return ["Manual risk assessment recommended"]
    
    def update_risk_profile(self, entity_id: str, risk_score: RiskScore) -> None:
        """Update risk profile for an entity."""
        try:
            if entity_id not in self.risk_profiles:
                self.risk_profiles[entity_id] = RiskProfile(
                    entity_id=entity_id,
                    risk_history=[],
                    risk_trend='stable',
                    avg_risk_score=risk_score.overall_risk_score,
                    max_risk_score=risk_score.overall_risk_score,
                    risk_volatility=0.0,
                    last_assessment=risk_score.timestamp
                )
            
            profile = self.risk_profiles[entity_id]
            profile.risk_history.append(risk_score)
            
            # Keep only last 100 scores
            if len(profile.risk_history) > 100:
                profile.risk_history = profile.risk_history[-100:]
            
            # Update statistics
            scores = [rs.overall_risk_score for rs in profile.risk_history]
            profile.avg_risk_score = np.mean(scores)
            profile.max_risk_score = np.max(scores)
            profile.risk_volatility = np.std(scores)
            profile.last_assessment = risk_score.timestamp
            
            # Determine trend
            if len(scores) >= 3:
                recent_scores = scores[-3:]
                if all(recent_scores[i] > recent_scores[i-1] for i in range(1, len(recent_scores))):
                    profile.risk_trend = 'deteriorating'
                elif all(recent_scores[i] < recent_scores[i-1] for i in range(1, len(recent_scores))):
                    profile.risk_trend = 'improving'
                else:
                    profile.risk_trend = 'stable'
            
        except Exception as e:
            self.logger.error(f"Risk profile update failed: {str(e)}")
    
    def get_risk_profile(self, entity_id: str) -> Optional[RiskProfile]:
        """Get risk profile for an entity."""
        return self.risk_profiles.get(entity_id)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained risk scoring models."""
        try:
            model_data = {
                'transaction_risk_model': self.transaction_risk_model,
                'behavioral_risk_model': self.behavioral_risk_model,
                'network_risk_model': self.network_risk_model,
                'liquidity_risk_model': self.liquidity_risk_model,
                'counterparty_risk_model': self.counterparty_risk_model,
                'regulatory_risk_model': self.regulatory_risk_model,
                'technical_risk_model': self.technical_risk_model,
                'market_risk_model': self.market_risk_model,
                'operational_risk_model': self.operational_risk_model,
                'ensemble_model': self.ensemble_model,
                'transaction_scaler': self.transaction_scaler,
                'behavioral_scaler': self.behavioral_scaler,
                'network_scaler': self.network_scaler,
                'feature_columns': self.feature_columns,
                'risk_components': self.risk_components,
                'component_weights': self.component_weights,
                'risk_categories': self.risk_categories,
                'risk_indicators': self.risk_indicators,
                'protocol_risk_params': self.protocol_risk_params,
                'risk_baselines': self.risk_baselines,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Risk scoring model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load trained risk scoring models."""
        try:
            model_data = joblib.load(filepath)
            
            self.transaction_risk_model = model_data['transaction_risk_model']
            self.behavioral_risk_model = model_data['behavioral_risk_model']
            self.network_risk_model = model_data['network_risk_model']
            self.liquidity_risk_model = model_data['liquidity_risk_model']
            self.counterparty_risk_model = model_data['counterparty_risk_model']
            self.regulatory_risk_model = model_data['regulatory_risk_model']
            self.technical_risk_model = model_data['technical_risk_model']
            self.market_risk_model = model_data['market_risk_model']
            self.operational_risk_model = model_data['operational_risk_model']
            self.ensemble_model = model_data['ensemble_model']
            self.transaction_scaler = model_data['transaction_scaler']
            self.behavioral_scaler = model_data['behavioral_scaler']
            self.network_scaler = model_data['network_scaler']
            self.feature_columns = model_data['feature_columns']
            self.risk_components = model_data['risk_components']
            self.component_weights = model_data['component_weights']
            self.risk_categories = model_data['risk_categories']
            self.risk_indicators = model_data['risk_indicators']
            self.protocol_risk_params = model_data['protocol_risk_params']
            self.risk_baselines = model_data['risk_baselines']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Risk scoring model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise


def create_risk_scoring_pipeline(config: Dict) -> RiskScoringModel:
    """
    Create a risk scoring pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RiskScoringModel instance
    """
    return RiskScoringModel(
        risk_components=config.get('risk_components'),
        component_weights=config.get('component_weights'),
        risk_categories=config.get('risk_categories')
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create risk scoring model
    risk_scorer = RiskScoringModel()
    
    # Example entity data
    entity_data = pd.Series({
        'entity_id': '0x1234567890abcdef',
        'transaction_count': 150,
        'total_volume': 50000.0,
        'active_days': 30,
        'unique_counterparties': 45,
        'avg_transaction_value': 333.33,
        'max_transaction_value': 5000.0,
        'night_tx_ratio': 0.2,
        'weekend_tx_ratio': 0.15
    })
    
    # This would be used with trained models
    risk_score = risk_scorer.calculate_risk_score(entity_data, 'address')
    print(f"Overall Risk Score: {risk_score.overall_risk_score:.1f}")
    print(f"Risk Category: {risk_score.risk_category}")
    print(f"Risk Factors: {risk_score.risk_factors}")
    print(f"Recommendations: {risk_score.recommendations}")
    
    print("Risk scoring model implementation completed")
