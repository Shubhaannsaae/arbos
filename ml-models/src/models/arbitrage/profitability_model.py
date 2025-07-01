"""
Profitability Model for Arbitrage Trading

This module implements advanced ML models for predicting and optimizing
arbitrage profitability using real-time gas costs, slippage calculations,
and market impact analysis with Chainlink Data Feeds integration.

Features:
- Dynamic profitability calculation with real-time parameters
- Gas cost optimization using network conditions
- Slippage prediction based on liquidity analysis
- MEV protection cost estimation
- Cross-chain arbitrage profitability modeling
- Risk-adjusted return calculations
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import joblib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from scipy import optimize
from scipy.stats import norm
import math

warnings.filterwarnings('ignore')

@dataclass
class ProfitabilityComponents:
    """Data class for profitability calculation components."""
    gross_profit: float
    gas_cost: float
    slippage_cost: float
    protocol_fees: float
    mev_protection_cost: float
    bridge_fees: float
    net_profit: float
    profit_margin: float
    roi: float
    risk_adjusted_return: float
    execution_probability: float

@dataclass
class MarketConditions:
    """Data class for current market conditions."""
    gas_price_gwei: float
    network_congestion: float
    volatility_index: float
    liquidity_depth: float
    spread_tightness: float
    mev_activity_level: float
    block_time_ms: float

class ProfitabilityModel:
    """
    Advanced profitability model for arbitrage opportunities using ML
    to predict optimal execution parameters and profitability outcomes.
    """
    
    def __init__(self, 
                 chains_config: Optional[Dict] = None,
                 dex_config: Optional[Dict] = None):
        """
        Initialize the profitability model.
        
        Args:
            chains_config: Configuration for different blockchain networks
            dex_config: Configuration for different DEX protocols
        """
        # Default chain configurations
        self.chains_config = chains_config or {
            1: {  # Ethereum
                'name': 'ethereum',
                'base_gas_limit': 150000,
                'priority_fee_multiplier': 1.2,
                'max_gas_price_gwei': 500,
                'block_time_ms': 12000
            },
            137: {  # Polygon
                'name': 'polygon',
                'base_gas_limit': 150000,
                'priority_fee_multiplier': 1.1,
                'max_gas_price_gwei': 1000,
                'block_time_ms': 2000
            },
            42161: {  # Arbitrum
                'name': 'arbitrum',
                'base_gas_limit': 1000000,
                'priority_fee_multiplier': 1.05,
                'max_gas_price_gwei': 10,
                'block_time_ms': 250
            },
            43114: {  # Avalanche
                'name': 'avalanche',
                'base_gas_limit': 200000,
                'priority_fee_multiplier': 1.15,
                'max_gas_price_gwei': 200,
                'block_time_ms': 2000
            }
        }
        
        # Default DEX configurations
        self.dex_config = dex_config or {
            'uniswap_v2': {
                'fee': 0.003,
                'gas_per_swap': 120000,
                'slippage_factor': 1.0,
                'liquidity_efficiency': 0.8
            },
            'uniswap_v3': {
                'fee': 0.0005,  # Variable, using 0.05% as base
                'gas_per_swap': 150000,
                'slippage_factor': 0.8,
                'liquidity_efficiency': 1.2
            },
            'sushiswap': {
                'fee': 0.003,
                'gas_per_swap': 130000,
                'slippage_factor': 1.0,
                'liquidity_efficiency': 0.85
            },
            'curve': {
                'fee': 0.0004,
                'gas_per_swap': 100000,
                'slippage_factor': 0.6,
                'liquidity_efficiency': 1.5
            },
            'balancer': {
                'fee': 0.001,  # Variable
                'gas_per_swap': 140000,
                'slippage_factor': 0.9,
                'liquidity_efficiency': 1.0
            }
        }
        
        # Model components
        self.gas_cost_predictor = None
        self.slippage_predictor = None
        self.execution_optimizer = None
        self.risk_adjuster = None
        
        # Feature scalers
        self.gas_scaler = StandardScaler()
        self.slippage_scaler = RobustScaler()
        self.profit_scaler = StandardScaler()
        
        # Model metadata
        self.feature_columns = []
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def prepare_profitability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for profitability prediction.
        
        Args:
            df: DataFrame with arbitrage opportunity data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            features_df = df.copy()
            
            # Basic profitability features
            features_df['trade_size_usd'] = features_df['trade_amount'] * features_df['source_price']
            features_df['price_impact_source'] = features_df['trade_amount'] / features_df['source_liquidity']
            features_df['price_impact_target'] = features_df['trade_amount'] / features_df['target_liquidity']
            features_df['combined_price_impact'] = features_df['price_impact_source'] + features_df['price_impact_target']
            
            # Liquidity ratios
            features_df['liquidity_ratio'] = features_df['target_liquidity'] / features_df['source_liquidity']
            features_df['min_liquidity'] = np.minimum(features_df['source_liquidity'], features_df['target_liquidity'])
            features_df['liquidity_depth_score'] = features_df['min_liquidity'] / features_df['trade_amount']
            
            # Gas cost features
            features_df['gas_cost_percentage'] = features_df['estimated_gas_cost'] / features_df['trade_size_usd'] * 100
            features_df['gas_efficiency_score'] = features_df['trade_size_usd'] / features_df['estimated_gas_cost']
            
            # Market timing features
            features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
            features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
            features_df['is_high_activity_hour'] = features_df['hour'].isin([8, 9, 10, 14, 15, 16, 20, 21]).astype(int)
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Network condition features
            features_df['network_congestion_score'] = self._calculate_network_congestion(features_df)
            features_df['gas_price_percentile'] = features_df.groupby('chain_id')['gas_price_gwei'].rank(pct=True)
            features_df['relative_gas_cost'] = features_df['gas_price_gwei'] / features_df.groupby('chain_id')['gas_price_gwei'].transform('median')
            
            # DEX-specific features
            features_df['source_dex_fee'] = features_df['source_dex'].map(
                lambda x: self.dex_config.get(x, {}).get('fee', 0.003)
            )
            features_df['target_dex_fee'] = features_df['target_dex'].map(
                lambda x: self.dex_config.get(x, {}).get('fee', 0.003)
            )
            features_df['total_protocol_fees'] = (features_df['source_dex_fee'] + features_df['target_dex_fee']) * features_df['trade_size_usd']
            
            # Volatility and risk features
            features_df['price_volatility'] = self._calculate_price_volatility(features_df)
            features_df['execution_risk_score'] = self._calculate_execution_risk(features_df)
            features_df['mev_risk_score'] = self._calculate_mev_risk(features_df)
            
            # Cross-chain features
            features_df['is_cross_chain'] = (features_df['source_chain'] != features_df['target_chain']).astype(int)
            features_df['bridge_cost_estimate'] = features_df.apply(self._estimate_bridge_cost, axis=1)
            
            # Competition features
            features_df['similar_opportunities_count'] = features_df.groupby(['token_pair', 'hour'])['token_pair'].transform('count')
            features_df['opportunity_uniqueness'] = 1 / (features_df['similar_opportunities_count'] + 1)
            
            # Technical indicators
            features_df['rsi'] = self._calculate_rsi(features_df)
            features_df['bollinger_position'] = self._calculate_bollinger_position(features_df)
            features_df['momentum_score'] = self._calculate_momentum(features_df)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise
    
    def _calculate_network_congestion(self, df: pd.DataFrame) -> pd.Series:
        """Calculate network congestion score."""
        try:
            # Use gas price and pending transactions as proxies
            base_congestion = df['gas_price_gwei'] / df.groupby('chain_id')['gas_price_gwei'].transform('median')
            
            # Add transaction count factor if available
            if 'pending_tx_count' in df.columns:
                tx_factor = df['pending_tx_count'] / df.groupby('chain_id')['pending_tx_count'].transform('median')
                congestion_score = (base_congestion * 0.7 + tx_factor * 0.3)
            else:
                congestion_score = base_congestion
            
            return np.clip(congestion_score, 0.1, 10.0)
            
        except Exception as e:
            self.logger.error(f"Network congestion calculation failed: {str(e)}")
            return pd.Series(1.0, index=df.index)
    
    def _calculate_price_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price volatility score."""
        try:
            # Use price change and volume as volatility proxies
            if 'price_change_1h' in df.columns:
                return np.abs(df['price_change_1h']) * 100
            else:
                # Use price difference as proxy
                price_diff = np.abs(df['target_price'] - df['source_price']) / df['source_price']
                return price_diff * 100
                
        except Exception as e:
            self.logger.error(f"Price volatility calculation failed: {str(e)}")
            return pd.Series(5.0, index=df.index)  # Default moderate volatility
    
    def _calculate_execution_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate execution risk score."""
        try:
            # Combine multiple risk factors
            liquidity_risk = np.clip(df['combined_price_impact'] * 100, 0, 100)
            gas_risk = np.clip(df['gas_cost_percentage'], 0, 100)
            timing_risk = np.clip(df.get('opportunity_age_seconds', 0) / 60 * 10, 0, 100)
            
            execution_risk = (liquidity_risk * 0.5 + gas_risk * 0.3 + timing_risk * 0.2)
            return np.clip(execution_risk, 0, 100)
            
        except Exception as e:
            self.logger.error(f"Execution risk calculation failed: {str(e)}")
            return pd.Series(50.0, index=df.index)
    
    def _calculate_mev_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MEV risk score."""
        try:
            # Higher profit margins attract more MEV attention
            profit_attractiveness = np.clip(df.get('expected_profit_percentage', 1) * 10, 0, 100)
            
            # Gas price competition factor
            gas_competition = df['gas_price_percentile'] * 50
            
            # Network activity factor
            network_activity = df['network_congestion_score'] * 20
            
            mev_risk = (profit_attractiveness * 0.4 + gas_competition * 0.4 + network_activity * 0.2)
            return np.clip(mev_risk, 0, 100)
            
        except Exception as e:
            self.logger.error(f"MEV risk calculation failed: {str(e)}")
            return pd.Series(30.0, index=df.index)
    
    def _estimate_bridge_cost(self, row) -> float:
        """Estimate cross-chain bridge cost."""
        try:
            if row.get('is_cross_chain', 0) == 0:
                return 0.0
            
            # Base bridge fee (varies by bridge protocol)
            base_fee = 0.001  # 0.1% base fee
            
            # Gas cost on destination chain
            dest_chain_gas = 50000 * row.get('target_gas_price_gwei', 20) * 1e-9  # Convert to ETH
            
            # Bridge-specific fees
            bridge_fee = row.get('trade_size_usd', 1000) * base_fee
            
            return bridge_fee + dest_chain_gas
            
        except Exception as e:
            self.logger.error(f"Bridge cost estimation failed: {str(e)}")
            return 50.0  # Default bridge cost
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            if 'price_changes' in df.columns:
                delta = df['price_changes']
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi.fillna(50)
            else:
                return pd.Series(50, index=df.index)  # Neutral RSI
                
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {str(e)}")
            return pd.Series(50, index=df.index)
    
    def _calculate_bollinger_position(self, df: pd.DataFrame) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        try:
            if 'price_20_sma' in df.columns and 'price_20_std' in df.columns:
                upper_band = df['price_20_sma'] + (df['price_20_std'] * 2)
                lower_band = df['price_20_sma'] - (df['price_20_std'] * 2)
                current_price = df.get('current_price', df.get('source_price', 0))
                
                position = (current_price - lower_band) / (upper_band - lower_band)
                return np.clip(position, 0, 1)
            else:
                return pd.Series(0.5, index=df.index)  # Neutral position
                
        except Exception as e:
            self.logger.error(f"Bollinger position calculation failed: {str(e)}")
            return pd.Series(0.5, index=df.index)
    
    def _calculate_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score."""
        try:
            # Use volume and price change as momentum indicators
            volume_momentum = df.get('volume_ratio', 1.0)  # Current vs average volume
            price_momentum = df.get('price_change_percentage', 0) / 100
            
            momentum = (volume_momentum * 0.6 + price_momentum * 0.4)
            return np.clip(momentum, -2, 2)
            
        except Exception as e:
            self.logger.error(f"Momentum calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def calculate_profitability(self, 
                              opportunity_data: Dict,
                              market_conditions: MarketConditions) -> ProfitabilityComponents:
        """
        Calculate detailed profitability for an arbitrage opportunity.
        
        Args:
            opportunity_data: Dictionary with opportunity details
            market_conditions: Current market conditions
            
        Returns:
            ProfitabilityComponents with detailed breakdown
        """
        try:
            # Extract basic parameters
            trade_amount = opportunity_data['trade_amount']
            source_price = opportunity_data['source_price']
            target_price = opportunity_data['target_price']
            source_liquidity = opportunity_data['source_liquidity']
            target_liquidity = opportunity_data['target_liquidity']
            chain_id = opportunity_data.get('chain_id', 1)
            
            # Calculate gross profit
            gross_profit = (target_price - source_price) * trade_amount
            
            # Calculate gas costs
            gas_cost = self._calculate_gas_cost(
                opportunity_data, market_conditions, chain_id
            )
            
            # Calculate slippage costs
            slippage_cost = self._calculate_slippage_cost(
                trade_amount, source_liquidity, target_liquidity,
                source_price, target_price
            )
            
            # Calculate protocol fees
            protocol_fees = self._calculate_protocol_fees(
                opportunity_data, trade_amount
            )
            
            # Calculate MEV protection cost
            mev_protection_cost = self._calculate_mev_protection_cost(
                gross_profit, market_conditions
            )
            
            # Calculate bridge fees for cross-chain
            bridge_fees = self._calculate_bridge_fees(opportunity_data)
            
            # Calculate net profit
            total_costs = (gas_cost + slippage_cost + protocol_fees + 
                          mev_protection_cost + bridge_fees)
            net_profit = gross_profit - total_costs
            
            # Calculate ratios
            trade_value = trade_amount * source_price
            profit_margin = (net_profit / trade_value) * 100 if trade_value > 0 else 0
            roi = (net_profit / trade_value) * 100 if trade_value > 0 else 0
            
            # Calculate risk-adjusted return
            risk_score = self._calculate_total_risk_score(opportunity_data, market_conditions)
            risk_adjustment_factor = 1 - (risk_score / 100) * 0.5  # Max 50% risk adjustment
            risk_adjusted_return = roi * risk_adjustment_factor
            
            # Calculate execution probability
            execution_probability = self._calculate_execution_probability(
                opportunity_data, market_conditions, net_profit
            )
            
            return ProfitabilityComponents(
                gross_profit=gross_profit,
                gas_cost=gas_cost,
                slippage_cost=slippage_cost,
                protocol_fees=protocol_fees,
                mev_protection_cost=mev_protection_cost,
                bridge_fees=bridge_fees,
                net_profit=net_profit,
                profit_margin=profit_margin,
                roi=roi,
                risk_adjusted_return=risk_adjusted_return,
                execution_probability=execution_probability
            )
            
        except Exception as e:
            self.logger.error(f"Profitability calculation failed: {str(e)}")
            raise
    
    def _calculate_gas_cost(self, 
                           opportunity_data: Dict,
                           market_conditions: MarketConditions,
                           chain_id: int) -> float:
        """Calculate gas cost for the arbitrage execution."""
        try:
            chain_config = self.chains_config.get(chain_id, self.chains_config[1])
            
            # Base gas limit for arbitrage transaction
            base_gas_limit = chain_config['base_gas_limit']
            
            # Add DEX-specific gas costs
            source_dex = opportunity_data.get('source_dex', 'uniswap_v2')
            target_dex = opportunity_data.get('target_dex', 'uniswap_v2')
            
            source_gas = self.dex_config.get(source_dex, {}).get('gas_per_swap', 120000)
            target_gas = self.dex_config.get(target_dex, {}).get('gas_per_swap', 120000)
            
            total_gas_limit = base_gas_limit + source_gas + target_gas
            
            # Adjust for cross-chain operations
            if opportunity_data.get('is_cross_chain', False):
                total_gas_limit *= 1.5  # Additional overhead for cross-chain
            
            # Current gas price with priority fee
            gas_price_wei = market_conditions.gas_price_gwei * 1e9
            priority_fee = gas_price_wei * chain_config['priority_fee_multiplier']
            total_gas_price = gas_price_wei + priority_fee
            
            # Calculate gas cost in ETH
            gas_cost_wei = total_gas_limit * total_gas_price
            gas_cost_eth = gas_cost_wei / 1e18
            
            # Convert to USD (using ETH price from opportunity data)
            eth_price_usd = opportunity_data.get('eth_price_usd', 3000)
            gas_cost_usd = gas_cost_eth * eth_price_usd
            
            return gas_cost_usd
            
        except Exception as e:
            self.logger.error(f"Gas cost calculation failed: {str(e)}")
            return 50.0  # Default gas cost
    
    def _calculate_slippage_cost(self, 
                               trade_amount: float,
                               source_liquidity: float,
                               target_liquidity: float,
                               source_price: float,
                               target_price: float) -> float:
        """Calculate slippage cost based on liquidity and trade size."""
        try:
            # Source slippage (buying)
            source_impact = trade_amount / source_liquidity
            source_slippage = source_impact ** 0.5 * 0.01  # Square root price impact model
            source_slippage_cost = trade_amount * source_price * source_slippage
            
            # Target slippage (selling)
            target_impact = trade_amount / target_liquidity
            target_slippage = target_impact ** 0.5 * 0.01
            target_slippage_cost = trade_amount * target_price * target_slippage
            
            total_slippage_cost = source_slippage_cost + target_slippage_cost
            return total_slippage_cost
            
        except Exception as e:
            self.logger.error(f"Slippage cost calculation failed: {str(e)}")
            return trade_amount * source_price * 0.005  # Default 0.5% slippage
    
    def _calculate_protocol_fees(self, 
                               opportunity_data: Dict,
                               trade_amount: float) -> float:
        """Calculate protocol fees for DEX transactions."""
        try:
            source_dex = opportunity_data.get('source_dex', 'uniswap_v2')
            target_dex = opportunity_data.get('target_dex', 'uniswap_v2')
            
            source_fee_rate = self.dex_config.get(source_dex, {}).get('fee', 0.003)
            target_fee_rate = self.dex_config.get(target_dex, {}).get('fee', 0.003)
            
            trade_value = trade_amount * opportunity_data.get('source_price', 0)
            
            source_fee = trade_value * source_fee_rate
            target_fee = trade_value * target_fee_rate
            
            return source_fee + target_fee
            
        except Exception as e:
            self.logger.error(f"Protocol fees calculation failed: {str(e)}")
            return trade_amount * opportunity_data.get('source_price', 0) * 0.006  # Default 0.6%
    
    def _calculate_mev_protection_cost(self, 
                                     gross_profit: float,
                                     market_conditions: MarketConditions) -> float:
        """Calculate cost of MEV protection."""
        try:
            # Base MEV protection cost (private mempool, etc.)
            base_protection_cost = 10.0  # $10 base cost
            
            # Dynamic cost based on profit attractiveness
            profit_based_cost = gross_profit * 0.1  # 10% of gross profit
            
            # Network congestion multiplier
            congestion_multiplier = 1 + (market_conditions.network_congestion / 10)
            
            # MEV activity level multiplier
            mev_multiplier = 1 + (market_conditions.mev_activity_level / 10)
            
            total_protection_cost = (
                base_protection_cost + profit_based_cost
            ) * congestion_multiplier * mev_multiplier
            
            return min(total_protection_cost, gross_profit * 0.3)  # Cap at 30% of gross profit
            
        except Exception as e:
            self.logger.error(f"MEV protection cost calculation failed: {str(e)}")
            return max(10.0, gross_profit * 0.05)  # Default 5% with minimum
    
    def _calculate_bridge_fees(self, opportunity_data: Dict) -> float:
        """Calculate cross-chain bridge fees."""
        try:
            if not opportunity_data.get('is_cross_chain', False):
                return 0.0
            
            trade_value = opportunity_data['trade_amount'] * opportunity_data['source_price']
            
            # Base bridge fee percentage
            bridge_fee_rate = 0.001  # 0.1%
            
            # Bridge-specific adjustments
            bridge_protocol = opportunity_data.get('bridge_protocol', 'ccip')
            if bridge_protocol == 'ccip':
                bridge_fee_rate *= 0.8  # CCIP is more efficient
            
            bridge_fee = trade_value * bridge_fee_rate
            
            # Add fixed gas cost for destination chain
            dest_gas_cost = 30.0  # $30 average
            
            return bridge_fee + dest_gas_cost
            
        except Exception as e:
            self.logger.error(f"Bridge fees calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_total_risk_score(self, 
                                  opportunity_data: Dict,
                                  market_conditions: MarketConditions) -> float:
        """Calculate total risk score for the opportunity."""
        try:
            # Liquidity risk
            min_liquidity = min(opportunity_data['source_liquidity'], 
                              opportunity_data['target_liquidity'])
            trade_value = opportunity_data['trade_amount'] * opportunity_data['source_price']
            liquidity_risk = min(50, (trade_value / min_liquidity) * 100)
            
            # Volatility risk
            volatility_risk = min(30, market_conditions.volatility_index * 30)
            
            # Execution risk (timing, gas price changes)
            execution_risk = min(25, market_conditions.network_congestion * 2.5)
            
            # MEV risk
            mev_risk = min(20, market_conditions.mev_activity_level * 2)
            
            # Cross-chain risk
            cross_chain_risk = 15 if opportunity_data.get('is_cross_chain', False) else 0
            
            total_risk = (liquidity_risk + volatility_risk + execution_risk + 
                         mev_risk + cross_chain_risk)
            
            return min(100, total_risk)
            
        except Exception as e:
            self.logger.error(f"Risk score calculation failed: {str(e)}")
            return 50.0  # Default moderate risk
    
    def _calculate_execution_probability(self, 
                                       opportunity_data: Dict,
                                       market_conditions: MarketConditions,
                                       net_profit: float) -> float:
        """Calculate probability of successful execution."""
        try:
            # Base probability
            base_probability = 0.8
            
            # Profit margin factor
            profit_margin = net_profit / (opportunity_data['trade_amount'] * opportunity_data['source_price'])
            profit_factor = min(1.2, 1 + profit_margin * 2)  # Higher profit = higher probability
            
            # Liquidity factor
            min_liquidity = min(opportunity_data['source_liquidity'], 
                              opportunity_data['target_liquidity'])
            trade_value = opportunity_data['trade_amount'] * opportunity_data['source_price']
            liquidity_factor = min(1.1, min_liquidity / trade_value)
            
            # Network conditions factor
            congestion_factor = max(0.5, 1 - market_conditions.network_congestion / 20)
            
            # Time decay factor (opportunities decay over time)
            age_seconds = opportunity_data.get('age_seconds', 0)
            time_factor = max(0.3, 1 - age_seconds / 300)  # 5 minute decay
            
            execution_probability = (
                base_probability * profit_factor * liquidity_factor * 
                congestion_factor * time_factor
            )
            
            return min(0.99, max(0.01, execution_probability))
            
        except Exception as e:
            self.logger.error(f"Execution probability calculation failed: {str(e)}")
            return 0.5  # Default 50% probability
    
    def optimize_trade_parameters(self, 
                                opportunity_data: Dict,
                                market_conditions: MarketConditions,
                                max_trade_amount: Optional[float] = None) -> Dict:
        """
        Optimize trade parameters for maximum risk-adjusted return.
        
        Args:
            opportunity_data: Opportunity details
            market_conditions: Current market conditions
            max_trade_amount: Maximum allowed trade amount
            
        Returns:
            Dictionary with optimal parameters
        """
        try:
            # Define objective function
            def objective(trade_amount):
                opt_data = opportunity_data.copy()
                opt_data['trade_amount'] = trade_amount[0]
                
                profitability = self.calculate_profitability(opt_data, market_conditions)
                
                # Maximize risk-adjusted return
                return -profitability.risk_adjusted_return
            
            # Set bounds
            min_amount = opportunity_data['trade_amount'] * 0.1  # 10% of original
            max_amount = max_trade_amount or opportunity_data['trade_amount'] * 2.0
            
            # Constraint by available liquidity
            min_liquidity = min(opportunity_data['source_liquidity'], 
                              opportunity_data['target_liquidity'])
            max_amount = min(max_amount, min_liquidity * 0.1)  # Max 10% of liquidity
            
            bounds = [(min_amount, max_amount)]
            
            # Optimize
            result = optimize.minimize(
                objective,
                x0=[opportunity_data['trade_amount']],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                optimal_amount = result.x[0]
                
                # Calculate optimal profitability
                opt_data = opportunity_data.copy()
                opt_data['trade_amount'] = optimal_amount
                optimal_profitability = self.calculate_profitability(opt_data, market_conditions)
                
                return {
                    'optimal_trade_amount': optimal_amount,
                    'optimal_profitability': optimal_profitability,
                    'improvement_factor': (-result.fun) / opportunity_data.get('expected_return', 1),
                    'optimization_successful': True
                }
            else:
                return {
                    'optimal_trade_amount': opportunity_data['trade_amount'],
                    'optimal_profitability': self.calculate_profitability(opportunity_data, market_conditions),
                    'improvement_factor': 1.0,
                    'optimization_successful': False,
                    'error': result.message
                }
                
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {str(e)}")
            return {
                'optimal_trade_amount': opportunity_data['trade_amount'],
                'optimization_successful': False,
                'error': str(e)
            }
    
    def train_profitability_models(self, 
                                 training_data: pd.DataFrame,
                                 validation_split: float = 0.2) -> Dict:
        """
        Train ML models for profitability prediction components.
        
        Args:
            training_data: Historical arbitrage execution data
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        try:
            self.logger.info("Starting profitability model training...")
            
            # Prepare features
            features_df = self.prepare_profitability_features(training_data)
            
            # Select feature columns
            feature_cols = [col for col in features_df.columns 
                          if not col.startswith('actual_') and 
                          col not in ['timestamp', 'trade_id', 'token_pair']]
            
            self.feature_columns = feature_cols
            X = features_df[feature_cols].fillna(0)
            
            training_metrics = {}
            
            # Train gas cost predictor
            if 'actual_gas_cost' in training_data.columns:
                self.logger.info("Training gas cost predictor...")
                y_gas = training_data['actual_gas_cost']
                
                X_train, X_val, y_train_gas, y_val_gas = train_test_split(
                    X, y_gas, test_size=validation_split, random_state=42
                )
                
                X_train_gas_scaled = self.gas_scaler.fit_transform(X_train)
                X_val_gas_scaled = self.gas_scaler.transform(X_val)
                
                self.gas_cost_predictor = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    random_state=42
                )
                
                self.gas_cost_predictor.fit(
                    X_train_gas_scaled, y_train_gas,
                    eval_set=[(X_val_gas_scaled, y_val_gas)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                gas_pred = self.gas_cost_predictor.predict(X_val_gas_scaled)
                training_metrics['gas_predictor'] = {
                    'mae': mean_absolute_error(y_val_gas, gas_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val_gas, gas_pred)),
                    'r2': r2_score(y_val_gas, gas_pred)
                }
            
            # Train slippage predictor
            if 'actual_slippage_cost' in training_data.columns:
                self.logger.info("Training slippage predictor...")
                y_slippage = training_data['actual_slippage_cost']
                
                X_train_slip_scaled = self.slippage_scaler.fit_transform(X_train)
                X_val_slip_scaled = self.slippage_scaler.transform(X_val)
                
                self.slippage_predictor = lgb.LGBMRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    random_state=42
                )
                
                self.slippage_predictor.fit(
                    X_train_slip_scaled, y_train_gas,  # Using same split
                    eval_set=[(X_val_slip_scaled, y_val_gas)],
                    callbacks=[lgb.early_stopping(50)]
                )
                
                slip_pred = self.slippage_predictor.predict(X_val_slip_scaled)
                training_metrics['slippage_predictor'] = {
                    'mae': mean_absolute_error(y_val_gas, slip_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val_gas, slip_pred)),
                    'r2': r2_score(y_val_gas, slip_pred)
                }
            
            self.last_trained = datetime.now()
            self.logger.info("Profitability model training completed")
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """Save the trained models."""
        try:
            model_data = {
                'gas_cost_predictor': self.gas_cost_predictor,
                'slippage_predictor': self.slippage_predictor,
                'execution_optimizer': self.execution_optimizer,
                'risk_adjuster': self.risk_adjuster,
                'gas_scaler': self.gas_scaler,
                'slippage_scaler': self.slippage_scaler,
                'profit_scaler': self.profit_scaler,
                'feature_columns': self.feature_columns,
                'chains_config': self.chains_config,
                'dex_config': self.dex_config,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load trained models."""
        try:
            model_data = joblib.load(filepath)
            
            self.gas_cost_predictor = model_data['gas_cost_predictor']
            self.slippage_predictor = model_data['slippage_predictor']
            self.execution_optimizer = model_data['execution_optimizer']
            self.risk_adjuster = model_data['risk_adjuster']
            self.gas_scaler = model_data['gas_scaler']
            self.slippage_scaler = model_data['slippage_scaler']
            self.profit_scaler = model_data['profit_scaler']
            self.feature_columns = model_data['feature_columns']
            self.chains_config = model_data['chains_config']
            self.dex_config = model_data['dex_config']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise


def create_profitability_model_pipeline(config: Dict) -> ProfitabilityModel:
    """
    Create a profitability model pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ProfitabilityModel instance
    """
    return ProfitabilityModel(
        chains_config=config.get('chains_config'),
        dex_config=config.get('dex_config')
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = ProfitabilityModel()
    
    # Example market conditions
    market_conditions = MarketConditions(
        gas_price_gwei=25.0,
        network_congestion=0.7,
        volatility_index=0.3,
        liquidity_depth=0.8,
        spread_tightness=0.6,
        mev_activity_level=0.5,
        block_time_ms=12000
    )
    
    # Example opportunity data
    opportunity_data = {
        'trade_amount': 10.0,
        'source_price': 3800.0,
        'target_price': 3820.0,
        'source_liquidity': 1000000.0,
        'target_liquidity': 800000.0,
        'source_dex': 'uniswap_v2',
        'target_dex': 'sushiswap',
        'chain_id': 1,
        'is_cross_chain': False,
        'eth_price_usd': 3800.0
    }
    
    # Calculate profitability
    profitability = model.calculate_profitability(opportunity_data, market_conditions)
    print(f"Net Profit: ${profitability.net_profit:.2f}")
    print(f"ROI: {profitability.roi:.2f}%")
    print(f"Risk-Adjusted Return: {profitability.risk_adjusted_return:.2f}%")
    
    # Optimize parameters
    optimized = model.optimize_trade_parameters(opportunity_data, market_conditions)
    print(f"Optimal Trade Amount: {optimized['optimal_trade_amount']:.2f}")
    
    print("Profitability model implementation completed")
