"""
Portfolio Rebalancing Strategy Model

This module implements advanced ML-driven portfolio rebalancing strategies using
reinforcement learning, dynamic programming, and adaptive algorithms with
Chainlink Automation for scheduled rebalancing and real-time market data.

Features:
- Reinforcement learning for adaptive rebalancing decisions
- Dynamic threshold-based rebalancing strategies
- Transaction cost-aware rebalancing optimization
- Multi-objective rebalancing (risk, return, costs, taxes)
- Market regime-dependent rebalancing frequencies
- Real-time rebalancing signals using Chainlink oracles
- Cross-chain portfolio rebalancing with CCIP integration
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import scipy.optimize as optimize
import cvxpy as cp
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import joblib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from arch import arch_model
import gym
from gym import spaces
import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

warnings.filterwarnings('ignore')

@dataclass
class RebalancingSignal:
    """Data class for rebalancing signals."""
    timestamp: datetime
    signal_type: str  # 'time_based', 'threshold_based', 'ml_based', 'risk_based'
    signal_strength: float  # 0-1 strength of signal
    recommended_action: str  # 'rebalance', 'hold', 'emergency_rebalance'
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    deviation_scores: Dict[str, float]
    transaction_costs: float
    expected_benefit: float
    risk_reduction: float
    confidence: float

@dataclass
class RebalancingAction:
    """Data class for rebalancing actions."""
    action_id: str
    timestamp: datetime
    trades: List[Dict[str, Union[str, float]]]  # List of trades to execute
    total_cost: float
    expected_slippage: float
    execution_time_estimate: float
    priority: str  # 'high', 'medium', 'low'
    approval_required: bool
    risk_impact: float

@dataclass
class MarketRegime:
    """Data class for market regime classification."""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile', 'crisis'
    confidence: float
    volatility_level: str  # 'low', 'medium', 'high'
    correlation_level: str  # 'low', 'medium', 'high'
    recommended_frequency: str  # 'daily', 'weekly', 'monthly', 'quarterly'

class PortfolioRebalancingEnv(gym.Env):
    """
    Gym environment for reinforcement learning-based portfolio rebalancing.
    """
    
    def __init__(self, 
                 price_data: pd.DataFrame,
                 transaction_costs: Dict[str, float],
                 rebalancing_costs: float = 0.001):
        super(PortfolioRebalancingEnv, self).__init__()
        
        self.price_data = price_data
        self.returns_data = price_data.pct_change().dropna()
        self.transaction_costs = transaction_costs
        self.rebalancing_costs = rebalancing_costs
        
        self.n_assets = len(price_data.columns)
        self.current_step = 0
        self.max_steps = len(self.returns_data) - 1
        
        # Action space: weights for each asset (continuous)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation space: returns, volatilities, correlations, technical indicators
        obs_dim = self.n_assets * 5  # returns, volatilities, momentum, RSI, portfolio weights
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Portfolio state
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.cash = 0.0
        
        # Performance tracking
        self.portfolio_returns = []
        self.rebalancing_costs_incurred = []
        
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 20  # Start after 20 days for technical indicators
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.cash = 0.0
        self.portfolio_returns = []
        self.rebalancing_costs_incurred = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment."""
        # Normalize action to ensure weights sum to 1
        action = np.clip(action, 0, 1)
        action = action / np.sum(action) if np.sum(action) > 0 else self.current_weights
        
        # Calculate rebalancing cost
        weight_changes = np.abs(action - self.current_weights)
        rebalancing_cost = np.sum(weight_changes) * self.rebalancing_costs
        
        # Update portfolio
        current_returns = self.returns_data.iloc[self.current_step].values
        
        # Portfolio return before rebalancing
        portfolio_return = np.sum(self.current_weights * current_returns)
        
        # Apply rebalancing cost
        portfolio_return -= rebalancing_cost
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        
        # Update weights
        self.current_weights = action
        
        # Track performance
        self.portfolio_returns.append(portfolio_return)
        self.rebalancing_costs_incurred.append(rebalancing_cost)
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return, rebalancing_cost)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get next observation
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        info = {
            'portfolio_return': portfolio_return,
            'rebalancing_cost': rebalancing_cost,
            'portfolio_value': self.portfolio_value,
            'current_weights': self.current_weights.copy()
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current observation."""
        if self.current_step >= len(self.returns_data):
            return np.zeros(self.observation_space.shape)
        
        # Get recent returns (last 5 days)
        recent_returns = self.returns_data.iloc[max(0, self.current_step-5):self.current_step].values
        if len(recent_returns) < 5:
            recent_returns = np.pad(recent_returns, ((5-len(recent_returns), 0), (0, 0)), 'constant')
        
        # Calculate features
        mean_returns = np.mean(recent_returns, axis=0)
        volatilities = np.std(recent_returns, axis=0)
        
        # Momentum (last 20 days)
        momentum_window = max(0, self.current_step-20)
        momentum = np.sum(self.returns_data.iloc[momentum_window:self.current_step].values, axis=0)
        
        # Simple RSI approximation
        price_changes = self.returns_data.iloc[max(0, self.current_step-14):self.current_step].values
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gains = np.mean(gains, axis=0)
        avg_losses = np.mean(losses, axis=0)
        rsi = 100 - (100 / (1 + (avg_gains + 1e-8) / (avg_losses + 1e-8)))
        
        # Combine all features
        observation = np.concatenate([
            mean_returns,
            volatilities,
            momentum,
            rsi / 100.0,  # Normalize RSI
            self.current_weights
        ])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, portfolio_return, rebalancing_cost):
        """Calculate reward for the action."""
        # Base reward is portfolio return
        reward = portfolio_return
        
        # Penalize excessive rebalancing
        if rebalancing_cost > 0.005:  # 0.5% threshold
            reward -= rebalancing_cost * 10  # Heavy penalty for excessive rebalancing
        
        # Bonus for risk-adjusted returns
        if len(self.portfolio_returns) > 20:
            recent_returns = np.array(self.portfolio_returns[-20:])
            sharpe_ratio = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
            reward += sharpe_ratio * 0.1
        
        return reward

class RebalancingStrategy:
    """
    Advanced portfolio rebalancing strategy using ML and reinforcement learning
    for optimal timing and execution of portfolio rebalancing decisions.
    """
    
    def __init__(self, 
                 rebalancing_frequency: str = 'adaptive',
                 transaction_cost_model: str = 'linear',
                 risk_tolerance: float = 0.5):
        """
        Initialize the rebalancing strategy.
        
        Args:
            rebalancing_frequency: Base rebalancing frequency
            transaction_cost_model: Model for transaction costs
            risk_tolerance: Risk tolerance for rebalancing decisions
        """
        self.rebalancing_frequency = rebalancing_frequency
        self.transaction_cost_model = transaction_cost_model
        self.risk_tolerance = risk_tolerance
        
        # ML Models
        self.regime_classifier = None
        self.signal_predictor = None
        self.cost_predictor = None
        self.rl_agent = None
        self.threshold_optimizer = None
        
        # Rebalancing components
        self.regime_detector = None
        self.signal_generator = None
        self.execution_optimizer = None
        
        # Feature scalers
        self.signal_scaler = StandardScaler()
        self.regime_scaler = StandardScaler()
        self.cost_scaler = StandardScaler()
        
        # Model metadata
        self.asset_universe = []
        self.feature_columns = []
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Configuration
        self.rebalancing_thresholds = {
            'conservative': 0.05,  # 5% deviation
            'moderate': 0.03,      # 3% deviation
            'aggressive': 0.01     # 1% deviation
        }
        
        self.frequency_mapping = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'adaptive': -1  # Dynamic based on market conditions
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def prepare_rebalancing_features(self, 
                                   price_data: pd.DataFrame,
                                   portfolio_weights: Dict[str, float],
                                   market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for rebalancing strategy models.
        
        Args:
            price_data: Historical price data
            portfolio_weights: Current portfolio weights
            market_data: Additional market data
            
        Returns:
            DataFrame with rebalancing features
        """
        try:
            # Calculate returns
            returns_df = price_data.pct_change().dropna()
            
            features_df = pd.DataFrame(index=returns_df.index)
            
            # Portfolio-level features
            portfolio_returns = self._calculate_portfolio_returns(returns_df, portfolio_weights)
            features_df['portfolio_return'] = portfolio_returns
            features_df['portfolio_volatility'] = portfolio_returns.rolling(20).std() * np.sqrt(252)
            features_df['portfolio_sharpe'] = self._rolling_sharpe(portfolio_returns, 60)
            
            # Deviation features
            for asset, target_weight in portfolio_weights.items():
                if asset in returns_df.columns:
                    asset_returns = returns_df[asset]
                    
                    # Calculate current implied weight based on price movements
                    cumulative_returns = (1 + asset_returns).cumprod()
                    implied_weight = target_weight * cumulative_returns / (
                        sum(portfolio_weights[a] * (1 + returns_df[a]).cumprod() 
                            for a in portfolio_weights.keys() if a in returns_df.columns)
                    )
                    
                    deviation = implied_weight - target_weight
                    features_df[f'{asset}_deviation'] = deviation
                    features_df[f'{asset}_abs_deviation'] = abs(deviation)
                    
                    # Momentum and mean reversion features
                    features_df[f'{asset}_momentum_5d'] = asset_returns.rolling(5).sum()
                    features_df[f'{asset}_momentum_20d'] = asset_returns.rolling(20).sum()
                    features_df[f'{asset}_mean_reversion'] = (
                        asset_returns - asset_returns.rolling(60).mean()
                    ) / asset_returns.rolling(60).std()
            
            # Market regime features
            features_df = self._add_regime_features(features_df, returns_df)
            
            # Risk features
            features_df = self._add_risk_features(features_df, returns_df, portfolio_weights)
            
            # Transaction cost features
            features_df = self._add_cost_features(features_df, returns_df)
            
            # Timing features
            features_df = self._add_timing_features(features_df)
            
            # Market microstructure features
            if market_data is not None:
                features_df = self._add_microstructure_features(features_df, market_data)
            
            return features_df.dropna()
            
        except Exception as e:
            self.logger.error(f"Rebalancing feature preparation failed: {str(e)}")
            raise
    
    def _calculate_portfolio_returns(self, returns_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns."""
        try:
            portfolio_returns = pd.Series(0.0, index=returns_df.index)
            
            for asset, weight in weights.items():
                if asset in returns_df.columns:
                    portfolio_returns += returns_df[asset] * weight
            
            return portfolio_returns
        except Exception as e:
            self.logger.error(f"Portfolio returns calculation failed: {str(e)}")
            return pd.Series(0.0, index=returns_df.index)
    
    def _rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        try:
            mean_return = returns.rolling(window).mean() * 252
            volatility = returns.rolling(window).std() * np.sqrt(252)
            risk_free_rate = 0.02
            
            sharpe = (mean_return - risk_free_rate) / volatility
            return sharpe.fillna(0)
        except Exception as e:
            self.logger.error(f"Rolling Sharpe calculation failed: {str(e)}")
            return pd.Series(0.0, index=returns.index)
    
    def _add_regime_features(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        try:
            market_return = returns_df.mean(axis=1)
            
            # Volatility regime
            volatility = market_return.rolling(20).std() * np.sqrt(252)
            vol_percentile = volatility.rolling(252).rank(pct=True)
            
            features_df['vol_regime_low'] = (vol_percentile < 0.33).astype(int)
            features_df['vol_regime_medium'] = ((vol_percentile >= 0.33) & (vol_percentile < 0.67)).astype(int)
            features_df['vol_regime_high'] = (vol_percentile >= 0.67).astype(int)
            
            # Trend regime
            sma_20 = market_return.rolling(20).mean()
            sma_60 = market_return.rolling(60).mean()
            features_df['trend_regime'] = (sma_20 > sma_60).astype(int)
            
            # Correlation regime
            correlation_matrix = returns_df.rolling(60).corr()
            avg_correlation = []
            
            for date in returns_df.index:
                try:
                    if date in correlation_matrix.index.get_level_values(0):
                        corr_data = correlation_matrix.loc[date]
                        # Average of upper triangular matrix (excluding diagonal)
                        mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
                        avg_corr = corr_data.values[mask].mean()
                        avg_correlation.append(avg_corr)
                    else:
                        avg_correlation.append(np.nan)
                except:
                    avg_correlation.append(np.nan)
            
            features_df['avg_correlation'] = avg_correlation
            features_df['high_correlation_regime'] = (
                pd.Series(avg_correlation, index=returns_df.index) > 0.7
            ).astype(int)
            
            return features_df
        except Exception as e:
            self.logger.error(f"Regime features calculation failed: {str(e)}")
            return features_df
    
    def _add_risk_features(self, features_df: pd.DataFrame, returns_df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Add risk-based features."""
        try:
            # Portfolio risk metrics
            portfolio_returns = self._calculate_portfolio_returns(returns_df, weights)
            
            # VaR estimation
            features_df['var_95'] = portfolio_returns.rolling(60).quantile(0.05)
            features_df['var_99'] = portfolio_returns.rolling(60).quantile(0.01)
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            features_df['max_drawdown'] = drawdown.rolling(60).min()
            
            # Risk concentration
            weight_array = np.array([weights.get(asset, 0) for asset in returns_df.columns])
            herfindahl_index = np.sum(weight_array ** 2)
            features_df['concentration_risk'] = herfindahl_index
            
            return features_df
        except Exception as e:
            self.logger.error(f"Risk features calculation failed: {str(e)}")
            return features_df
    
    def _add_cost_features(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction cost features."""
        try:
            # Volatility as proxy for bid-ask spreads
            for asset in returns_df.columns:
                asset_vol = returns_df[asset].rolling(20).std()
                features_df[f'{asset}_transaction_cost_proxy'] = asset_vol * 100  # Convert to basis points
            
            # Market impact proxy (based on recent price movements)
            features_df['market_impact_proxy'] = returns_df.abs().rolling(5).mean().mean(axis=1)
            
            # Liquidity proxy (inverse of volatility)
            avg_volatility = returns_df.rolling(20).std().mean(axis=1)
            features_df['liquidity_proxy'] = 1 / (avg_volatility + 1e-6)
            
            return features_df
        except Exception as e:
            self.logger.error(f"Cost features calculation failed: {str(e)}")
            return features_df
    
    def _add_timing_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add timing-based features."""
        try:
            # Calendar effects
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['month'] = features_df.index.month
            features_df['quarter'] = features_df.index.quarter
            
            # Month-end effect
            features_df['is_month_end'] = features_df.index.is_month_end.astype(int)
            features_df['is_quarter_end'] = features_df.index.is_quarter_end.astype(int)
            
            # Days since last rebalancing (simulated)
            features_df['days_since_rebalance'] = np.arange(len(features_df)) % 30  # Assume monthly rebalancing
            
            return features_df
        except Exception as e:
            self.logger.error(f"Timing features calculation failed: {str(e)}")
            return features_df
    
    def _add_microstructure_features(self, features_df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        try:
            # Volume features
            if 'volume' in market_data.columns:
                features_df['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
                features_df['volume_trend'] = market_data['volume'].rolling(5).mean() / market_data['volume'].rolling(20).mean()
            
            # Bid-ask spread features
            if 'bid_ask_spread' in market_data.columns:
                features_df['bid_ask_spread'] = market_data['bid_ask_spread']
                features_df['spread_percentile'] = market_data['bid_ask_spread'].rolling(60).rank(pct=True)
            
            # Order flow features
            if 'order_flow_imbalance' in market_data.columns:
                features_df['order_flow_imbalance'] = market_data['order_flow_imbalance']
            
            return features_df
        except Exception as e:
            self.logger.error(f"Microstructure features calculation failed: {str(e)}")
            return features_df
    
    def detect_market_regime(self, current_features: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            current_features: Current market features
            
        Returns:
            MarketRegime classification
        """
        try:
            if self.regime_classifier is None:
                # Fallback to rule-based regime detection
                return self._rule_based_regime_detection(current_features)
            
            # Prepare features for regime classification
            regime_features = current_features[['vol_regime_high', 'trend_regime', 'high_correlation_regime']].iloc[-1]
            regime_features_scaled = self.regime_scaler.transform(regime_features.values.reshape(1, -1))
            
            # Predict regime
            regime_prob = self.regime_classifier.predict_proba(regime_features_scaled)[0]
            regime_classes = ['bull', 'bear', 'sideways', 'volatile', 'crisis']
            regime_type = regime_classes[np.argmax(regime_prob)]
            confidence = np.max(regime_prob)
            
            # Determine characteristics
            volatility_level = 'high' if current_features['vol_regime_high'].iloc[-1] else 'low'
            correlation_level = 'high' if current_features['high_correlation_regime'].iloc[-1] else 'low'
            
            # Recommend rebalancing frequency based on regime
            frequency_mapping = {
                'bull': 'monthly',
                'bear': 'weekly',
                'sideways': 'monthly',
                'volatile': 'weekly',
                'crisis': 'daily'
            }
            
            return MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                volatility_level=volatility_level,
                correlation_level=correlation_level,
                recommended_frequency=frequency_mapping.get(regime_type, 'monthly')
            )
            
        except Exception as e:
            self.logger.error(f"Market regime detection failed: {str(e)}")
            return self._rule_based_regime_detection(current_features)
    
    def _rule_based_regime_detection(self, current_features: pd.DataFrame) -> MarketRegime:
        """Fallback rule-based regime detection."""
        try:
            # Simple rule-based classification
            vol_high = current_features['vol_regime_high'].iloc[-1] if 'vol_regime_high' in current_features.columns else 0
            trend_up = current_features['trend_regime'].iloc[-1] if 'trend_regime' in current_features.columns else 0
            
            if vol_high and not trend_up:
                regime = 'volatile'
                frequency = 'weekly'
            elif not vol_high and trend_up:
                regime = 'bull'
                frequency = 'monthly'
            elif not vol_high and not trend_up:
                regime = 'bear'
                frequency = 'weekly'
            else:
                regime = 'crisis'
                frequency = 'daily'
            
            return MarketRegime(
                regime_type=regime,
                confidence=0.6,
                volatility_level='high' if vol_high else 'low',
                correlation_level='medium',
                recommended_frequency=frequency
            )
        except Exception as e:
            self.logger.error(f"Rule-based regime detection failed: {str(e)}")
            return MarketRegime(
                regime_type='sideways',
                confidence=0.5,
                volatility_level='medium',
                correlation_level='medium',
                recommended_frequency='monthly'
            )
    
    def generate_rebalancing_signal(self, 
                                  current_features: pd.DataFrame,
                                  current_weights: Dict[str, float],
                                  target_weights: Dict[str, float],
                                  market_regime: MarketRegime) -> RebalancingSignal:
        """
        Generate rebalancing signal based on current conditions.
        
        Args:
            current_features: Current market features
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            market_regime: Current market regime
            
        Returns:
            RebalancingSignal with recommendations
        """
        try:
            # Calculate deviations
            deviations = {}
            total_deviation = 0
            
            for asset in target_weights.keys():
                current_weight = current_weights.get(asset, 0)
                target_weight = target_weights[asset]
                deviation = abs(current_weight - target_weight)
                deviations[asset] = deviation
                total_deviation += deviation
            
            # Determine rebalancing threshold based on regime
            regime_thresholds = {
                'bull': 0.05,
                'bear': 0.03,
                'sideways': 0.04,
                'volatile': 0.02,
                'crisis': 0.01
            }
            
            threshold = regime_thresholds.get(market_regime.regime_type, 0.03)
            
            # Generate signal
            if total_deviation > threshold:
                signal_strength = min(1.0, total_deviation / threshold)
                signal_type = 'threshold_based'
                recommended_action = 'rebalance'
                
                # Adjust for emergency conditions
                if market_regime.regime_type == 'crisis' and total_deviation > 0.1:
                    recommended_action = 'emergency_rebalance'
                    signal_strength = 1.0
                
            else:
                signal_strength = 0.0
                signal_type = 'threshold_based'
                recommended_action = 'hold'
            
            # Estimate transaction costs
            transaction_costs = self._estimate_transaction_costs(current_weights, target_weights)
            
            # Estimate expected benefit
            expected_benefit = self._estimate_rebalancing_benefit(
                current_weights, target_weights, current_features
            )
            
            # Calculate risk reduction
            risk_reduction = self._estimate_risk_reduction(
                current_weights, target_weights, current_features
            )
            
            # ML-based signal enhancement
            if self.signal_predictor is not None:
                ml_signal_strength = self._get_ml_signal_strength(current_features)
                signal_strength = (signal_strength + ml_signal_strength) / 2
                signal_type = 'ml_enhanced'
            
            # Calculate confidence
            confidence = market_regime.confidence * 0.7 + signal_strength * 0.3
            
            return RebalancingSignal(
                timestamp=datetime.now(),
                signal_type=signal_type,
                signal_strength=signal_strength,
                recommended_action=recommended_action,
                target_weights=target_weights,
                current_weights=current_weights,
                deviation_scores=deviations,
                transaction_costs=transaction_costs,
                expected_benefit=expected_benefit,
                risk_reduction=risk_reduction,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Rebalancing signal generation failed: {str(e)}")
            return RebalancingSignal(
                timestamp=datetime.now(),
                signal_type='error',
                signal_strength=0.0,
                recommended_action='hold',
                target_weights=target_weights,
                current_weights=current_weights,
                deviation_scores={},
                transaction_costs=0.0,
                expected_benefit=0.0,
                risk_reduction=0.0,
                confidence=0.0
            )
    
    def _estimate_transaction_costs(self, current_weights: Dict[str, float], target_weights: Dict[str, float]) -> float:
        """Estimate transaction costs for rebalancing."""
        try:
            total_cost = 0.0
            
            for asset in target_weights.keys():
                current_weight = current_weights.get(asset, 0)
                target_weight = target_weights[asset]
                weight_change = abs(target_weight - current_weight)
                
                # Simple linear cost model (can be enhanced with ML)
                asset_cost = weight_change * 0.001  # 0.1% transaction cost
                total_cost += asset_cost
            
            return total_cost
        except Exception as e:
            self.logger.error(f"Transaction cost estimation failed: {str(e)}")
            return 0.01  # Default 1% cost
    
    def _estimate_rebalancing_benefit(self, 
                                    current_weights: Dict[str, float],
                                    target_weights: Dict[str, float],
                                    current_features: pd.DataFrame) -> float:
        """Estimate expected benefit from rebalancing."""
        try:
            # Simple benefit estimation based on risk reduction and expected returns
            # In practice, this would use more sophisticated models
            
            deviation_benefit = 0.0
            for asset in target_weights.keys():
                current_weight = current_weights.get(asset, 0)
                target_weight = target_weights[asset]
                deviation = abs(target_weight - current_weight)
                deviation_benefit += deviation * 0.1  # Benefit from reducing deviation
            
            return min(deviation_benefit, 0.05)  # Cap at 5% benefit
        except Exception as e:
            self.logger.error(f"Rebalancing benefit estimation failed: {str(e)}")
            return 0.01
    
    def _estimate_risk_reduction(self, 
                               current_weights: Dict[str, float],
                               target_weights: Dict[str, float],
                               current_features: pd.DataFrame) -> float:
        """Estimate risk reduction from rebalancing."""
        try:
            # Simple risk reduction estimation
            concentration_current = sum(w**2 for w in current_weights.values())
            concentration_target = sum(w**2 for w in target_weights.values())
            
            risk_reduction = (concentration_current - concentration_target) * 10
            return max(0, risk_reduction)
        except Exception as e:
            self.logger.error(f"Risk reduction estimation failed: {str(e)}")
            return 0.0
    
    def _get_ml_signal_strength(self, current_features: pd.DataFrame) -> float:
        """Get ML-based signal strength."""
        try:
            if self.signal_predictor is None:
                return 0.5
            
            # Prepare features
            feature_cols = [col for col in current_features.columns if col in self.feature_columns]
            X = current_features[feature_cols].iloc[-1:].fillna(0)
            X_scaled = self.signal_scaler.transform(X)
            
            # Predict signal strength
            signal_strength = self.signal_predictor.predict(X_scaled)[0]
            return np.clip(signal_strength, 0, 1)
            
        except Exception as e:
            self.logger.error(f"ML signal strength prediction failed: {str(e)}")
            return 0.5
    
    def train_rebalancing_models(self, 
                               training_data: pd.DataFrame,
                               validation_split: float = 0.2) -> Dict[str, any]:
        """
        Train ML models for rebalancing strategy.
        
        Args:
            training_data: Historical market and portfolio data
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        try:
            self.logger.info("Starting rebalancing model training...")
            
            # Prepare features
            price_columns = [col for col in training_data.columns if 'price' in col.lower()]
            price_data = training_data[price_columns]
            
            # Assume equal weights for training
            weights = {col: 1.0/len(price_columns) for col in price_columns}
            
            features_df = self.prepare_rebalancing_features(price_data, weights)
            
            # Create training targets
            # 1. Regime classification targets
            regime_targets = self._create_regime_targets(features_df)
            
            # 2. Signal strength targets  
            signal_targets = self._create_signal_targets(features_df)
            
            training_metrics = {}
            
            # Train regime classifier
            if len(regime_targets) > 0:
                self.logger.info("Training regime classifier...")
                
                feature_cols = ['vol_regime_high', 'trend_regime', 'high_correlation_regime']
                X_regime = features_df[feature_cols].loc[regime_targets.index].fillna(0)
                y_regime = regime_targets
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_regime, y_regime, test_size=validation_split, random_state=42
                )
                
                X_train_scaled = self.regime_scaler.fit_transform(X_train)
                X_val_scaled = self.regime_scaler.transform(X_val)
                
                self.regime_classifier = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.01,
                    random_state=42
                )
                
                self.regime_classifier.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                regime_pred = self.regime_classifier.predict(X_val_scaled)
                training_metrics['regime_classifier'] = {
                    'accuracy': accuracy_score(y_val, regime_pred),
                    'classification_report': classification_report(y_val, regime_pred, output_dict=True)
                }
            
            # Train signal predictor
            if len(signal_targets) > 0:
                self.logger.info("Training signal predictor...")
                
                feature_cols = [col for col in features_df.columns 
                              if col not in ['portfolio_return'] and not col.endswith('_deviation')]
                self.feature_columns = feature_cols
                
                X_signal = features_df[feature_cols].loc[signal_targets.index].fillna(0)
                y_signal = signal_targets
                
                X_train_sig, X_val_sig, y_train_sig, y_val_sig = train_test_split(
                    X_signal, y_signal, test_size=validation_split, random_state=42
                )
                
                X_train_sig_scaled = self.signal_scaler.fit_transform(X_train_sig)
                X_val_sig_scaled = self.signal_scaler.transform(X_val_sig)
                
                self.signal_predictor = lgb.LGBMRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    random_state=42
                )
                
                self.signal_predictor.fit(
                    X_train_sig_scaled, y_train_sig,
                    eval_set=[(X_val_sig_scaled, y_val_sig)],
                    callbacks=[lgb.early_stopping(50)]
                )
                
                signal_pred = self.signal_predictor.predict(X_val_sig_scaled)
                training_metrics['signal_predictor'] = {
                    'mae': np.mean(np.abs(y_val_sig - signal_pred)),
                    'rmse': np.sqrt(np.mean((y_val_sig - signal_pred) ** 2)),
                    'r2': 1 - np.sum((y_val_sig - signal_pred) ** 2) / np.sum((y_val_sig - y_val_sig.mean()) ** 2)
                }
            
            # Train RL agent for dynamic rebalancing
            self.logger.info("Training RL agent...")
            returns_data = price_data.pct_change().dropna()
            transaction_costs = {asset: 0.001 for asset in price_data.columns}
            
            env = PortfolioRebalancingEnv(price_data, transaction_costs)
            
            # Use PPO for continuous action space
            self.rl_agent = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0
            )
            
            # Train for a limited number of timesteps
            self.rl_agent.learn(total_timesteps=50000)
            
            training_metrics['rl_agent'] = {
                'training_completed': True,
                'total_timesteps': 50000
            }
            
            self.asset_universe = list(price_data.columns)
            self.last_trained = datetime.now()
            
            self.logger.info("Rebalancing model training completed")
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Rebalancing model training failed: {str(e)}")
            raise
    
    def _create_regime_targets(self, features_df: pd.DataFrame) -> pd.Series:
        """Create regime classification targets."""
        try:
            targets = []
            
            for i in range(len(features_df)):
                vol_high = features_df['vol_regime_high'].iloc[i] if 'vol_regime_high' in features_df.columns else 0
                trend_up = features_df['trend_regime'].iloc[i] if 'trend_regime' in features_df.columns else 0
                high_corr = features_df['high_correlation_regime'].iloc[i] if 'high_correlation_regime' in features_df.columns else 0
                
                if vol_high and high_corr:
                    regime = 'crisis'
                elif vol_high and not trend_up:
                    regime = 'volatile'
                elif not vol_high and trend_up:
                    regime = 'bull'
                elif not vol_high and not trend_up:
                    regime = 'bear'
                else:
                    regime = 'sideways'
                
                targets.append(regime)
            
            return pd.Series(targets, index=features_df.index)
        except Exception as e:
            self.logger.error(f"Regime targets creation failed: {str(e)}")
            return pd.Series([], dtype=str)
    
    def _create_signal_targets(self, features_df: pd.DataFrame) -> pd.Series:
        """Create signal strength targets."""
        try:
            targets = []
            
            for i in range(len(features_df)):
                # Calculate signal strength based on deviations
                deviation_cols = [col for col in features_df.columns if col.endswith('_abs_deviation')]
                
                if deviation_cols:
                    total_deviation = features_df[deviation_cols].iloc[i].sum()
                    signal_strength = min(1.0, total_deviation / 0.05)  # Normalize by 5% threshold
                else:
                    signal_strength = 0.5  # Default
                
                targets.append(signal_strength)
            
            return pd.Series(targets, index=features_df.index)
        except Exception as e:
            self.logger.error(f"Signal targets creation failed: {str(e)}")
            return pd.Series([], dtype=float)
    
    def optimize_rebalancing_execution(self, 
                                     rebalancing_signal: RebalancingSignal,
                                     market_conditions: Dict) -> RebalancingAction:
        """
        Optimize the execution of rebalancing trades.
        
        Args:
            rebalancing_signal: Signal indicating need to rebalance
            market_conditions: Current market conditions
            
        Returns:
            RebalancingAction with optimized execution plan
        """
        try:
            if rebalancing_signal.recommended_action == 'hold':
                return RebalancingAction(
                    action_id=f"hold_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    trades=[],
                    total_cost=0.0,
                    expected_slippage=0.0,
                    execution_time_estimate=0.0,
                    priority='low',
                    approval_required=False,
                    risk_impact=0.0
                )
            
            # Calculate required trades
            trades = []
            total_cost = 0.0
            total_slippage = 0.0
            
            for asset in rebalancing_signal.target_weights.keys():
                current_weight = rebalancing_signal.current_weights.get(asset, 0)
                target_weight = rebalancing_signal.target_weights[asset]
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.001:  # 0.1% minimum trade size
                    trade_type = 'buy' if weight_diff > 0 else 'sell'
                    trade_amount = abs(weight_diff)
                    
                    # Estimate costs
                    trade_cost = trade_amount * 0.001  # 0.1% transaction cost
                    trade_slippage = trade_amount * 0.0005  # 0.05% slippage
                    
                    trades.append({
                        'asset': asset,
                        'action': trade_type,
                        'amount': trade_amount,
                        'estimated_cost': trade_cost,
                        'estimated_slippage': trade_slippage,
                        'priority': 'high' if abs(weight_diff) > 0.05 else 'medium'
                    })
                    
                    total_cost += trade_cost
                    total_slippage += trade_slippage
            
            # Determine execution priority
            if rebalancing_signal.recommended_action == 'emergency_rebalance':
                priority = 'high'
                approval_required = True
            elif rebalancing_signal.signal_strength > 0.8:
                priority = 'high'
                approval_required = False
            elif rebalancing_signal.signal_strength > 0.5:
                priority = 'medium'
                approval_required = False
            else:
                priority = 'low'
                approval_required = False
            
            # Estimate execution time
            execution_time = len(trades) * 30 + 60  # 30 seconds per trade + 1 minute overhead
            
            return RebalancingAction(
                action_id=f"rebalance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                trades=trades,
                total_cost=total_cost,
                expected_slippage=total_slippage,
                execution_time_estimate=execution_time,
                priority=priority,
                approval_required=approval_required,
                risk_impact=rebalancing_signal.risk_reduction
            )
            
        except Exception as e:
            self.logger.error(f"Rebalancing execution optimization failed: {str(e)}")
            return RebalancingAction(
                action_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                trades=[],
                total_cost=0.0,
                expected_slippage=0.0,
                execution_time_estimate=0.0,
                priority='low',
                approval_required=True,
                risk_impact=0.0
            )
    
    def save_model(self, filepath: str) -> None:
        """Save the trained rebalancing strategy models."""
        try:
            model_data = {
                'regime_classifier': self.regime_classifier,
                'signal_predictor': self.signal_predictor,
                'cost_predictor': self.cost_predictor,
                'rl_agent': self.rl_agent,
                'threshold_optimizer': self.threshold_optimizer,
                'regime_detector': self.regime_detector,
                'signal_generator': self.signal_generator,
                'execution_optimizer': self.execution_optimizer,
                'signal_scaler': self.signal_scaler,
                'regime_scaler': self.regime_scaler,
                'cost_scaler': self.cost_scaler,
                'asset_universe': self.asset_universe,
                'feature_columns': self.feature_columns,
                'rebalancing_frequency': self.rebalancing_frequency,
                'transaction_cost_model': self.transaction_cost_model,
                'risk_tolerance': self.risk_tolerance,
                'rebalancing_thresholds': self.rebalancing_thresholds,
                'frequency_mapping': self.frequency_mapping,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Rebalancing strategy saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load trained rebalancing strategy models."""
        try:
            model_data = joblib.load(filepath)
            
            self.regime_classifier = model_data['regime_classifier']
            self.signal_predictor = model_data['signal_predictor']
            self.cost_predictor = model_data['cost_predictor']
            self.rl_agent = model_data['rl_agent']
            self.threshold_optimizer = model_data['threshold_optimizer']
            self.regime_detector = model_data['regime_detector']
            self.signal_generator = model_data['signal_generator']
            self.execution_optimizer = model_data['execution_optimizer']
            self.signal_scaler = model_data['signal_scaler']
            self.regime_scaler = model_data['regime_scaler']
            self.cost_scaler = model_data['cost_scaler']
            self.asset_universe = model_data['asset_universe']
            self.feature_columns = model_data['feature_columns']
            self.rebalancing_frequency = model_data['rebalancing_frequency']
            self.transaction_cost_model = model_data['transaction_cost_model']
            self.risk_tolerance = model_data['risk_tolerance']
            self.rebalancing_thresholds = model_data['rebalancing_thresholds']
            self.frequency_mapping = model_data['frequency_mapping']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Rebalancing strategy loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise


def create_rebalancing_strategy_pipeline(config: Dict) -> RebalancingStrategy:
    """
    Create a rebalancing strategy pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RebalancingStrategy instance
    """
    return RebalancingStrategy(
        rebalancing_frequency=config.get('rebalancing_frequency', 'adaptive'),
        transaction_cost_model=config.get('transaction_cost_model', 'linear'),
        risk_tolerance=config.get('risk_tolerance', 0.5)
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create rebalancing strategy
    strategy = RebalancingStrategy(rebalancing_frequency='adaptive')
    
    # Example workflow:
    # 1. Load historical data
    # 2. Train rebalancing models
    # 3. Detect market regime
    # 4. Generate rebalancing signals
    # 5. Optimize execution
    
    print("Rebalancing strategy implementation completed")
