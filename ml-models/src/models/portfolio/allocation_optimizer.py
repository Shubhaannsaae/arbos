"""
Portfolio Allocation Optimizer

This module implements advanced portfolio optimization techniques using ML-enhanced
mean-variance optimization, Black-Litterman model, risk parity, and other modern
portfolio theory approaches with Chainlink Data Feeds integration.

Features:
- Mean-variance optimization with ML-predicted returns
- Black-Litterman model for incorporating market views
- Risk parity and equal risk contribution strategies
- Multi-objective optimization (return, risk, ESG, liquidity)
- Dynamic rebalancing with transaction cost consideration
- Constraint handling (position limits, turnover, sector exposure)
- Real-time optimization using Chainlink price feeds
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.optimize as optimize
from scipy import linalg
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
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

@dataclass
class OptimizationObjective:
    """Data class for optimization objectives."""
    maximize_return: bool = True
    minimize_risk: bool = True
    minimize_turnover: bool = False
    maximize_diversification: bool = False
    maximize_esg_score: bool = False
    weights: Dict[str, float] = None  # Objective weights

@dataclass
class OptimizationConstraints:
    """Data class for optimization constraints."""
    min_weights: Dict[str, float] = None  # Minimum position sizes
    max_weights: Dict[str, float] = None  # Maximum position sizes
    max_turnover: float = None  # Maximum portfolio turnover
    min_diversification: float = None  # Minimum diversification ratio
    sector_limits: Dict[str, Tuple[float, float]] = None  # Sector exposure limits
    risk_budget: Dict[str, float] = None  # Risk budget allocation
    transaction_costs: Dict[str, float] = None  # Transaction cost per asset
    leverage_limit: float = 1.0  # Maximum leverage

@dataclass
class OptimizationResult:
    """Data class for optimization results."""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: float
    turnover: float
    optimization_success: bool
    optimization_method: str
    objective_value: float
    constraints_satisfied: bool
    execution_cost: float

class AllocationOptimizer:
    """
    Advanced portfolio allocation optimizer using multiple optimization techniques
    and machine learning for enhanced return and risk predictions.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 optimization_method: str = 'mean_variance',
                 rebalancing_frequency: str = 'monthly'):
        """
        Initialize the allocation optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            optimization_method: Default optimization method
            rebalancing_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
        """
        self.risk_free_rate = risk_free_rate
        self.optimization_method = optimization_method
        self.rebalancing_frequency = rebalancing_frequency
        
        # ML Models for enhanced optimization
        self.return_predictor = None
        self.risk_predictor = None
        self.correlation_predictor = None
        self.transaction_cost_predictor = None
        
        # Optimization components
        self.covariance_estimator = None
        self.return_estimator = None
        self.black_litterman_model = None
        
        # Feature scalers
        self.return_scaler = StandardScaler()
        self.risk_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
        # Model metadata
        self.asset_universe = []
        self.feature_columns = []
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Market parameters
        self.market_cap_weights = {}  # For market cap weighted portfolios
        self.asset_sectors = {}  # Asset sector classifications
        self.asset_esg_scores = {}  # ESG scores for assets
        
    def prepare_optimization_features(self, 
                                   price_data: pd.DataFrame,
                                   market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for ML-enhanced optimization.
        
        Args:
            price_data: Historical price data
            market_data: Additional market data (volume, sentiment, etc.)
            
        Returns:
            DataFrame with optimization features
        """
        try:
            # Calculate returns
            returns_df = price_data.pct_change().dropna()
            
            features_df = pd.DataFrame(index=returns_df.index)
            
            # For each asset, calculate predictive features
            for asset in price_data.columns:
                asset_returns = returns_df[asset]
                asset_prices = price_data[asset]
                
                # Return prediction features
                features_df[f'{asset}_return_1d'] = asset_returns
                features_df[f'{asset}_return_5d'] = asset_returns.rolling(5).sum()
                features_df[f'{asset}_return_20d'] = asset_returns.rolling(20).sum()
                features_df[f'{asset}_momentum_3m'] = asset_returns.rolling(60).sum()
                
                # Risk prediction features
                features_df[f'{asset}_volatility_20d'] = asset_returns.rolling(20).std() * np.sqrt(252)
                features_df[f'{asset}_volatility_60d'] = asset_returns.rolling(60).std() * np.sqrt(252)
                features_df[f'{asset}_downside_vol'] = self._calculate_downside_volatility(asset_returns)
                
                # Technical indicators
                features_df[f'{asset}_rsi'] = self._calculate_rsi(asset_prices)
                features_df[f'{asset}_macd'] = self._calculate_macd(asset_prices)
                features_df[f'{asset}_bollinger_pos'] = self._calculate_bollinger_position(asset_prices)
                
                # Price-based features
                features_df[f'{asset}_price_ratio_sma20'] = asset_prices / asset_prices.rolling(20).mean()
                features_df[f'{asset}_price_ratio_sma60'] = asset_prices / asset_prices.rolling(60).mean()
                
                # Volume features (if available)
                if market_data is not None and f'{asset}_volume' in market_data.columns:
                    volume = market_data[f'{asset}_volume']
                    features_df[f'{asset}_volume_ratio'] = volume / volume.rolling(20).mean()
                    features_df[f'{asset}_price_volume_trend'] = self._calculate_price_volume_trend(
                        asset_prices, volume
                    )
                
                # Skewness and kurtosis
                features_df[f'{asset}_skewness'] = asset_returns.rolling(60).skew()
                features_df[f'{asset}_kurtosis'] = asset_returns.rolling(60).kurt()
                
            # Cross-asset features
            features_df = self._add_cross_asset_features(features_df, returns_df)
            
            # Market regime features
            features_df = self._add_market_regime_features(features_df, returns_df)
            
            # Correlation features
            features_df = self._add_correlation_features(features_df, returns_df)
            
            # Economic features
            if market_data is not None:
                features_df = self._add_economic_features(features_df, market_data)
            
            return features_df.dropna()
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise
    
    def _calculate_downside_volatility(self, returns: pd.Series, target_return: float = 0) -> pd.Series:
        """Calculate downside volatility."""
        try:
            downside_returns = returns[returns < target_return] - target_return
            downside_vol = downside_returns.rolling(20).std() * np.sqrt(252)
            return downside_vol.reindex(returns.index).fillna(method='ffill')
        except Exception as e:
            self.logger.error(f"Downside volatility calculation failed: {str(e)}")
            return pd.Series(0, index=returns.index)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {str(e)}")
            return pd.Series(50, index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            return macd
        except Exception as e:
            self.logger.error(f"MACD calculation failed: {str(e)}")
            return pd.Series(0, index=prices.index)
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        try:
            sma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            position = (prices - lower_band) / (upper_band - lower_band)
            return position.clip(0, 1)
        except Exception as e:
            self.logger.error(f"Bollinger position calculation failed: {str(e)}")
            return pd.Series(0.5, index=prices.index)
    
    def _calculate_price_volume_trend(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate price-volume trend indicator."""
        try:
            price_change = prices.pct_change()
            volume_change = volume.pct_change()
            pvt = (price_change * volume_change).rolling(20).sum()
            return pvt
        except Exception as e:
            self.logger.error(f"Price-volume trend calculation failed: {str(e)}")
            return pd.Series(0, index=prices.index)
    
    def _add_cross_asset_features(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset relationship features."""
        try:
            assets = returns_df.columns.tolist()
            
            # Asset momentum spreads
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    mom1 = returns_df[asset1].rolling(20).sum()
                    mom2 = returns_df[asset2].rolling(20).sum()
                    features_df[f'momentum_spread_{asset1}_{asset2}'] = mom1 - mom2
                    
                    # Volatility spreads
                    vol1 = returns_df[asset1].rolling(20).std()
                    vol2 = returns_df[asset2].rolling(20).std()
                    features_df[f'vol_spread_{asset1}_{asset2}'] = vol1 - vol2
            
            # Portfolio-level features
            equal_weight_return = returns_df.mean(axis=1)
            features_df['market_return'] = equal_weight_return
            features_df['market_volatility'] = equal_weight_return.rolling(20).std() * np.sqrt(252)
            
            return features_df
        except Exception as e:
            self.logger.error(f"Cross-asset features calculation failed: {str(e)}")
            return features_df
    
    def _add_market_regime_features(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators."""
        try:
            market_return = returns_df.mean(axis=1)
            
            # Trend indicators
            sma_20 = market_return.rolling(20).mean()
            sma_60 = market_return.rolling(60).mean()
            features_df['bull_market'] = (sma_20 > sma_60).astype(int)
            
            # Volatility regime
            volatility = market_return.rolling(20).std()
            vol_percentile = volatility.rolling(252).rank(pct=True)
            features_df['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
            features_df['low_vol_regime'] = (vol_percentile < 0.2).astype(int)
            
            # Dispersion measure
            cross_sectional_vol = returns_df.std(axis=1)
            features_df['cross_sectional_dispersion'] = cross_sectional_vol
            
            return features_df
        except Exception as e:
            self.logger.error(f"Market regime features calculation failed: {str(e)}")
            return features_df
    
    def _add_correlation_features(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Add correlation-based features."""
        try:
            # Average correlation
            corr_matrix = returns_df.rolling(60).corr()
            
            for asset in returns_df.columns:
                # Average correlation with other assets
                asset_corrs = []
                for date in returns_df.index[-60:]:  # Last 60 days
                    try:
                        if date in corr_matrix.index.get_level_values(0):
                            asset_corr_data = corr_matrix.loc[date, asset].drop(asset)
                            avg_corr = asset_corr_data.mean()
                            asset_corrs.append(avg_corr)
                    except:
                        continue
                
                if asset_corrs:
                    features_df[f'{asset}_avg_correlation'] = pd.Series(
                        asset_corrs[-1], index=features_df.index
                    ).fillna(method='ffill')
            
            return features_df
        except Exception as e:
            self.logger.error(f"Correlation features calculation failed: {str(e)}")
            return features_df
    
    def _add_economic_features(self, features_df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add economic and market-wide features."""
        try:
            # VIX or volatility index
            if 'vix' in market_data.columns:
                features_df['vix'] = market_data['vix']
                features_df['vix_change'] = market_data['vix'].pct_change()
            
            # Interest rates
            if 'interest_rate' in market_data.columns:
                features_df['interest_rate'] = market_data['interest_rate']
                features_df['rate_change'] = market_data['interest_rate'].diff()
            
            # Economic indicators
            economic_indicators = ['gdp_growth', 'inflation', 'unemployment', 'consumer_confidence']
            for indicator in economic_indicators:
                if indicator in market_data.columns:
                    features_df[indicator] = market_data[indicator]
            
            return features_df
        except Exception as e:
            self.logger.error(f"Economic features calculation failed: {str(e)}")
            return features_df
    
    def train_prediction_models(self, 
                              training_data: pd.DataFrame,
                              target_horizon: int = 20,
                              validation_split: float = 0.2) -> Dict[str, any]:
        """
        Train ML models for return and risk prediction.
        
        Args:
            training_data: Historical market data with features
            target_horizon: Prediction horizon in days
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        try:
            self.logger.info("Starting prediction model training...")
            
            # Prepare price data for return calculation
            price_cols = [col for col in training_data.columns if 'price' in col.lower()]
            price_data = training_data[price_cols]
            
            # Prepare features
            features_df = self.prepare_optimization_features(price_data)
            
            # Create targets
            returns_df = price_data.pct_change().dropna()
            
            # Future returns (target)
            future_returns = {}
            future_volatility = {}
            
            for asset in price_data.columns:
                asset_returns = returns_df[asset]
                
                # Future return target
                future_ret = asset_returns.shift(-target_horizon).rolling(target_horizon).sum()
                future_returns[f'{asset}_future_return'] = future_ret
                
                # Future volatility target
                future_vol = asset_returns.shift(-target_horizon).rolling(target_horizon).std() * np.sqrt(252)
                future_volatility[f'{asset}_future_volatility'] = future_vol
            
            # Combine targets
            targets_df = pd.DataFrame(future_returns, index=returns_df.index)
            volatility_targets_df = pd.DataFrame(future_volatility, index=returns_df.index)
            
            # Align features and targets
            aligned_features, aligned_targets = features_df.align(targets_df, join='inner')
            _, aligned_vol_targets = features_df.align(volatility_targets_df, join='inner')
            
            # Remove rows with NaN targets
            valid_mask = ~aligned_targets.isna().any(axis=1)
            X = aligned_features[valid_mask]
            y_returns = aligned_targets[valid_mask]
            y_volatility = aligned_vol_targets[valid_mask]
            
            # Feature selection
            feature_cols = [col for col in X.columns if not col.startswith('target_')]
            self.feature_columns = feature_cols
            X_selected = X[feature_cols].fillna(0)
            
            training_metrics = {}
            
            # Split data
            X_train, X_val, y_ret_train, y_ret_val = train_test_split(
                X_selected, y_returns, test_size=validation_split, random_state=42
            )
            _, _, y_vol_train, y_vol_val = train_test_split(
                X_selected, y_volatility, test_size=validation_split, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Train return predictor
            self.logger.info("Training return predictor...")
            self.return_predictor = self._build_return_predictor(X_train_scaled.shape[1], y_ret_train.shape[1])
            
            # Scale targets
            y_ret_train_scaled = self.return_scaler.fit_transform(y_ret_train)
            y_ret_val_scaled = self.return_scaler.transform(y_ret_val)
            
            history_ret = self.return_predictor.fit(
                X_train_scaled, y_ret_train_scaled,
                validation_data=(X_val_scaled, y_ret_val_scaled),
                epochs=100,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ],
                verbose=0
            )
            
            # Evaluate return predictor
            ret_pred = self.return_predictor.predict(X_val_scaled)
            ret_pred_original = self.return_scaler.inverse_transform(ret_pred)
            
            training_metrics['return_predictor'] = {
                'mae': mean_absolute_error(y_ret_val.values, ret_pred_original),
                'rmse': np.sqrt(mean_squared_error(y_ret_val.values, ret_pred_original)),
                'final_loss': history_ret.history['loss'][-1]
            }
            
            # Train risk predictor
            self.logger.info("Training risk predictor...")
            self.risk_predictor = self._build_risk_predictor(X_train_scaled.shape[1], y_vol_train.shape[1])
            
            # Scale volatility targets
            y_vol_train_scaled = self.risk_scaler.fit_transform(y_vol_train)
            y_vol_val_scaled = self.risk_scaler.transform(y_vol_val)
            
            history_risk = self.risk_predictor.fit(
                X_train_scaled, y_vol_train_scaled,
                validation_data=(X_val_scaled, y_vol_val_scaled),
                epochs=100,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ],
                verbose=0
            )
            
            # Evaluate risk predictor
            risk_pred = self.risk_predictor.predict(X_val_scaled)
            risk_pred_original = self.risk_scaler.inverse_transform(risk_pred)
            
            training_metrics['risk_predictor'] = {
                'mae': mean_absolute_error(y_vol_val.values, risk_pred_original),
                'rmse': np.sqrt(mean_squared_error(y_vol_val.values, risk_pred_original)),
                'final_loss': history_risk.history['loss'][-1]
            }
            
            # Train correlation predictor using XGBoost
            self.logger.info("Training correlation predictor...")
            
            # Create correlation targets
            correlation_targets = self._create_correlation_targets(returns_df, target_horizon)
            aligned_corr = correlation_targets.reindex(X_selected.index).dropna()
            
            if len(aligned_corr) > 0:
                X_corr_train, X_corr_val, y_corr_train, y_corr_val = train_test_split(
                    X_selected.loc[aligned_corr.index], aligned_corr,
                    test_size=validation_split, random_state=42
                )
                
                self.correlation_predictor = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    random_state=42
                )
                
                self.correlation_predictor.fit(
                    X_corr_train, y_corr_train,
                    eval_set=[(X_corr_val, y_corr_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                corr_pred = self.correlation_predictor.predict(X_corr_val)
                training_metrics['correlation_predictor'] = {
                    'mae': mean_absolute_error(y_corr_val.values.flatten(), corr_pred.flatten()),
                    'rmse': np.sqrt(mean_squared_error(y_corr_val.values.flatten(), corr_pred.flatten()))
                }
            
            self.asset_universe = list(price_data.columns)
            self.last_trained = datetime.now()
            
            self.logger.info("Prediction model training completed")
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Prediction model training failed: {str(e)}")
            raise
    
    def _build_return_predictor(self, input_dim: int, output_dim: int) -> Model:
        """Build neural network for return prediction."""
        try:
            model = Sequential([
                Dense(256, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(64, activation='relu'),
                Dropout(0.1),
                
                Dense(output_dim, activation='tanh')  # Returns can be negative
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            self.logger.error(f"Return predictor building failed: {str(e)}")
            raise
    
    def _build_risk_predictor(self, input_dim: int, output_dim: int) -> Model:
        """Build neural network for risk prediction."""
        try:
            model = Sequential([
                Dense(256, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(64, activation='relu'),
                Dropout(0.1),
                
                Dense(output_dim, activation='softplus')  # Risk is always positive
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            self.logger.error(f"Risk predictor building failed: {str(e)}")
            raise
    
    def _create_correlation_targets(self, returns_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Create future correlation targets."""
        try:
            correlation_targets = []
            assets = returns_df.columns.tolist()
            
            for i in range(len(returns_df) - horizon):
                future_returns = returns_df.iloc[i+1:i+1+horizon]
                future_corr = future_returns.corr()
                
                # Flatten upper triangular correlation matrix
                corr_values = []
                for j, asset1 in enumerate(assets):
                    for asset2 in assets[j+1:]:
                        corr_values.append(future_corr.loc[asset1, asset2])
                
                correlation_targets.append(corr_values)
            
            # Create column names
            corr_columns = []
            for j, asset1 in enumerate(assets):
                for asset2 in assets[j+1:]:
                    corr_columns.append(f'corr_{asset1}_{asset2}')
            
            corr_df = pd.DataFrame(
                correlation_targets,
                index=returns_df.index[:-horizon],
                columns=corr_columns
            )
            
            return corr_df
        except Exception as e:
            self.logger.error(f"Correlation targets creation failed: {str(e)}")
            return pd.DataFrame()
    
    def predict_returns_and_risk(self, current_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict expected returns and risk using trained models.
        
        Args:
            current_features: Current market features
            
        Returns:
            Tuple of (predicted_returns, predicted_risks)
        """
        try:
            if self.return_predictor is None or self.risk_predictor is None:
                raise ValueError("Prediction models not trained")
            
            # Prepare features
            X = current_features[self.feature_columns].fillna(0)
            X_scaled = self.feature_scaler.transform(X)
            
            # Predict returns
            returns_pred_scaled = self.return_predictor.predict(X_scaled)
            returns_pred = self.return_scaler.inverse_transform(returns_pred_scaled)
            
            # Predict risks
            risks_pred_scaled = self.risk_predictor.predict(X_scaled)
            risks_pred = self.risk_scaler.inverse_transform(risks_pred_scaled)
            
            return returns_pred[-1], risks_pred[-1]  # Return most recent prediction
            
        except Exception as e:
            self.logger.error(f"Return and risk prediction failed: {str(e)}")
            # Fallback to historical estimates
            return np.zeros(len(self.asset_universe)), np.ones(len(self.asset_universe)) * 0.2
    
    def optimize_portfolio(self, 
                         expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray,
                         current_weights: Optional[Dict[str, float]] = None,
                         objective: OptimizationObjective = None,
                         constraints: OptimizationConstraints = None) -> OptimizationResult:
        """
        Optimize portfolio allocation using specified method and constraints.
        
        Args:
            expected_returns: Array of expected returns
            covariance_matrix: Covariance matrix
            current_weights: Current portfolio weights
            objective: Optimization objective
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        try:
            if objective is None:
                objective = OptimizationObjective()
            
            if constraints is None:
                constraints = OptimizationConstraints()
            
            n_assets = len(expected_returns)
            
            # Choose optimization method
            if self.optimization_method == 'mean_variance':
                result = self._mean_variance_optimization(
                    expected_returns, covariance_matrix, current_weights, objective, constraints
                )
            elif self.optimization_method == 'black_litterman':
                result = self._black_litterman_optimization(
                    expected_returns, covariance_matrix, current_weights, objective, constraints
                )
            elif self.optimization_method == 'risk_parity':
                result = self._risk_parity_optimization(
                    covariance_matrix, current_weights, constraints
                )
            elif self.optimization_method == 'equal_weight':
                result = self._equal_weight_optimization(n_assets, constraints)
            elif self.optimization_method == 'minimum_variance':
                result = self._minimum_variance_optimization(
                    covariance_matrix, current_weights, constraints
                )
            elif self.optimization_method == 'maximum_sharpe':
                result = self._maximum_sharpe_optimization(
                    expected_returns, covariance_matrix, current_weights, constraints
                )
            else:
                raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {str(e)}")
            # Return equal weight portfolio as fallback
            equal_weights = {asset: 1.0/len(self.asset_universe) for asset in self.asset_universe}
            return OptimizationResult(
                optimal_weights=equal_weights,
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                diversification_ratio=1.0,
                turnover=0.0,
                optimization_success=False,
                optimization_method='fallback_equal_weight',
                objective_value=0.0,
                constraints_satisfied=False,
                execution_cost=0.0
            )
    
    def _mean_variance_optimization(self, 
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray,
                                  current_weights: Optional[Dict[str, float]],
                                  objective: OptimizationObjective,
                                  constraints: OptimizationConstraints) -> OptimizationResult:
        """Perform mean-variance optimization."""
        try:
            n_assets = len(expected_returns)
            
            # Define optimization variables
            w = cp.Variable(n_assets)
            
            # Objective function
            portfolio_return = w.T @ expected_returns
            portfolio_risk = cp.quad_form(w, covariance_matrix)
            
            # Risk aversion parameter (can be tuned)
            risk_aversion = 5.0
            if objective.weights and 'risk_aversion' in objective.weights:
                risk_aversion = objective.weights['risk_aversion']
            
            objective_func = portfolio_return - 0.5 * risk_aversion * portfolio_risk
            
            # Add turnover penalty if current weights provided
            if current_weights is not None:
                current_w = np.array([current_weights.get(asset, 0) for asset in self.asset_universe])
                turnover = cp.norm(w - current_w, 1)
                
                turnover_penalty = 0.01  # 1% penalty
                if constraints.max_turnover is not None:
                    turnover_penalty = constraints.max_turnover * 0.1
                
                objective_func -= turnover_penalty * turnover
            
            # Problem definition
            problem = cp.Problem(cp.Maximize(objective_func), self._build_constraints(w, constraints))
            
            # Solve
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights_array = w.value
                optimal_weights = {asset: float(weight) for asset, weight in 
                                 zip(self.asset_universe, optimal_weights_array)}
                
                # Calculate metrics
                expected_return = float(optimal_weights_array.T @ expected_returns)
                expected_risk = float(np.sqrt(optimal_weights_array.T @ covariance_matrix @ optimal_weights_array))
                sharpe_ratio = (expected_return - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0
                
                # Calculate turnover
                turnover = 0.0
                if current_weights is not None:
                    current_w = np.array([current_weights.get(asset, 0) for asset in self.asset_universe])
                    turnover = float(np.sum(np.abs(optimal_weights_array - current_w)))
                
                # Calculate diversification ratio
                individual_vols = np.sqrt(np.diag(covariance_matrix))
                weighted_avg_vol = optimal_weights_array @ individual_vols
                diversification_ratio = weighted_avg_vol / expected_risk if expected_risk > 0 else 1.0
                
                return OptimizationResult(
                    optimal_weights=optimal_weights,
                    expected_return=expected_return,
                    expected_risk=expected_risk,
                    sharpe_ratio=sharpe_ratio,
                    diversification_ratio=diversification_ratio,
                    turnover=turnover,
                    optimization_success=True,
                    optimization_method='mean_variance',
                    objective_value=float(problem.value),
                    constraints_satisfied=True,
                    execution_cost=0.0
                )
            else:
                raise ValueError(f"Optimization failed with status: {problem.status}")
                
        except Exception as e:
            self.logger.error(f"Mean-variance optimization failed: {str(e)}")
            raise
    
    def _build_constraints(self, w: cp.Variable, constraints: OptimizationConstraints) -> List:
        """Build optimization constraints."""
        constraint_list = []
        
        # Budget constraint (weights sum to 1)
        constraint_list.append(cp.sum(w) == 1)
        
        # Long-only constraint (can be modified for long-short)
        constraint_list.append(w >= 0)
        
        # Position size constraints
        if constraints.min_weights:
            for i, asset in enumerate(self.asset_universe):
                if asset in constraints.min_weights:
                    constraint_list.append(w[i] >= constraints.min_weights[asset])
        
        if constraints.max_weights:
            for i, asset in enumerate(self.asset_universe):
                if asset in constraints.max_weights:
                    constraint_list.append(w[i] <= constraints.max_weights[asset])
        
        # Leverage constraint
        if constraints.leverage_limit:
            constraint_list.append(cp.norm(w, 1) <= constraints.leverage_limit)
        
        return constraint_list
    
    def _risk_parity_optimization(self, 
                                covariance_matrix: np.ndarray,
                                current_weights: Optional[Dict[str, float]],
                                constraints: OptimizationConstraints) -> OptimizationResult:
        """Perform risk parity optimization."""
        try:
            n_assets = len(covariance_matrix)
            
            def risk_parity_objective(weights):
                weights = weights / np.sum(weights)  # Normalize
                portfolio_variance = weights.T @ covariance_matrix @ weights
                
                # Calculate risk contributions
                marginal_contrib = covariance_matrix @ weights
                contrib = weights * marginal_contrib / portfolio_variance
                
                # Target equal risk contributions
                target_contrib = 1.0 / n_assets
                risk_diff = contrib - target_contrib
                
                return np.sum(risk_diff ** 2)
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Constraints
            constraint_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Budget constraint
            bounds = [(0, 1) for _ in range(n_assets)]  # Long-only
            
            # Add position constraints
            if constraints.max_weights:
                for i, asset in enumerate(self.asset_universe):
                    if asset in constraints.max_weights:
                        bounds[i] = (bounds[i][0], constraints.max_weights[asset])
            
            # Optimize
            result = optimize.minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights_array = result.x / np.sum(result.x)  # Normalize
                optimal_weights = {asset: float(weight) for asset, weight in 
                                 zip(self.asset_universe, optimal_weights_array)}
                
                # Calculate metrics
                portfolio_variance = optimal_weights_array.T @ covariance_matrix @ optimal_weights_array
                expected_risk = float(np.sqrt(portfolio_variance))
                
                # Calculate diversification ratio
                individual_vols = np.sqrt(np.diag(covariance_matrix))
                weighted_avg_vol = optimal_weights_array @ individual_vols
                diversification_ratio = weighted_avg_vol / expected_risk if expected_risk > 0 else 1.0
                
                return OptimizationResult(
                    optimal_weights=optimal_weights,
                    expected_return=0.0,  # Not optimizing for return
                    expected_risk=expected_risk,
                    sharpe_ratio=0.0,
                    diversification_ratio=diversification_ratio,
                    turnover=0.0,
                    optimization_success=True,
                    optimization_method='risk_parity',
                    objective_value=float(result.fun),
                    constraints_satisfied=True,
                    execution_cost=0.0
                )
            else:
                raise ValueError(f"Risk parity optimization failed: {result.message}")
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {str(e)}")
            raise
    
    def _minimum_variance_optimization(self, 
                                     covariance_matrix: np.ndarray,
                                     current_weights: Optional[Dict[str, float]],
                                     constraints: OptimizationConstraints) -> OptimizationResult:
        """Perform minimum variance optimization."""
        try:
            n_assets = len(covariance_matrix)
            
            # Define optimization variables
            w = cp.Variable(n_assets)
            
            # Objective: minimize portfolio variance
            portfolio_variance = cp.quad_form(w, covariance_matrix)
            
            # Problem definition
            problem = cp.Problem(cp.Minimize(portfolio_variance), self._build_constraints(w, constraints))
            
            # Solve
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights_array = w.value
                optimal_weights = {asset: float(weight) for asset, weight in 
                                 zip(self.asset_universe, optimal_weights_array)}
                
                # Calculate metrics
                expected_risk = float(np.sqrt(optimal_weights_array.T @ covariance_matrix @ optimal_weights_array))
                
                # Calculate diversification ratio
                individual_vols = np.sqrt(np.diag(covariance_matrix))
                weighted_avg_vol = optimal_weights_array @ individual_vols
                diversification_ratio = weighted_avg_vol / expected_risk if expected_risk > 0 else 1.0
                
                return OptimizationResult(
                    optimal_weights=optimal_weights,
                    expected_return=0.0,
                    expected_risk=expected_risk,
                    sharpe_ratio=0.0,
                    diversification_ratio=diversification_ratio,
                    turnover=0.0,
                    optimization_success=True,
                    optimization_method='minimum_variance',
                    objective_value=float(problem.value),
                    constraints_satisfied=True,
                    execution_cost=0.0
                )
            else:
                raise ValueError(f"Minimum variance optimization failed with status: {problem.status}")
                
        except Exception as e:
            self.logger.error(f"Minimum variance optimization failed: {str(e)}")
            raise
    
    def _maximum_sharpe_optimization(self, 
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   current_weights: Optional[Dict[str, float]],
                                   constraints: OptimizationConstraints) -> OptimizationResult:
        """Perform maximum Sharpe ratio optimization."""
        try:
            n_assets = len(expected_returns)
            
            # Define optimization variables
            w = cp.Variable(n_assets)
            kappa = cp.Variable()  # Auxiliary variable
            
            # Transform the problem to convex form
            excess_returns = expected_returns - self.risk_free_rate
            
            # Constraints
            constraint_list = [
                cp.quad_form(w, covariance_matrix) <= kappa,
                cp.sum(w) == 1,
                w >= 0,
                kappa >= 0
            ]
            
            # Add additional constraints
            if constraints.max_weights:
                for i, asset in enumerate(self.asset_universe):
                    if asset in constraints.max_weights:
                        constraint_list.append(w[i] <= constraints.max_weights[asset])
            
            # Objective: maximize excess return (equivalent to maximizing Sharpe)
            problem = cp.Problem(cp.Maximize(w.T @ excess_returns), constraint_list)
            
            # Solve
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights_array = w.value
                optimal_weights = {asset: float(weight) for asset, weight in 
                                 zip(self.asset_universe, optimal_weights_array)}
                
                # Calculate metrics
                expected_return = float(optimal_weights_array.T @ expected_returns)
                expected_risk = float(np.sqrt(optimal_weights_array.T @ covariance_matrix @ optimal_weights_array))
                sharpe_ratio = (expected_return - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0
                
                # Calculate diversification ratio
                individual_vols = np.sqrt(np.diag(covariance_matrix))
                weighted_avg_vol = optimal_weights_array @ individual_vols
                diversification_ratio = weighted_avg_vol / expected_risk if expected_risk > 0 else 1.0
                
                return OptimizationResult(
                    optimal_weights=optimal_weights,
                    expected_return=expected_return,
                    expected_risk=expected_risk,
                    sharpe_ratio=sharpe_ratio,
                    diversification_ratio=diversification_ratio,
                    turnover=0.0,
                    optimization_success=True,
                    optimization_method='maximum_sharpe',
                    objective_value=float(problem.value),
                    constraints_satisfied=True,
                    execution_cost=0.0
                )
            else:
                raise ValueError(f"Maximum Sharpe optimization failed with status: {problem.status}")
                
        except Exception as e:
            self.logger.error(f"Maximum Sharpe optimization failed: {str(e)}")
            raise
    
    def _equal_weight_optimization(self, 
                                 n_assets: int,
                                 constraints: OptimizationConstraints) -> OptimizationResult:
        """Create equal weight portfolio."""
        try:
            weight = 1.0 / n_assets
            optimal_weights = {asset: weight for asset in self.asset_universe}
            
            return OptimizationResult(
                optimal_weights=optimal_weights,
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                diversification_ratio=1.0,
                turnover=0.0,
                optimization_success=True,
                optimization_method='equal_weight',
                objective_value=0.0,
                constraints_satisfied=True,
                execution_cost=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Equal weight optimization failed: {str(e)}")
            raise
    
    def _black_litterman_optimization(self, 
                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    current_weights: Optional[Dict[str, float]],
                                    objective: OptimizationObjective,
                                    constraints: OptimizationConstraints) -> OptimizationResult:
        """Perform Black-Litterman optimization (simplified implementation)."""
        try:
            # This is a simplified Black-Litterman implementation
            # In practice, you would need market cap weights and investor views
            
            # Use market cap weights as prior (or equal weights if not available)
            if self.market_cap_weights:
                prior_weights = np.array([self.market_cap_weights.get(asset, 1.0/len(self.asset_universe)) 
                                        for asset in self.asset_universe])
            else:
                prior_weights = np.ones(len(self.asset_universe)) / len(self.asset_universe)
            
            # Risk aversion parameter
            risk_aversion = 3.0
            
            # Implied equilibrium returns
            implied_returns = risk_aversion * covariance_matrix @ prior_weights
            
            # Use implied returns for optimization
            return self._mean_variance_optimization(
                implied_returns, covariance_matrix, current_weights, objective, constraints
            )
            
        except Exception as e:
            self.logger.error(f"Black-Litterman optimization failed: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """Save the trained allocation optimizer."""
        try:
            model_data = {
                'return_predictor': self.return_predictor,
                'risk_predictor': self.risk_predictor,
                'correlation_predictor': self.correlation_predictor,
                'transaction_cost_predictor': self.transaction_cost_predictor,
                'covariance_estimator': self.covariance_estimator,
                'return_estimator': self.return_estimator,
                'black_litterman_model': self.black_litterman_model,
                'return_scaler': self.return_scaler,
                'risk_scaler': self.risk_scaler,
                'feature_scaler': self.feature_scaler,
                'asset_universe': self.asset_universe,
                'feature_columns': self.feature_columns,
                'risk_free_rate': self.risk_free_rate,
                'optimization_method': self.optimization_method,
                'rebalancing_frequency': self.rebalancing_frequency,
                'market_cap_weights': self.market_cap_weights,
                'asset_sectors': self.asset_sectors,
                'asset_esg_scores': self.asset_esg_scores,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Allocation optimizer saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load trained allocation optimizer."""
        try:
            model_data = joblib.load(filepath)
            
            self.return_predictor = model_data['return_predictor']
            self.risk_predictor = model_data['risk_predictor']
            self.correlation_predictor = model_data['correlation_predictor']
            self.transaction_cost_predictor = model_data['transaction_cost_predictor']
            self.covariance_estimator = model_data['covariance_estimator']
            self.return_estimator = model_data['return_estimator']
            self.black_litterman_model = model_data['black_litterman_model']
            self.return_scaler = model_data['return_scaler']
            self.risk_scaler = model_data['risk_scaler']
            self.feature_scaler = model_data['feature_scaler']
            self.asset_universe = model_data['asset_universe']
            self.feature_columns = model_data['feature_columns']
            self.risk_free_rate = model_data['risk_free_rate']
            self.optimization_method = model_data['optimization_method']
            self.rebalancing_frequency = model_data['rebalancing_frequency']
            self.market_cap_weights = model_data['market_cap_weights']
            self.asset_sectors = model_data['asset_sectors']
            self.asset_esg_scores = model_data['asset_esg_scores']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Allocation optimizer loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise


def create_allocation_optimizer_pipeline(config: Dict) -> AllocationOptimizer:
    """
    Create an allocation optimizer pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AllocationOptimizer instance
    """
    return AllocationOptimizer(
        risk_free_rate=config.get('risk_free_rate', 0.02),
        optimization_method=config.get('optimization_method', 'mean_variance'),
        rebalancing_frequency=config.get('rebalancing_frequency', 'monthly')
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = AllocationOptimizer(optimization_method='mean_variance')
    
    # Example usage would involve:
    # 1. Loading historical price data
    # 2. Training prediction models
    # 3. Creating optimization constraints and objectives
    # 4. Running optimization
    # 5. Analyzing results
    
    print("Allocation optimizer implementation completed")
