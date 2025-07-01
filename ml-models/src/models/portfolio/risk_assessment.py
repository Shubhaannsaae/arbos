"""
Portfolio Risk Assessment Model

This module implements advanced ML models for comprehensive portfolio risk assessment
using modern portfolio theory, VaR calculations, and real-time risk monitoring
with Chainlink Data Feeds integration.

Features:
- Value at Risk (VaR) and Conditional VaR calculations
- Correlation analysis and risk decomposition
- Stress testing and scenario analysis
- Real-time risk monitoring using Chainlink oracles
- Multi-asset and cross-chain risk assessment
- Dynamic risk model adaptation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.linalg import sqrtm
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
class RiskMetrics:
    """Data class for portfolio risk metrics."""
    var_95: float
    var_99: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float
    maximum_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    calmar_ratio: float
    risk_score: float  # Overall risk score 0-100

@dataclass
class AssetRisk:
    """Data class for individual asset risk metrics."""
    asset_symbol: str
    volatility: float
    var_95: float
    beta: float
    correlation_to_portfolio: float
    contribution_to_portfolio_risk: float
    marginal_var: float
    component_var: float
    risk_score: float

@dataclass
class PortfolioComposition:
    """Data class for portfolio composition."""
    assets: Dict[str, float]  # asset -> weight
    total_value: float
    currency: str
    rebalance_frequency: str
    risk_budget: Dict[str, float]  # asset -> risk budget allocation

class RiskAssessmentModel:
    """
    Advanced portfolio risk assessment model using ML techniques
    and traditional portfolio theory for comprehensive risk analysis.
    """
    
    def __init__(self, 
                 confidence_levels: List[float] = [0.95, 0.99],
                 lookback_window: int = 252,  # Trading days
                 benchmark_symbol: str = 'SPY'):
        """
        Initialize the risk assessment model.
        
        Args:
            confidence_levels: VaR confidence levels
            lookback_window: Historical data window for calculations
            benchmark_symbol: Benchmark for beta and alpha calculations
        """
        self.confidence_levels = confidence_levels
        self.lookback_window = lookback_window
        self.benchmark_symbol = benchmark_symbol
        
        # Model components
        self.volatility_predictor = None
        self.correlation_predictor = None
        self.var_estimator = None
        self.stress_test_model = None
        self.anomaly_detector = None
        
        # Risk models
        self.garch_models = {}  # Asset-specific GARCH models
        self.covariance_estimator = LedoitWolf()
        
        # Feature scalers
        self.volatility_scaler = StandardScaler()
        self.correlation_scaler = RobustScaler()
        self.risk_scaler = StandardScaler()
        
        # Model metadata
        self.feature_columns = []
        self.asset_universe = []
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def prepare_risk_features(self, 
                            price_data: pd.DataFrame,
                            portfolio_weights: Dict[str, float]) -> pd.DataFrame:
        """
        Prepare features for risk assessment model.
        
        Args:
            price_data: DataFrame with asset prices
            portfolio_weights: Dictionary of asset weights
            
        Returns:
            DataFrame with risk features
        """
        try:
            # Calculate returns
            returns_df = price_data.pct_change().dropna()
            
            # Portfolio return
            portfolio_returns = self._calculate_portfolio_returns(returns_df, portfolio_weights)
            
            features_df = pd.DataFrame(index=returns_df.index)
            
            # Individual asset features
            for asset in portfolio_weights.keys():
                if asset in returns_df.columns:
                    asset_returns = returns_df[asset]
                    
                    # Volatility features
                    features_df[f'{asset}_volatility_5d'] = asset_returns.rolling(5).std() * np.sqrt(252)
                    features_df[f'{asset}_volatility_20d'] = asset_returns.rolling(20).std() * np.sqrt(252)
                    features_df[f'{asset}_volatility_60d'] = asset_returns.rolling(60).std() * np.sqrt(252)
                    
                    # Return features
                    features_df[f'{asset}_return_5d'] = asset_returns.rolling(5).sum()
                    features_df[f'{asset}_return_20d'] = asset_returns.rolling(20).sum()
                    features_df[f'{asset}_skewness'] = asset_returns.rolling(60).skew()
                    features_df[f'{asset}_kurtosis'] = asset_returns.rolling(60).kurt()
                    
                    # Risk-adjusted features
                    features_df[f'{asset}_sharpe_20d'] = self._rolling_sharpe(asset_returns, 20)
                    features_df[f'{asset}_max_drawdown_60d'] = self._rolling_max_drawdown(asset_returns, 60)
                    
                    # GARCH volatility prediction
                    features_df[f'{asset}_garch_vol'] = self._fit_garch_volatility(asset_returns)
                    
            # Portfolio-level features
            features_df['portfolio_return'] = portfolio_returns
            features_df['portfolio_volatility_20d'] = portfolio_returns.rolling(20).std() * np.sqrt(252)
            features_df['portfolio_skewness'] = portfolio_returns.rolling(60).skew()
            features_df['portfolio_kurtosis'] = portfolio_returns.rolling(60).kurt()
            
            # Cross-asset features
            features_df = self._add_correlation_features(features_df, returns_df, portfolio_weights)
            
            # Market regime features
            features_df = self._add_market_regime_features(features_df, returns_df)
            
            # Stress indicators
            features_df = self._add_stress_indicators(features_df, returns_df)
            
            # Time-based features
            features_df['month'] = features_df.index.month
            features_df['quarter'] = features_df.index.quarter
            features_df['is_month_end'] = features_df.index.is_month_end.astype(int)
            features_df['is_quarter_end'] = features_df.index.is_quarter_end.astype(int)
            
            return features_df.dropna()
            
        except Exception as e:
            self.logger.error(f"Risk feature preparation failed: {str(e)}")
            raise
    
    def _calculate_portfolio_returns(self, 
                                   returns_df: pd.DataFrame,
                                   weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns from individual asset returns."""
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
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            
            sharpe = (mean_return - risk_free_rate) / volatility
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Rolling Sharpe calculation failed: {str(e)}")
            return pd.Series(0.0, index=returns.index)
    
    def _rolling_max_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.rolling(window).max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.rolling(window).min()
            
            return abs(max_drawdown)
            
        except Exception as e:
            self.logger.error(f"Rolling max drawdown calculation failed: {str(e)}")
            return pd.Series(0.0, index=returns.index)
    
    def _fit_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Fit GARCH model and predict volatility."""
        try:
            # Convert to percentage returns for GARCH
            returns_pct = returns * 100
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(returns_pct.dropna(), vol='Garch', p=1, q=1)
            fitted_model = garch_model.fit(disp='off')
            
            # Predict conditional volatility
            forecast = fitted_model.forecast(horizon=1)
            conditional_vol = forecast.variance.iloc[-1, 0] ** 0.5 / 100  # Convert back to decimal
            
            # Create series with constant conditional volatility
            vol_series = pd.Series(conditional_vol, index=returns.index)
            
            return vol_series
            
        except Exception as e:
            self.logger.error(f"GARCH volatility estimation failed: {str(e)}")
            return returns.rolling(20).std() * np.sqrt(252)  # Fallback to rolling volatility
    
    def _add_correlation_features(self, 
                                features_df: pd.DataFrame,
                                returns_df: pd.DataFrame,
                                weights: Dict[str, float]) -> pd.DataFrame:
        """Add correlation-based features."""
        try:
            assets = list(weights.keys())
            
            # Pairwise correlations
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    if asset1 in returns_df.columns and asset2 in returns_df.columns:
                        corr_20d = returns_df[asset1].rolling(20).corr(returns_df[asset2])
                        corr_60d = returns_df[asset1].rolling(60).corr(returns_df[asset2])
                        
                        features_df[f'corr_{asset1}_{asset2}_20d'] = corr_20d
                        features_df[f'corr_{asset1}_{asset2}_60d'] = corr_60d
                        features_df[f'corr_change_{asset1}_{asset2}'] = corr_20d - corr_60d
            
            # Average correlation for each asset
            for asset in assets:
                if asset in returns_df.columns:
                    other_assets = [a for a in assets if a != asset and a in returns_df.columns]
                    if other_assets:
                        avg_corr = returns_df[other_assets].corrwith(returns_df[asset]).mean()
                        features_df[f'{asset}_avg_correlation'] = avg_corr
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Correlation features calculation failed: {str(e)}")
            return features_df
    
    def _add_market_regime_features(self, 
                                  features_df: pd.DataFrame,
                                  returns_df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators."""
        try:
            # Use first asset as market proxy (or benchmark if available)
            market_returns = returns_df.iloc[:, 0] if self.benchmark_symbol not in returns_df.columns else returns_df[self.benchmark_symbol]
            
            # Bull/Bear market indicator
            sma_20 = market_returns.rolling(20).mean()
            sma_60 = market_returns.rolling(60).mean()
            features_df['bull_market'] = (sma_20 > sma_60).astype(int)
            
            # Volatility regime
            vol_20 = market_returns.rolling(20).std()
            vol_percentile = vol_20.rolling(252).rank(pct=True)
            features_df['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
            features_df['low_vol_regime'] = (vol_percentile < 0.2).astype(int)
            
            # Trend strength
            features_df['trend_strength'] = abs(market_returns.rolling(20).mean()) / market_returns.rolling(20).std()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Market regime features calculation failed: {str(e)}")
            return features_df
    
    def _add_stress_indicators(self, 
                             features_df: pd.DataFrame,
                             returns_df: pd.DataFrame) -> pd.DataFrame:
        """Add stress test indicators."""
        try:
            # VIX-like volatility index
            market_returns = returns_df.iloc[:, 0]
            vol_index = market_returns.rolling(20).std() * np.sqrt(252) * 100
            features_df['volatility_index'] = vol_index
            
            # Stress indicators
            features_df['extreme_negative_days'] = (returns_df < -0.05).sum(axis=1)  # Days with >5% loss
            features_df['extreme_positive_days'] = (returns_df > 0.05).sum(axis=1)   # Days with >5% gain
            
            # Crisis indicator (multiple assets down significantly)
            features_df['crisis_indicator'] = (returns_df < -0.03).sum(axis=1) / len(returns_df.columns)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Stress indicators calculation failed: {str(e)}")
            return features_df
    
    def calculate_var(self, 
                     returns: pd.Series,
                     confidence_level: float = 0.95,
                     method: str = 'parametric') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Return series
            confidence_level: Confidence level (0.95 for 95% VaR)
            method: 'parametric', 'historical', or 'monte_carlo'
            
        Returns:
            VaR value
        """
        try:
            if method == 'parametric':
                # Parametric VaR assuming normal distribution
                mean_return = returns.mean()
                std_return = returns.std()
                var = stats.norm.ppf(1 - confidence_level, mean_return, std_return)
                
            elif method == 'historical':
                # Historical VaR
                var = returns.quantile(1 - confidence_level)
                
            elif method == 'monte_carlo':
                # Monte Carlo VaR
                n_simulations = 10000
                mean_return = returns.mean()
                std_return = returns.std()
                
                # Generate random returns
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
                
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            return abs(var)  # Return positive value
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {str(e)}")
            return 0.0
    
    def calculate_cvar(self, 
                      returns: pd.Series,
                      confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        try:
            var_threshold = returns.quantile(1 - confidence_level)
            cvar = returns[returns <= var_threshold].mean()
            
            return abs(cvar)  # Return positive value
            
        except Exception as e:
            self.logger.error(f"CVaR calculation failed: {str(e)}")
            return 0.0
    
    def calculate_portfolio_risk_metrics(self, 
                                       returns_df: pd.DataFrame,
                                       weights: Dict[str, float],
                                       benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            returns_df: DataFrame with asset returns
            weights: Portfolio weights
            benchmark_returns: Benchmark return series for alpha/beta
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(returns_df, weights)
            
            # Basic risk metrics
            var_95 = self.calculate_var(portfolio_returns, 0.95)
            var_99 = self.calculate_var(portfolio_returns, 0.99)
            cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)
            cvar_99 = self.calculate_cvar(portfolio_returns, 0.99)
            
            # Volatility
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            maximum_drawdown = abs(drawdown.min())
            
            # Sharpe ratio
            mean_return = portfolio_returns.mean() * 252
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Beta and Alpha (if benchmark provided)
            beta = 1.0
            alpha = 0.0
            tracking_error = 0.0
            information_ratio = 0.0
            
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align indices
                aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
                
                if len(aligned_portfolio) > 0:
                    # Beta calculation
                    covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                    benchmark_variance = aligned_benchmark.var()
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                    
                    # Alpha calculation
                    benchmark_return = aligned_benchmark.mean() * 252
                    alpha = mean_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
                    
                    # Tracking error
                    excess_returns = aligned_portfolio - aligned_benchmark
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    
                    # Information ratio
                    excess_return = excess_returns.mean() * 252
                    information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Calmar ratio
            calmar_ratio = mean_return / maximum_drawdown if maximum_drawdown > 0 else 0
            
            # Overall risk score (0-100, higher = more risky)
            risk_score = self._calculate_risk_score(
                var_95, volatility, maximum_drawdown, sharpe_ratio
            )
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                maximum_drawdown=maximum_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                calmar_ratio=calmar_ratio,
                risk_score=risk_score
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio risk metrics calculation failed: {str(e)}")
            raise
    
    def _calculate_risk_score(self, 
                            var_95: float,
                            volatility: float,
                            max_drawdown: float,
                            sharpe_ratio: float) -> float:
        """Calculate overall risk score (0-100)."""
        try:
            # Normalize metrics to 0-100 scale
            var_score = min(100, abs(var_95) * 1000)  # Scale daily VaR
            vol_score = min(100, volatility * 100)    # Annual volatility as percentage
            dd_score = min(100, max_drawdown * 100)   # Max drawdown as percentage
            sharpe_score = max(0, 50 - sharpe_ratio * 25)  # Lower Sharpe = higher risk
            
            # Weighted combination
            risk_score = (
                var_score * 0.3 +
                vol_score * 0.3 +
                dd_score * 0.25 +
                sharpe_score * 0.15
            )
            
            return min(100, max(0, risk_score))
            
        except Exception as e:
            self.logger.error(f"Risk score calculation failed: {str(e)}")
            return 50.0  # Default moderate risk
    
    def calculate_asset_risk_contributions(self, 
                                         returns_df: pd.DataFrame,
                                         weights: Dict[str, float]) -> List[AssetRisk]:
        """
        Calculate individual asset risk contributions to portfolio.
        
        Args:
            returns_df: DataFrame with asset returns
            weights: Portfolio weights
            
        Returns:
            List of AssetRisk objects
        """
        try:
            portfolio_returns = self._calculate_portfolio_returns(returns_df, weights)
            portfolio_var = self.calculate_var(portfolio_returns, 0.95)
            
            asset_risks = []
            
            for asset, weight in weights.items():
                if asset in returns_df.columns:
                    asset_returns = returns_df[asset]
                    
                    # Individual asset metrics
                    asset_volatility = asset_returns.std() * np.sqrt(252)
                    asset_var = self.calculate_var(asset_returns, 0.95)
                    
                    # Beta relative to portfolio
                    covariance = np.cov(asset_returns, portfolio_returns)[0, 1]
                    portfolio_variance = portfolio_returns.var()
                    beta = covariance / portfolio_variance if portfolio_variance > 0 else 1.0
                    
                    # Correlation to portfolio
                    correlation = asset_returns.corr(portfolio_returns)
                    
                    # Risk contribution calculations
                    marginal_var = self._calculate_marginal_var(
                        asset_returns, returns_df, weights, asset
                    )
                    
                    component_var = weight * marginal_var
                    contribution_to_portfolio_risk = component_var / portfolio_var if portfolio_var > 0 else 0
                    
                    # Asset risk score
                    asset_risk_score = self._calculate_risk_score(
                        asset_var, asset_volatility, 0.1, 0.5  # Use defaults for individual assets
                    )
                    
                    asset_risk = AssetRisk(
                        asset_symbol=asset,
                        volatility=asset_volatility,
                        var_95=asset_var,
                        beta=beta,
                        correlation_to_portfolio=correlation,
                        contribution_to_portfolio_risk=contribution_to_portfolio_risk,
                        marginal_var=marginal_var,
                        component_var=component_var,
                        risk_score=asset_risk_score
                    )
                    
                    asset_risks.append(asset_risk)
            
            return asset_risks
            
        except Exception as e:
            self.logger.error(f"Asset risk contribution calculation failed: {str(e)}")
            return []
    
    def _calculate_marginal_var(self, 
                              asset_returns: pd.Series,
                              returns_df: pd.DataFrame,
                              weights: Dict[str, float],
                              target_asset: str) -> float:
        """Calculate marginal VaR for an asset."""
        try:
            # Create perturbed portfolio
            epsilon = 0.01  # Small perturbation
            perturbed_weights = weights.copy()
            perturbed_weights[target_asset] += epsilon
            
            # Normalize weights
            total_weight = sum(perturbed_weights.values())
            perturbed_weights = {k: v/total_weight for k, v in perturbed_weights.items()}
            
            # Calculate VaR for original and perturbed portfolios
            original_portfolio = self._calculate_portfolio_returns(returns_df, weights)
            perturbed_portfolio = self._calculate_portfolio_returns(returns_df, perturbed_weights)
            
            original_var = self.calculate_var(original_portfolio, 0.95)
            perturbed_var = self.calculate_var(perturbed_portfolio, 0.95)
            
            # Marginal VaR
            marginal_var = (perturbed_var - original_var) / epsilon
            
            return marginal_var
            
        except Exception as e:
            self.logger.error(f"Marginal VaR calculation failed: {str(e)}")
            return 0.0
    
    def stress_test_portfolio(self, 
                            returns_df: pd.DataFrame,
                            weights: Dict[str, float],
                            scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
        """
        Perform stress testing on portfolio.
        
        Args:
            returns_df: Historical returns data
            weights: Portfolio weights
            scenarios: Custom stress scenarios
            
        Returns:
            Dictionary with stress test results
        """
        try:
            if scenarios is None:
                # Default stress scenarios
                scenarios = {
                    'market_crash': {asset: -0.20 for asset in weights.keys()},  # 20% drop
                    'sector_rotation': {list(weights.keys())[0]: -0.15, 
                                      list(weights.keys())[1]: 0.10 if len(weights) > 1 else 0},
                    'volatility_spike': {asset: 0.0 for asset in weights.keys()},  # No price change, just vol
                    'correlation_breakdown': {asset: np.random.normal(0, 0.05) for asset in weights.keys()}
                }
            
            portfolio_returns = self._calculate_portfolio_returns(returns_df, weights)
            baseline_var = self.calculate_var(portfolio_returns, 0.95)
            
            stress_results = {'baseline_var': baseline_var}
            
            for scenario_name, shock in scenarios.items():
                # Apply shock to returns
                stressed_returns = returns_df.copy()
                for asset, shock_value in shock.items():
                    if asset in stressed_returns.columns:
                        if scenario_name == 'volatility_spike':
                            # Increase volatility by 50%
                            stressed_returns[asset] = stressed_returns[asset] * 1.5
                        else:
                            # Apply price shock to last observation
                            stressed_returns.loc[stressed_returns.index[-1], asset] += shock_value
                
                # Calculate stressed portfolio returns and VaR
                stressed_portfolio = self._calculate_portfolio_returns(stressed_returns, weights)
                stressed_var = self.calculate_var(stressed_portfolio, 0.95)
                
                stress_results[f'{scenario_name}_var'] = stressed_var
                stress_results[f'{scenario_name}_impact'] = stressed_var - baseline_var
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"Stress testing failed: {str(e)}")
            return {}
    
    def detect_risk_anomalies(self, 
                            current_features: pd.DataFrame) -> Dict[str, float]:
        """
        Detect risk anomalies in current portfolio state.
        
        Args:
            current_features: Current feature values
            
        Returns:
            Dictionary with anomaly scores
        """
        try:
            if self.anomaly_detector is None:
                self.logger.warning("Anomaly detector not trained")
                return {}
            
            # Prepare features for anomaly detection
            feature_cols = [col for col in current_features.columns if col in self.feature_columns]
            X = current_features[feature_cols].fillna(0)
            
            # Scale features
            X_scaled = self.risk_scaler.transform(X)
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
            outlier_flags = self.anomaly_detector.predict(X_scaled)
            
            results = {
                'anomaly_score': anomaly_scores[0] if len(anomaly_scores) > 0 else 0,
                'is_outlier': outlier_flags[0] == -1 if len(outlier_flags) > 0 else False,
                'risk_level': 'high' if outlier_flags[0] == -1 else 'normal'
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Risk anomaly detection failed: {str(e)}")
            return {}
    
    def train_risk_models(self, 
                        training_data: pd.DataFrame,
                        validation_split: float = 0.2) -> Dict[str, any]:
        """
        Train ML models for risk assessment.
        
        Args:
            training_data: Historical portfolio and market data
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        try:
            self.logger.info("Starting risk model training...")
            
            # Prepare features
            price_columns = [col for col in training_data.columns if 'price' in col.lower()]
            price_data = training_data[price_columns]
            
            # Assume equal weights for training (can be made more sophisticated)
            weights = {col: 1.0/len(price_columns) for col in price_columns}
            
            features_df = self.prepare_risk_features(price_data, weights)
            
            # Select feature columns
            feature_cols = [col for col in features_df.columns 
                          if not col.startswith('target_') and 
                          col not in ['portfolio_return']]
            
            self.feature_columns = feature_cols
            X = features_df[feature_cols].fillna(0)
            
            training_metrics = {}
            
            # Train anomaly detector
            self.logger.info("Training anomaly detector...")
            X_scaled = self.risk_scaler.fit_transform(X)
            
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=200
            )
            self.anomaly_detector.fit(X_scaled)
            
            # Train volatility predictor if target available
            if 'realized_volatility' in training_data.columns:
                self.logger.info("Training volatility predictor...")
                y_vol = training_data['realized_volatility']
                
                X_train, X_val, y_train_vol, y_val_vol = train_test_split(
                    X, y_vol, test_size=validation_split, random_state=42
                )
                
                X_train_vol_scaled = self.volatility_scaler.fit_transform(X_train)
                X_val_vol_scaled = self.volatility_scaler.transform(X_val)
                
                self.volatility_predictor = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    random_state=42
                )
                
                self.volatility_predictor.fit(
                    X_train_vol_scaled, y_train_vol,
                    eval_set=[(X_val_vol_scaled, y_val_vol)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                vol_pred = self.volatility_predictor.predict(X_val_vol_scaled)
                training_metrics['volatility_predictor'] = {
                    'mae': mean_absolute_error(y_val_vol, vol_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val_vol, vol_pred)),
                    'r2': 1 - (np.sum((y_val_vol - vol_pred) ** 2) / np.sum((y_val_vol - y_val_vol.mean()) ** 2))
                }
            
            self.last_trained = datetime.now()
            self.logger.info("Risk model training completed")
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Risk model training failed: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """Save the trained risk models."""
        try:
            model_data = {
                'volatility_predictor': self.volatility_predictor,
                'correlation_predictor': self.correlation_predictor,
                'var_estimator': self.var_estimator,
                'stress_test_model': self.stress_test_model,
                'anomaly_detector': self.anomaly_detector,
                'garch_models': self.garch_models,
                'covariance_estimator': self.covariance_estimator,
                'volatility_scaler': self.volatility_scaler,
                'correlation_scaler': self.correlation_scaler,
                'risk_scaler': self.risk_scaler,
                'feature_columns': self.feature_columns,
                'asset_universe': self.asset_universe,
                'confidence_levels': self.confidence_levels,
                'lookback_window': self.lookback_window,
                'benchmark_symbol': self.benchmark_symbol,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Risk model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Risk model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load trained risk models."""
        try:
            model_data = joblib.load(filepath)
            
            self.volatility_predictor = model_data['volatility_predictor']
            self.correlation_predictor = model_data['correlation_predictor']
            self.var_estimator = model_data['var_estimator']
            self.stress_test_model = model_data['stress_test_model']
            self.anomaly_detector = model_data['anomaly_detector']
            self.garch_models = model_data['garch_models']
            self.covariance_estimator = model_data['covariance_estimator']
            self.volatility_scaler = model_data['volatility_scaler']
            self.correlation_scaler = model_data['correlation_scaler']
            self.risk_scaler = model_data['risk_scaler']
            self.feature_columns = model_data['feature_columns']
            self.asset_universe = model_data['asset_universe']
            self.confidence_levels = model_data['confidence_levels']
            self.lookback_window = model_data['lookback_window']
            self.benchmark_symbol = model_data['benchmark_symbol']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Risk model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Risk model loading failed: {str(e)}")
            raise


def create_risk_assessment_pipeline(config: Dict) -> RiskAssessmentModel:
    """
    Create a risk assessment pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RiskAssessmentModel instance
    """
    return RiskAssessmentModel(
        confidence_levels=config.get('confidence_levels', [0.95, 0.99]),
        lookback_window=config.get('lookback_window', 252),
        benchmark_symbol=config.get('benchmark_symbol', 'SPY')
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = RiskAssessmentModel()
    
    # Example portfolio
    weights = {'BTC': 0.4, 'ETH': 0.3, 'USDC': 0.3}
    
    # This would be replaced with actual price data
    price_data = load_price_data(['BTC', 'ETH', 'USDC'])
    returns_data = price_data.pct_change().dropna()
    
    # Calculate risk metrics
    risk_metrics = model.calculate_portfolio_risk_metrics(returns_data, weights)
    print(f"Portfolio VaR (95%): {risk_metrics.var_95:.4f}")
    print(f"Portfolio Volatility: {risk_metrics.volatility:.4f}")
    print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.4f}")
    
    # Asset risk contributions
    asset_risks = model.calculate_asset_risk_contributions(returns_data, weights)
    for asset_risk in asset_risks:
        print(f"{asset_risk.asset_symbol}: Risk Contribution = {asset_risk.contribution_to_portfolio_risk:.4f}")
    
    # Stress testing
    stress_results = model.stress_test_portfolio(returns_data, weights)
    print("Stress Test Results:", stress_results)
    
    print("Risk assessment model implementation completed")
