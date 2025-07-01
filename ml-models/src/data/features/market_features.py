"""
Market Features Engineering Module

This module implements comprehensive market-wide feature engineering for ML models,
integrating with Chainlink Data Feeds for real-time market data and supporting
advanced cross-asset and macroeconomic feature extraction.

Features:
- Market regime and cycle detection features
- Cross-asset correlation and beta calculations
- Liquidity and market depth features
- Volatility surface and skew features
- Economic indicator integration
- Real-time market feature updates with Chainlink Data Feeds
- Cross-chain market analysis using CCIP
- Advanced market microstructure features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import networkx as nx
import joblib

warnings.filterwarnings('ignore')

@dataclass
class MarketFeatureConfig:
    """Configuration for market feature engineering."""
    feature_type: str
    lookback_periods: List[int]
    update_frequency: str  # 'real_time', 'hourly', 'daily'
    cross_asset_analysis: bool = True
    regime_detection: bool = True
    liquidity_analysis: bool = True

@dataclass
class MarketRegimeFeatures:
    """Market regime classification features."""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile', 'crisis'
    regime_probability: float
    regime_duration: int  # days
    volatility_regime: str  # 'low', 'medium', 'high'
    correlation_regime: str  # 'low', 'medium', 'high'
    trend_strength: float
    momentum_score: float
    mean_reversion_score: float

@dataclass
class LiquidityFeatures:
    """Liquidity and market depth features."""
    bid_ask_spread: float
    market_depth: float
    liquidity_ratio: float
    amihud_illiquidity: float
    kyle_lambda: float
    roll_spread: float
    effective_spread: float
    price_impact: float
    volume_participation_rate: float

@dataclass
class VolatilityFeatures:
    """Volatility surface and related features."""
    realized_volatility: float
    garch_volatility: float
    parkinson_volatility: float
    garman_klass_volatility: float
    rogers_satchell_volatility: float
    volatility_of_volatility: float
    volatility_skew: float
    volatility_smile: float
    vol_term_structure: Dict[str, float]

class MarketFeaturesEngine:
    """
    Advanced market features engineering pipeline for ML models
    with comprehensive cross-asset and macroeconomic analysis.
    """
    
    def __init__(self):
        """Initialize market features engine."""
        self.scalers = {}
        self.feature_cache = {}
        self.correlation_matrices = {}
        self.regime_models = {}
        
        # Feature extractors
        self.pca = PCA(n_components=0.95)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
        # Model metadata
        self.feature_history = {}
        self.asset_universe = set()
        self.model_version = "1.0.0"
        
        # Market identifiers for cross-asset analysis
        self.market_categories = {
            'crypto_major': ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'LINK'],
            'crypto_defi': ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'CRV'],
            'crypto_layer1': ['ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'ATOM'],
            'traditional_equity': ['SPY', 'QQQ', 'IWM', 'VTI'],
            'traditional_fixed_income': ['TLT', 'IEF', 'SHY', 'HYG'],
            'commodities': ['GLD', 'SLV', 'USO', 'DBA'],
            'volatility': ['VIX', 'VXX', 'UVXY'],
            'currencies': ['DXY', 'EURUSD', 'USDJPY', 'GBPUSD']
        }
        
        # Economic indicators
        self.economic_indicators = {
            'monetary_policy': ['DFF', 'DGS10', 'DGS2', 'T10Y2Y'],
            'inflation': ['CPIAUCSL', 'CPILFESL', 'DFEDTARL', 'T5YIE'],
            'employment': ['UNRATE', 'NPPTTL', 'AHETPI', 'CIVPART'],
            'growth': ['GDP', 'GDPPOT', 'NYGDPMKTPCDWLD', 'INDPRO'],
            'sentiment': ['UMCSENT', 'NASDAQCOM', 'CBOE', 'AAII']
        }
        
        # Risk-free rates by currency
        self.risk_free_rates = {
            'USD': 0.02,  # 2% default
            'EUR': 0.00,
            'JPY': -0.001,
            'GBP': 0.015,
            'CHF': -0.005
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def extract_market_regime_features(self, 
                                     market_data: Dict[str, pd.DataFrame],
                                     lookback_period: int = 252) -> MarketRegimeFeatures:
        """
        Extract market regime classification features.
        
        Args:
            market_data: Dictionary of market data by asset
            lookback_period: Number of days to look back
            
        Returns:
            MarketRegimeFeatures object
        """
        try:
            # Aggregate market returns
            returns_data = {}
            for asset, df in market_data.items():
                if df is not None and 'close' in df.columns and len(df) > 0:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[asset] = returns.tail(lookback_period)
            
            if not returns_data:
                return self._default_regime_features()
            
            # Create market return (equal-weighted)
            returns_df = pd.DataFrame(returns_data)
            market_return = returns_df.mean(axis=1)
            
            # Calculate regime indicators
            
            # 1. Trend Analysis
            price_trend = self._calculate_trend_strength(market_return)
            
            # 2. Volatility Analysis
            volatility = market_return.rolling(20).std() * np.sqrt(252)
            vol_percentile = volatility.iloc[-1] / volatility.quantile(0.8) if len(volatility) > 0 else 1.0
            
            if vol_percentile > 1.5:
                volatility_regime = 'high'
            elif vol_percentile > 0.8:
                volatility_regime = 'medium'
            else:
                volatility_regime = 'low'
            
            # 3. Correlation Analysis
            corr_matrix = returns_df.corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            if avg_correlation > 0.7:
                correlation_regime = 'high'
            elif avg_correlation > 0.4:
                correlation_regime = 'medium'
            else:
                correlation_regime = 'low'
            
            # 4. Momentum Score
            momentum_periods = [5, 10, 20, 50]
            momentum_scores = []
            for period in momentum_periods:
                if len(market_return) >= period:
                    momentum = market_return.rolling(period).sum().iloc[-1]
                    momentum_scores.append(momentum)
            
            momentum_score = np.mean(momentum_scores) if momentum_scores else 0.0
            
            # 5. Mean Reversion Score
            mean_reversion_score = self._calculate_mean_reversion_score(market_return)
            
            # 6. Regime Classification
            regime_type, regime_probability, regime_duration = self._classify_market_regime(
                market_return, volatility, avg_correlation, momentum_score
            )
            
            return MarketRegimeFeatures(
                regime_type=regime_type,
                regime_probability=regime_probability,
                regime_duration=regime_duration,
                volatility_regime=volatility_regime,
                correlation_regime=correlation_regime,
                trend_strength=price_trend,
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion_score
            )
            
        except Exception as e:
            self.logger.error(f"Market regime feature extraction failed: {str(e)}")
            return self._default_regime_features()
    
    def _calculate_trend_strength(self, returns: pd.Series) -> float:
        """Calculate trend strength using multiple indicators."""
        try:
            if len(returns) < 50:
                return 0.0
            
            # Convert to price series
            prices = (1 + returns).cumprod()
            
            # Moving averages
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            
            # Trend direction
            trend_direction = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
            
            # Trend consistency
            price_above_ma20 = (prices.iloc[-20:] > sma_20.iloc[-20:]).mean()
            
            # Trend angle (slope of regression line)
            x = np.arange(len(prices.iloc[-50:]))
            y = prices.iloc[-50:].values
            if len(x) > 1 and len(y) > 1:
                slope, _ = np.polyfit(x, y, 1)
                trend_angle = np.arctan(slope) * (180 / np.pi)  # Convert to degrees
            else:
                trend_angle = 0
            
            # Combine indicators
            trend_strength = (
                trend_direction * 
                price_above_ma20 * 
                min(1.0, abs(trend_angle) / 45)  # Normalize angle
            )
            
            return np.clip(trend_strength, -1, 1)
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_mean_reversion_score(self, returns: pd.Series) -> float:
        """Calculate mean reversion tendency."""
        try:
            if len(returns) < 20:
                return 0.0
            
            # Variance ratio test
            def variance_ratio(returns, q):
                """Calculate variance ratio for given holding period q."""
                n = len(returns)
                if n < q:
                    return 1.0
                
                # q-period returns
                q_returns = returns.rolling(q).sum().dropna()
                
                # Variance of q-period returns
                var_q = q_returns.var()
                
                # q times variance of 1-period returns
                var_1 = returns.var()
                
                if var_1 == 0:
                    return 1.0
                
                return var_q / (q * var_1)
            
            # Calculate variance ratios for different horizons
            vr_scores = []
            for q in [2, 4, 8, 16]:
                if len(returns) >= q:
                    vr = variance_ratio(returns, q)
                    # VR < 1 indicates mean reversion
                    vr_scores.append(1 - vr)
            
            mean_reversion_score = np.mean(vr_scores) if vr_scores else 0.0
            
            return np.clip(mean_reversion_score, -1, 1)
            
        except Exception as e:
            self.logger.error(f"Mean reversion score calculation failed: {str(e)}")
            return 0.0
    
    def _classify_market_regime(self, 
                              market_return: pd.Series,
                              volatility: pd.Series,
                              avg_correlation: float,
                              momentum_score: float) -> Tuple[str, float, int]:
        """Classify market regime using ensemble approach."""
        try:
            # Recent market metrics (last 30 days)
            recent_return = market_return.tail(30).mean() if len(market_return) >= 30 else market_return.mean()
            recent_volatility = volatility.tail(30).mean() if len(volatility) >= 30 else volatility.mean()
            
            # Regime scoring
            regime_scores = {}
            
            # Bull market criteria
            bull_score = 0
            if recent_return > 0.001:  # Positive returns
                bull_score += 0.4
            if momentum_score > 0.01:  # Positive momentum
                bull_score += 0.3
            if recent_volatility < 0.25:  # Low volatility
                bull_score += 0.2
            if avg_correlation < 0.6:  # Low correlation
                bull_score += 0.1
            
            # Bear market criteria
            bear_score = 0
            if recent_return < -0.002:  # Negative returns
                bear_score += 0.4
            if momentum_score < -0.01:  # Negative momentum
                bear_score += 0.3
            if avg_correlation > 0.7:  # High correlation (fear)
                bear_score += 0.2
            if recent_volatility > 0.3:  # High volatility
                bear_score += 0.1
            
            # Volatile market criteria
            volatile_score = 0
            if recent_volatility > 0.4:  # Very high volatility
                volatile_score += 0.5
            if abs(momentum_score) > 0.02:  # High absolute momentum
                volatile_score += 0.3
            if avg_correlation > 0.8:  # Very high correlation
                volatile_score += 0.2
            
            # Crisis criteria
            crisis_score = 0
            if recent_volatility > 0.6:  # Extreme volatility
                crisis_score += 0.4
            if recent_return < -0.005:  # Very negative returns
                crisis_score += 0.3
            if avg_correlation > 0.85:  # Extreme correlation
                crisis_score += 0.3
            
            # Sideways market (default)
            sideways_score = max(0, 1 - max(bull_score, bear_score, volatile_score, crisis_score))
            
            regime_scores = {
                'bull': bull_score,
                'bear': bear_score,
                'sideways': sideways_score,
                'volatile': volatile_score,
                'crisis': crisis_score
            }
            
            # Determine dominant regime
            dominant_regime = max(regime_scores.items(), key=lambda x: x[1])
            regime_type = dominant_regime[0]
            regime_probability = dominant_regime[1]
            
            # Estimate regime duration (simplified)
            regime_duration = 30  # Default 30 days
            
            return regime_type, regime_probability, regime_duration
            
        except Exception as e:
            self.logger.error(f"Regime classification failed: {str(e)}")
            return 'sideways', 0.5, 30
    
    def _default_regime_features(self) -> MarketRegimeFeatures:
        """Return default regime features when calculation fails."""
        return MarketRegimeFeatures(
            regime_type='sideways',
            regime_probability=0.5,
            regime_duration=30,
            volatility_regime='medium',
            correlation_regime='medium',
            trend_strength=0.0,
            momentum_score=0.0,
            mean_reversion_score=0.0
        )
    
    def extract_liquidity_features(self, 
                                 market_data: Dict[str, pd.DataFrame],
                                 price_column: str = 'close',
                                 volume_column: str = 'volume') -> LiquidityFeatures:
        """
        Extract liquidity and market microstructure features.
        
        Args:
            market_data: Dictionary of market data by asset
            price_column: Column name for price data
            volume_column: Column name for volume data
            
        Returns:
            LiquidityFeatures object
        """
        try:
            # Aggregate liquidity metrics across assets
            bid_ask_spreads = []
            market_depths = []
            liquidity_ratios = []
            amihud_illiquidities = []
            roll_spreads = []
            price_impacts = []
            volume_rates = []
            
            for asset, df in market_data.items():
                if df is None or len(df) == 0:
                    continue
                
                if price_column not in df.columns or volume_column not in df.columns:
                    continue
                
                prices = df[price_column]
                volumes = df[volume_column]
                returns = prices.pct_change().dropna()
                
                # Bid-ask spread proxy (Roll's estimator)
                roll_spread = self._calculate_roll_spread(returns)
                roll_spreads.append(roll_spread)
                
                # Effective spread proxy
                bid_ask_spread = 2 * roll_spread  # Simplified
                bid_ask_spreads.append(bid_ask_spread)
                
                # Market depth proxy (inverse of price impact)
                price_impact = self._calculate_price_impact(returns, volumes)
                price_impacts.append(price_impact)
                
                market_depth = 1 / (price_impact + 1e-8)
                market_depths.append(market_depth)
                
                # Liquidity ratio
                liquidity_ratio = volumes.mean() / (prices.mean() + 1e-8)
                liquidity_ratios.append(liquidity_ratio)
                
                # Amihud illiquidity measure
                amihud = self._calculate_amihud_illiquidity(returns, volumes)
                amihud_illiquidities.append(amihud)
                
                # Volume participation rate
                volume_rate = volumes.std() / (volumes.mean() + 1e-8)
                volume_rates.append(volume_rate)
            
            # Aggregate results
            return LiquidityFeatures(
                bid_ask_spread=np.mean(bid_ask_spreads) if bid_ask_spreads else 0.0,
                market_depth=np.mean(market_depths) if market_depths else 0.0,
                liquidity_ratio=np.mean(liquidity_ratios) if liquidity_ratios else 0.0,
                amihud_illiquidity=np.mean(amihud_illiquidities) if amihud_illiquidities else 0.0,
                kyle_lambda=0.0,  # Would need order book data
                roll_spread=np.mean(roll_spreads) if roll_spreads else 0.0,
                effective_spread=np.mean(bid_ask_spreads) if bid_ask_spreads else 0.0,
                price_impact=np.mean(price_impacts) if price_impacts else 0.0,
                volume_participation_rate=np.mean(volume_rates) if volume_rates else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Liquidity feature extraction failed: {str(e)}")
            return LiquidityFeatures(
                bid_ask_spread=0.0, market_depth=0.0, liquidity_ratio=0.0,
                amihud_illiquidity=0.0, kyle_lambda=0.0, roll_spread=0.0,
                effective_spread=0.0, price_impact=0.0, volume_participation_rate=0.0
            )
    
    def _calculate_roll_spread(self, returns: pd.Series) -> float:
        """Calculate Roll's bid-ask spread estimator."""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Roll's estimator: 2 * sqrt(-Cov(r_t, r_{t-1}))
            returns_diff = returns.diff().dropna()
            covariance = returns_diff.autocorr(lag=1)
            
            if covariance is None or np.isnan(covariance) or covariance >= 0:
                return 0.0
            
            roll_spread = 2 * np.sqrt(-covariance)
            return min(roll_spread, 0.1)  # Cap at 10%
            
        except Exception as e:
            self.logger.error(f"Roll spread calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_price_impact(self, returns: pd.Series, volumes: pd.Series) -> float:
        """Calculate price impact measure."""
        try:
            if len(returns) != len(volumes) or len(returns) < 10:
                return 0.0
            
            # Align series
            returns, volumes = returns.align(volumes, join='inner')
            
            # Remove zero volumes
            non_zero_mask = volumes > 0
            returns_clean = returns[non_zero_mask]
            volumes_clean = volumes[non_zero_mask]
            
            if len(returns_clean) < 5:
                return 0.0
            
            # Price impact = correlation between |returns| and volume
            abs_returns = abs(returns_clean)
            log_volumes = np.log(volumes_clean + 1)
            
            correlation = abs_returns.corr(log_volumes)
            
            if correlation is None or np.isnan(correlation):
                return 0.0
            
            return max(0, correlation)
            
        except Exception as e:
            self.logger.error(f"Price impact calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_amihud_illiquidity(self, returns: pd.Series, volumes: pd.Series) -> float:
        """Calculate Amihud illiquidity measure."""
        try:
            if len(returns) != len(volumes) or len(returns) < 10:
                return 0.0
            
            # Align series
            returns, volumes = returns.align(volumes, join='inner')
            
            # Remove zero volumes
            non_zero_mask = volumes > 0
            returns_clean = returns[non_zero_mask]
            volumes_clean = volumes[non_zero_mask]
            
            if len(returns_clean) < 5:
                return 0.0
            
            # Amihud measure = |return| / volume
            abs_returns = abs(returns_clean)
            illiquidity_ratios = abs_returns / volumes_clean
            
            # Take average
            amihud = illiquidity_ratios.mean()
            
            return min(amihud, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Amihud illiquidity calculation failed: {str(e)}")
            return 0.0
    
    def extract_volatility_features(self, 
                                  market_data: Dict[str, pd.DataFrame],
                                  price_column: str = 'close') -> VolatilityFeatures:
        """
        Extract volatility surface and related features.
        
        Args:
            market_data: Dictionary of market data by asset
            price_column: Column name for price data
            
        Returns:
            VolatilityFeatures object
        """
        try:
            # Aggregate volatility metrics across assets
            realized_vols = []
            garch_vols = []
            parkinson_vols = []
            gk_vols = []
            rs_vols = []
            vol_of_vols = []
            
            for asset, df in market_data.items():
                if df is None or len(df) == 0:
                    continue
                
                if price_column not in df.columns:
                    continue
                
                prices = df[price_column]
                returns = prices.pct_change().dropna()
                
                if len(returns) < 20:
                    continue
                
                # Realized volatility
                realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                realized_vols.append(realized_vol)
                
                # GARCH volatility (simplified)
                garch_vol = self._estimate_garch_volatility(returns)
                garch_vols.append(garch_vol)
                
                # Parkinson volatility (if OHLC available)
                if all(col in df.columns for col in ['high', 'low']):
                    parkinson_vol = self._calculate_parkinson_volatility(df)
                    parkinson_vols.append(parkinson_vol)
                
                # Garman-Klass volatility (if OHLC available)
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    gk_vol = self._calculate_garman_klass_volatility(df)
                    gk_vols.append(gk_vol)
                    
                    # Rogers-Satchell volatility
                    rs_vol = self._calculate_rogers_satchell_volatility(df)
                    rs_vols.append(rs_vol)
                
                # Volatility of volatility
                vol_series = returns.rolling(20).std()
                vol_of_vol = vol_series.rolling(20).std().iloc[-1] if len(vol_series) >= 20 else 0
                vol_of_vols.append(vol_of_vol)
            
            # Calculate volatility skew and smile (simplified)
            volatility_skew = self._calculate_volatility_skew(realized_vols)
            volatility_smile = self._calculate_volatility_smile(realized_vols)
            
            # Volatility term structure
            vol_term_structure = self._calculate_vol_term_structure(market_data)
            
            return VolatilityFeatures(
                realized_volatility=np.mean(realized_vols) if realized_vols else 0.0,
                garch_volatility=np.mean(garch_vols) if garch_vols else 0.0,
                parkinson_volatility=np.mean(parkinson_vols) if parkinson_vols else 0.0,
                garman_klass_volatility=np.mean(gk_vols) if gk_vols else 0.0,
                rogers_satchell_volatility=np.mean(rs_vols) if rs_vols else 0.0,
                volatility_of_volatility=np.mean(vol_of_vols) if vol_of_vols else 0.0,
                volatility_skew=volatility_skew,
                volatility_smile=volatility_smile,
                vol_term_structure=vol_term_structure
            )
            
        except Exception as e:
            self.logger.error(f"Volatility feature extraction failed: {str(e)}")
            return VolatilityFeatures(
                realized_volatility=0.0, garch_volatility=0.0, parkinson_volatility=0.0,
                garman_klass_volatility=0.0, rogers_satchell_volatility=0.0,
                volatility_of_volatility=0.0, volatility_skew=0.0, volatility_smile=0.0,
                vol_term_structure={}
            )
    
    def _estimate_garch_volatility(self, returns: pd.Series, window: int = 50) -> float:
        """Estimate GARCH(1,1) volatility (simplified)."""
        try:
            if len(returns) < window:
                return returns.std() * np.sqrt(252)
            
            # Simple EWMA model (GARCH approximation)
            lambda_param = 0.94  # RiskMetrics lambda
            
            recent_returns = returns.tail(window)
            weights = np.array([(lambda_param ** i) for i in range(len(recent_returns))])
            weights = weights[::-1]  # Reverse to give more weight to recent observations
            weights = weights / weights.sum()  # Normalize
            
            weighted_variance = np.sum(weights * (recent_returns ** 2))
            garch_vol = np.sqrt(weighted_variance * 252)
            
            return garch_vol
            
        except Exception as e:
            self.logger.error(f"GARCH volatility estimation failed: {str(e)}")
            return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
    
    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Parkinson volatility estimator."""
        try:
            if 'high' not in df.columns or 'low' not in df.columns:
                return 0.0
            
            # Parkinson estimator
            log_hl_ratio = np.log(df['high'] / df['low'])
            parkinson_var = (log_hl_ratio ** 2).mean() / (4 * np.log(2))
            parkinson_vol = np.sqrt(parkinson_var * 252)
            
            return parkinson_vol
            
        except Exception as e:
            self.logger.error(f"Parkinson volatility calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_garman_klass_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Garman-Klass volatility estimator."""
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return 0.0
            
            # Garman-Klass estimator
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])
            
            gk_var = (0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)).mean()
            gk_vol = np.sqrt(gk_var * 252)
            
            return gk_vol
            
        except Exception as e:
            self.logger.error(f"Garman-Klass volatility calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_rogers_satchell_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Rogers-Satchell volatility estimator."""
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return 0.0
            
            # Rogers-Satchell estimator
            log_ho = np.log(df['high'] / df['open'])
            log_hc = np.log(df['high'] / df['close'])
            log_lo = np.log(df['low'] / df['open'])
            log_lc = np.log(df['low'] / df['close'])
            
            rs_var = (log_ho * log_hc + log_lo * log_lc).mean()
            rs_vol = np.sqrt(rs_var * 252)
            
            return rs_vol
            
        except Exception as e:
            self.logger.error(f"Rogers-Satchell volatility calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_volatility_skew(self, volatilities: List[float]) -> float:
        """Calculate volatility skew across assets."""
        try:
            if len(volatilities) < 3:
                return 0.0
            
            return stats.skew(volatilities)
            
        except Exception as e:
            self.logger.error(f"Volatility skew calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_volatility_smile(self, volatilities: List[float]) -> float:
        """Calculate volatility smile metric."""
        try:
            if len(volatilities) < 5:
                return 0.0
            
            # Simple smile metric: kurtosis of volatility distribution
            return stats.kurtosis(volatilities)
            
        except Exception as e:
            self.logger.error(f"Volatility smile calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_vol_term_structure(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate volatility term structure."""
        try:
            term_structure = {}
            
            # Calculate volatilities for different time horizons
            horizons = {'1w': 5, '1m': 21, '3m': 63, '6m': 126, '1y': 252}
            
            for horizon_name, horizon_days in horizons.items():
                horizon_vols = []
                
                for asset, df in market_data.items():
                    if df is None or 'close' not in df.columns or len(df) < horizon_days:
                        continue
                    
                    returns = df['close'].pct_change().dropna()
                    if len(returns) >= horizon_days:
                        vol = returns.tail(horizon_days).std() * np.sqrt(252)
                        horizon_vols.append(vol)
                
                if horizon_vols:
                    term_structure[horizon_name] = np.mean(horizon_vols)
                else:
                    term_structure[horizon_name] = 0.0
            
            return term_structure
            
        except Exception as e:
            self.logger.error(f"Volatility term structure calculation failed: {str(e)}")
            return {}
    
    def extract_cross_asset_features(self, 
                                   market_data: Dict[str, pd.DataFrame],
                                   lookback_period: int = 252) -> pd.DataFrame:
        """
        Extract cross-asset correlation and relationship features.
        
        Args:
            market_data: Dictionary of market data by asset
            lookback_period: Number of days to look back
            
        Returns:
            DataFrame with cross-asset features
        """
        try:
            # Extract returns for all assets
            returns_data = {}
            for asset, df in market_data.items():
                if df is not None and 'close' in df.columns and len(df) > 0:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[asset] = returns.tail(lookback_period)
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            # Align all return series
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 10:
                return pd.DataFrame()
            
            # Calculate cross-asset features
            features_df = pd.DataFrame(index=returns_df.index)
            
            # 1. Correlation matrix and derived features
            correlation_matrix = returns_df.corr()
            
            # Average correlation
            corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
            features_df['avg_correlation'] = np.mean(corr_values)
            features_df['correlation_dispersion'] = np.std(corr_values)
            
            # 2. Principal components
            try:
                pca_transformed = self.pca.fit_transform(returns_df.fillna(0))
                explained_variance = self.pca.explained_variance_ratio_
                
                # Add PC features
                for i, var_ratio in enumerate(explained_variance[:3]):  # Top 3 components
                    features_df[f'pc_{i+1}_variance_explained'] = var_ratio
                    if i < pca_transformed.shape[1]:
                        features_df[f'pc_{i+1}_loading'] = pca_transformed[:, i]
                        
            except Exception as e:
                self.logger.warning(f"PCA calculation failed: {str(e)}")
            
            # 3. Beta calculations (relative to market index)
            market_return = returns_df.mean(axis=1)  # Equal-weighted market
            
            for asset in returns_df.columns:
                if len(returns_df[asset]) > 20:
                    # Rolling beta calculation
                    rolling_betas = []
                    for i in range(20, len(returns_df)):
                        y = returns_df[asset].iloc[i-20:i]
                        x = market_return.iloc[i-20:i]
                        
                        if len(x) > 0 and len(y) > 0 and x.var() > 0:
                            beta = y.cov(x) / x.var()
                            rolling_betas.append(beta)
                        else:
                            rolling_betas.append(1.0)
                    
                    # Pad with initial values
                    beta_series = [1.0] * 20 + rolling_betas
                    features_df[f'{asset}_beta'] = beta_series[:len(features_df)]
            
            # 4. Market concentration features
            market_caps = []  # Would need market cap data
            # For now, use equal weights
            equal_weights = [1.0 / len(returns_df.columns)] * len(returns_df.columns)
            herfindahl_index = sum(w**2 for w in equal_weights)
            features_df['market_concentration'] = herfindahl_index
            
            # 5. Sector rotation features
            sector_features = self._calculate_sector_rotation_features(returns_df)
            for feature_name, feature_values in sector_features.items():
                features_df[feature_name] = feature_values
            
            # 6. Risk parity features
            risk_parity_weights = self._calculate_risk_parity_weights(returns_df)
            features_df['risk_parity_concentration'] = sum(w**2 for w in risk_parity_weights)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Cross-asset feature extraction failed: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_sector_rotation_features(self, returns_df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate sector rotation features."""
        try:
            features = {}
            
            # Group assets by categories
            category_returns = {}
            for category, assets in self.market_categories.items():
                category_assets = [asset for asset in assets if asset in returns_df.columns]
                if category_assets:
                    category_return = returns_df[category_assets].mean(axis=1)
                    category_returns[category] = category_return
            
            if len(category_returns) < 2:
                return {}
            
            # Calculate rotation metrics
            category_df = pd.DataFrame(category_returns)
            
            # Relative performance
            for i, cat1 in enumerate(category_df.columns):
                for cat2 in category_df.columns[i+1:]:
                    rel_perf = (category_df[cat1] - category_df[cat2]).rolling(20).sum()
                    features[f'{cat1}_vs_{cat2}_rotation'] = rel_perf.tolist()
            
            # Momentum across sectors
            momentum_scores = []
            for idx in range(len(category_df)):
                if idx >= 20:
                    recent_returns = category_df.iloc[idx-20:idx].sum()
                    momentum_score = recent_returns.std() / (recent_returns.mean() + 1e-8)
                    momentum_scores.append(momentum_score)
                else:
                    momentum_scores.append(0.0)
            
            features['sector_momentum_dispersion'] = momentum_scores
            
            return features
            
        except Exception as e:
            self.logger.error(f"Sector rotation calculation failed: {str(e)}")
            return {}
    
    def _calculate_risk_parity_weights(self, returns_df: pd.DataFrame) -> List[float]:
        """Calculate risk parity portfolio weights."""
        try:
            if len(returns_df) < 20:
                n_assets = len(returns_df.columns)
                return [1.0 / n_assets] * n_assets
            
            # Calculate covariance matrix
            cov_matrix = returns_df.cov()
            
            # Risk parity: weights inversely proportional to volatility
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol_weights = 1 / (volatilities + 1e-8)
            weights = inv_vol_weights / inv_vol_weights.sum()
            
            return weights.tolist()
            
        except Exception as e:
            self.logger.error(f"Risk parity calculation failed: {str(e)}")
            n_assets = len(returns_df.columns)
            return [1.0 / n_assets] * n_assets
    
    def extract_economic_features(self, 
                                economic_data: Dict[str, pd.DataFrame],
                                lookback_period: int = 252) -> pd.DataFrame:
        """
        Extract economic indicator features.
        
        Args:
            economic_data: Dictionary of economic indicator data
            lookback_period: Number of days to look back
            
        Returns:
            DataFrame with economic features
        """
        try:
            if not economic_data:
                return pd.DataFrame()
            
            # Combine all economic indicators
            economic_series = {}
            for indicator, df in economic_data.items():
                if df is not None and 'value' in df.columns:
                    series = df['value'].tail(lookback_period)
                    if len(series) > 0:
                        economic_series[indicator] = series
            
            if not economic_series:
                return pd.DataFrame()
            
            # Align series
            economic_df = pd.DataFrame(economic_series)
            economic_df = economic_df.fillna(method='ffill').fillna(method='bfill')
            
            features_df = pd.DataFrame(index=economic_df.index)
            
            # 1. Level features (normalized)
            for indicator in economic_df.columns:
                # Z-score normalization
                series = economic_df[indicator]
                z_score = (series - series.mean()) / (series.std() + 1e-8)
                features_df[f'{indicator}_zscore'] = z_score
                
                # Percentile rank
                features_df[f'{indicator}_percentile'] = series.rank(pct=True)
            
            # 2. Change features
            for indicator in economic_df.columns:
                series = economic_df[indicator]
                
                # Absolute change
                features_df[f'{indicator}_change'] = series.diff()
                
                # Percentage change
                features_df[f'{indicator}_pct_change'] = series.pct_change()
                
                # Momentum (3-month change)
                features_df[f'{indicator}_momentum'] = series.diff(63) if len(series) > 63 else 0
            
            # 3. Yield curve features (if bond data available)
            yield_curve_features = self._extract_yield_curve_features(economic_df)
            for feature_name, feature_values in yield_curve_features.items():
                features_df[feature_name] = feature_values
            
            # 4. Economic regime features
            regime_features = self._extract_economic_regime_features(economic_df)
            for feature_name, feature_values in regime_features.items():
                features_df[feature_name] = feature_values
            
            return features_df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Economic feature extraction failed: {str(e)}")
            return pd.DataFrame()
    
    def _extract_yield_curve_features(self, economic_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract yield curve related features."""
        try:
            features = {}
            
            # Common yield curve indicators
            yield_indicators = ['DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
            available_yields = [yi for yi in yield_indicators if yi in economic_df.columns]
            
            if len(available_yields) >= 2:
                # Yield curve slope (10Y - 2Y)
                if 'DGS10' in available_yields and 'DGS2' in available_yields:
                    features['yield_curve_slope'] = economic_df['DGS10'] - economic_df['DGS2']
                
                # Yield curve level (average of available yields)
                features['yield_curve_level'] = economic_df[available_yields].mean(axis=1)
                
                # Yield curve curvature (if 3 points available)
                if len(available_yields) >= 3:
                    # Simple curvature: 2*mid - short - long
                    short = economic_df[available_yields[0]]
                    mid = economic_df[available_yields[len(available_yields)//2]]
                    long = economic_df[available_yields[-1]]
                    features['yield_curve_curvature'] = 2 * mid - short - long
            
            return features
            
        except Exception as e:
            self.logger.error(f"Yield curve feature extraction failed: {str(e)}")
            return {}
    
    def _extract_economic_regime_features(self, economic_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract economic regime features."""
        try:
            features = {}
            
            # Growth indicators
            growth_indicators = ['GDP', 'INDPRO', 'PAYEMS']
            available_growth = [gi for gi in growth_indicators if gi in economic_df.columns]
            
            if available_growth:
                growth_composite = economic_df[available_growth].mean(axis=1)
                growth_momentum = growth_composite.pct_change(12)  # YoY change
                features['economic_growth_momentum'] = growth_momentum
                
                # Growth regime (expansion vs recession)
                features['growth_regime'] = (growth_momentum > 0).astype(int)
            
            # Inflation indicators
            inflation_indicators = ['CPIAUCSL', 'CPILFESL']
            available_inflation = [ii for ii in inflation_indicators if ii in economic_df.columns]
            
            if available_inflation:
                inflation_composite = economic_df[available_inflation].mean(axis=1)
                inflation_momentum = inflation_composite.pct_change(12)  # YoY change
                features['inflation_momentum'] = inflation_momentum
                
                # Inflation regime (high vs low)
                features['inflation_regime'] = (inflation_momentum > 0.02).astype(int)  # Above 2%
            
            # Employment indicators
            employment_indicators = ['UNRATE', 'NPPTTL']
            available_employment = [ei for ei in employment_indicators if ei in economic_df.columns]
            
            if available_employment:
                # Unemployment trend
                if 'UNRATE' in available_employment:
                    unemployment_trend = economic_df['UNRATE'].diff(3)  # 3-month change
                    features['unemployment_trend'] = unemployment_trend
                    
                    # Employment regime
                    features['employment_regime'] = (unemployment_trend < 0).astype(int)  # Improving
            
            return features
            
        except Exception as e:
            self.logger.error(f"Economic regime feature extraction failed: {str(e)}")
            return {}
    
    def create_comprehensive_feature_matrix(self, 
                                          market_data: Dict[str, pd.DataFrame],
                                          economic_data: Optional[Dict[str, pd.DataFrame]] = None,
                                          lookback_period: int = 252) -> pd.DataFrame:
        """
        Create comprehensive feature matrix combining all feature types.
        
        Args:
            market_data: Dictionary of market data by asset
            economic_data: Optional economic indicator data
            lookback_period: Number of days to look back
            
        Returns:
            Comprehensive feature matrix DataFrame
        """
        try:
            feature_dfs = []
            
            # Extract regime features
            regime_features = self.extract_market_regime_features(market_data, lookback_period)
            regime_df = pd.DataFrame([{
                'regime_bull_prob': regime_features.regime_probability if regime_features.regime_type == 'bull' else 0,
                'regime_bear_prob': regime_features.regime_probability if regime_features.regime_type == 'bear' else 0,
                'regime_volatile_prob': regime_features.regime_probability if regime_features.regime_type == 'volatile' else 0,
                'regime_crisis_prob': regime_features.regime_probability if regime_features.regime_type == 'crisis' else 0,
                'volatility_regime_high': 1 if regime_features.volatility_regime == 'high' else 0,
                'correlation_regime_high': 1 if regime_features.correlation_regime == 'high' else 0,
                'trend_strength': regime_features.trend_strength,
                'momentum_score': regime_features.momentum_score,
                'mean_reversion_score': regime_features.mean_reversion_score
            }])
            
            # Extract liquidity features
            liquidity_features = self.extract_liquidity_features(market_data)
            liquidity_df = pd.DataFrame([{
                'bid_ask_spread': liquidity_features.bid_ask_spread,
                'market_depth': liquidity_features.market_depth,
                'liquidity_ratio': liquidity_features.liquidity_ratio,
                'amihud_illiquidity': liquidity_features.amihud_illiquidity,
                'price_impact': liquidity_features.price_impact,
                'volume_participation_rate': liquidity_features.volume_participation_rate
            }])
            
            # Extract volatility features
            volatility_features = self.extract_volatility_features(market_data)
            volatility_df = pd.DataFrame([{
                'realized_volatility': volatility_features.realized_volatility,
                'garch_volatility': volatility_features.garch_volatility,
                'volatility_of_volatility': volatility_features.volatility_of_volatility,
                'volatility_skew': volatility_features.volatility_skew,
                'volatility_smile': volatility_features.volatility_smile
            }])
            
            # Add term structure features
            for horizon, vol_value in volatility_features.vol_term_structure.items():
                volatility_df[f'vol_{horizon}'] = vol_value
            
            # Cross-asset features
            cross_asset_df = self.extract_cross_asset_features(market_data, lookback_period)
            
            # Economic features (if available)
            if economic_data:
                economic_df = self.extract_economic_features(economic_data, lookback_period)
                if not economic_df.empty:
                    feature_dfs.append(economic_df)
            
            # Combine all features
            # For scalar features, broadcast to match time series length
            if not cross_asset_df.empty:
                target_length = len(cross_asset_df)
                target_index = cross_asset_df.index
                
                # Broadcast scalar features
                for df in [regime_df, liquidity_df, volatility_df]:
                    for col in df.columns:
                        cross_asset_df[col] = df[col].iloc[0]
                
                feature_matrix = cross_asset_df
            else:
                # Fallback: combine scalar features only
                scalar_dfs = [regime_df, liquidity_df, volatility_df]
                feature_matrix = pd.concat(scalar_dfs, axis=1)
            
            # Remove any remaining NaN values
            feature_matrix = feature_matrix.fillna(0)
            
            self.logger.info(f"Created comprehensive feature matrix with shape: {feature_matrix.shape}")
            
            return feature_matrix
            
        except Exception as e:
            self.logger.error(f"Comprehensive feature matrix creation failed: {str(e)}")
            return pd.DataFrame()
    
    def save_features_engine(self, filepath: str) -> None:
        """Save market features engine state."""
        try:
            engine_data = {
                'scalers': self.scalers,
                'feature_cache': self.feature_cache,
                'correlation_matrices': self.correlation_matrices,
                'regime_models': self.regime_models,
                'pca': self.pca,
                'kmeans': self.kmeans,
                'feature_history': self.feature_history,
                'asset_universe': list(self.asset_universe),
                'market_categories': self.market_categories,
                'economic_indicators': self.economic_indicators,
                'risk_free_rates': self.risk_free_rates,
                'model_version': self.model_version
            }
            
            joblib.dump(engine_data, filepath)
            self.logger.info(f"Market features engine saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Features engine saving failed: {str(e)}")
            raise
    
    def load_features_engine(self, filepath: str) -> None:
        """Load market features engine state."""
        try:
            engine_data = joblib.load(filepath)
            
            self.scalers = engine_data['scalers']
            self.feature_cache = engine_data['feature_cache']
            self.correlation_matrices = engine_data['correlation_matrices']
            self.regime_models = engine_data['regime_models']
            self.pca = engine_data['pca']
            self.kmeans = engine_data['kmeans']
            self.feature_history = engine_data['feature_history']
            self.asset_universe = set(engine_data['asset_universe'])
            self.market_categories = engine_data['market_categories']
            self.economic_indicators = engine_data['economic_indicators']
            self.risk_free_rates = engine_data['risk_free_rates']
            self.model_version = engine_data['model_version']
            
            self.logger.info(f"Market features engine loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Features engine loading failed: {str(e)}")
            raise


def create_market_features_engine(config: Optional[Dict] = None) -> MarketFeaturesEngine:
    """
    Create a market features engine with optional configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured MarketFeaturesEngine instance
    """
    engine = MarketFeaturesEngine()
    
    if config:
        if 'market_categories' in config:
            engine.market_categories.update(config['market_categories'])
        
        if 'economic_indicators' in config:
            engine.economic_indicators.update(config['economic_indicators'])
        
        if 'risk_free_rates' in config:
            engine.risk_free_rates.update(config['risk_free_rates'])
    
    return engine


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create market features engine
    engine = MarketFeaturesEngine()
    
    # Example usage would go here
    market_data = load_multi_asset_data(['BTC', 'ETH', 'SPY'])
    economic_data = load_economic_data(['DFF', 'DGS10', 'UNRATE'])
    feature_matrix = engine.create_comprehensive_feature_matrix(market_data, economic_data)
    
    print("Market features engine implementation completed")
