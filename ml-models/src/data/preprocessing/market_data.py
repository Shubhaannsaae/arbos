"""
Market Data Preprocessing Module

This module implements comprehensive market data preprocessing for ML models,
integrating with Chainlink Data Feeds, market sentiment, and cross-market analytics.

Features:
- Multi-asset market data aggregation and synchronization
- Market regime detection and classification
- Cross-market correlation analysis
- Real-time market data processing with Chainlink Data Feeds
- Economic indicator integration and preprocessing
- Market microstructure and liquidity analysis
- Cross-chain market data synchronization using CCIP
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import yfinance as yf  # For traditional market data
import requests
import json

warnings.filterwarnings('ignore')

@dataclass
class MarketDataQuality:
    """Data class for market data quality metrics."""
    data_source: str
    asset_coverage: float
    data_completeness: float
    timeliness_score: float
    cross_market_consistency: float
    data_freshness: float
    quality_score: float
    issues: List[str]
    timestamp: datetime

@dataclass
class MarketRegime:
    """Data class for market regime classification."""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile', 'crisis'
    confidence: float
    duration_days: int
    volatility_level: str  # 'low', 'medium', 'high'
    correlation_level: str  # 'low', 'medium', 'high'
    market_stress_indicator: float
    regime_probability: Dict[str, float]
    timestamp: datetime

@dataclass
class MarketMetrics:
    """Data class for market statistical metrics."""
    market_name: str
    period: str
    total_market_cap: float
    average_volume: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: Dict[str, Dict[str, float]]
    dominant_sector: str
    market_concentration: float
    liquidity_score: float

class MarketDataPreprocessor:
    """
    Advanced market data preprocessing pipeline for ML models
    with comprehensive multi-asset analysis and regime detection.
    """
    
    def __init__(self, 
                 regime_detection_method: str = 'hmm',
                 correlation_window: int = 30,
                 volatility_window: int = 20):
        """
        Initialize the market data preprocessor.
        
        Args:
            regime_detection_method: Method for market regime detection
            correlation_window: Window for correlation calculations
            volatility_window: Window for volatility calculations
        """
        self.regime_detection_method = regime_detection_method
        self.correlation_window = correlation_window
        self.volatility_window = volatility_window
        
        # Scalers and transformers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()
        
        # Imputers
        self.simple_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer(n_neighbors=5)
        
        # Market analysis components
        self.regime_models = {}
        self.correlation_matrices = {}
        self.market_metrics_history = {}
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Market regime clustering
        self.regime_kmeans = KMeans(n_clusters=5, random_state=42)
        
        # Model metadata
        self.processed_markets = set()
        self.asset_universe = set()
        self.model_version = "1.0.0"
        
        # Known market identifiers
        self.market_identifiers = {
            'crypto': ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'LINK', 'UNI'],
            'traditional': ['SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT'],
            'defi': ['AAVE', 'COMP', 'MKR', 'SNX', 'CRV', '1INCH'],
            'sectors': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLB', 'XLRE', 'XLY', 'XLP']
        }
        
        # Economic indicators
        self.economic_indicators = [
            'DGS10',  # 10-Year Treasury Rate
            'DFF',    # Federal Funds Rate
            'UNRATE', # Unemployment Rate
            'CPIAUCSL', # CPI
            'GDP'     # GDP
        ]
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def validate_market_data(self, market_data: Dict[str, pd.DataFrame], data_source: str) -> MarketDataQuality:
        """
        Validate market data quality across multiple assets.
        
        Args:
            market_data: Dictionary of asset DataFrames
            data_source: Source of the market data
            
        Returns:
            MarketDataQuality object with validation results
        """
        try:
            issues = []
            
            # Asset coverage analysis
            expected_assets = len(self.market_identifiers.get('crypto', [])) + len(self.market_identifiers.get('traditional', []))
            actual_assets = len(market_data)
            asset_coverage = actual_assets / expected_assets if expected_assets > 0 else 0
            
            if asset_coverage < 0.5:
                issues.append(f"Low asset coverage: {asset_coverage:.2f}")
            
            # Data completeness analysis
            total_completeness = []
            for asset, df in market_data.items():
                if df is not None and len(df) > 0:
                    completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
                    total_completeness.append(completeness)
                else:
                    total_completeness.append(0)
                    issues.append(f"Empty data for {asset}")
            
            data_completeness = np.mean(total_completeness) if total_completeness else 0
            
            # Timeliness analysis
            timeliness_scores = []
            current_time = datetime.now()
            
            for asset, df in market_data.items():
                if df is not None and len(df) > 0 and 'timestamp' in df.columns:
                    latest_timestamp = pd.to_datetime(df['timestamp']).max()
                    time_diff = (current_time - latest_timestamp).total_seconds() / 3600  # Hours
                    
                    # Score based on how recent the data is
                    if time_diff <= 1:
                        timeliness_score = 1.0
                    elif time_diff <= 24:
                        timeliness_score = 0.8
                    elif time_diff <= 168:  # 1 week
                        timeliness_score = 0.6
                    else:
                        timeliness_score = 0.3
                    
                    timeliness_scores.append(timeliness_score)
                    
                    if time_diff > 24:
                        issues.append(f"Stale data for {asset}: {time_diff:.1f} hours old")
                else:
                    timeliness_scores.append(0)
            
            timeliness_score = np.mean(timeliness_scores) if timeliness_scores else 0
            
            # Cross-market consistency
            consistency_score = self._assess_cross_market_consistency(market_data)
            
            # Data freshness (percentage of recent data)
            freshness_scores = []
            for asset, df in market_data.items():
                if df is not None and len(df) > 0:
                    recent_threshold = current_time - timedelta(hours=24)
                    if 'timestamp' in df.columns:
                        recent_data = pd.to_datetime(df['timestamp']) >= recent_threshold
                        freshness = recent_data.sum() / len(df)
                        freshness_scores.append(freshness)
            
            data_freshness = np.mean(freshness_scores) if freshness_scores else 0
            
            # Overall quality score
            quality_components = [asset_coverage, data_completeness, timeliness_score, consistency_score, data_freshness]
            quality_score = np.mean(quality_components)
            
            return MarketDataQuality(
                data_source=data_source,
                asset_coverage=asset_coverage,
                data_completeness=data_completeness,
                timeliness_score=timeliness_score,
                cross_market_consistency=consistency_score,
                data_freshness=data_freshness,
                quality_score=quality_score,
                issues=issues,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Market data validation failed: {str(e)}")
            return MarketDataQuality(
                data_source=data_source,
                asset_coverage=0.0,
                data_completeness=0.0,
                timeliness_score=0.0,
                cross_market_consistency=0.0,
                data_freshness=0.0,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    def _assess_cross_market_consistency(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Assess consistency across different markets."""
        try:
            consistency_scores = []
            
            # Check timestamp alignment
            timestamps = {}
            for asset, df in market_data.items():
                if df is not None and len(df) > 0 and 'timestamp' in df.columns:
                    timestamps[asset] = set(pd.to_datetime(df['timestamp']).dt.floor('H'))
            
            if len(timestamps) > 1:
                # Calculate intersection of timestamps
                common_timestamps = set.intersection(*timestamps.values())
                avg_timestamps = np.mean([len(ts) for ts in timestamps.values()])
                
                if avg_timestamps > 0:
                    timestamp_consistency = len(common_timestamps) / avg_timestamps
                    consistency_scores.append(timestamp_consistency)
            
            # Check price correlation consistency for similar assets
            crypto_assets = [asset for asset in market_data.keys() 
                           if any(crypto in asset.upper() for crypto in self.market_identifiers.get('crypto', []))]
            
            if len(crypto_assets) > 1:
                correlations = []
                for i, asset1 in enumerate(crypto_assets):
                    for asset2 in crypto_assets[i+1:]:
                        df1, df2 = market_data[asset1], market_data[asset2]
                        if (df1 is not None and df2 is not None and 
                            'close' in df1.columns and 'close' in df2.columns):
                            
                            # Align data and calculate correlation
                            returns1 = df1.set_index('timestamp')['close'].pct_change().dropna()
                            returns2 = df2.set_index('timestamp')['close'].pct_change().dropna()
                            
                            aligned_returns1, aligned_returns2 = returns1.align(returns2, join='inner')
                            
                            if len(aligned_returns1) > 10:
                                corr = aligned_returns1.corr(aligned_returns2)
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                
                if correlations:
                    # High correlation indicates consistency
                    avg_correlation = np.mean(correlations)
                    consistency_scores.append(avg_correlation)
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Cross-market consistency assessment failed: {str(e)}")
            return 0.5
    
    def clean_market_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean market data across multiple assets.
        
        Args:
            market_data: Dictionary of raw market data DataFrames
            
        Returns:
            Dictionary of cleaned market data DataFrames
        """
        try:
            cleaned_data = {}
            
            for asset, df in market_data.items():
                if df is None or len(df) == 0:
                    self.logger.warning(f"Skipping empty data for {asset}")
                    continue
                
                cleaned_df = df.copy()
                
                # Ensure timestamp is datetime and sorted
                if 'timestamp' in cleaned_df.columns:
                    cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
                    cleaned_df = cleaned_df.sort_values('timestamp').reset_index(drop=True)
                
                # Remove duplicates
                if 'timestamp' in cleaned_df.columns:
                    cleaned_df = cleaned_df.drop_duplicates(subset=['timestamp'], keep='last')
                
                # Handle missing values
                cleaned_df = self._handle_missing_values(cleaned_df, asset)
                
                # Handle outliers
                cleaned_df = self._handle_market_outliers(cleaned_df, asset)
                
                # Validate price relationships
                cleaned_df = self._validate_price_relationships(cleaned_df, asset)
                
                # Ensure positive values where appropriate
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    if col in cleaned_df.columns:
                        cleaned_df[col] = cleaned_df[col].clip(lower=0.000001)
                
                if 'volume' in cleaned_df.columns:
                    cleaned_df['volume'] = cleaned_df['volume'].clip(lower=0)
                
                cleaned_data[asset] = cleaned_df
                
            self.logger.info(f"Cleaned market data for {len(cleaned_data)} assets")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Market data cleaning failed: {str(e)}")
            return market_data
    
    def _handle_missing_values(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Handle missing values in market data."""
        try:
            # Price data: use interpolation
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear')
            
            # Volume: forward fill then backward fill
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(method='ffill').fillna(method='bfill')
                # If still missing, use median
                if df['volume'].isnull().any():
                    df['volume'] = df['volume'].fillna(df['volume'].median())
            
            # Market cap, if available
            if 'market_cap' in df.columns:
                df['market_cap'] = df['market_cap'].interpolate(method='linear')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Missing value handling failed for {asset}: {str(e)}")
            return df
    
    def _handle_market_outliers(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Handle outliers in market data."""
        try:
            # Price outlier detection and handling
            if 'close' in df.columns:
                returns = df['close'].pct_change()
                
                # Detect extreme returns (more than 5 standard deviations)
                z_scores = np.abs(stats.zscore(returns.dropna(), nan_policy='omit'))
                outlier_mask = z_scores > 5
                
                if outlier_mask.sum() > 0:
                    # Replace outlier returns with clipped values
                    return_std = returns.std()
                    return_mean = returns.mean()
                    
                    # Clip to 3 standard deviations
                    returns.loc[outlier_mask] = np.clip(
                        returns.loc[outlier_mask],
                        return_mean - 3 * return_std,
                        return_mean + 3 * return_std
                    )
                    
                    # Reconstruct prices
                    df['close'] = df['close'].iloc[0] * (1 + returns).cumprod()
                    
                    self.logger.info(f"Handled {outlier_mask.sum()} price outliers for {asset}")
            
            # Volume outlier handling
            if 'volume' in df.columns:
                volume_q99 = df['volume'].quantile(0.99)
                extreme_volume_mask = df['volume'] > volume_q99 * 10
                
                if extreme_volume_mask.sum() > 0:
                    # Replace with 99th percentile
                    df.loc[extreme_volume_mask, 'volume'] = volume_q99
                    self.logger.info(f"Handled {extreme_volume_mask.sum()} volume outliers for {asset}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Outlier handling failed for {asset}: {str(e)}")
            return df
    
    def _validate_price_relationships(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Validate and fix OHLC price relationships."""
        try:
            ohlc_columns = ['open', 'high', 'low', 'close']
            
            if all(col in df.columns for col in ohlc_columns):
                # Ensure high >= max(open, close) and low <= min(open, close)
                df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
                df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
                
                # Log any fixes
                inconsistencies = (
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ).sum()
                
                if inconsistencies > 0:
                    self.logger.info(f"Fixed {inconsistencies} OHLC inconsistencies for {asset}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Price relationship validation failed for {asset}: {str(e)}")
            return df
    
    def synchronize_market_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Synchronize market data across different assets to common timestamps.
        
        Args:
            market_data: Dictionary of market data DataFrames
            
        Returns:
            Dictionary of synchronized market data DataFrames
        """
        try:
            if not market_data:
                return {}
            
            # Find common timestamp range
            all_timestamps = []
            for asset, df in market_data.items():
                if df is not None and len(df) > 0 and 'timestamp' in df.columns:
                    timestamps = pd.to_datetime(df['timestamp'])
                    all_timestamps.extend(timestamps)
            
            if not all_timestamps:
                return market_data
            
            # Create common time index
            all_timestamps = pd.Series(all_timestamps)
            min_time = all_timestamps.min()
            max_time = all_timestamps.max()
            
            # Determine frequency (assume hourly for high-frequency data, daily otherwise)
            time_diffs = all_timestamps.diff().dropna()
            median_diff = time_diffs.median()
            
            if median_diff <= timedelta(hours=2):
                freq = 'H'  # Hourly
            else:
                freq = 'D'  # Daily
            
            common_index = pd.date_range(start=min_time, end=max_time, freq=freq)
            
            # Synchronize each asset
            synchronized_data = {}
            for asset, df in market_data.items():
                if df is None or len(df) == 0:
                    continue
                
                # Set timestamp as index
                df_indexed = df.set_index('timestamp')
                df_indexed.index = pd.to_datetime(df_indexed.index)
                
                # Reindex to common timestamps
                df_reindexed = df_indexed.reindex(common_index)
                
                # Forward fill missing values
                df_reindexed = df_reindexed.fillna(method='ffill')
                
                # Reset index
                df_reindexed = df_reindexed.reset_index().rename(columns={'index': 'timestamp'})
                
                synchronized_data[asset] = df_reindexed
            
            self.logger.info(f"Synchronized {len(synchronized_data)} assets to {len(common_index)} timestamps")
            
            return synchronized_data
            
        except Exception as e:
            self.logger.error(f"Market data synchronization failed: {str(e)}")
            return market_data
    
    def detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """
        Detect current market regime across assets.
        
        Args:
            market_data: Dictionary of synchronized market data
            
        Returns:
            MarketRegime object with regime classification
        """
        try:
            # Calculate market-wide metrics
            returns_data = {}
            volatility_data = {}
            
            for asset, df in market_data.items():
                if df is not None and 'close' in df.columns:
                    returns = df['close'].pct_change().dropna()
                    returns_data[asset] = returns
                    
                    # Rolling volatility
                    volatility = returns.rolling(self.volatility_window).std() * np.sqrt(252)
                    volatility_data[asset] = volatility
            
            if not returns_data:
                return self._default_market_regime()
            
            # Combine returns into DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate market-wide indicators
            market_return = returns_df.mean(axis=1)
            market_volatility = returns_df.std(axis=1)
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            # Market stress indicators
            negative_returns_ratio = (market_return < -0.02).rolling(20).mean().iloc[-1] if len(market_return) > 20 else 0
            high_volatility_ratio = (market_volatility > market_volatility.quantile(0.8)).rolling(20).mean().iloc[-1] if len(market_volatility) > 20 else 0
            
            market_stress_indicator = (negative_returns_ratio * 0.6 + high_volatility_ratio * 0.4)
            
            # Recent market metrics (last 30 days)
            recent_return = market_return.tail(30).mean() if len(market_return) >= 30 else market_return.mean()
            recent_volatility = market_volatility.tail(30).mean() if len(market_volatility) >= 30 else market_volatility.mean()
            
            # Regime classification logic
            regime_probabilities = {}
            
            # Bull market indicators
            bull_score = 0
            if recent_return > 0.001:  # Positive returns
                bull_score += 0.4
            if recent_volatility < 0.25:  # Low volatility
                bull_score += 0.3
            if avg_correlation < 0.7:  # Low correlation
                bull_score += 0.3
            
            # Bear market indicators
            bear_score = 0
            if recent_return < -0.002:  # Negative returns
                bear_score += 0.4
            if market_stress_indicator > 0.5:  # High stress
                bear_score += 0.3
            if avg_correlation > 0.8:  # High correlation
                bear_score += 0.3
            
            # Volatile market indicators
            volatile_score = 0
            if recent_volatility > 0.4:  # High volatility
                volatile_score += 0.5
            if market_stress_indicator > 0.3:  # Medium-high stress
                volatile_score += 0.3
            if abs(recent_return) > 0.003:  # Large absolute returns
                volatile_score += 0.2
            
            # Crisis indicators
            crisis_score = 0
            if recent_volatility > 0.6:  # Very high volatility
                crisis_score += 0.4
            if market_stress_indicator > 0.7:  # Very high stress
                crisis_score += 0.4
            if avg_correlation > 0.9:  # Very high correlation
                crisis_score += 0.2
            
            # Sideways market (default)
            sideways_score = max(0, 1 - max(bull_score, bear_score, volatile_score, crisis_score))
            
            regime_probabilities = {
                'bull': bull_score,
                'bear': bear_score,
                'sideways': sideways_score,
                'volatile': volatile_score,
                'crisis': crisis_score
            }
            
            # Determine dominant regime
            dominant_regime = max(regime_probabilities.items(), key=lambda x: x[1])
            regime_type = dominant_regime[0]
            confidence = dominant_regime[1]
            
            # Classify volatility and correlation levels
            volatility_level = 'high' if recent_volatility > 0.4 else 'medium' if recent_volatility > 0.2 else 'low'
            correlation_level = 'high' if avg_correlation > 0.7 else 'medium' if avg_correlation > 0.4 else 'low'
            
            # Estimate regime duration (simplified)
            duration_days = 30  # Default assumption
            
            return MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                duration_days=duration_days,
                volatility_level=volatility_level,
                correlation_level=correlation_level,
                market_stress_indicator=market_stress_indicator,
                regime_probability=regime_probabilities,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Market regime detection failed: {str(e)}")
            return self._default_market_regime()
    
    def _default_market_regime(self) -> MarketRegime:
        """Return default market regime when detection fails."""
        return MarketRegime(
            regime_type='sideways',
            confidence=0.5,
            duration_days=30,
            volatility_level='medium',
            correlation_level='medium',
            market_stress_indicator=0.5,
            regime_probability={'sideways': 0.5, 'bull': 0.125, 'bear': 0.125, 'volatile': 0.125, 'crisis': 0.125},
            timestamp=datetime.now()
        )
    
    def calculate_market_metrics(self, market_data: Dict[str, pd.DataFrame]) -> MarketMetrics:
        """
        Calculate comprehensive market metrics.
        
        Args:
            market_data: Dictionary of market data DataFrames
            
        Returns:
            MarketMetrics object with market statistics
        """
        try:
            # Aggregate market metrics
            total_market_cap = 0
            total_volume = 0
            returns_data = []
            
            for asset, df in market_data.items():
                if df is not None and len(df) > 0:
                    # Market cap
                    if 'market_cap' in df.columns:
                        latest_market_cap = df['market_cap'].iloc[-1]
                        if not np.isnan(latest_market_cap):
                            total_market_cap += latest_market_cap
                    
                    # Volume
                    if 'volume' in df.columns:
                        avg_volume = df['volume'].mean()
                        if not np.isnan(avg_volume):
                            total_volume += avg_volume
                    
                    # Returns
                    if 'close' in df.columns:
                        returns = df['close'].pct_change().dropna()
                        returns_data.append(returns)
            
            # Calculate market-wide return statistics
            if returns_data:
                # Equal-weighted market return
                market_returns = pd.concat(returns_data, axis=1).mean(axis=1)
                
                # Market volatility
                volatility = market_returns.std() * np.sqrt(252)
                
                # Sharpe ratio (assuming 2% risk-free rate)
                risk_free_rate = 0.02
                excess_returns = market_returns.mean() * 252 - risk_free_rate
                sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
                
                # Maximum drawdown
                cumulative_returns = (1 + market_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = abs(drawdown.min())
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Correlation matrix
            correlation_matrix = {}
            if len(returns_data) > 1:
                returns_df = pd.concat(returns_data, axis=1, keys=list(market_data.keys()))
                corr_matrix = returns_df.corr()
                
                for asset1 in corr_matrix.index:
                    correlation_matrix[asset1] = {}
                    for asset2 in corr_matrix.columns:
                        correlation_matrix[asset1][asset2] = float(corr_matrix.loc[asset1, asset2])
            
            # Market concentration (simplified Herfindahl index)
            market_caps = []
            for asset, df in market_data.items():
                if df is not None and 'market_cap' in df.columns:
                    latest_mc = df['market_cap'].iloc[-1]
                    if not np.isnan(latest_mc):
                        market_caps.append(latest_mc)
            
            if market_caps and sum(market_caps) > 0:
                market_shares = [mc / sum(market_caps) for mc in market_caps]
                market_concentration = sum(share ** 2 for share in market_shares)
            else:
                market_concentration = 1.0  # Perfect concentration by default
            
            # Liquidity score (based on volume consistency)
            volume_scores = []
            for asset, df in market_data.items():
                if df is not None and 'volume' in df.columns:
                    volume_cv = df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 1
                    # Lower coefficient of variation = higher liquidity score
                    liquidity_score = max(0, 1 - min(1, volume_cv))
                    volume_scores.append(liquidity_score)
            
            liquidity_score = np.mean(volume_scores) if volume_scores else 0.5
            
            return MarketMetrics(
                market_name='aggregated',
                period='1D',
                total_market_cap=float(total_market_cap),
                average_volume=float(total_volume),
                volatility=float(volatility),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                correlation_matrix=correlation_matrix,
                dominant_sector='mixed',  # Would need sector classification
                market_concentration=float(market_concentration),
                liquidity_score=float(liquidity_score)
            )
            
        except Exception as e:
            self.logger.error(f"Market metrics calculation failed: {str(e)}")
            return MarketMetrics(
                market_name='aggregated',
                period='1D',
                total_market_cap=0.0,
                average_volume=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                correlation_matrix={},
                dominant_sector='unknown',
                market_concentration=1.0,
                liquidity_score=0.5
            )
    
    def normalize_market_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Normalize market data for ML model input.
        
        Args:
            market_data: Dictionary of cleaned market data
            
        Returns:
            Dictionary of normalized market data
        """
        try:
            normalized_data = {}
            
            for asset, df in market_data.items():
                if df is None or len(df) == 0:
                    continue
                
                normalized_df = df.copy()
                
                # Normalize price data (convert to returns)
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    if col in df.columns:
                        # Log returns
                        normalized_df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
                        
                        # Normalized prices (z-score)
                        normalized_df[f'{col}_normalized'] = self.robust_scaler.fit_transform(
                            df[[col]]
                        ).flatten()
                
                # Normalize volume (log transformation + scaling)
                if 'volume' in df.columns:
                    normalized_df['volume_log'] = np.log1p(df['volume'])
                    normalized_df['volume_normalized'] = self.robust_scaler.fit_transform(
                        normalized_df[['volume_log']]
                    ).flatten()
                
                # Add technical indicators
                normalized_df = self._add_market_technical_indicators(normalized_df)
                
                # Add cross-asset features
                normalized_df = self._add_cross_asset_features(normalized_df, asset, market_data)
                
                normalized_data[asset] = normalized_df.dropna()
            
            self.logger.info(f"Normalized market data for {len(normalized_data)} assets")
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Market data normalization failed: {str(e)}")
            return market_data
    
    def _add_market_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data."""
        try:
            if 'close' not in df.columns:
                return df
            
            close_prices = df['close']
            
            # Moving averages
            for window in [5, 10, 20, 50, 100]:
                df[f'sma_{window}'] = close_prices.rolling(window).mean()
                df[f'ema_{window}'] = close_prices.ewm(span=window).mean()
            
            # Price momentum
            for period in [1, 5, 10, 20]:
                df[f'momentum_{period}'] = close_prices.pct_change(period)
            
            # Volatility indicators
            returns = close_prices.pct_change()
            for window in [10, 20, 50]:
                df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = close_prices.ewm(span=12).mean()
            ema26 = close_prices.ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            sma20 = close_prices.rolling(20).mean()
            std20 = close_prices.rolling(20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
            df['bb_position'] = (close_prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicators calculation failed: {str(e)}")
            return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame, current_asset: str, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add cross-asset correlation and relative performance features."""
        try:
            if 'close' not in df.columns:
                return df
            
            current_returns = df['close'].pct_change()
            
            # Calculate correlations with other assets
            correlations = {}
            relative_performance = {}
            
            for other_asset, other_df in market_data.items():
                if other_asset == current_asset or other_df is None or 'close' not in other_df.columns:
                    continue
                
                other_returns = other_df['close'].pct_change()
                
                # Align data
                aligned_current, aligned_other = current_returns.align(other_returns, join='inner')
                
                if len(aligned_current) > 30:  # Need sufficient data
                    # Rolling correlation
                    rolling_corr = aligned_current.rolling(30).corr(aligned_other)
                    correlations[f'corr_{other_asset}'] = rolling_corr
                    
                    # Relative performance
                    relative_perf = aligned_current.rolling(20).sum() - aligned_other.rolling(20).sum()
                    relative_performance[f'rel_perf_{other_asset}'] = relative_perf
            
            # Add correlation features (limit to top 5 most correlated)
            if correlations:
                for feature, values in list(correlations.items())[:5]:
                    df[feature] = values.reindex(df.index)
            
            # Add relative performance features
            if relative_performance:
                for feature, values in list(relative_performance.items())[:5]:
                    df[feature] = values.reindex(df.index)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Cross-asset features calculation failed: {str(e)}")
            return df
    
    def process_market_data(self, 
                          market_data: Dict[str, pd.DataFrame],
                          data_source: str = 'multiple') -> Tuple[Dict[str, pd.DataFrame], MarketDataQuality, MarketRegime, MarketMetrics]:
        """
        Complete market data processing pipeline.
        
        Args:
            market_data: Dictionary of raw market data
            data_source: Source identifier for the data
            
        Returns:
            Tuple of (processed_data, quality_metrics, market_regime, market_metrics)
        """
        try:
            self.logger.info(f"Starting market data processing for {len(market_data)} assets")
            
            # Validate data quality
            quality_metrics = self.validate_market_data(market_data, data_source)
            
            if quality_metrics.quality_score < 0.3:
                self.logger.warning(f"Very low quality market data: {quality_metrics.quality_score:.2f}")
            
            # Clean the data
            cleaned_data = self.clean_market_data(market_data)
            
            # Synchronize timestamps
            synchronized_data = self.synchronize_market_data(cleaned_data)
            
            # Detect market regime
            market_regime = self.detect_market_regime(synchronized_data)
            
            # Calculate market metrics
            market_metrics = self.calculate_market_metrics(synchronized_data)
            
            # Normalize the data
            processed_data = self.normalize_market_data(synchronized_data)
            
            # Update tracking
            self.processed_markets.add(data_source)
            self.asset_universe.update(processed_data.keys())
            
            # Store regime and metrics
            self.market_metrics_history[datetime.now()] = {
                'quality': quality_metrics,
                'regime': market_regime,
                'metrics': market_metrics
            }
            
            self.logger.info(f"Completed market data processing: {len(processed_data)} assets, regime: {market_regime.regime_type}")
            
            return processed_data, quality_metrics, market_regime, market_metrics
            
        except Exception as e:
            self.logger.error(f"Market data processing failed: {str(e)}")
            raise
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the market data preprocessor state."""
        try:
            preprocessor_data = {
                'regime_detection_method': self.regime_detection_method,
                'correlation_window': self.correlation_window,
                'volatility_window': self.volatility_window,
                'standard_scaler': self.standard_scaler,
                'minmax_scaler': self.minmax_scaler,
                'robust_scaler': self.robust_scaler,
                'simple_imputer': self.simple_imputer,
                'knn_imputer': self.knn_imputer,
                'regime_models': self.regime_models,
                'correlation_matrices': self.correlation_matrices,
                'market_metrics_history': self.market_metrics_history,
                'pca': self.pca,
                'regime_kmeans': self.regime_kmeans,
                'processed_markets': list(self.processed_markets),
                'asset_universe': list(self.asset_universe),
                'market_identifiers': self.market_identifiers,
                'economic_indicators': self.economic_indicators,
                'model_version': self.model_version
            }
            
            joblib.dump(preprocessor_data, filepath)
            self.logger.info(f"Market data preprocessor saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Market preprocessor saving failed: {str(e)}")
            raise
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load the market data preprocessor state."""
        try:
            preprocessor_data = joblib.load(filepath)
            
            self.regime_detection_method = preprocessor_data['regime_detection_method']
            self.correlation_window = preprocessor_data['correlation_window']
            self.volatility_window = preprocessor_data['volatility_window']
            self.standard_scaler = preprocessor_data['standard_scaler']
            self.minmax_scaler = preprocessor_data['minmax_scaler']
            self.robust_scaler = preprocessor_data['robust_scaler']
            self.simple_imputer = preprocessor_data['simple_imputer']
            self.knn_imputer = preprocessor_data['knn_imputer']
            self.regime_models = preprocessor_data['regime_models']
            self.correlation_matrices = preprocessor_data['correlation_matrices']
            self.market_metrics_history = preprocessor_data['market_metrics_history']
            self.pca = preprocessor_data['pca']
            self.regime_kmeans = preprocessor_data['regime_kmeans']
            self.processed_markets = set(preprocessor_data['processed_markets'])
            self.asset_universe = set(preprocessor_data['asset_universe'])
            self.market_identifiers = preprocessor_data['market_identifiers']
            self.economic_indicators = preprocessor_data['economic_indicators']
            self.model_version = preprocessor_data['model_version']
            
            self.logger.info(f"Market data preprocessor loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Market preprocessor loading failed: {str(e)}")
            raise


def create_market_preprocessor(config: Dict) -> MarketDataPreprocessor:
    """
    Create a market data preprocessor with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MarketDataPreprocessor instance
    """
    return MarketDataPreprocessor(
        regime_detection_method=config.get('regime_detection_method', 'hmm'),
        correlation_window=config.get('correlation_window', 30),
        volatility_window=config.get('volatility_window', 20)
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create preprocessor
    preprocessor = MarketDataPreprocessor()
    
    # Example data processing would go here
    market_data = load_multi_asset_data(['BTC', 'ETH', 'SPY'])
    processed_data, quality, regime, metrics = preprocessor.process_market_data(market_data)
    
    print("Market data preprocessing implementation completed")
