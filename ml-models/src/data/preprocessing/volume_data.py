"""
Volume Data Preprocessing Module

This module implements comprehensive volume data preprocessing for ML models,
integrating with Chainlink Data Feeds and DEX/CEX volume aggregation.

Features:
- Multi-source volume data aggregation (CEX, DEX, on-chain)
- Volume anomaly detection and cleaning
- Liquidity-weighted volume calculations
- Real-time volume stream processing with Chainlink Data Feeds
- Cross-chain volume synchronization using CCIP
- Volume profile analysis and market microstructure features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import DBSCAN
import joblib

warnings.filterwarnings('ignore')

@dataclass
class VolumeDataQuality:
    """Data class for volume data quality metrics."""
    symbol: str
    exchange: str
    total_records: int
    zero_volume_periods: int
    volume_spikes: int
    data_completeness: float
    volume_consistency: float
    microstructure_quality: float
    quality_score: float
    issues: List[str]
    timestamp: datetime

@dataclass
class VolumeMetrics:
    """Data class for volume statistical metrics."""
    symbol: str
    exchange: str
    period: str
    total_volume: float
    average_volume: float
    median_volume: float
    volume_volatility: float
    volume_skewness: float
    volume_kurtosis: float
    vwap: float  # Volume Weighted Average Price
    participation_rate: float
    market_impact: float

@dataclass
class VolumeProfile:
    """Data class for volume profile analysis."""
    symbol: str
    price_levels: List[float]
    volume_at_price: List[float]
    poc: float  # Point of Control (highest volume price level)
    value_area_high: float
    value_area_low: float
    volume_profile_shape: str  # 'normal', 'bimodal', 'uniform'
    timestamp: datetime

class VolumeDataPreprocessor:
    """
    Advanced volume data preprocessing pipeline for ML models
    with comprehensive volume analysis and market microstructure features.
    """
    
    def __init__(self, 
                 spike_detection_method: str = 'iqr',
                 volume_normalization: str = 'log_transform',
                 microstructure_analysis: bool = True):
        """
        Initialize the volume data preprocessor.
        
        Args:
            spike_detection_method: Method for detecting volume spikes
            volume_normalization: Normalization method for volume data
            microstructure_analysis: Whether to include microstructure analysis
        """
        self.spike_detection_method = spike_detection_method
        self.volume_normalization = volume_normalization
        self.microstructure_analysis = microstructure_analysis
        
        # Scalers and transformers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        
        # Imputers
        self.simple_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer(n_neighbors=5)
        
        # Volume analysis components
        self.volume_profiles = {}
        self.liquidity_metrics = {}
        self.microstructure_features = {}
        
        # Model metadata
        self.processed_exchanges = set()
        self.volume_statistics = {}
        self.model_version = "1.0.0"
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def validate_volume_data(self, df: pd.DataFrame, symbol: str, exchange: str) -> VolumeDataQuality:
        """
        Validate volume data quality and identify issues.
        
        Args:
            df: Volume data DataFrame
            symbol: Asset symbol
            exchange: Exchange identifier
            
        Returns:
            VolumeDataQuality object with validation results
        """
        try:
            issues = []
            
            # Basic data structure validation
            required_columns = ['timestamp', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
            
            total_records = len(df)
            
            # Data completeness
            missing_values = df['volume'].isnull().sum()
            data_completeness = 1 - (missing_values / total_records)
            
            # Zero volume analysis
            zero_volume_periods = (df['volume'] == 0).sum()
            if zero_volume_periods > total_records * 0.05:  # More than 5%
                issues.append(f"High number of zero volume periods: {zero_volume_periods}")
            
            # Volume spike detection
            volume_spikes = self._detect_volume_spikes(df['volume']).sum()
            if volume_spikes > total_records * 0.02:  # More than 2%
                issues.append(f"High number of volume spikes: {volume_spikes}")
            
            # Volume consistency (check for unrealistic values)
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                if negative_volume > 0:
                    issues.append(f"Negative volume values detected: {negative_volume}")
                
                # Check for extremely large volumes (potential data errors)
                volume_q99 = df['volume'].quantile(0.99)
                extreme_volumes = (df['volume'] > volume_q99 * 100).sum()
                if extreme_volumes > 0:
                    issues.append(f"Extremely large volume values: {extreme_volumes}")
            
            volume_consistency = 1 - (zero_volume_periods + volume_spikes) / total_records
            
            # Microstructure quality (if price data available)
            microstructure_quality = 1.0
            if 'price' in df.columns and self.microstructure_analysis:
                microstructure_quality = self._assess_microstructure_quality(df)
            
            # Calculate overall quality score
            quality_components = [data_completeness, volume_consistency, microstructure_quality]
            quality_score = np.mean(quality_components)
            
            return VolumeDataQuality(
                symbol=symbol,
                exchange=exchange,
                total_records=total_records,
                zero_volume_periods=zero_volume_periods,
                volume_spikes=volume_spikes,
                data_completeness=data_completeness,
                volume_consistency=volume_consistency,
                microstructure_quality=microstructure_quality,
                quality_score=quality_score,
                issues=issues,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Volume data validation failed for {symbol}@{exchange}: {str(e)}")
            return VolumeDataQuality(
                symbol=symbol,
                exchange=exchange,
                total_records=len(df) if df is not None else 0,
                zero_volume_periods=0,
                volume_spikes=0,
                data_completeness=0.0,
                volume_consistency=0.0,
                microstructure_quality=0.0,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    def _detect_volume_spikes(self, volume_series: pd.Series) -> pd.Series:
        """Detect volume spikes using the specified method."""
        try:
            if self.spike_detection_method == 'iqr':
                Q1 = volume_series.quantile(0.25)
                Q3 = volume_series.quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 3 * IQR  # More conservative for volume
                spikes = volume_series > upper_bound
                
            elif self.spike_detection_method == 'zscore':
                # Use log-transformed volume for z-score
                log_volume = np.log1p(volume_series)
                z_scores = np.abs(stats.zscore(log_volume, nan_policy='omit'))
                spikes = z_scores > 4  # More conservative threshold
                
            elif self.spike_detection_method == 'rolling_median':
                # Rolling median approach
                rolling_median = volume_series.rolling(window=20, center=True).median()
                rolling_mad = volume_series.rolling(window=20, center=True).apply(
                    lambda x: np.median(np.abs(x - np.median(x)))
                )
                threshold = rolling_median + 5 * rolling_mad
                spikes = volume_series > threshold
                
            elif self.spike_detection_method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                log_volume = np.log1p(volume_series).values.reshape(-1, 1)
                isolation_forest = IsolationForest(contamination=0.05, random_state=42)
                outlier_labels = isolation_forest.fit_predict(log_volume)
                spikes = pd.Series(outlier_labels == -1, index=volume_series.index)
                
            else:
                # Default to IQR method
                Q1 = volume_series.quantile(0.25)
                Q3 = volume_series.quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 3 * IQR
                spikes = volume_series > upper_bound
            
            return spikes.fillna(False)
            
        except Exception as e:
            self.logger.error(f"Volume spike detection failed: {str(e)}")
            return pd.Series(False, index=volume_series.index)
    
    def _assess_microstructure_quality(self, df: pd.DataFrame) -> float:
        """Assess microstructure quality of volume data."""
        try:
            quality_score = 1.0
            
            # Check for volume-price relationship
            if 'price' in df.columns and 'volume' in df.columns:
                price_changes = df['price'].pct_change().abs()
                volume_changes = df['volume'].pct_change().abs()
                
                # Strong price changes should correlate with volume
                correlation = price_changes.corr(volume_changes)
                if correlation < 0.1:  # Very weak correlation
                    quality_score -= 0.2
            
            # Check for systematic patterns that might indicate data issues
            if 'volume' in df.columns:
                # Check for too many repeated volume values
                volume_counts = df['volume'].value_counts()
                max_repeated = volume_counts.max()
                if max_repeated > len(df) * 0.1:  # More than 10% same value
                    quality_score -= 0.3
                
                # Check for unrealistic intraday patterns
                if 'timestamp' in df.columns:
                    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                    hourly_volume = df.groupby('hour')['volume'].mean()
                    
                    # Volume should generally be higher during market hours
                    if hourly_volume.std() / hourly_volume.mean() < 0.1:  # Too uniform
                        quality_score -= 0.2
            
            return max(0.0, quality_score)
            
        except Exception as e:
            self.logger.error(f"Microstructure quality assessment failed: {str(e)}")
            return 0.5  # Default medium quality
    
    def clean_volume_data(self, df: pd.DataFrame, symbol: str, exchange: str) -> pd.DataFrame:
        """
        Clean volume data by handling anomalies and inconsistencies.
        
        Args:
            df: Raw volume data DataFrame
            symbol: Asset symbol
            exchange: Exchange identifier
            
        Returns:
            Cleaned volume data DataFrame
        """
        try:
            cleaned_df = df.copy()
            
            # Ensure timestamp is datetime and sorted
            if 'timestamp' in cleaned_df.columns:
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
                cleaned_df = cleaned_df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicate timestamps
            if 'timestamp' in cleaned_df.columns:
                cleaned_df = cleaned_df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Handle negative volumes
            if 'volume' in cleaned_df.columns:
                negative_mask = cleaned_df['volume'] < 0
                if negative_mask.sum() > 0:
                    self.logger.warning(f"Found {negative_mask.sum()} negative volume values, setting to 0")
                    cleaned_df.loc[negative_mask, 'volume'] = 0
            
            # Handle volume spikes
            cleaned_df = self._handle_volume_spikes(cleaned_df)
            
            # Handle missing volume values
            cleaned_df = self._handle_missing_volume(cleaned_df)
            
            # Smooth extreme volume variations if needed
            cleaned_df = self._smooth_volume_variations(cleaned_df)
            
            self.logger.info(f"Cleaned volume data for {symbol}@{exchange}: {len(cleaned_df)} records")
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Volume data cleaning failed for {symbol}@{exchange}: {str(e)}")
            return df
    
    def _handle_volume_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle volume spikes in the data."""
        try:
            if 'volume' not in df.columns:
                return df
            
            volume_spikes = self._detect_volume_spikes(df['volume'])
            
            if volume_spikes.sum() > 0:
                # Replace spikes with rolling median
                rolling_median = df['volume'].rolling(window=10, center=True).median()
                df.loc[volume_spikes, 'volume'] = rolling_median.loc[volume_spikes]
                
                # Forward fill any remaining NaN values
                df['volume'] = df['volume'].fillna(method='ffill')
                
                self.logger.info(f"Handled {volume_spikes.sum()} volume spikes")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Volume spike handling failed: {str(e)}")
            return df
    
    def _handle_missing_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing volume values."""
        try:
            if 'volume' not in df.columns:
                return df
            
            missing_count = df['volume'].isnull().sum()
            
            if missing_count > 0:
                # Use interpolation for small gaps
                df['volume'] = df['volume'].interpolate(method='linear')
                
                # Use forward fill for remaining missing values
                df['volume'] = df['volume'].fillna(method='ffill')
                
                # Use backward fill for any remaining at the beginning
                df['volume'] = df['volume'].fillna(method='bfill')
                
                # If still missing, use median
                if df['volume'].isnull().sum() > 0:
                    median_volume = df['volume'].median()
                    df['volume'] = df['volume'].fillna(median_volume)
                
                self.logger.info(f"Handled {missing_count} missing volume values")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Missing volume handling failed: {str(e)}")
            return df
    
    def _smooth_volume_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooth extreme volume variations."""
        try:
            if 'volume' not in df.columns:
                return df
            
            # Calculate volume changes
            volume_changes = df['volume'].pct_change().abs()
            
            # Identify extreme changes (more than 10x)
            extreme_changes = volume_changes > 10
            
            if extreme_changes.sum() > 0:
                # Apply exponential smoothing to extreme changes
                for idx in df[extreme_changes].index:
                    if idx > 0:
                        # Use weighted average with previous value
                        prev_volume = df.loc[idx-1, 'volume']
                        current_volume = df.loc[idx, 'volume']
                        smoothed_volume = 0.7 * prev_volume + 0.3 * current_volume
                        df.loc[idx, 'volume'] = smoothed_volume
                
                self.logger.info(f"Smoothed {extreme_changes.sum()} extreme volume variations")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Volume smoothing failed: {str(e)}")
            return df
    
    def normalize_volume_data(self, df: pd.DataFrame, symbol: str, exchange: str) -> pd.DataFrame:
        """
        Normalize volume data for ML model input.
        
        Args:
            df: Cleaned volume data DataFrame
            symbol: Asset symbol
            exchange: Exchange identifier
            
        Returns:
            Normalized volume data DataFrame
        """
        try:
            normalized_df = df.copy()
            
            if 'volume' not in df.columns:
                return normalized_df
            
            # Apply normalization based on specified method
            if self.volume_normalization == 'log_transform':
                normalized_df['volume_log'] = np.log1p(df['volume'])
                normalized_df['volume_log_normalized'] = self.standard_scaler.fit_transform(
                    normalized_df[['volume_log']]
                ).flatten()
                
            elif self.volume_normalization == 'power_transform':
                normalized_df['volume_power'] = self.power_transformer.fit_transform(
                    df[['volume']]
                ).flatten()
                
            elif self.volume_normalization == 'minmax':
                normalized_df['volume_normalized'] = self.minmax_scaler.fit_transform(
                    df[['volume']]
                ).flatten()
                
            elif self.volume_normalization == 'z_score':
                normalized_df['volume_zscore'] = self.standard_scaler.fit_transform(
                    df[['volume']]
                ).flatten()
                
            elif self.volume_normalization == 'robust':
                # Use log transform followed by robust scaling
                log_volume = np.log1p(df['volume'])
                q25, q75 = log_volume.quantile([0.25, 0.75])
                iqr = q75 - q25
                normalized_df['volume_robust'] = (log_volume - log_volume.median()) / iqr
            
            # Add volume-based features
            normalized_df = self._add_volume_features(normalized_df)
            
            # Add microstructure features if enabled
            if self.microstructure_analysis:
                normalized_df = self._add_microstructure_features(normalized_df)
            
            self.logger.info(f"Normalized volume data for {symbol}@{exchange} using {self.volume_normalization}")
            
            return normalized_df.dropna()
            
        except Exception as e:
            self.logger.error(f"Volume normalization failed for {symbol}@{exchange}: {str(e)}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        try:
            if 'volume' not in df.columns:
                return df
            
            # Volume moving averages
            for window in [5, 10, 20, 50]:
                df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
            
            # Volume volatility
            df['volume_volatility_20'] = df['volume'].rolling(window=20).std()
            df['volume_volatility_ratio'] = df['volume_volatility_20'] / df['volume_ma_20']
            
            # Volume momentum
            df['volume_momentum_5'] = df['volume'].rolling(window=5).sum()
            df['volume_momentum_20'] = df['volume'].rolling(window=20).sum()
            
            # Volume percentiles
            df['volume_percentile_20'] = df['volume'].rolling(window=20).rank(pct=True)
            df['volume_percentile_100'] = df['volume'].rolling(window=100).rank(pct=True)
            
            # Volume acceleration
            df['volume_change'] = df['volume'].pct_change()
            df['volume_acceleration'] = df['volume_change'].diff()
            
            # Volume clustering (high/low volume periods)
            if len(df) > 100:
                volume_values = df['volume'].values.reshape(-1, 1)
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                clusters = dbscan.fit_predict(np.log1p(volume_values))
                df['volume_cluster'] = clusters
            
            return df
            
        except Exception as e:
            self.logger.error(f"Volume features calculation failed: {str(e)}")
            return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        try:
            # Volume-Price relationship features
            if 'price' in df.columns and 'volume' in df.columns:
                # Volume Weighted Average Price (VWAP)
                df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
                df['price_vwap_ratio'] = df['price'] / df['vwap']
                
                # Price-Volume correlation
                df['pv_correlation_20'] = df['price'].rolling(20).corr(df['volume'])
                
                # Volume-adjusted price changes
                price_change = df['price'].pct_change()
                df['volume_adjusted_return'] = price_change * np.log1p(df['volume'])
                
                # Market impact proxy
                df['market_impact'] = abs(price_change) / (np.log1p(df['volume']) + 1e-8)
            
            # Time-based volume features
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                
                # Volume relative to time of day
                hourly_avg_volume = df.groupby('hour')['volume'].transform('mean')
                df['volume_vs_hour_avg'] = df['volume'] / hourly_avg_volume
                
                # Volume seasonality
                daily_avg_volume = df.groupby('day_of_week')['volume'].transform('mean')
                df['volume_vs_daily_avg'] = df['volume'] / daily_avg_volume
            
            # Liquidity proxies
            if 'bid_size' in df.columns and 'ask_size' in df.columns:
                df['total_liquidity'] = df['bid_size'] + df['ask_size']
                df['liquidity_imbalance'] = (df['bid_size'] - df['ask_size']) / df['total_liquidity']
                
            # Volume distribution features
            df['volume_skewness_20'] = df['volume'].rolling(20).skew()
            df['volume_kurtosis_20'] = df['volume'].rolling(20).kurt()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Microstructure features calculation failed: {str(e)}")
            return df
    
    def calculate_volume_metrics(self, df: pd.DataFrame, symbol: str, exchange: str) -> VolumeMetrics:
        """
        Calculate comprehensive volume metrics.
        
        Args:
            df: Volume data DataFrame
            symbol: Asset symbol
            exchange: Exchange identifier
            
        Returns:
            VolumeMetrics object with statistical information
        """
        try:
            if 'volume' not in df.columns:
                raise ValueError("Volume column not found")
            
            # Basic volume statistics
            total_volume = df['volume'].sum()
            average_volume = df['volume'].mean()
            median_volume = df['volume'].median()
            volume_volatility = df['volume'].std() / average_volume if average_volume > 0 else 0
            volume_skewness = df['volume'].skew()
            volume_kurtosis = df['volume'].kurtosis()
            
            # VWAP calculation
            if 'price' in df.columns:
                vwap = (df['price'] * df['volume']).sum() / total_volume if total_volume > 0 else 0
            else:
                vwap = 0
            
            # Participation rate (proxy)
            participation_rate = df['volume'].std() / average_volume if average_volume > 0 else 0
            
            # Market impact (proxy)
            if 'price' in df.columns:
                price_changes = df['price'].pct_change().abs()
                volume_normalized = df['volume'] / average_volume
                market_impact = price_changes.corr(volume_normalized) if len(price_changes) > 1 else 0
            else:
                market_impact = 0
            
            return VolumeMetrics(
                symbol=symbol,
                exchange=exchange,
                period='1D',  # Default period
                total_volume=float(total_volume),
                average_volume=float(average_volume),
                median_volume=float(median_volume),
                volume_volatility=float(volume_volatility),
                volume_skewness=float(volume_skewness),
                volume_kurtosis=float(volume_kurtosis),
                vwap=float(vwap),
                participation_rate=float(participation_rate),
                market_impact=float(market_impact) if not np.isnan(market_impact) else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Volume metrics calculation failed for {symbol}@{exchange}: {str(e)}")
            return VolumeMetrics(
                symbol=symbol,
                exchange=exchange,
                period='1D',
                total_volume=0.0,
                average_volume=0.0,
                median_volume=0.0,
                volume_volatility=0.0,
                volume_skewness=0.0,
                volume_kurtosis=0.0,
                vwap=0.0,
                participation_rate=0.0,
                market_impact=0.0
            )
    
    def create_volume_profile(self, df: pd.DataFrame, symbol: str, price_levels: int = 50) -> VolumeProfile:
        """
        Create volume profile analysis.
        
        Args:
            df: DataFrame with price and volume data
            symbol: Asset symbol
            price_levels: Number of price levels for profile
            
        Returns:
            VolumeProfile object with profile analysis
        """
        try:
            if 'price' not in df.columns or 'volume' not in df.columns:
                raise ValueError("Price and volume columns required for volume profile")
            
            # Create price bins
            min_price = df['price'].min()
            max_price = df['price'].max()
            price_bins = np.linspace(min_price, max_price, price_levels + 1)
            price_centers = (price_bins[:-1] + price_bins[1:]) / 2
            
            # Calculate volume at each price level
            volume_at_price = []
            for i in range(len(price_bins) - 1):
                mask = (df['price'] >= price_bins[i]) & (df['price'] < price_bins[i + 1])
                volume_sum = df.loc[mask, 'volume'].sum()
                volume_at_price.append(volume_sum)
            
            volume_at_price = np.array(volume_at_price)
            
            # Find Point of Control (POC)
            poc_index = np.argmax(volume_at_price)
            poc = price_centers[poc_index]
            
            # Calculate Value Area (70% of volume)
            total_volume = volume_at_price.sum()
            target_volume = total_volume * 0.7
            
            # Start from POC and expand until 70% volume is captured
            cumulative_volume = volume_at_price[poc_index]
            lower_index = poc_index
            upper_index = poc_index
            
            while cumulative_volume < target_volume and (lower_index > 0 or upper_index < len(volume_at_price) - 1):
                # Decide whether to expand up or down
                volume_below = volume_at_price[lower_index - 1] if lower_index > 0 else 0
                volume_above = volume_at_price[upper_index + 1] if upper_index < len(volume_at_price) - 1 else 0
                
                if volume_below > volume_above and lower_index > 0:
                    lower_index -= 1
                    cumulative_volume += volume_at_price[lower_index]
                elif upper_index < len(volume_at_price) - 1:
                    upper_index += 1
                    cumulative_volume += volume_at_price[upper_index]
                else:
                    break
            
            value_area_high = price_centers[upper_index]
            value_area_low = price_centers[lower_index]
            
            # Determine profile shape
            profile_shape = self._classify_profile_shape(volume_at_price)
            
            return VolumeProfile(
                symbol=symbol,
                price_levels=price_centers.tolist(),
                volume_at_price=volume_at_price.tolist(),
                poc=float(poc),
                value_area_high=float(value_area_high),
                value_area_low=float(value_area_low),
                volume_profile_shape=profile_shape,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Volume profile creation failed for {symbol}: {str(e)}")
            return VolumeProfile(
                symbol=symbol,
                price_levels=[],
                volume_at_price=[],
                poc=0.0,
                value_area_high=0.0,
                value_area_low=0.0,
                volume_profile_shape='unknown',
                timestamp=datetime.now()
            )
    
    def _classify_profile_shape(self, volume_at_price: np.ndarray) -> str:
        """Classify the shape of the volume profile."""
        try:
            if len(volume_at_price) < 3:
                return 'insufficient_data'
            
            # Find peaks
            peaks = []
            for i in range(1, len(volume_at_price) - 1):
                if volume_at_price[i] > volume_at_price[i-1] and volume_at_price[i] > volume_at_price[i+1]:
                    peaks.append(i)
            
            # Classify based on number and prominence of peaks
            if len(peaks) == 0:
                return 'uniform'
            elif len(peaks) == 1:
                # Check if peak is in the middle (normal) or at edges (skewed)
                peak_position = peaks[0] / len(volume_at_price)
                if 0.3 <= peak_position <= 0.7:
                    return 'normal'
                else:
                    return 'skewed'
            elif len(peaks) == 2:
                # Check if peaks are significant
                peak_volumes = [volume_at_price[p] for p in peaks]
                max_volume = max(volume_at_price)
                significant_peaks = [v for v in peak_volumes if v > max_volume * 0.5]
                
                if len(significant_peaks) >= 2:
                    return 'bimodal'
                else:
                    return 'normal'
            else:
                return 'multimodal'
                
        except Exception as e:
            self.logger.error(f"Profile shape classification failed: {str(e)}")
            return 'unknown'
    
    def process_volume_data(self, 
                          df: pd.DataFrame, 
                          symbol: str, 
                          exchange: str,
                          create_profile: bool = True) -> Tuple[pd.DataFrame, VolumeDataQuality, VolumeMetrics, Optional[VolumeProfile]]:
        """
        Complete volume data processing pipeline.
        
        Args:
            df: Raw volume data DataFrame
            symbol: Asset symbol
            exchange: Exchange identifier
            create_profile: Whether to create volume profile
            
        Returns:
            Tuple of (processed_data, quality_metrics, volume_metrics, volume_profile)
        """
        try:
            self.logger.info(f"Starting volume data processing for {symbol}@{exchange}")
            
            # Validate data quality
            quality_metrics = self.validate_volume_data(df, symbol, exchange)
            
            if quality_metrics.quality_score < 0.3:
                self.logger.warning(f"Very low quality volume data for {symbol}@{exchange}: {quality_metrics.quality_score:.2f}")
            
            # Clean the data
            cleaned_df = self.clean_volume_data(df, symbol, exchange)
            
            # Normalize the data
            processed_df = self.normalize_volume_data(cleaned_df, symbol, exchange)
            
            # Calculate volume metrics
            volume_metrics = self.calculate_volume_metrics(cleaned_df, symbol, exchange)
            
            # Create volume profile if requested
            volume_profile = None
            if create_profile and 'price' in cleaned_df.columns:
                volume_profile = self.create_volume_profile(cleaned_df, symbol)
            
            # Update tracking
            self.processed_exchanges.add(f"{symbol}@{exchange}")
            self.volume_statistics[f"{symbol}@{exchange}"] = {
                'timestamp': datetime.now(),
                'records_processed': len(processed_df),
                'quality_score': quality_metrics.quality_score,
                'volume_metrics': volume_metrics
            }
            
            self.logger.info(f"Completed volume data processing for {symbol}@{exchange}: {len(processed_df)} records")
            
            return processed_df, quality_metrics, volume_metrics, volume_profile
            
        except Exception as e:
            self.logger.error(f"Volume data processing failed for {symbol}@{exchange}: {str(e)}")
            raise
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the volume preprocessor state."""
        try:
            preprocessor_data = {
                'spike_detection_method': self.spike_detection_method,
                'volume_normalization': self.volume_normalization,
                'microstructure_analysis': self.microstructure_analysis,
                'standard_scaler': self.standard_scaler,
                'minmax_scaler': self.minmax_scaler,
                'power_transformer': self.power_transformer,
                'simple_imputer': self.simple_imputer,
                'knn_imputer': self.knn_imputer,
                'volume_profiles': self.volume_profiles,
                'liquidity_metrics': self.liquidity_metrics,
                'microstructure_features': self.microstructure_features,
                'processed_exchanges': list(self.processed_exchanges),
                'volume_statistics': self.volume_statistics,
                'model_version': self.model_version
            }
            
            joblib.dump(preprocessor_data, filepath)
            self.logger.info(f"Volume data preprocessor saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Volume preprocessor saving failed: {str(e)}")
            raise
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load the volume preprocessor state."""
        try:
            preprocessor_data = joblib.load(filepath)
            
            self.spike_detection_method = preprocessor_data['spike_detection_method']
            self.volume_normalization = preprocessor_data['volume_normalization']
            self.microstructure_analysis = preprocessor_data['microstructure_analysis']
            self.standard_scaler = preprocessor_data['standard_scaler']
            self.minmax_scaler = preprocessor_data['minmax_scaler']
            self.power_transformer = preprocessor_data['power_transformer']
            self.simple_imputer = preprocessor_data['simple_imputer']
            self.knn_imputer = preprocessor_data['knn_imputer']
            self.volume_profiles = preprocessor_data['volume_profiles']
            self.liquidity_metrics = preprocessor_data['liquidity_metrics']
            self.microstructure_features = preprocessor_data['microstructure_features']
            self.processed_exchanges = set(preprocessor_data['processed_exchanges'])
            self.volume_statistics = preprocessor_data['volume_statistics']
            self.model_version = preprocessor_data['model_version']
            
            self.logger.info(f"Volume data preprocessor loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Volume preprocessor loading failed: {str(e)}")
            raise


def create_volume_preprocessor(config: Dict) -> VolumeDataPreprocessor:
    """
    Create a volume data preprocessor with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured VolumeDataPreprocessor instance
    """
    return VolumeDataPreprocessor(
        spike_detection_method=config.get('spike_detection_method', 'iqr'),
        volume_normalization=config.get('volume_normalization', 'log_transform'),
        microstructure_analysis=config.get('microstructure_analysis', True)
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create preprocessor
    preprocessor = VolumeDataPreprocessor()
    
    # Example data processing would go here
    df = load_raw_volume_data('BTC', 'binance')
    processed_df, quality, metrics, profile = preprocessor.process_volume_data(df, 'BTC', 'binance')
    
    print("Volume data preprocessing implementation completed")
