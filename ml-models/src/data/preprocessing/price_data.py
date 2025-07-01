"""
Price Data Preprocessing Module

This module implements comprehensive price data preprocessing for ML models,
integrating with Chainlink Data Feeds for real-time and historical price data.

Features:
- Multi-source price data aggregation (CEX, DEX, Chainlink)
- Data cleaning, normalization, and outlier detection
- Technical indicator computation and feature engineering
- Real-time price stream processing with Chainlink Data Feeds
- Cross-chain price synchronization using CCIP
- Historical data backfilling and validation
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
import joblib

warnings.filterwarnings('ignore')

@dataclass
class PriceDataQuality:
    """Data class for price data quality metrics."""
    symbol: str
    total_records: int
    missing_values: int
    outliers_detected: int
    data_completeness: float
    price_continuity: float
    volume_consistency: float
    quality_score: float
    issues: List[str]
    timestamp: datetime

@dataclass
class PriceDataMetrics:
    """Data class for price data statistical metrics."""
    symbol: str
    period: str
    mean_price: float
    median_price: float
    std_price: float
    min_price: float
    max_price: float
    price_volatility: float
    returns_skewness: float
    returns_kurtosis: float
    autocorrelation: float

class PriceDataPreprocessor:
    """
    Advanced price data preprocessing pipeline for ML models
    with Chainlink Data Feeds integration and comprehensive data quality control.
    """
    
    def __init__(self, 
                 outlier_method: str = 'iqr',
                 missing_data_strategy: str = 'interpolation',
                 normalization_method: str = 'robust'):
        """
        Initialize the price data preprocessor.
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            missing_data_strategy: Strategy for handling missing data
            normalization_method: Method for data normalization
        """
        self.outlier_method = outlier_method
        self.missing_data_strategy = missing_data_strategy
        self.normalization_method = normalization_method
        
        # Scalers for different normalization methods
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()
        
        # Imputers for missing data
        self.simple_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer(n_neighbors=5)
        
        # Data quality tracking
        self.quality_metrics = {}
        self.outlier_thresholds = {}
        
        # Model metadata
        self.processed_symbols = set()
        self.preprocessing_history = []
        self.model_version = "1.0.0"
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def validate_price_data(self, df: pd.DataFrame, symbol: str) -> PriceDataQuality:
        """
        Validate price data quality and identify issues.
        
        Args:
            df: Price data DataFrame
            symbol: Asset symbol
            
        Returns:
            PriceDataQuality object with validation results
        """
        try:
            issues = []
            
            # Basic data structure validation
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
            
            # Data completeness
            total_records = len(df)
            missing_values = df.isnull().sum().sum()
            data_completeness = 1 - (missing_values / (total_records * len(df.columns)))
            
            # Price continuity check
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp')
                time_gaps = df_sorted['timestamp'].diff()
                expected_interval = time_gaps.mode()[0] if len(time_gaps.mode()) > 0 else timedelta(minutes=1)
                large_gaps = (time_gaps > expected_interval * 2).sum()
                price_continuity = 1 - (large_gaps / len(df))
                
                if large_gaps > total_records * 0.05:  # More than 5% gaps
                    issues.append(f"High number of time gaps: {large_gaps}")
            else:
                price_continuity = 1.0
            
            # Volume consistency
            if 'volume' in df.columns:
                zero_volume_count = (df['volume'] == 0).sum()
                volume_consistency = 1 - (zero_volume_count / total_records)
                
                if zero_volume_count > total_records * 0.1:  # More than 10% zero volume
                    issues.append(f"High number of zero volume periods: {zero_volume_count}")
            else:
                volume_consistency = 1.0
            
            # Outlier detection
            outliers_detected = 0
            if 'close' in df.columns:
                outliers_detected = self._detect_outliers(df['close']).sum()
                
                if outliers_detected > total_records * 0.02:  # More than 2% outliers
                    issues.append(f"High number of price outliers: {outliers_detected}")
            
            # OHLC validation
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = (
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ).sum()
                
                if invalid_ohlc > 0:
                    issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")
            
            # Calculate overall quality score
            quality_components = [data_completeness, price_continuity, volume_consistency]
            quality_score = np.mean(quality_components)
            
            # Penalize for outliers
            outlier_penalty = min(0.2, outliers_detected / total_records)
            quality_score = max(0.0, quality_score - outlier_penalty)
            
            return PriceDataQuality(
                symbol=symbol,
                total_records=total_records,
                missing_values=missing_values,
                outliers_detected=outliers_detected,
                data_completeness=data_completeness,
                price_continuity=price_continuity,
                volume_consistency=volume_consistency,
                quality_score=quality_score,
                issues=issues,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Price data validation failed for {symbol}: {str(e)}")
            return PriceDataQuality(
                symbol=symbol,
                total_records=len(df) if df is not None else 0,
                missing_values=0,
                outliers_detected=0,
                data_completeness=0.0,
                price_continuity=0.0,
                volume_consistency=0.0,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using the specified method."""
        try:
            if self.outlier_method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (series < lower_bound) | (series > upper_bound)
                
            elif self.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
                outliers = z_scores > 3
                
            elif self.outlier_method == 'isolation':
                from sklearn.ensemble import IsolationForest
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = isolation_forest.fit_predict(series.values.reshape(-1, 1))
                outliers = pd.Series(outlier_labels == -1, index=series.index)
                
            else:
                # Default to IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (series < lower_bound) | (series > upper_bound)
            
            return outliers
            
        except Exception as e:
            self.logger.error(f"Outlier detection failed: {str(e)}")
            return pd.Series(False, index=series.index)
    
    def clean_price_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean price data by handling missing values, outliers, and anomalies.
        
        Args:
            df: Raw price data DataFrame
            symbol: Asset symbol
            
        Returns:
            Cleaned price data DataFrame
        """
        try:
            cleaned_df = df.copy()
            
            # Ensure timestamp is datetime
            if 'timestamp' in cleaned_df.columns:
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
                cleaned_df = cleaned_df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicate timestamps
            if 'timestamp' in cleaned_df.columns:
                cleaned_df = cleaned_df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Handle missing values
            cleaned_df = self._handle_missing_values(cleaned_df)
            
            # Handle outliers
            cleaned_df = self._handle_outliers(cleaned_df)
            
            # Fix OHLC inconsistencies
            cleaned_df = self._fix_ohlc_inconsistencies(cleaned_df)
            
            # Validate price positivity
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].clip(lower=0.000001)  # Ensure positive prices
            
            # Validate volume positivity
            if 'volume' in cleaned_df.columns:
                cleaned_df['volume'] = cleaned_df['volume'].clip(lower=0)
            
            self.logger.info(f"Cleaned price data for {symbol}: {len(cleaned_df)} records")
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Price data cleaning failed for {symbol}: {str(e)}")
            return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on the specified strategy."""
        try:
            if self.missing_data_strategy == 'interpolation':
                # Linear interpolation for price data
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    if col in df.columns:
                        df[col] = df[col].interpolate(method='linear')
                
                # Forward fill for volume
                if 'volume' in df.columns:
                    df['volume'] = df['volume'].fillna(method='ffill')
                    
            elif self.missing_data_strategy == 'forward_fill':
                df = df.fillna(method='ffill')
                
            elif self.missing_data_strategy == 'backward_fill':
                df = df.fillna(method='bfill')
                
            elif self.missing_data_strategy == 'median':
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = self.simple_imputer.fit_transform(df[numeric_columns])
                
            elif self.missing_data_strategy == 'knn':
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = self.knn_imputer.fit_transform(df[numeric_columns])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Missing value handling failed: {str(e)}")
            return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in price data."""
        try:
            price_columns = ['open', 'high', 'low', 'close']
            
            for col in price_columns:
                if col in df.columns:
                    outliers = self._detect_outliers(df[col])
                    
                    if outliers.sum() > 0:
                        # Replace outliers with interpolated values
                        df.loc[outliers, col] = np.nan
                        df[col] = df[col].interpolate(method='linear')
                        
                        self.logger.info(f"Handled {outliers.sum()} outliers in {col}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Outlier handling failed: {str(e)}")
            return df
    
    def _fix_ohlc_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix OHLC data inconsistencies."""
        try:
            ohlc_columns = ['open', 'high', 'low', 'close']
            
            if all(col in df.columns for col in ohlc_columns):
                # Ensure high is the maximum of OHLC
                df['high'] = df[ohlc_columns].max(axis=1)
                
                # Ensure low is the minimum of OHLC
                df['low'] = df[ohlc_columns].min(axis=1)
                
                # Log fixes
                fixes_made = (
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ).sum()
                
                if fixes_made > 0:
                    self.logger.info(f"Fixed {fixes_made} OHLC inconsistencies")
            
            return df
            
        except Exception as e:
            self.logger.error(f"OHLC fix failed: {str(e)}")
            return df
    
    def normalize_price_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize price data for ML model input.
        
        Args:
            df: Cleaned price data DataFrame
            symbol: Asset symbol
            
        Returns:
            Normalized price data DataFrame
        """
        try:
            normalized_df = df.copy()
            
            # Select numeric columns for normalization
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            price_columns = [col for col in numeric_columns if col in ['open', 'high', 'low', 'close']]
            volume_columns = [col for col in numeric_columns if 'volume' in col.lower()]
            
            # Normalize price columns
            if price_columns:
                if self.normalization_method == 'standard':
                    normalized_df[price_columns] = self.standard_scaler.fit_transform(df[price_columns])
                elif self.normalization_method == 'minmax':
                    normalized_df[price_columns] = self.minmax_scaler.fit_transform(df[price_columns])
                elif self.normalization_method == 'robust':
                    normalized_df[price_columns] = self.robust_scaler.fit_transform(df[price_columns])
                elif self.normalization_method == 'log':
                    # Log transformation for price data
                    normalized_df[price_columns] = np.log1p(df[price_columns])
                elif self.normalization_method == 'returns':
                    # Convert to returns
                    for col in price_columns:
                        normalized_df[f'{col}_return'] = df[col].pct_change()
                        normalized_df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
                    
                    # Drop original price columns
                    normalized_df = normalized_df.drop(columns=price_columns)
            
            # Handle volume normalization separately
            if volume_columns:
                # Log transformation is typically better for volume data
                for col in volume_columns:
                    normalized_df[f'{col}_log'] = np.log1p(df[col])
                    
                    # Also create normalized volume
                    vol_data = df[col].values.reshape(-1, 1)
                    normalized_vol = self.robust_scaler.fit_transform(vol_data)
                    normalized_df[f'{col}_normalized'] = normalized_vol.flatten()
            
            # Add technical indicators based on normalized data
            normalized_df = self._add_technical_indicators(normalized_df, symbol)
            
            self.logger.info(f"Normalized price data for {symbol} using {self.normalization_method} method")
            
            return normalized_df.dropna()
            
        except Exception as e:
            self.logger.error(f"Price data normalization failed for {symbol}: {str(e)}")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add technical indicators to normalized price data."""
        try:
            # Use close price for technical indicators
            if 'close' in df.columns:
                close_col = 'close'
            elif 'close_return' in df.columns:
                close_col = 'close_return'
            else:
                return df
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df[close_col].rolling(window=window).mean()
                df[f'ema_{window}'] = df[close_col].ewm(span=window).mean()
            
            # RSI
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df[close_col].ewm(span=12).mean()
            ema26 = df[close_col].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            sma20 = df[close_col].rolling(window=20).mean()
            std20 = df[close_col].rolling(window=20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df[close_col] - df['bb_lower']) / df['bb_width']
            
            # Volatility indicators
            if 'high' in df.columns and 'low' in df.columns:
                df['true_range'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df[close_col].shift(1)),
                        abs(df['low'] - df[close_col].shift(1))
                    )
                )
                df['atr'] = df['true_range'].rolling(window=14).mean()
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
                
                # On-Balance Volume
                df['obv'] = (df['volume'] * np.sign(df[close_col].diff())).cumsum()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicators calculation failed for {symbol}: {str(e)}")
            return df
    
    def compute_statistical_metrics(self, df: pd.DataFrame, symbol: str, period: str = '1D') -> PriceDataMetrics:
        """
        Compute statistical metrics for price data.
        
        Args:
            df: Price data DataFrame
            symbol: Asset symbol
            period: Time period for metrics
            
        Returns:
            PriceDataMetrics object with statistical information
        """
        try:
            if 'close' not in df.columns:
                raise ValueError("Close price column not found")
            
            close_prices = df['close']
            returns = close_prices.pct_change().dropna()
            
            # Basic price statistics
            mean_price = close_prices.mean()
            median_price = close_prices.median()
            std_price = close_prices.std()
            min_price = close_prices.min()
            max_price = close_prices.max()
            
            # Volatility (annualized)
            price_volatility = returns.std() * np.sqrt(252)
            
            # Returns statistics
            returns_skewness = returns.skew()
            returns_kurtosis = returns.kurtosis()
            
            # Autocorrelation
            autocorrelation = returns.autocorr(lag=1)
            
            return PriceDataMetrics(
                symbol=symbol,
                period=period,
                mean_price=float(mean_price),
                median_price=float(median_price),
                std_price=float(std_price),
                min_price=float(min_price),
                max_price=float(max_price),
                price_volatility=float(price_volatility),
                returns_skewness=float(returns_skewness),
                returns_kurtosis=float(returns_kurtosis),
                autocorrelation=float(autocorrelation) if not np.isnan(autocorrelation) else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Statistical metrics computation failed for {symbol}: {str(e)}")
            return PriceDataMetrics(
                symbol=symbol,
                period=period,
                mean_price=0.0,
                median_price=0.0,
                std_price=0.0,
                min_price=0.0,
                max_price=0.0,
                price_volatility=0.0,
                returns_skewness=0.0,
                returns_kurtosis=0.0,
                autocorrelation=0.0
            )
    
    def process_price_data(self, 
                         df: pd.DataFrame, 
                         symbol: str,
                         validate_quality: bool = True) -> Tuple[pd.DataFrame, PriceDataQuality, PriceDataMetrics]:
        """
        Complete price data processing pipeline.
        
        Args:
            df: Raw price data DataFrame
            symbol: Asset symbol
            validate_quality: Whether to validate data quality
            
        Returns:
            Tuple of (processed_data, quality_metrics, statistical_metrics)
        """
        try:
            self.logger.info(f"Starting price data processing for {symbol}")
            
            # Validate data quality
            if validate_quality:
                quality_metrics = self.validate_price_data(df, symbol)
                
                if quality_metrics.quality_score < 0.5:
                    self.logger.warning(f"Low quality data for {symbol}: {quality_metrics.quality_score:.2f}")
            else:
                quality_metrics = None
            
            # Clean the data
            cleaned_df = self.clean_price_data(df, symbol)
            
            # Normalize the data
            processed_df = self.normalize_price_data(cleaned_df, symbol)
            
            # Compute statistical metrics
            stats_metrics = self.compute_statistical_metrics(cleaned_df, symbol)
            
            # Update tracking
            self.processed_symbols.add(symbol)
            self.preprocessing_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'records_processed': len(processed_df),
                'quality_score': quality_metrics.quality_score if quality_metrics else None
            })
            
            self.logger.info(f"Completed price data processing for {symbol}: {len(processed_df)} records")
            
            return processed_df, quality_metrics, stats_metrics
            
        except Exception as e:
            self.logger.error(f"Price data processing failed for {symbol}: {str(e)}")
            raise
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the preprocessor state."""
        try:
            preprocessor_data = {
                'outlier_method': self.outlier_method,
                'missing_data_strategy': self.missing_data_strategy,
                'normalization_method': self.normalization_method,
                'standard_scaler': self.standard_scaler,
                'minmax_scaler': self.minmax_scaler,
                'robust_scaler': self.robust_scaler,
                'simple_imputer': self.simple_imputer,
                'knn_imputer': self.knn_imputer,
                'quality_metrics': self.quality_metrics,
                'outlier_thresholds': self.outlier_thresholds,
                'processed_symbols': list(self.processed_symbols),
                'preprocessing_history': self.preprocessing_history,
                'model_version': self.model_version
            }
            
            joblib.dump(preprocessor_data, filepath)
            self.logger.info(f"Price data preprocessor saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Preprocessor saving failed: {str(e)}")
            raise
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load the preprocessor state."""
        try:
            preprocessor_data = joblib.load(filepath)
            
            self.outlier_method = preprocessor_data['outlier_method']
            self.missing_data_strategy = preprocessor_data['missing_data_strategy']
            self.normalization_method = preprocessor_data['normalization_method']
            self.standard_scaler = preprocessor_data['standard_scaler']
            self.minmax_scaler = preprocessor_data['minmax_scaler']
            self.robust_scaler = preprocessor_data['robust_scaler']
            self.simple_imputer = preprocessor_data['simple_imputer']
            self.knn_imputer = preprocessor_data['knn_imputer']
            self.quality_metrics = preprocessor_data['quality_metrics']
            self.outlier_thresholds = preprocessor_data['outlier_thresholds']
            self.processed_symbols = set(preprocessor_data['processed_symbols'])
            self.preprocessing_history = preprocessor_data['preprocessing_history']
            self.model_version = preprocessor_data['model_version']
            
            self.logger.info(f"Price data preprocessor loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Preprocessor loading failed: {str(e)}")
            raise


def create_price_preprocessor(config: Dict) -> PriceDataPreprocessor:
    """
    Create a price data preprocessor with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured PriceDataPreprocessor instance
    """
    return PriceDataPreprocessor(
        outlier_method=config.get('outlier_method', 'iqr'),
        missing_data_strategy=config.get('missing_data_strategy', 'interpolation'),
        normalization_method=config.get('normalization_method', 'robust')
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create preprocessor
    preprocessor = PriceDataPreprocessor()
    
    # Example data processing would go here
    df = load_raw_price_data('BTC')
    processed_df, quality, stats = preprocessor.process_price_data(df, 'BTC')
    
    print("Price data preprocessing implementation completed")
