"""
Data Utilities

Production data processing utilities for ML pipelines with Chainlink
integration and comprehensive data validation.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import aiohttp
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    dataset_name: str
    total_records: int
    missing_values: Dict[str, int]
    duplicate_records: int
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    quality_score: float
    recommendations: List[str]
    timestamp: datetime

class DataValidator:
    """Comprehensive data validation utilities."""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> DataQualityReport:
        """Validate OHLCV market data."""
        issues = []
        
        # Required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # OHLC relationship validation
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
        
        # Volume validation
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume values: {negative_volume}")
        
        # Calculate quality score
        quality_score = max(0, 1 - len(issues) * 0.2)
        
        return DataQualityReport(
            dataset_name="OHLCV",
            total_records=len(df),
            missing_values=df.isnull().sum().to_dict(),
            duplicate_records=df.duplicated().sum(),
            outliers={},
            data_types=df.dtypes.astype(str).to_dict(),
            quality_score=quality_score,
            recommendations=issues,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def detect_outliers(series: pd.Series, method: str = 'iqr') -> pd.Series:
        """Detect outliers in data series."""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > 3
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            outlier_detector = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = outlier_detector.fit_predict(series.values.reshape(-1, 1))
            return pd.Series(outlier_labels == -1, index=series.index)
        
        return pd.Series(False, index=series.index)

class DataTransformer:
    """Data transformation utilities."""
    
    @staticmethod
    def normalize_price_data(df: pd.DataFrame, method: str = 'min_max') -> pd.DataFrame:
        """Normalize price data."""
        result_df = df.copy()
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col in df.columns:
                if method == 'min_max':
                    min_val, max_val = df[col].min(), df[col].max()
                    result_df[col] = (df[col] - min_val) / (max_val - min_val)
                elif method == 'z_score':
                    result_df[col] = (df[col] - df[col].mean()) / df[col].std()
                elif method == 'log':
                    result_df[col] = np.log1p(df[col])
        
        return result_df
    
    @staticmethod
    def create_features_from_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features from price data."""
        result_df = df.copy()
        
        if 'close' in df.columns:
            # Returns
            result_df['returns'] = df['close'].pct_change()
            result_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                result_df[f'sma_{window}'] = df['close'].rolling(window).mean()
                result_df[f'price_sma_{window}_ratio'] = df['close'] / result_df[f'sma_{window}']
            
            # Volatility
            result_df['volatility_20'] = result_df['returns'].rolling(20).std()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df['rsi'] = 100 - (100 / (1 + rs))
        
        return result_df.dropna()
    
    @staticmethod
    def align_multiple_series(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Align multiple time series with common index."""
        # Find common date range
        start_dates = [s.index.min() for s in series_dict.values()]
        end_dates = [s.index.max() for s in series_dict.values()]
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Align series
        aligned_series = {}
        for name, series in series_dict.items():
            aligned_series[name] = series.loc[common_start:common_end]
        
        return pd.DataFrame(aligned_series)

class DataCache:
    """Simple data caching utility."""
    
    def __init__(self, cache_dir: str = "./data_cache"):
        """Initialize data cache."""
        import os
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, identifier: str) -> str:
        """Generate cache key."""
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def cache_dataframe(self, df: pd.DataFrame, identifier: str, ttl_hours: int = 24):
        """Cache DataFrame."""
        cache_key = self._get_cache_key(identifier)
        cache_file = f"{self.cache_dir}/{cache_key}.parquet"
        
        # Add metadata
        metadata = {
            'cached_at': datetime.now().isoformat(),
            'ttl_hours': ttl_hours,
            'shape': df.shape,
            'columns': list(df.columns)
        }
        
        # Save DataFrame and metadata
        df.to_parquet(cache_file)
        with open(f"{cache_file}.meta", 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Cached data: {identifier}")
    
    def get_cached_dataframe(self, identifier: str) -> Optional[pd.DataFrame]:
        """Retrieve cached DataFrame."""
        cache_key = self._get_cache_key(identifier)
        cache_file = f"{self.cache_dir}/{cache_key}.parquet"
        meta_file = f"{cache_file}.meta"
        
        if not (os.path.exists(cache_file) and os.path.exists(meta_file)):
            return None
        
        # Check TTL
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        cached_at = datetime.fromisoformat(metadata['cached_at'])
        ttl = timedelta(hours=metadata['ttl_hours'])
        
        if datetime.now() - cached_at > ttl:
            return None
        
        return pd.read_parquet(cache_file)

class DataPipeline:
    """Production data pipeline utilities."""
    
    def __init__(self):
        """Initialize data pipeline."""
        self.cache = DataCache()
        self.validator = DataValidator()
        self.transformer = DataTransformer()
    
    async def process_price_data_pipeline(self, 
                                        raw_data: pd.DataFrame, 
                                        symbol: str) -> pd.DataFrame:
        """Complete price data processing pipeline."""
        logger.info(f"Processing price data pipeline for {symbol}")
        
        # Validate data
        quality_report = self.validator.validate_ohlcv_data(raw_data)
        if quality_report.quality_score < 0.5:
            logger.warning(f"Low quality data for {symbol}: {quality_report.quality_score}")
        
        # Clean data
        cleaned_data = self._clean_price_data(raw_data)
        
        # Transform data
        transformed_data = self.transformer.create_features_from_prices(cleaned_data)
        
        # Cache results
        self.cache.cache_dataframe(transformed_data, f"processed_{symbol}")
        
        return transformed_data
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data."""
        cleaned_df = df.copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].interpolate()
        
        # Remove outliers
        for col in ['open', 'high', 'low', 'close']:
            if col in cleaned_df.columns:
                outliers = self.validator.detect_outliers(cleaned_df[col])
                if outliers.sum() > 0:
                    cleaned_df.loc[outliers, col] = np.nan
                    cleaned_df[col] = cleaned_df[col].interpolate()
        
        return cleaned_df

def create_train_test_split_temporal(df: pd.DataFrame, 
                                   test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create temporal train-test split."""
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def resample_data(df: pd.DataFrame, 
                 freq: str, 
                 agg_method: Dict[str, str] = None) -> pd.DataFrame:
    """Resample time series data."""
    if agg_method is None:
        agg_method = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # Filter columns that exist
    existing_agg = {k: v for k, v in agg_method.items() if k in df.columns}
    
    if 'timestamp' in df.columns:
        df_indexed = df.set_index('timestamp')
    else:
        df_indexed = df
    
    return df_indexed.resample(freq).agg(existing_agg)

async def fetch_multiple_sources(urls: List[str], 
                               headers: Dict[str, str] = None) -> List[Optional[Dict]]:
    """Fetch data from multiple sources concurrently."""
    async def fetch_single(session: aiohttp.ClientSession, url: str) -> Optional[Dict]:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
        return None
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if not isinstance(r, Exception) else None for r in results]

if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.exponential(1000, 100)
    })
    
    # Process data
    processed = asyncio.run(pipeline.process_price_data_pipeline(sample_data, 'BTC'))
    print(f"Processed data shape: {processed.shape}")
    print(f"Processed columns: {list(processed.columns)}")
