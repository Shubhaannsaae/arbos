"""
Market Data Loader

Production-ready loader for traditional and crypto market data
from multiple sources with caching and rate limiting.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import aiohttp
import time
from urllib.parse import urlencode

@dataclass
class MarketData:
    """Market data point."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str

class MarketLoader:
    """
    Production market data loader with multiple data sources,
    rate limiting, and intelligent caching.
    """
    
    def __init__(self, api_keys: Dict[str, str], cache_ttl: int = 300):
        """Initialize market loader."""
        self.api_keys = api_keys
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.rate_limits = {}
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.endpoints = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'polygon': 'https://api.polygon.io/v2/aggs/ticker',
            'coinapi': 'https://rest.coinapi.io/v1/ohlcv',
            'coingecko': 'https://api.coingecko.com/api/v3'
        }
        
        # Rate limits (requests per minute)
        self.rate_limit_config = {
            'alpha_vantage': 5,
            'polygon': 5,
            'coinapi': 100,
            'coingecko': 50
        }
    
    async def _check_rate_limit(self, source: str) -> bool:
        """Check if rate limit allows request."""
        now = time.time()
        minute = int(now / 60)
        
        if source not in self.rate_limits:
            self.rate_limits[source] = {}
        
        if minute not in self.rate_limits[source]:
            self.rate_limits[source] = {minute: 0}
        
        # Clean old entries
        self.rate_limits[source] = {
            k: v for k, v in self.rate_limits[source].items() 
            if k >= minute - 1
        }
        
        current_count = self.rate_limits[source].get(minute, 0)
        limit = self.rate_limit_config.get(source, 60)
        
        if current_count >= limit:
            return False
        
        self.rate_limits[source][minute] = current_count + 1
        return True
    
    def _get_cache_key(self, symbol: str, source: str, timeframe: str) -> str:
        """Generate cache key."""
        return f"{source}:{symbol}:{timeframe}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        return (time.time() - cache_entry['timestamp']) < self.cache_ttl
    
    async def get_stock_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Get stock data from Alpha Vantage."""
        try:
            if 'alpha_vantage' not in self.api_keys:
                raise ValueError("Alpha Vantage API key required")
            
            cache_key = self._get_cache_key(symbol, 'alpha_vantage', f'{days}d')
            if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                return self.cache[cache_key]['data']
            
            if not await self._check_rate_limit('alpha_vantage'):
                self.logger.warning("Alpha Vantage rate limit exceeded")
                return []
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_keys['alpha_vantage'],
                'outputsize': 'compact'
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.endpoints['alpha_vantage']}?{urlencode(params)}"
                async with session.get(url) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    time_series = data.get('Time Series (Daily)', {})
                    
                    market_data = []
                    for date_str, values in list(time_series.items())[:days]:
                        market_data.append(MarketData(
                            symbol=symbol,
                            timestamp=datetime.strptime(date_str, '%Y-%m-%d'),
                            open=float(values['1. open']),
                            high=float(values['2. high']),
                            low=float(values['3. low']),
                            close=float(values['4. close']),
                            volume=float(values['5. volume']),
                            source='alpha_vantage'
                        ))
                    
                    # Cache result
                    self.cache[cache_key] = {
                        'data': market_data,
                        'timestamp': time.time()
                    }
                    
                    return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {e}")
            return []
    
    async def get_crypto_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Get crypto data from CoinGecko."""
        try:
            cache_key = self._get_cache_key(symbol, 'coingecko', f'{days}d')
            if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                return self.cache[cache_key]['data']
            
            if not await self._check_rate_limit('coingecko'):
                self.logger.warning("CoinGecko rate limit exceeded")
                return []
            
            # Convert symbol to CoinGecko ID
            symbol_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'ADA': 'cardano',
                'SOL': 'solana',
                'AVAX': 'avalanche-2',
                'LINK': 'chainlink'
            }
            
            coin_id = symbol_map.get(symbol.upper(), symbol.lower())
            url = f"{self.endpoints['coingecko']}/coins/{coin_id}/ohlc"
            params = {'vs_currency': 'usd', 'days': str(days)}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    market_data = []
                    
                    for entry in data:
                        timestamp = datetime.fromtimestamp(entry[0] / 1000)
                        market_data.append(MarketData(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(entry[1]),
                            high=float(entry[2]),
                            low=float(entry[3]),
                            close=float(entry[4]),
                            volume=0.0,  # CoinGecko OHLC doesn't include volume
                            source='coingecko'
                        ))
                    
                    # Cache result
                    self.cache[cache_key] = {
                        'data': market_data,
                        'timestamp': time.time()
                    }
                    
                    return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return []
    
    async def get_forex_data(self, from_currency: str, to_currency: str, days: int = 30) -> List[MarketData]:
        """Get forex data from Alpha Vantage."""
        try:
            if 'alpha_vantage' not in self.api_keys:
                raise ValueError("Alpha Vantage API key required")
            
            symbol = f"{from_currency}{to_currency}"
            cache_key = self._get_cache_key(symbol, 'alpha_vantage_fx', f'{days}d')
            
            if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                return self.cache[cache_key]['data']
            
            if not await self._check_rate_limit('alpha_vantage'):
                return []
            
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'apikey': self.api_keys['alpha_vantage']
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.endpoints['alpha_vantage']}?{urlencode(params)}"
                async with session.get(url) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    time_series = data.get('Time Series FX (Daily)', {})
                    
                    market_data = []
                    for date_str, values in list(time_series.items())[:days]:
                        market_data.append(MarketData(
                            symbol=symbol,
                            timestamp=datetime.strptime(date_str, '%Y-%m-%d'),
                            open=float(values['1. open']),
                            high=float(values['2. high']),
                            low=float(values['3. low']),
                            close=float(values['4. close']),
                            volume=0.0,  # Forex doesn't have volume
                            source='alpha_vantage_fx'
                        ))
                    
                    # Cache result
                    self.cache[cache_key] = {
                        'data': market_data,
                        'timestamp': time.time()
                    }
                    
                    return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching forex data for {symbol}: {e}")
            return []
    
    async def get_multiple_assets(self, symbols: List[str], asset_type: str = 'crypto') -> Dict[str, List[MarketData]]:
        """Get data for multiple assets concurrently."""
        try:
            if asset_type == 'crypto':
                tasks = [self.get_crypto_data(symbol) for symbol in symbols]
            elif asset_type == 'stock':
                tasks = [self.get_stock_data(symbol) for symbol in symbols]
            else:
                raise ValueError(f"Unsupported asset type: {asset_type}")
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, list):
                    data[symbol] = result
                else:
                    self.logger.error(f"Failed to get data for {symbol}: {result}")
                    data[symbol] = []
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching multiple assets: {e}")
            return {}
    
    def to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to pandas DataFrame."""
        if not market_data:
            return pd.DataFrame()
        
        data = []
        for item in market_data:
            data.append({
                'timestamp': item.timestamp,
                'symbol': item.symbol,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume,
                'source': item.source
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


async def create_market_loader(config: Dict) -> MarketLoader:
    """Create and initialize market loader."""
    return MarketLoader(
        api_keys=config.get('api_keys', {}),
        cache_ttl=config.get('cache_ttl', 300)
    )


if __name__ == "__main__":
    async def main():
        loader = MarketLoader({'alpha_vantage': 'demo'})
        
        # Test crypto data
        btc_data = await loader.get_crypto_data('BTC', days=7)
        if btc_data:
            print(f"Got {len(btc_data)} BTC data points")
            print(f"Latest: ${btc_data[0].close:,.2f}")
    
    asyncio.run(main())
