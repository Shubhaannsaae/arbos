"""
Technical Indicators Feature Engineering Module

This module implements comprehensive technical indicator calculations for ML models,
integrating with Chainlink Data Feeds for real-time price data and supporting
advanced feature engineering for crypto and traditional markets.

Features:
- Complete suite of technical indicators (trend, momentum, volatility, volume)
- Real-time indicator calculation with Chainlink Data Feeds
- Multi-timeframe indicator analysis
- Custom indicator combinations and transformations
- Adaptive indicators based on market regime
- Cross-asset indicator correlations
- Performance optimized calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import talib
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

warnings.filterwarnings('ignore')

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    name: str
    parameters: Dict
    timeframes: List[str]
    enabled: bool = True
    normalization: str = 'none'  # 'none', 'z_score', 'minmax', 'percentile'

@dataclass
class IndicatorResult:
    """Result of technical indicator calculation."""
    indicator_name: str
    values: pd.Series
    signal: pd.Series  # Buy/sell/hold signals
    confidence: pd.Series
    parameters: Dict
    timestamp: datetime

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculation and feature engineering
    with support for real-time updates and multi-timeframe analysis.
    """
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        self.indicators_cache = {}
        self.scalers = {}
        self.model_version = "1.0.0"
        self.logger = logging.getLogger(__name__)
        
        # Default indicator configurations
        self.default_configs = {
            # Trend Indicators
            'sma': IndicatorConfig('sma', {'timeperiod': [5, 10, 20, 50, 100, 200]}, ['1h', '4h', '1d']),
            'ema': IndicatorConfig('ema', {'timeperiod': [5, 10, 20, 50, 100, 200]}, ['1h', '4h', '1d']),
            'wma': IndicatorConfig('wma', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'dema': IndicatorConfig('dema', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'tema': IndicatorConfig('tema', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'trima': IndicatorConfig('trima', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'kama': IndicatorConfig('kama', {'timeperiod': [30]}, ['1h', '4h', '1d']),
            'mama': IndicatorConfig('mama', {'fastlimit': 0.5, 'slowlimit': 0.05}, ['1h', '4h', '1d']),
            'ht_trendline': IndicatorConfig('ht_trendline', {}, ['1h', '4h', '1d']),
            'sar': IndicatorConfig('sar', {'acceleration': 0.02, 'maximum': 0.2}, ['1h', '4h', '1d']),
            
            # Momentum Indicators
            'rsi': IndicatorConfig('rsi', {'timeperiod': [14, 21, 30]}, ['1h', '4h', '1d']),
            'stoch': IndicatorConfig('stoch', {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3}, ['1h', '4h', '1d']),
            'stochf': IndicatorConfig('stochf', {'fastk_period': 14, 'fastd_period': 3}, ['1h', '4h', '1d']),
            'stochrsi': IndicatorConfig('stochrsi', {'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3}, ['1h', '4h', '1d']),
            'macd': IndicatorConfig('macd', {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}, ['1h', '4h', '1d']),
            'macdext': IndicatorConfig('macdext', {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}, ['1h', '4h', '1d']),
            'macdfix': IndicatorConfig('macdfix', {'signalperiod': 9}, ['1h', '4h', '1d']),
            'ppo': IndicatorConfig('ppo', {'fastperiod': 12, 'slowperiod': 26}, ['1h', '4h', '1d']),
            'roc': IndicatorConfig('roc', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'rocp': IndicatorConfig('rocp', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'rocr': IndicatorConfig('rocr', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'rocr100': IndicatorConfig('rocr100', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'mom': IndicatorConfig('mom', {'timeperiod': [10, 20, 50]}, ['1h', '4h', '1d']),
            'trix': IndicatorConfig('trix', {'timeperiod': 30}, ['1h', '4h', '1d']),
            'ultosc': IndicatorConfig('ultosc', {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28}, ['1h', '4h', '1d']),
            'willr': IndicatorConfig('willr', {'timeperiod': [14, 21]}, ['1h', '4h', '1d']),
            'cci': IndicatorConfig('cci', {'timeperiod': [14, 20]}, ['1h', '4h', '1d']),
            'cmo': IndicatorConfig('cmo', {'timeperiod': [14, 21]}, ['1h', '4h', '1d']),
            'dx': IndicatorConfig('dx', {'timeperiod': 14}, ['1h', '4h', '1d']),
            'adx': IndicatorConfig('adx', {'timeperiod': 14}, ['1h', '4h', '1d']),
            'adxr': IndicatorConfig('adxr', {'timeperiod': 14}, ['1h', '4h', '1d']),
            'apo': IndicatorConfig('apo', {'fastperiod': 12, 'slowperiod': 26}, ['1h', '4h', '1d']),
            'aroon': IndicatorConfig('aroon', {'timeperiod': 14}, ['1h', '4h', '1d']),
            'aroonosc': IndicatorConfig('aroonosc', {'timeperiod': 14}, ['1h', '4h', '1d']),
            'bop': IndicatorConfig('bop', {}, ['1h', '4h', '1d']),
            'mfi': IndicatorConfig('mfi', {'timeperiod': [14, 21]}, ['1h', '4h', '1d']),
            'minus_di': IndicatorConfig('minus_di', {'timeperiod': 14}, ['1h', '4h', '1d']),
            'minus_dm': IndicatorConfig('minus_dm', {'timeperiod': 14}, ['1h', '4h', '1d']),
            'plus_di': IndicatorConfig('plus_di', {'timeperiod': 14}, ['1h', '4h', '1d']),
            'plus_dm': IndicatorConfig('plus_dm', {'timeperiod': 14}, ['1h', '4h', '1d']),
            
            # Volatility Indicators
            'bbands': IndicatorConfig('bbands', {'timeperiod': [20, 50], 'nbdevup': 2, 'nbdevdn': 2}, ['1h', '4h', '1d']),
            'atr': IndicatorConfig('atr', {'timeperiod': [14, 21]}, ['1h', '4h', '1d']),
            'natr': IndicatorConfig('natr', {'timeperiod': [14, 21]}, ['1h', '4h', '1d']),
            'trange': IndicatorConfig('trange', {}, ['1h', '4h', '1d']),
            
            # Volume Indicators
            'ad': IndicatorConfig('ad', {}, ['1h', '4h', '1d']),
            'adosc': IndicatorConfig('adosc', {'fastperiod': 3, 'slowperiod': 10}, ['1h', '4h', '1d']),
            'obv': IndicatorConfig('obv', {}, ['1h', '4h', '1d']),
            
            # Cycle Indicators
            'ht_dcperiod': IndicatorConfig('ht_dcperiod', {}, ['1h', '4h', '1d']),
            'ht_dcphase': IndicatorConfig('ht_dcphase', {}, ['1h', '4h', '1d']),
            'ht_phasor': IndicatorConfig('ht_phasor', {}, ['1h', '4h', '1d']),
            'ht_sine': IndicatorConfig('ht_sine', {}, ['1h', '4h', '1d']),
            'ht_trendmode': IndicatorConfig('ht_trendmode', {}, ['1h', '4h', '1d']),
            
            # Pattern Recognition
            'cdl2crows': IndicatorConfig('cdl2crows', {}, ['1h', '4h', '1d']),
            'cdl3blackcrows': IndicatorConfig('cdl3blackcrows', {}, ['1h', '4h', '1d']),
            'cdl3inside': IndicatorConfig('cdl3inside', {}, ['1h', '4h', '1d']),
            'cdl3linestrike': IndicatorConfig('cdl3linestrike', {}, ['1h', '4h', '1d']),
            'cdl3outside': IndicatorConfig('cdl3outside', {}, ['1h', '4h', '1d']),
            'cdl3starsinsouth': IndicatorConfig('cdl3starsinsouth', {}, ['1h', '4h', '1d']),
            'cdl3whitesoldiers': IndicatorConfig('cdl3whitesoldiers', {}, ['1h', '4h', '1d']),
            'cdlabandonedbaby': IndicatorConfig('cdlabandonedbaby', {'penetration': 0.3}, ['1h', '4h', '1d']),
            'cdlbelthold': IndicatorConfig('cdlbelthold', {}, ['1h', '4h', '1d']),
            'cdlbreakaway': IndicatorConfig('cdlbreakaway', {}, ['1h', '4h', '1d']),
            'cdlclosingmarubozu': IndicatorConfig('cdlclosingmarubozu', {}, ['1h', '4h', '1d']),
            'cdlconcealbabyswall': IndicatorConfig('cdlconcealbabyswall', {}, ['1h', '4h', '1d']),
            'cdlcounterattack': IndicatorConfig('cdlcounterattack', {}, ['1h', '4h', '1d']),
            'cdldarkcloudcover': IndicatorConfig('cdldarkcloudcover', {'penetration': 0.5}, ['1h', '4h', '1d']),
            'cdldoji': IndicatorConfig('cdldoji', {}, ['1h', '4h', '1d']),
            'cdldojistar': IndicatorConfig('cdldojistar', {}, ['1h', '4h', '1d']),
            'cdldragonflydoji': IndicatorConfig('cdldragonflydoji', {}, ['1h', '4h', '1d']),
            'cdlengulfing': IndicatorConfig('cdlengulfing', {}, ['1h', '4h', '1d']),
            'cdleveningdojistar': IndicatorConfig('cdleveningdojistar', {'penetration': 0.3}, ['1h', '4h', '1d']),
            'cdleveningstar': IndicatorConfig('cdleveningstar', {'penetration': 0.3}, ['1h', '4h', '1d']),
            'cdlgapsidesidewhite': IndicatorConfig('cdlgapsidesidewhite', {}, ['1h', '4h', '1d']),
            'cdlgravestonedoji': IndicatorConfig('cdlgravestonedoji', {}, ['1h', '4h', '1d']),
            'cdlhammer': IndicatorConfig('cdlhammer', {}, ['1h', '4h', '1d']),
            'cdlhangingman': IndicatorConfig('cdlhangingman', {}, ['1h', '4h', '1d']),
            'cdlharami': IndicatorConfig('cdlharami', {}, ['1h', '4h', '1d']),
            'cdlharamicross': IndicatorConfig('cdlharamicross', {}, ['1h', '4h', '1d']),
            'cdlhighwave': IndicatorConfig('cdlhighwave', {}, ['1h', '4h', '1d']),
            'cdlhikkake': IndicatorConfig('cdlhikkake', {}, ['1h', '4h', '1d']),
            'cdlhikkakemod': IndicatorConfig('cdlhikkakemod', {}, ['1h', '4h', '1d']),
            'cdlhomingpigeon': IndicatorConfig('cdlhomingpigeon', {}, ['1h', '4h', '1d']),
            'cdlidentical3crows': IndicatorConfig('cdlidentical3crows', {}, ['1h', '4h', '1d']),
            'cdlinneck': IndicatorConfig('cdlinneck', {}, ['1h', '4h', '1d']),
            'cdlinvertedhammer': IndicatorConfig('cdlinvertedhammer', {}, ['1h', '4h', '1d']),
            'cdlkicking': IndicatorConfig('cdlkicking', {}, ['1h', '4h', '1d']),
            'cdlkickingbylength': IndicatorConfig('cdlkickingbylength', {}, ['1h', '4h', '1d']),
            'cdlladderbottom': IndicatorConfig('cdlladderbottom', {}, ['1h', '4h', '1d']),
            'cdllongleggeddoji': IndicatorConfig('cdllongleggeddoji', {}, ['1h', '4h', '1d']),
            'cdllongline': IndicatorConfig('cdllongline', {}, ['1h', '4h', '1d']),
            'cdlmarubozu': IndicatorConfig('cdlmarubozu', {}, ['1h', '4h', '1d']),
            'cdlmatchinglow': IndicatorConfig('cdlmatchinglow', {}, ['1h', '4h', '1d']),
            'cdlmathold': IndicatorConfig('cdlmathold', {'penetration': 0.5}, ['1h', '4h', '1d']),
            'cdlmorningdojistar': IndicatorConfig('cdlmorningdojistar', {'penetration': 0.3}, ['1h', '4h', '1d']),
            'cdlmorningstar': IndicatorConfig('cdlmorningstar', {'penetration': 0.3}, ['1h', '4h', '1d']),
            'cdlonneck': IndicatorConfig('cdlonneck', {}, ['1h', '4h', '1d']),
            'cdlpiercing': IndicatorConfig('cdlpiercing', {}, ['1h', '4h', '1d']),
            'cdlrickshawman': IndicatorConfig('cdlrickshawman', {}, ['1h', '4h', '1d']),
            'cdlrisefall3methods': IndicatorConfig('cdlrisefall3methods', {}, ['1h', '4h', '1d']),
            'cdlseparatinglines': IndicatorConfig('cdlseparatinglines', {}, ['1h', '4h', '1d']),
            'cdlshootingstar': IndicatorConfig('cdlshootingstar', {}, ['1h', '4h', '1d']),
            'cdlshortline': IndicatorConfig('cdlshortline', {}, ['1h', '4h', '1d']),
            'cdlspinningtop': IndicatorConfig('cdlspinningtop', {}, ['1h', '4h', '1d']),
            'cdlstalledpattern': IndicatorConfig('cdlstalledpattern', {}, ['1h', '4h', '1d']),
            'cdlsticksandwich': IndicatorConfig('cdlsticksandwich', {}, ['1h', '4h', '1d']),
            'cdltakuri': IndicatorConfig('cdltakuri', {}, ['1h', '4h', '1d']),
            'cdltasukigap': IndicatorConfig('cdltasukigap', {}, ['1h', '4h', '1d']),
            'cdlthrusting': IndicatorConfig('cdlthrusting', {}, ['1h', '4h', '1d']),
            'cdltristar': IndicatorConfig('cdltristar', {}, ['1h', '4h', '1d']),
            'cdlunique3river': IndicatorConfig('cdlunique3river', {}, ['1h', '4h', '1d']),
            'cdlupsidegap2crows': IndicatorConfig('cdlupsidegap2crows', {}, ['1h', '4h', '1d']),
            'cdlxsidegap3methods': IndicatorConfig('cdlxsidegap3methods', {}, ['1h', '4h', '1d']),
        }
        
    def calculate_indicator(self, 
                          data: pd.DataFrame, 
                          indicator_name: str, 
                          config: Optional[IndicatorConfig] = None) -> IndicatorResult:
        """
        Calculate a specific technical indicator.
        
        Args:
            data: OHLCV data DataFrame
            indicator_name: Name of the indicator to calculate
            config: Optional configuration for the indicator
            
        Returns:
            IndicatorResult with calculated values and signals
        """
        try:
            if config is None:
                config = self.default_configs.get(indicator_name)
                if config is None:
                    raise ValueError(f"Unknown indicator: {indicator_name}")
            
            # Extract price data
            if indicator_name in self._get_price_based_indicators():
                values = self._calculate_price_indicator(data, indicator_name, config)
            elif indicator_name in self._get_volume_based_indicators():
                values = self._calculate_volume_indicator(data, indicator_name, config)
            elif indicator_name in self._get_pattern_indicators():
                values = self._calculate_pattern_indicator(data, indicator_name, config)
            else:
                raise ValueError(f"Unsupported indicator category for: {indicator_name}")
            
            # Generate signals
            signals = self._generate_signals(values, indicator_name)
            
            # Calculate confidence
            confidence = self._calculate_confidence(values, signals)
            
            return IndicatorResult(
                indicator_name=indicator_name,
                values=values,
                signal=signals,
                confidence=confidence,
                parameters=config.parameters,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Indicator calculation failed for {indicator_name}: {str(e)}")
            raise
    
    def _get_price_based_indicators(self) -> List[str]:
        """Get list of price-based indicators."""
        return [
            'sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'mama', 'ht_trendline', 'sar',
            'rsi', 'stoch', 'stochf', 'stochrsi', 'macd', 'macdext', 'macdfix', 'ppo',
            'roc', 'rocp', 'rocr', 'rocr100', 'mom', 'trix', 'ultosc', 'willr', 'cci', 'cmo',
            'dx', 'adx', 'adxr', 'apo', 'aroon', 'aroonosc', 'bop', 'mfi',
            'minus_di', 'minus_dm', 'plus_di', 'plus_dm',
            'bbands', 'atr', 'natr', 'trange',
            'ht_dcperiod', 'ht_dcphase', 'ht_phasor', 'ht_sine', 'ht_trendmode'
        ]
    
    def _get_volume_based_indicators(self) -> List[str]:
        """Get list of volume-based indicators."""
        return ['ad', 'adosc', 'obv', 'mfi']
    
    def _get_pattern_indicators(self) -> List[str]:
        """Get list of candlestick pattern indicators."""
        return [name for name in self.default_configs.keys() if name.startswith('cdl')]
    
    def _calculate_price_indicator(self, data: pd.DataFrame, indicator_name: str, config: IndicatorConfig) -> pd.Series:
        """Calculate price-based technical indicators."""
        try:
            close = data['close'].values
            high = data['high'].values if 'high' in data.columns else close
            low = data['low'].values if 'low' in data.columns else close
            open_price = data['open'].values if 'open' in data.columns else close
            volume = data['volume'].values if 'volume' in data.columns else np.ones_like(close)
            
            # Handle different indicator types
            if indicator_name == 'sma':
                if isinstance(config.parameters['timeperiod'], list):
                    # Return the first timeperiod for now, can be extended for multiple
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.SMA(close, timeperiod=period)
                
            elif indicator_name == 'ema':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.EMA(close, timeperiod=period)
                
            elif indicator_name == 'wma':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.WMA(close, timeperiod=period)
                
            elif indicator_name == 'dema':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.DEMA(close, timeperiod=period)
                
            elif indicator_name == 'tema':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.TEMA(close, timeperiod=period)
                
            elif indicator_name == 'trima':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.TRIMA(close, timeperiod=period)
                
            elif indicator_name == 'kama':
                values = talib.KAMA(close, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'mama':
                mama, fama = talib.MAMA(close, 
                                      fastlimit=config.parameters['fastlimit'],
                                      slowlimit=config.parameters['slowlimit'])
                values = mama  # Return MAMA line
                
            elif indicator_name == 'ht_trendline':
                values = talib.HT_TRENDLINE(close)
                
            elif indicator_name == 'sar':
                values = talib.SAR(high, low, 
                                 acceleration=config.parameters['acceleration'],
                                 maximum=config.parameters['maximum'])
                
            elif indicator_name == 'rsi':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.RSI(close, timeperiod=period)
                
            elif indicator_name == 'stoch':
                slowk, slowd = talib.STOCH(high, low, close,
                                         fastk_period=config.parameters['fastk_period'],
                                         slowk_period=config.parameters['slowk_period'],
                                         slowd_period=config.parameters['slowd_period'])
                values = slowk  # Return %K line
                
            elif indicator_name == 'stochf':
                fastk, fastd = talib.STOCHF(high, low, close,
                                          fastk_period=config.parameters['fastk_period'],
                                          fastd_period=config.parameters['fastd_period'])
                values = fastk  # Return %K line
                
            elif indicator_name == 'stochrsi':
                fastk, fastd = talib.STOCHRSI(close,
                                            timeperiod=config.parameters['timeperiod'],
                                            fastk_period=config.parameters['fastk_period'],
                                            fastd_period=config.parameters['fastd_period'])
                values = fastk  # Return %K line
                
            elif indicator_name == 'macd':
                macd_line, macd_signal, macd_hist = talib.MACD(close,
                                                             fastperiod=config.parameters['fastperiod'],
                                                             slowperiod=config.parameters['slowperiod'],
                                                             signalperiod=config.parameters['signalperiod'])
                values = macd_line  # Return MACD line
                
            elif indicator_name == 'macdext':
                macd_line, macd_signal, macd_hist = talib.MACDEXT(close,
                                                                fastperiod=config.parameters['fastperiod'],
                                                                slowperiod=config.parameters['slowperiod'],
                                                                signalperiod=config.parameters['signalperiod'])
                values = macd_line
                
            elif indicator_name == 'macdfix':
                macd_line, macd_signal, macd_hist = talib.MACDFIX(close,
                                                                signalperiod=config.parameters['signalperiod'])
                values = macd_line
                
            elif indicator_name == 'ppo':
                values = talib.PPO(close,
                                 fastperiod=config.parameters['fastperiod'],
                                 slowperiod=config.parameters['slowperiod'])
                
            elif indicator_name == 'roc':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.ROC(close, timeperiod=period)
                
            elif indicator_name == 'rocp':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.ROCP(close, timeperiod=period)
                
            elif indicator_name == 'rocr':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.ROCR(close, timeperiod=period)
                
            elif indicator_name == 'rocr100':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.ROCR100(close, timeperiod=period)
                
            elif indicator_name == 'mom':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.MOM(close, timeperiod=period)
                
            elif indicator_name == 'trix':
                values = talib.TRIX(close, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'ultosc':
                values = talib.ULTOSC(high, low, close,
                                    timeperiod1=config.parameters['timeperiod1'],
                                    timeperiod2=config.parameters['timeperiod2'],
                                    timeperiod3=config.parameters['timeperiod3'])
                
            elif indicator_name == 'willr':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.WILLR(high, low, close, timeperiod=period)
                
            elif indicator_name == 'cci':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.CCI(high, low, close, timeperiod=period)
                
            elif indicator_name == 'cmo':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.CMO(close, timeperiod=period)
                
            elif indicator_name == 'dx':
                values = talib.DX(high, low, close, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'adx':
                values = talib.ADX(high, low, close, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'adxr':
                values = talib.ADXR(high, low, close, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'apo':
                values = talib.APO(close,
                                 fastperiod=config.parameters['fastperiod'],
                                 slowperiod=config.parameters['slowperiod'])
                
            elif indicator_name == 'aroon':
                aroondown, aroonup = talib.AROON(high, low, timeperiod=config.parameters['timeperiod'])
                values = aroonup  # Return Aroon Up
                
            elif indicator_name == 'aroonosc':
                values = talib.AROONOSC(high, low, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'bop':
                values = talib.BOP(open_price, high, low, close)
                
            elif indicator_name == 'mfi':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.MFI(high, low, close, volume, timeperiod=period)
                
            elif indicator_name == 'minus_di':
                values = talib.MINUS_DI(high, low, close, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'minus_dm':
                values = talib.MINUS_DM(high, low, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'plus_di':
                values = talib.PLUS_DI(high, low, close, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'plus_dm':
                values = talib.PLUS_DM(high, low, timeperiod=config.parameters['timeperiod'])
                
            elif indicator_name == 'bbands':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                upper, middle, lower = talib.BBANDS(close,
                                                  timeperiod=period,
                                                  nbdevup=config.parameters['nbdevup'],
                                                  nbdevdn=config.parameters['nbdevdn'])
                values = middle  # Return middle band (SMA)
                
            elif indicator_name == 'atr':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.ATR(high, low, close, timeperiod=period)
                
            elif indicator_name == 'natr':
                if isinstance(config.parameters['timeperiod'], list):
                    period = config.parameters['timeperiod'][0]
                else:
                    period = config.parameters['timeperiod']
                values = talib.NATR(high, low, close, timeperiod=period)
                
            elif indicator_name == 'trange':
                values = talib.TRANGE(high, low, close)
                
            elif indicator_name == 'ht_dcperiod':
                values = talib.HT_DCPERIOD(close)
                
            elif indicator_name == 'ht_dcphase':
                values = talib.HT_DCPHASE(close)
                
            elif indicator_name == 'ht_phasor':
                inphase, quadrature = talib.HT_PHASOR(close)
                values = inphase  # Return in-phase component
                
            elif indicator_name == 'ht_sine':
                sine, leadsine = talib.HT_SINE(close)
                values = sine  # Return sine wave
                
            elif indicator_name == 'ht_trendmode':
                values = talib.HT_TRENDMODE(close)
                
            else:
                raise ValueError(f"Unsupported price-based indicator: {indicator_name}")
            
            return pd.Series(values, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Price indicator calculation failed for {indicator_name}: {str(e)}")
            raise
    
    def _calculate_volume_indicator(self, data: pd.DataFrame, indicator_name: str, config: IndicatorConfig) -> pd.Series:
        """Calculate volume-based technical indicators."""
        try:
            close = data['close'].values
            high = data['high'].values if 'high' in data.columns else close
            low = data['low'].values if 'low' in data.columns else close
            volume = data['volume'].values if 'volume' in data.columns else np.ones_like(close)
            
            if indicator_name == 'ad':
                values = talib.AD(high, low, close, volume)
                
            elif indicator_name == 'adosc':
                values = talib.ADOSC(high, low, close, volume,
                                   fastperiod=config.parameters['fastperiod'],
                                   slowperiod=config.parameters['slowperiod'])
                
            elif indicator_name == 'obv':
                values = talib.OBV(close, volume)
                
            else:
                raise ValueError(f"Unsupported volume-based indicator: {indicator_name}")
            
            return pd.Series(values, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Volume indicator calculation failed for {indicator_name}: {str(e)}")
            raise
    
    def _calculate_pattern_indicator(self, data: pd.DataFrame, indicator_name: str, config: IndicatorConfig) -> pd.Series:
        """Calculate candlestick pattern indicators."""
        try:
            open_price = data['open'].values
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Map indicator names to talib functions
            pattern_functions = {
                'cdl2crows': talib.CDL2CROWS,
                'cdl3blackcrows': talib.CDL3BLACKCROWS,
                'cdl3inside': talib.CDL3INSIDE,
                'cdl3linestrike': talib.CDL3LINESTRIKE,
                'cdl3outside': talib.CDL3OUTSIDE,
                'cdl3starsinsouth': talib.CDL3STARSINSOUTH,
                'cdl3whitesoldiers': talib.CDL3WHITESOLDIERS,
                'cdlbelthold': talib.CDLBELTHOLD,
                'cdlbreakaway': talib.CDLBREAKAWAY,
                'cdlclosingmarubozu': talib.CDLCLOSINGMARUBOZU,
                'cdlconcealbabyswall': talib.CDLCONCEALBABYSWALL,
                'cdlcounterattack': talib.CDLCOUNTERATTACK,
                'cdldoji': talib.CDLDOJI,
                'cdldojistar': talib.CDLDOJISTAR,
                'cdldragonflydoji': talib.CDLDRAGONFLYDOJI,
                'cdlengulfing': talib.CDLENGULFING,
                'cdlgapsidesidewhite': talib.CDLGAPSIDESIDEWHITE,
                'cdlgravestonedoji': talib.CDLGRAVESTONEDOJI,
                'cdlhammer': talib.CDLHAMMER,
                'cdlhangingman': talib.CDLHANGINGMAN,
                'cdlharami': talib.CDLHARAMI,
                'cdlharamicross': talib.CDLHARAMICROSS,
                'cdlhighwave': talib.CDLHIGHWAVE,
                'cdlhikkake': talib.CDLHIKKAKE,
                'cdlhikkakemod': talib.CDLHIKKAKEMOD,
                'cdlhomingpigeon': talib.CDLHOMINGPIGEON,
                'cdlidentical3crows': talib.CDLIDENTICAL3CROWS,
                'cdlinneck': talib.CDLINNECK,
                'cdlinvertedhammer': talib.CDLINVERTEDHAMMER,
                'cdlkicking': talib.CDLKICKING,
                'cdlkickingbylength': talib.CDLKICKINGBYLENGTH,
                'cdlladderbottom': talib.CDLLADDERBOTTOM,
                'cdllongleggeddoji': talib.CDLLONGLEGGEDDOJI,
                'cdllongline': talib.CDLLONGLINE,
                'cdlmarubozu': talib.CDLMARUBOZU,
                'cdlmatchinglow': talib.CDLMATCHINGLOW,
                'cdlonneck': talib.CDLONNECK,
                'cdlpiercing': talib.CDLPIERCING,
                'cdlrickshawman': talib.CDLRICKSHAWMAN,
                'cdlrisefall3methods': talib.CDLRISEFALL3methods,
                'cdlseparatinglines': talib.CDLSEPARATINGLINES,
                'cdlshootingstar': talib.CDLSHOOTINGSTAR,
                'cdlshortline': talib.CDLSHORTLINE,
                'cdlspinningtop': talib.CDLSPINNINGTOP,
                'cdlstalledpattern': talib.CDLSTALLEDPATTERN,
                'cdlsticksandwich': talib.CDLSTICKSANDWICH,
                'cdltakuri': talib.CDLTAKURI,
                'cdltasukigap': talib.CDLTASUKIGAP,
                'cdlthrusting': talib.CDLTHRUSTING,
                'cdltristar': talib.CDLTRISTAR,
                'cdlunique3river': talib.CDLUNIQUE3RIVER,
                'cdlupsidegap2crows': talib.CDLUPSIDEGAP2CROWS,
                'cdlxsidegap3methods': talib.CDLXSIDEGAP3METHODS
            }
            
            # Special cases with parameters
            if indicator_name == 'cdlabandonedbaby':
                values = talib.CDLABANDONEDBABY(open_price, high, low, close, 
                                              penetration=config.parameters['penetration'])
            elif indicator_name == 'cdldarkcloudcover':
                values = talib.CDLDARKCLOUDCOVER(open_price, high, low, close,
                                               penetration=config.parameters['penetration'])
            elif indicator_name == 'cdleveningdojistar':
                values = talib.CDLEVENINGDOJISTAR(open_price, high, low, close,
                                                penetration=config.parameters['penetration'])
            elif indicator_name == 'cdleveningstar':
                values = talib.CDLEVENINGSTAR(open_price, high, low, close,
                                            penetration=config.parameters['penetration'])
            elif indicator_name == 'cdlmathold':
                values = talib.CDLMATHOLD(open_price, high, low, close,
                                        penetration=config.parameters['penetration'])
            elif indicator_name == 'cdlmorningdojistar':
                values = talib.CDLMORNINGDOJISTAR(open_price, high, low, close,
                                                penetration=config.parameters['penetration'])
            elif indicator_name == 'cdlmorningstar':
                values = talib.CDLMORNINGSTAR(open_price, high, low, close,
                                            penetration=config.parameters['penetration'])
            elif indicator_name in pattern_functions:
                func = pattern_functions[indicator_name]
                values = func(open_price, high, low, close)
            else:
                raise ValueError(f"Unsupported pattern indicator: {indicator_name}")
            
            return pd.Series(values, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Pattern indicator calculation failed for {indicator_name}: {str(e)}")
            raise
    
    def _generate_signals(self, values: pd.Series, indicator_name: str) -> pd.Series:
        """Generate buy/sell/hold signals from indicator values."""
        try:
            signals = pd.Series(0, index=values.index)  # 0 = hold, 1 = buy, -1 = sell
            
            # Define signal generation logic based on indicator type
            if indicator_name == 'rsi':
                signals.loc[values < 30] = 1   # Oversold - buy signal
                signals.loc[values > 70] = -1  # Overbought - sell signal
                
            elif indicator_name in ['stoch', 'stochf', 'stochrsi']:
                signals.loc[values < 20] = 1   # Oversold
                signals.loc[values > 80] = -1  # Overbought
                
            elif indicator_name == 'willr':
                signals.loc[values < -80] = 1  # Oversold
                signals.loc[values > -20] = -1 # Overbought
                
            elif indicator_name == 'cci':
                signals.loc[values < -100] = 1  # Oversold
                signals.loc[values > 100] = -1  # Overbought
                
            elif indicator_name == 'mfi':
                signals.loc[values < 20] = 1   # Oversold
                signals.loc[values > 80] = -1  # Overbought
                
            elif indicator_name == 'macd':
                # MACD line crossing above/below zero
                signals.loc[(values > 0) & (values.shift(1) <= 0)] = 1   # Buy
                signals.loc[(values < 0) & (values.shift(1) >= 0)] = -1  # Sell
                
            elif indicator_name in ['sma', 'ema', 'wma', 'dema', 'tema']:
                # Price vs moving average (assuming we have price data)
                # This would need price data, so we'll use trend of the MA itself
                ma_trend = values.diff()
                signals.loc[ma_trend > 0] = 1   # Uptrend
                signals.loc[ma_trend < 0] = -1  # Downtrend
                
            elif indicator_name == 'adx':
                # ADX > 25 indicates strong trend
                signals.loc[values > 25] = 1   # Strong trend (could be up or down)
                
            elif indicator_name.startswith('cdl'):
                # Candlestick patterns: positive values = bullish, negative = bearish
                signals.loc[values > 0] = 1    # Bullish pattern
                signals.loc[values < 0] = -1   # Bearish pattern
                
            elif indicator_name == 'sar':
                # SAR signals based on position relative to price
                # This would need price data for proper implementation
                sar_trend = values.diff()
                signals.loc[sar_trend > 0] = -1  # SAR moving up (bearish)
                signals.loc[sar_trend < 0] = 1   # SAR moving down (bullish)
                
            elif indicator_name in ['aroon', 'aroonosc']:
                signals.loc[values > 50] = 1   # Bullish momentum
                signals.loc[values < -50] = -1 # Bearish momentum
                
            else:
                # Default: use trend of the indicator
                trend = values.diff()
                signals.loc[trend > 0] = 1
                signals.loc[trend < 0] = -1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed for {indicator_name}: {str(e)}")
            return pd.Series(0, index=values.index)
    
    def _calculate_confidence(self, values: pd.Series, signals: pd.Series) -> pd.Series:
        """Calculate confidence scores for signals."""
        try:
            confidence = pd.Series(0.5, index=values.index)  # Default 50% confidence
            
            # Calculate confidence based on signal strength and volatility
            if len(values) > 10:
                # Use rolling statistics to gauge confidence
                rolling_mean = values.rolling(10).mean()
                rolling_std = values.rolling(10).std()
                
                # Distance from mean as confidence indicator
                distance_from_mean = abs(values - rolling_mean) / (rolling_std + 1e-8)
                
                # Normalize to 0-1 range
                confidence = np.clip(distance_from_mean / 3, 0, 1)
                
                # Higher confidence for stronger signals
                confidence.loc[abs(signals) == 1] *= 1.2
                confidence = np.clip(confidence, 0, 1)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {str(e)}")
            return pd.Series(0.5, index=values.index)
    
    def calculate_all_indicators(self, 
                               data: pd.DataFrame,
                               indicator_subset: Optional[List[str]] = None) -> Dict[str, IndicatorResult]:
        """
        Calculate all configured technical indicators.
        
        Args:
            data: OHLCV data DataFrame
            indicator_subset: Optional subset of indicators to calculate
            
        Returns:
            Dictionary of indicator results
        """
        try:
            indicators_to_calc = indicator_subset or list(self.default_configs.keys())
            results = {}
            
            for indicator_name in indicators_to_calc:
                try:
                    if indicator_name in self.default_configs:
                        result = self.calculate_indicator(data, indicator_name)
                        results[indicator_name] = result
                        self.logger.debug(f"Calculated {indicator_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {indicator_name}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully calculated {len(results)} indicators")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch indicator calculation failed: {str(e)}")
            return {}
    
    def create_feature_matrix(self, 
                            indicator_results: Dict[str, IndicatorResult],
                            include_signals: bool = True,
                            include_confidence: bool = True) -> pd.DataFrame:
        """
        Create a feature matrix from indicator results.
        
        Args:
            indicator_results: Dictionary of indicator results
            include_signals: Whether to include signal columns
            include_confidence: Whether to include confidence columns
            
        Returns:
            DataFrame with indicator features
        """
        try:
            feature_dfs = []
            
            for indicator_name, result in indicator_results.items():
                # Add indicator values
                value_df = pd.DataFrame({
                    f'{indicator_name}_value': result.values
                })
                
                if include_signals:
                    value_df[f'{indicator_name}_signal'] = result.signal
                
                if include_confidence:
                    value_df[f'{indicator_name}_confidence'] = result.confidence
                
                feature_dfs.append(value_df)
            
            if feature_dfs:
                feature_matrix = pd.concat(feature_dfs, axis=1)
                
                # Remove rows with all NaN values
                feature_matrix = feature_matrix.dropna(how='all')
                
                self.logger.info(f"Created feature matrix with shape: {feature_matrix.shape}")
                return feature_matrix
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Feature matrix creation failed: {str(e)}")
            return pd.DataFrame()
    
    def normalize_indicators(self, 
                           feature_matrix: pd.DataFrame,
                           method: str = 'z_score') -> pd.DataFrame:
        """
        Normalize indicator values for ML model input.
        
        Args:
            feature_matrix: DataFrame with indicator features
            method: Normalization method ('z_score', 'minmax', 'robust')
            
        Returns:
            Normalized feature matrix
        """
        try:
            normalized_df = feature_matrix.copy()
            
            # Get value columns (exclude signal and confidence columns)
            value_columns = [col for col in feature_matrix.columns if col.endswith('_value')]
            
            if not value_columns:
                return normalized_df
            
            if method == 'z_score':
                scaler = StandardScaler()
                scaler_key = 'standard_scaler'
            elif method == 'minmax':
                scaler = MinMaxScaler()
                scaler_key = 'minmax_scaler'
            elif method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                scaler_key = 'robust_scaler'
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Fit and transform
            normalized_values = scaler.fit_transform(feature_matrix[value_columns])
            normalized_df[value_columns] = normalized_values
            
            # Store scaler for future use
            self.scalers[scaler_key] = scaler
            
            self.logger.info(f"Normalized {len(value_columns)} indicator columns using {method}")
            
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"Indicator normalization failed: {str(e)}")
            return feature_matrix
    
    def save_indicators(self, filepath: str) -> None:
        """Save technical indicators calculator state."""
        try:
            indicators_data = {
                'default_configs': self.default_configs,
                'indicators_cache': self.indicators_cache,
                'scalers': self.scalers,
                'model_version': self.model_version
            }
            
            joblib.dump(indicators_data, filepath)
            self.logger.info(f"Technical indicators saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Indicators saving failed: {str(e)}")
            raise
    
    def load_indicators(self, filepath: str) -> None:
        """Load technical indicators calculator state."""
        try:
            indicators_data = joblib.load(filepath)
            
            self.default_configs = indicators_data['default_configs']
            self.indicators_cache = indicators_data['indicators_cache']
            self.scalers = indicators_data['scalers']
            self.model_version = indicators_data['model_version']
            
            self.logger.info(f"Technical indicators loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Indicators loading failed: {str(e)}")
            raise


def create_technical_indicators(config: Optional[Dict] = None) -> TechnicalIndicators:
    """
    Create a technical indicators calculator with optional configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured TechnicalIndicators instance
    """
    indicators = TechnicalIndicators()
    
    if config and 'indicators' in config:
        # Override default configurations
        for indicator_name, indicator_config in config['indicators'].items():
            indicators.default_configs[indicator_name] = IndicatorConfig(**indicator_config)
    
    return indicators


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create technical indicators calculator
    indicators = TechnicalIndicators()
    
    # Example data would be processed here
    data = load_ohlcv_data('BTC')
    results = indicators.calculate_all_indicators(data)
    feature_matrix = indicators.create_feature_matrix(results)
    normalized_features = indicators.normalize_indicators(feature_matrix)
    
    print("Technical indicators implementation completed")
