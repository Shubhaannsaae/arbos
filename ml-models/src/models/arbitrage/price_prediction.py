"""
Price Prediction Model for Arbitrage Opportunities

This module implements advanced ML models for predicting price movements
across DEXs using Chainlink Data Feeds and historical market data.

Features:
- LSTM neural networks for time series prediction
- Ensemble methods combining multiple models
- Real-time price prediction using Chainlink oracles
- Multi-asset and cross-chain price forecasting
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union
import joblib
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class PricePredictionModel:
    """
    Advanced price prediction model using ensemble methods and deep learning
    for arbitrage opportunity detection across multiple DEXs and chains.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_horizon: int = 5,
                 ensemble_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the price prediction model.
        
        Args:
            sequence_length: Number of historical data points to use for prediction
            prediction_horizon: Number of future time steps to predict
            ensemble_weights: Weights for ensemble model combination
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.ensemble_weights = ensemble_weights or {
            'lstm': 0.4,
            'transformer': 0.3,
            'xgboost': 0.2,
            'lightgbm': 0.1
        }
        
        # Model components
        self.lstm_model = None
        self.transformer_model = None
        self.xgb_model = None
        self.lgb_model = None
        
        # Scalers for data normalization
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        
        # Model metadata
        self.feature_columns = []
        self.target_columns = []
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def prepare_features(self, 
                        df: pd.DataFrame, 
                        target_assets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for model training.
        
        Args:
            df: DataFrame with price and market data
            target_assets: List of assets to predict prices for
            
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        try:
            # Technical indicators
            df = self._add_technical_indicators(df)
            
            # Market features
            df = self._add_market_features(df)
            
            # Cross-asset features
            df = self._add_cross_asset_features(df, target_assets)
            
            # Time-based features
            df = self._add_temporal_features(df)
            
            # Select feature columns
            feature_cols = [col for col in df.columns 
                          if col not in target_assets and not col.startswith('target_')]
            
            self.feature_columns = feature_cols
            self.target_columns = target_assets
            
            # Create sequences for time series prediction
            X, y = self._create_sequences(df[feature_cols].values, 
                                        df[target_assets].values)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        try:
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Volatility indicators
            df['volatility'] = df['close'].rolling(window=20).std()
            df['atr'] = self._calculate_atr(df)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change features
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_20'] = df['close'].pct_change(20)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicator calculation failed: {str(e)}")
            raise
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high_low = df['high'] - df['low']
            high_close_prev = np.abs(df['high'] - df['close'].shift(1))
            low_close_prev = np.abs(df['low'] - df['close'].shift(1))
            
            true_range = np.maximum(high_low, 
                                  np.maximum(high_close_prev, low_close_prev))
            
            return true_range.rolling(window=14).mean()
            
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide features."""
        try:
            # Market momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # Support and resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            
            # Trend strength
            df['trend_strength'] = np.abs(df['close'].rolling(window=20).corr(
                pd.Series(range(20), index=df.index)))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market feature calculation failed: {str(e)}")
            raise
    
    def _add_cross_asset_features(self, df: pd.DataFrame, target_assets: List[str]) -> pd.DataFrame:
        """Add cross-asset correlation and spread features."""
        try:
            # Asset correlations
            for i, asset1 in enumerate(target_assets):
                for asset2 in target_assets[i+1:]:
                    if asset1 in df.columns and asset2 in df.columns:
                        corr_col = f'corr_{asset1}_{asset2}'
                        df[corr_col] = df[asset1].rolling(window=20).corr(df[asset2])
                        
                        # Price spreads
                        spread_col = f'spread_{asset1}_{asset2}'
                        df[spread_col] = df[asset1] - df[asset2]
                        
                        # Spread ratios
                        ratio_col = f'ratio_{asset1}_{asset2}'
                        df[ratio_col] = df[asset1] / df[asset2]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Cross-asset feature calculation failed: {str(e)}")
            raise
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        try:
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Temporal feature calculation failed: {str(e)}")
            raise
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        try:
            X_seq, y_seq = [], []
            
            for i in range(self.sequence_length, len(X) - self.prediction_horizon + 1):
                X_seq.append(X[i-self.sequence_length:i])
                y_seq.append(y[i:i+self.prediction_horizon])
            
            return np.array(X_seq), np.array(y_seq)
            
        except Exception as e:
            self.logger.error(f"Sequence creation failed: {str(e)}")
            raise
    
    def build_lstm_model(self, input_shape: Tuple[int, int], output_shape: int) -> Model:
        """Build LSTM model for price prediction."""
        try:
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                BatchNormalization(),
                
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                BatchNormalization(),
                
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                
                Dense(64, activation='relu'),
                Dropout(0.1),
                Dense(32, activation='relu'),
                Dense(output_shape)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"LSTM model building failed: {str(e)}")
            raise
    
    def build_transformer_model(self, input_shape: Tuple[int, int], output_shape: int) -> Model:
        """Build Transformer model for price prediction."""
        try:
            inputs = tf.keras.Input(shape=input_shape)
            
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=64
            )(inputs, inputs)
            
            # Add & Norm
            attention_output = tf.keras.layers.LayerNormalization()(
                inputs + attention_output
            )
            
            # Feed forward
            ffn_output = tf.keras.layers.Dense(256, activation='relu')(attention_output)
            ffn_output = tf.keras.layers.Dense(input_shape[-1])(ffn_output)
            
            # Add & Norm
            ffn_output = tf.keras.layers.LayerNormalization()(
                attention_output + ffn_output
            )
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
            
            # Final layers
            outputs = tf.keras.layers.Dense(64, activation='relu')(pooled)
            outputs = tf.keras.layers.Dropout(0.1)(outputs)
            outputs = tf.keras.layers.Dense(output_shape)(outputs)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Transformer model building failed: {str(e)}")
            raise
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray, 
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32) -> Dict[str, any]:
        """
        Train all ensemble models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics
        """
        try:
            self.logger.info("Starting model training...")
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
            
            X_val_scaled = self.feature_scaler.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)
            
            # Scale targets
            y_train_scaled = self.price_scaler.fit_transform(
                y_train.reshape(-1, y_train.shape[-1])
            ).reshape(y_train.shape)
            
            y_val_scaled = self.price_scaler.transform(
                y_val.reshape(-1, y_val.shape[-1])
            ).reshape(y_val.shape)
            
            training_metrics = {}
            
            # Train LSTM model
            self.logger.info("Training LSTM model...")
            self.lstm_model = self.build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_shape=y_train.shape[1] * y_train.shape[2]
            )
            
            lstm_history = self.lstm_model.fit(
                X_train_scaled,
                y_train_scaled.reshape(y_train_scaled.shape[0], -1),
                validation_data=(
                    X_val_scaled,
                    y_val_scaled.reshape(y_val_scaled.shape[0], -1)
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ],
                verbose=1
            )
            
            training_metrics['lstm'] = {
                'final_loss': lstm_history.history['loss'][-1],
                'final_val_loss': lstm_history.history['val_loss'][-1],
                'epochs_trained': len(lstm_history.history['loss'])
            }
            
            # Train Transformer model
            self.logger.info("Training Transformer model...")
            self.transformer_model = self.build_transformer_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_shape=y_train.shape[1] * y_train.shape[2]
            )
            
            transformer_history = self.transformer_model.fit(
                X_train_scaled,
                y_train_scaled.reshape(y_train_scaled.shape[0], -1),
                validation_data=(
                    X_val_scaled,
                    y_val_scaled.reshape(y_val_scaled.shape[0], -1)
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ],
                verbose=1
            )
            
            training_metrics['transformer'] = {
                'final_loss': transformer_history.history['loss'][-1],
                'final_val_loss': transformer_history.history['val_loss'][-1],
                'epochs_trained': len(transformer_history.history['loss'])
            }
            
            # Prepare data for tree-based models
            X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
            X_val_flat = X_val_scaled.reshape(X_val_scaled.shape[0], -1)
            y_train_flat = y_train_scaled.reshape(y_train_scaled.shape[0], -1)
            y_val_flat = y_val_scaled.reshape(y_val_scaled.shape[0], -1)
            
            # Train XGBoost model
            self.logger.info("Training XGBoost model...")
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=50
            )
            
            self.xgb_model.fit(
                X_train_flat, y_train_flat,
                eval_set=[(X_val_flat, y_val_flat)],
                verbose=False
            )
            
            xgb_val_pred = self.xgb_model.predict(X_val_flat)
            training_metrics['xgboost'] = {
                'val_mse': mean_squared_error(y_val_flat, xgb_val_pred),
                'val_mae': mean_absolute_error(y_val_flat, xgb_val_pred)
            }
            
            # Train LightGBM model
            self.logger.info("Training LightGBM model...")
            self.lgb_model = lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.lgb_model.fit(
                X_train_flat, y_train_flat,
                eval_set=[(X_val_flat, y_val_flat)],
                callbacks=[lgb.early_stopping(50)]
            )
            
            lgb_val_pred = self.lgb_model.predict(X_val_flat)
            training_metrics['lightgbm'] = {
                'val_mse': mean_squared_error(y_val_flat, lgb_val_pred),
                'val_mae': mean_absolute_error(y_val_flat, lgb_val_pred)
            }
            
            self.last_trained = datetime.now()
            self.logger.info("Model training completed successfully")
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Ensemble predictions
        """
        try:
            # Scale input features
            X_scaled = self.feature_scaler.transform(
                X.reshape(-1, X.shape[-1])
            ).reshape(X.shape)
            
            predictions = {}
            
            # LSTM predictions
            if self.lstm_model is not None:
                lstm_pred = self.lstm_model.predict(X_scaled, verbose=0)
                predictions['lstm'] = lstm_pred.reshape(lstm_pred.shape[0], 
                                                       self.prediction_horizon, -1)
            
            # Transformer predictions
            if self.transformer_model is not None:
                transformer_pred = self.transformer_model.predict(X_scaled, verbose=0)
                predictions['transformer'] = transformer_pred.reshape(
                    transformer_pred.shape[0], self.prediction_horizon, -1)
            
            # Tree-based model predictions
            X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
            
            if self.xgb_model is not None:
                xgb_pred = self.xgb_model.predict(X_flat)
                predictions['xgboost'] = xgb_pred.reshape(xgb_pred.shape[0], 
                                                        self.prediction_horizon, -1)
            
            if self.lgb_model is not None:
                lgb_pred = self.lgb_model.predict(X_flat)
                predictions['lightgbm'] = lgb_pred.reshape(lgb_pred.shape[0], 
                                                         self.prediction_horizon, -1)
            
            # Ensemble predictions
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            
            for model_name, pred in predictions.items():
                weight = self.ensemble_weights.get(model_name, 0)
                ensemble_pred += weight * pred
            
            # Inverse transform predictions
            ensemble_pred_flat = ensemble_pred.reshape(-1, ensemble_pred.shape[-1])
            ensemble_pred_original = self.price_scaler.inverse_transform(ensemble_pred_flat)
            ensemble_pred_final = ensemble_pred_original.reshape(ensemble_pred.shape)
            
            return ensemble_pred_final
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            predictions = self.predict(X_test)
            
            # Flatten for evaluation
            y_test_flat = y_test.reshape(-1)
            predictions_flat = predictions.reshape(-1)
            
            metrics = {
                'mse': mean_squared_error(y_test_flat, predictions_flat),
                'mae': mean_absolute_error(y_test_flat, predictions_flat),
                'rmse': np.sqrt(mean_squared_error(y_test_flat, predictions_flat)),
                'r2': r2_score(y_test_flat, predictions_flat),
                'mape': np.mean(np.abs((y_test_flat - predictions_flat) / y_test_flat)) * 100
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        try:
            model_data = {
                'lstm_model': self.lstm_model,
                'transformer_model': self.transformer_model,
                'xgb_model': self.xgb_model,
                'lgb_model': self.lgb_model,
                'price_scaler': self.price_scaler,
                'feature_scaler': self.feature_scaler,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'ensemble_weights': self.ensemble_weights,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'model_version': self.model_version,
                'last_trained': self.last_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        try:
            model_data = joblib.load(filepath)
            
            self.lstm_model = model_data['lstm_model']
            self.transformer_model = model_data['transformer_model']
            self.xgb_model = model_data['xgb_model']
            self.lgb_model = model_data['lgb_model']
            self.price_scaler = model_data['price_scaler']
            self.feature_scaler = model_data['feature_scaler']
            self.sequence_length = model_data['sequence_length']
            self.prediction_horizon = model_data['prediction_horizon']
            self.ensemble_weights = model_data['ensemble_weights']
            self.feature_columns = model_data['feature_columns']
            self.target_columns = model_data['target_columns']
            self.model_version = model_data['model_version']
            self.last_trained = model_data['last_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models."""
        try:
            importance = {}
            
            if self.xgb_model is not None:
                importance['xgboost'] = self.xgb_model.feature_importances_
            
            if self.lgb_model is not None:
                importance['lightgbm'] = self.lgb_model.feature_importances_
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {str(e)}")
            return {}


def create_price_prediction_pipeline(config: Dict) -> PricePredictionModel:
    """
    Create a price prediction pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured PricePredictionModel instance
    """
    return PricePredictionModel(
        sequence_length=config.get('sequence_length', 60),
        prediction_horizon=config.get('prediction_horizon', 5),
        ensemble_weights=config.get('ensemble_weights')
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = PricePredictionModel()
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()
    
    training_metrics = model.train(X_train, y_train, X_val, y_val)
    
    test_metrics = model.evaluate(X_test, y_test)
    
    model.save_model('price_prediction_model.pkl')
    
    print("Price prediction model implementation completed")
