"""
Model Utilities

Production utilities for model management, versioning, and optimization
with Chainlink integration support.
"""

import logging
import pickle
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Production model registry for versioning and deployment."""
    
    def __init__(self, registry_path: str = "./model_registry"):
        """Initialize model registry."""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "versions": {}}
    
    def _save_metadata(self):
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def register_model(self, 
                      model: Any, 
                      name: str, 
                      version: str,
                      metrics: Dict[str, float],
                      description: str = "") -> str:
        """Register a new model version."""
        # Create model hash for integrity
        model_hash = self._calculate_model_hash(model)
        
        # Save model
        model_path = self.registry_path / f"{name}_v{version}.joblib"
        joblib.dump(model, model_path)
        
        # Update metadata
        if name not in self.metadata["models"]:
            self.metadata["models"][name] = {
                "current_version": version,
                "versions": {}
            }
        
        self.metadata["models"][name]["versions"][version] = {
            "path": str(model_path),
            "hash": model_hash,
            "metrics": metrics,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        self._save_metadata()
        logger.info(f"Registered model {name} version {version}")
        return model_hash
    
    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """Load model from registry."""
        if name not in self.metadata["models"]:
            raise ValueError(f"Model {name} not found in registry")
        
        if version is None:
            version = self.metadata["models"][name]["current_version"]
        
        model_info = self.metadata["models"][name]["versions"][version]
        model_path = Path(model_info["path"])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)
    
    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of model for integrity checking."""
        if hasattr(model, 'get_params'):
            # Scikit-learn model
            model_str = str(model.get_params())
        else:
            # Generic model
            model_str = str(model)
        
        return hashlib.sha256(model_str.encode()).hexdigest()[:16]
    
    def list_models(self) -> Dict[str, Dict]:
        """List all registered models."""
        return self.metadata["models"]

class ModelOptimizer:
    """Model optimization utilities."""
    
    @staticmethod
    def optimize_sklearn_model(model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Optimize scikit-learn model parameters."""
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grids for common models
        param_grids = {
            'RandomForestRegressor': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForestClassifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBRegressor': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        model_name = type(model).__name__
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=3, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        
        return model
    
    @staticmethod
    def optimize_tensorflow_model(model: tf.keras.Model, 
                                X: np.ndarray, 
                                y: np.ndarray) -> tf.keras.Model:
        """Optimize TensorFlow model."""
        # Learning rate scheduling
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train with optimization
        history = model.fit(
            X, y,
            validation_split=0.2,
            epochs=100,
            callbacks=[lr_schedule, early_stop],
            verbose=0
        )
        
        return model

class ModelMetrics:
    """Comprehensive model metrics calculation."""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        }
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted')),
            'recall': float(recall_score(y_true, y_pred, average='weighted')),
            'f1': float(f1_score(y_true, y_pred, average='weighted'))
        }
        
        # Add AUC if binary classification
        if len(np.unique(y_true)) == 2:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred))
            except:
                pass
        
        return metrics
    
    @staticmethod
    def calculate_trading_metrics(returns: np.ndarray, benchmark_returns: np.ndarray = None) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        metrics = {
            'total_return': float(np.prod(1 + returns) - 1),
            'annualized_return': float(np.mean(returns) * 252),
            'volatility': float(np.std(returns) * np.sqrt(252)),
            'sharpe_ratio': float((np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))),
            'max_drawdown': float(ModelMetrics._calculate_max_drawdown(returns)),
            'win_rate': float(np.mean(returns > 0))
        }
        
        if benchmark_returns is not None:
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            alpha = metrics['annualized_return'] - beta * np.mean(benchmark_returns) * 252
            metrics['beta'] = float(beta)
            metrics['alpha'] = float(alpha)
        
        return metrics
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

class ModelValidator:
    """Model validation utilities."""
    
    @staticmethod
    def time_series_split_validate(model: Any, 
                                  X: pd.DataFrame, 
                                  y: pd.Series,
                                  n_splits: int = 5) -> Dict[str, List[float]]:
        """Time series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {'train_score': [], 'test_score': []}
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Calculate scores
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            scores['train_score'].append(train_score)
            scores['test_score'].append(test_score)
        
        return scores
    
    @staticmethod
    def validate_model_stability(model: Any, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               n_iterations: int = 10) -> Dict[str, float]:
        """Validate model stability across multiple training runs."""
        from sklearn.model_selection import train_test_split
        
        scores = []
        for _ in range(n_iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=None
            )
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        return {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'stability_ratio': float(1 - np.std(scores) / np.mean(scores))
        }

def save_model_with_metadata(model: Any, 
                           filepath: str, 
                           metadata: Dict[str, Any]):
    """Save model with comprehensive metadata."""
    model_data = {
        'model': model,
        'metadata': {
            **metadata,
            'saved_at': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'version': '1.0.0'
        }
    }
    
    joblib.dump(model_data, filepath)
    logger.info(f"Model saved with metadata to {filepath}")

def load_model_with_metadata(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """Load model with metadata."""
    model_data = joblib.load(filepath)
    return model_data['model'], model_data['metadata']

if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # Register a dummy model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    
    metrics = {'mse': 0.1, 'r2': 0.9}
    model_hash = registry.register_model(
        model, 'test_model', '1.0', metrics, 'Test model'
    )
    
    print(f"Model registered with hash: {model_hash}")
    print(f"Available models: {list(registry.list_models().keys())}")
