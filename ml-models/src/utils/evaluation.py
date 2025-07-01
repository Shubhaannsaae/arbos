"""
Model Evaluation Utilities

Comprehensive evaluation metrics and backtesting utilities for ML models
with trading-specific metrics and Chainlink integration support.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtest result data structure."""
    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    returns_series: pd.Series
    equity_curve: pd.Series
    drawdown_series: pd.Series

@dataclass
class ModelPerformanceReport:
    """Comprehensive model performance report."""
    model_name: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray]
    feature_importance: Optional[Dict[str, float]]
    prediction_distribution: Dict[str, Any]
    temporal_performance: pd.DataFrame
    recommendations: List[str]
    timestamp: datetime

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.evaluation_history = []
    
    def evaluate_classification(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Comprehensive classification evaluation."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            if y_pred_proba.ndim == 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        return metrics
    
    def evaluate_regression(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict[str, float]:
        """Comprehensive regression evaluation."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
            'directional_accuracy': np.mean(np.sign(y_true) == np.sign(y_pred))
        }
    
    def evaluate_trading_strategy(self, 
                                signals: pd.Series, 
                                prices: pd.Series,
                                transaction_cost: float = 0.001) -> BacktestResult:
        """Evaluate trading strategy performance."""
        # Align signals and prices
        aligned_data = pd.concat([signals, prices], axis=1, join='inner')
        aligned_data.columns = ['signal', 'price']
        
        # Calculate returns
        returns = self._calculate_strategy_returns(
            aligned_data['signal'], 
            aligned_data['price'], 
            transaction_cost
        )
        
        # Calculate equity curve
        equity_curve = (1 + returns).cumprod()
        
        # Calculate drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        # Calculate metrics
        total_return = equity_curve.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside volatility)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        
        max_drawdown = abs(drawdown.min())
        
        # Trade analysis
        trades = self._analyze_trades(aligned_data['signal'], returns)
        win_rate = trades['win_rate']
        profit_factor = trades['profit_factor']
        total_trades = trades['total_trades']
        avg_trade_duration = trades['avg_duration']
        
        return BacktestResult(
            strategy_name="Strategy",
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=avg_trade_duration,
            returns_series=returns,
            equity_curve=equity_curve,
            drawdown_series=drawdown
        )
    
    def _calculate_strategy_returns(self, 
                                  signals: pd.Series, 
                                  prices: pd.Series,
                                  transaction_cost: float) -> pd.Series:
        """Calculate strategy returns with transaction costs."""
        # Calculate price returns
        price_returns = prices.pct_change().fillna(0)
        
        # Apply signals (1 for long, -1 for short, 0 for neutral)
        strategy_returns = signals.shift(1) * price_returns
        
        # Apply transaction costs when position changes
        position_changes = signals.diff().abs()
        transaction_costs = position_changes * transaction_cost
        
        # Net returns after costs
        net_returns = strategy_returns - transaction_costs
        
        return net_returns.fillna(0)
    
    def _analyze_trades(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Analyze individual trades."""
        # Identify trade periods
        position_changes = signals.diff()
        trade_starts = position_changes[position_changes != 0].index
        
        trade_returns = []
        trade_durations = []
        
        for i in range(len(trade_starts) - 1):
            start_idx = trade_starts[i]
            end_idx = trade_starts[i + 1]
            
            # Calculate trade return
            trade_return = returns.loc[start_idx:end_idx].sum()
            trade_returns.append(trade_return)
            
            # Calculate trade duration
            duration = (end_idx - start_idx).days
            trade_durations.append(duration)
        
        if not trade_returns:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'avg_duration': 0.0
            }
        
        # Calculate metrics
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        win_rate = len(winning_trades) / len(trade_returns)
        
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 1e-8
        profit_factor = total_profit / total_loss
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trade_returns),
            'avg_duration': np.mean(trade_durations) if trade_durations else 0
        }
    
    def create_performance_report(self, 
                                model_name: str,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                model: Optional[Any] = None) -> ModelPerformanceReport:
        """Create comprehensive performance report."""
        # Determine task type
        if len(np.unique(y_true)) <= 10 and np.issubdtype(y_true.dtype, np.integer):
            task_type = 'classification'
            metrics = self.evaluate_classification(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)
        else:
            task_type = 'regression'
            metrics = self.evaluate_regression(y_true, y_pred)
            conf_matrix = None
        
        # Feature importance
        feature_importance = None
        if model and hasattr(model, 'feature_importances_') and feature_names:
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        # Prediction distribution
        prediction_dist = {
            'mean': float(np.mean(y_pred)),
            'std': float(np.std(y_pred)),
            'min': float(np.min(y_pred)),
            'max': float(np.max(y_pred)),
            'quantiles': {
                '25%': float(np.percentile(y_pred, 25)),
                '50%': float(np.percentile(y_pred, 50)),
                '75%': float(np.percentile(y_pred, 75))
            }
        }
        
        # Temporal performance (simplified)
        n_bins = min(10, len(y_true) // 100)
        if n_bins > 0:
            bin_size = len(y_true) // n_bins
            temporal_perf_data = []
            
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
                
                bin_y_true = y_true[start_idx:end_idx]
                bin_y_pred = y_pred[start_idx:end_idx]
                
                if task_type == 'classification':
                    bin_metrics = self.evaluate_classification(bin_y_true, bin_y_pred)
                else:
                    bin_metrics = self.evaluate_regression(bin_y_true, bin_y_pred)
                
                temporal_perf_data.append({
                    'bin': i,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    **bin_metrics
                })
            
            temporal_performance = pd.DataFrame(temporal_perf_data)
        else:
            temporal_performance = pd.DataFrame()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, task_type)
        
        return ModelPerformanceReport(
            model_name=model_name,
            metrics=metrics,
            confusion_matrix=conf_matrix,
            feature_importance=feature_importance,
            prediction_distribution=prediction_dist,
            temporal_performance=temporal_performance,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _generate_recommendations(self, metrics: Dict[str, float], task_type: str) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if task_type == 'classification':
            if metrics.get('accuracy', 0) < 0.7:
                recommendations.append("Consider feature engineering or model tuning - accuracy below 70%")
            
            if metrics.get('f1_score', 0) < 0.6:
                recommendations.append("F1 score is low - check class imbalance")
            
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            if abs(precision - recall) > 0.2:
                recommendations.append("Large precision-recall gap - consider threshold tuning")
        
        else:  # regression
            if metrics.get('r2_score', 0) < 0.5:
                recommendations.append("Low R² score - model may be underfitting")
            
            if metrics.get('mape', 100) > 20:
                recommendations.append("High MAPE - consider log transformation or outlier treatment")
            
            if metrics.get('directional_accuracy', 0) < 0.55:
                recommendations.append("Poor directional accuracy for trading applications")
        
        return recommendations

class CrossValidationEvaluator:
    """Cross-validation evaluation utilities."""
    
    @staticmethod
    def time_series_cv(model: Any, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      n_splits: int = 5) -> Dict[str, List[float]]:
        """Time series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    
    @staticmethod
    def walk_forward_validation(model: Any,
                              X: pd.DataFrame,
                              y: pd.Series,
                              initial_train_size: int = 252,
                              step_size: int = 21) -> Dict[str, Any]:
        """Walk-forward validation for time series."""
        scores = []
        predictions = []
        actual_values = []
        
        for i in range(initial_train_size, len(X), step_size):
            # Training data
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            
            # Test data
            test_end = min(i + step_size, len(X))
            X_test = X.iloc[i:test_end]
            y_test = y.iloc[i:test_end]
            
            if len(X_test) == 0:
                break
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store results
            score = model.score(X_test, y_test)
            scores.append(score)
            predictions.extend(y_pred)
            actual_values.extend(y_test.values)
        
        return {
            'scores': scores,
            'predictions': np.array(predictions),
            'actual_values': np.array(actual_values),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }

def plot_backtest_results(backtest_result: BacktestResult, 
                         save_path: Optional[str] = None):
    """Plot comprehensive backtest results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Backtest Results - {backtest_result.strategy_name}', fontsize=16)
    
    # Equity curve
    axes[0, 0].plot(backtest_result.equity_curve.index, backtest_result.equity_curve.values)
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_ylabel('Portfolio Value')
    axes[0, 0].grid(True)
    
    # Drawdown
    axes[0, 1].fill_between(backtest_result.drawdown_series.index, 
                           backtest_result.drawdown_series.values, 0, 
                           alpha=0.3, color='red')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown %')
    axes[0, 1].grid(True)
    
    # Returns distribution
    axes[1, 0].hist(backtest_result.returns_series.values, bins=50, alpha=0.7)
    axes[1, 0].set_title('Returns Distribution')
    axes[1, 0].set_xlabel('Daily Returns')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # Performance metrics
    metrics_text = f"""
    Total Return: {backtest_result.total_return:.2%}
    Annualized Return: {backtest_result.annualized_return:.2%}
    Volatility: {backtest_result.volatility:.2%}
    Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}
    Max Drawdown: {backtest_result.max_drawdown:.2%}
    Win Rate: {backtest_result.win_rate:.2%}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    
    # Evaluate classification
    metrics = evaluator.evaluate_classification(y_true, y_pred)
    print("Classification Metrics:", metrics)
    
    # Create performance report
    report = evaluator.create_performance_report("Test Model", y_true, y_pred)
    print(f"Performance Report: {report.model_name}")
    print(f"Recommendations: {report.recommendations}")
