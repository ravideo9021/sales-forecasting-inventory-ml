"""
Model evaluation module for the Sales Forecasting project.

This module provides comprehensive model evaluation capabilities including
various metrics, statistical tests, and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class ModelEvaluator:
    """
    Comprehensive model evaluator for forecasting models.
    
    This class provides various evaluation metrics, statistical tests,
    and visualization tools for assessing model performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        self.metrics_config = self.evaluation_config.get('metrics', ['mape', 'wmape', 'rmse', 'mae'])
        
        logger.info("ModelEvaluator initialized successfully")
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays are 1D and handle NaN values
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Remove NaN and infinite values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid data points for metric calculation")
            return {}
        
        metrics = {}
        
        # Mean Absolute Error
        metrics['mae'] = float(mean_absolute_error(y_true_clean, y_pred_clean))
        
        # Root Mean Square Error
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
        
        # Mean Absolute Percentage Error
        mask_nonzero = y_true_clean != 0
        if mask_nonzero.any():
            mape = np.mean(np.abs((y_true_clean[mask_nonzero] - y_pred_clean[mask_nonzero]) / 
                                y_true_clean[mask_nonzero])) * 100
            metrics['mape'] = float(mape)
        
        # Weighted Mean Absolute Percentage Error
        if np.sum(np.abs(y_true_clean)) > 0:
            wmape = np.sum(np.abs(y_true_clean - y_pred_clean)) / np.sum(np.abs(y_true_clean)) * 100
            metrics['wmape'] = float(wmape)
        
        # Symmetric Mean Absolute Percentage Error
        denominator = (np.abs(y_true_clean) + np.abs(y_pred_clean)) / 2
        mask_nonzero_denom = denominator != 0
        if mask_nonzero_denom.any():
            smape = np.mean(np.abs(y_true_clean[mask_nonzero_denom] - y_pred_clean[mask_nonzero_denom]) / 
                          denominator[mask_nonzero_denom]) * 100
            metrics['smape'] = float(smape)
        
        # R-squared
        if len(y_true_clean) > 1:
            metrics['r2'] = float(r2_score(y_true_clean, y_pred_clean))
        
        # Mean Bias Error
        metrics['mbe'] = float(np.mean(y_pred_clean - y_true_clean))
        
        # Mean Absolute Scaled Error (MASE)
        if len(y_true_clean) > 1:
            naive_error = np.mean(np.abs(np.diff(y_true_clean)))
            if naive_error > 0:
                metrics['mase'] = metrics['mae'] / naive_error
        
        # Directional Accuracy
        if len(y_true_clean) > 1:
            true_direction = np.diff(y_true_clean) > 0
            pred_direction = np.diff(y_pred_clean) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
            metrics['directional_accuracy'] = float(directional_accuracy)
        
        return metrics
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model performance evaluation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating performance for {model_name}")
        
        # Calculate basic metrics
        metrics = self.calculate_all_metrics(y_true, y_pred)
        
        # Calculate residuals
        residuals = y_pred - y_true
        
        # Residual analysis
        residual_analysis = {
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals))
        }
        
        # Statistical tests
        statistical_tests = self.perform_statistical_tests(y_true, y_pred, residuals)
        
        evaluation_results = {
            'model_name': model_name,
            'metrics': metrics,
            'residual_analysis': residual_analysis,
            'statistical_tests': statistical_tests,
            'sample_size': len(y_true)
        }
        
        logger.info(f"Evaluation completed for {model_name}")
        return evaluation_results
    
    def perform_statistical_tests(self, y_true: np.ndarray, y_pred: np.ndarray,
                                residuals: np.ndarray) -> Dict[str, Any]:
        """
        Perform statistical tests on model predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            residuals: Model residuals
            
        Returns:
            Dictionary containing test results
        """
        tests = {}
        
        try:
            if len(residuals) > 10:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
                lb_stat = lb_result['lb_stat'].iloc[-1]
                lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
                tests['ljung_box'] = {
                    'statistic': float(lb_stat),
                    'p_value': float(lb_pvalue),
                    'interpretation': 'No autocorrelation' if lb_pvalue > 0.05 else 'Autocorrelation detected'
                }
        except Exception as e:
            logger.warning(f"Ljung-Box test failed: {e}")
        
        try:
            # Jarque-Bera test for normality of residuals
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            tests['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_pvalue),
                'interpretation': 'Residuals are normal' if jb_pvalue > 0.05 else 'Residuals are not normal'
            }
        except Exception as e:
            logger.warning(f"Jarque-Bera test failed: {e}")
        
        try:
            # Breusch-Pagan test for heteroscedasticity
            # Simplified version using correlation between squared residuals and predictions
            correlation, p_value = stats.pearsonr(y_pred, residuals**2)
            tests['heteroscedasticity'] = {
                'correlation': float(correlation),
                'p_value': float(p_value),
                'interpretation': 'Homoscedastic' if p_value > 0.05 else 'Heteroscedastic'
            }
        except Exception as e:
            logger.warning(f"Heteroscedasticity test failed: {e}")
        
        return tests
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_results: Dictionary of model evaluation results
            
        Returns:
            DataFrame comparing model performance
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            row.update(results.get('metrics', {}))
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by primary metric (WMAPE)
        if 'wmape' in comparison_df.columns:
            comparison_df['wmape_rank'] = comparison_df['wmape'].rank()
        
        return comparison_df
    
    def generate_evaluation_report(self, model_results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_results: Model evaluation results
            save_path: Optional path to save report
            
        Returns:
            Dictionary containing the report
        """
        report = {
            'summary': {
                'model_name': model_results.get('model_name', 'Unknown'),
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'sample_size': model_results.get('sample_size', 0)
            },
            'performance_metrics': model_results.get('metrics', {}),
            'residual_analysis': model_results.get('residual_analysis', {}),
            'statistical_tests': model_results.get('statistical_tests', {}),
            'recommendations': []
        }
        
        # Generate recommendations based on results
        metrics = model_results.get('metrics', {})
        
        if 'wmape' in metrics:
            if metrics['wmape'] < 10:
                report['recommendations'].append("Excellent forecast accuracy - model is production ready")
            elif metrics['wmape'] < 20:
                report['recommendations'].append("Good forecast accuracy - minor improvements possible")
            else:
                report['recommendations'].append("Poor forecast accuracy - model needs significant improvement")
        
        if 'r2' in metrics:
            if metrics['r2'] < 0.5:
                report['recommendations'].append("Low R² indicates poor model fit - consider feature engineering")
        
        # Check statistical tests
        tests = model_results.get('statistical_tests', {})
        if 'ljung_box' in tests and tests['ljung_box']['p_value'] < 0.05:
            report['recommendations'].append("Autocorrelation detected in residuals - consider time series modeling")
        
        if 'jarque_bera' in tests and tests['jarque_bera']['p_value'] < 0.05:
            report['recommendations'].append("Non-normal residuals detected - check for outliers or model specification")
        
        # Save report if path provided
        if save_path:
            try:
                import json
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Evaluation report saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report 