"""
Base model class for the Sales Forecasting project.

This module defines the abstract base class that all forecasting models
must inherit from, ensuring consistent interface and behavior.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from loguru import logger
import joblib
import os


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    This class defines the common interface and shared functionality
    that all forecasting models must implement.
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Initialize the base model.
        
        Args:
            config: Configuration dictionary
            model_name: Name of the model
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}
        self.validation_metrics = {}
        
        logger.info(f"{self.model_name} model initialized")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional arguments specific to each model
            
        Returns:
            Self (trained model)
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional arguments specific to each model
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate probabilistic predictions with confidence intervals.
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional arguments specific to each model
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        pass
    
    def validate_data(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Validate input data format and quality.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
        """
        # Check if data is pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Check for empty data
        if X.empty:
            raise ValueError("X cannot be empty")
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected in X")
        
        # Check target data if provided
        if y is not None:
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise TypeError("y must be a pandas Series or numpy array")
            
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")
            
            if np.isinf(y).any() if hasattr(y, 'any') else np.isinf(y).any():
                logger.warning("Infinite values detected in y")
    
    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training or prediction.
        
        Args:
            X: Raw feature matrix
            
        Returns:
            Processed feature matrix
        """
        # Create a copy to avoid modifying original data
        X_processed = X.copy()
        
        if X_processed.isnull().any().any():
            logger.warning("Missing values detected, filling with group mean then 0")
            X_processed = X_processed.ffill().bfill().fillna(0)
        
        # Store feature names for later use
        if self.feature_names is None:
            self.feature_names = list(X_processed.columns)
        
        return X_processed
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Ensure arrays are 1D
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Remove any NaN or infinite values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            logger.warning("No valid predictions for metric calculation")
            return {}
        
        metrics = {}
        
        # Mean Absolute Error
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        
        # Root Mean Square Error
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # Mean Absolute Percentage Error
        mask_nonzero = y_true != 0
        if mask_nonzero.any():
            mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
            metrics['mape'] = float(mape)
        
        # Weighted Mean Absolute Percentage Error
        if np.sum(np.abs(y_true)) > 0:
            wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
            metrics['wmape'] = float(wmape)
        
        # Symmetric Mean Absolute Percentage Error
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask_nonzero_denom = denominator != 0
        if mask_nonzero_denom.any():
            smape = np.mean(np.abs(y_true[mask_nonzero_denom] - y_pred[mask_nonzero_denom]) / 
                          denominator[mask_nonzero_denom]) * 100
            metrics['smape'] = float(smape)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot > 0:
            metrics['r2'] = float(1 - (ss_res / ss_tot))
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model with metadata
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'config': self.config,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self (loaded model)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.validation_metrics = model_data['validation_metrics']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Default implementation returns empty DataFrame
        # Subclasses should override this method if they support feature importance
        logger.warning(f"{self.model_name} does not support feature importance")
        return pd.DataFrame()
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5, 
                      metric: str = 'wmape') -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_folds: Number of cross-validation folds
            metric: Metric to optimize
            
        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_model = self.__class__(self.config)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = fold_model.predict(X_val)
            
            # Calculate metrics
            fold_metrics = self.calculate_metrics(y_val, y_pred)
            cv_scores.append(fold_metrics.get(metric, np.inf))
        
        cv_results = {
            f'cv_{metric}_mean': np.mean(cv_scores),
            f'cv_{metric}_std': np.std(cv_scores),
            f'cv_{metric}_scores': cv_scores
        }
        
        logger.info(f"Cross-validation completed: {metric} = {cv_results[f'cv_{metric}_mean']:.4f} ± {cv_results[f'cv_{metric}_std']:.4f}")
        
        return cv_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "not trained"
        return f"{self.model_name} ({status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', is_trained={self.is_trained})" 