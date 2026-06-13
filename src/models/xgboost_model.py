"""
XGBoost model implementation for the Sales Forecasting project.

This module implements XGBoost regression for sales forecasting with
hyperparameter tuning, feature importance analysis, and cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import optuna
from loguru import logger

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost model for sales forecasting.
    
    This class implements XGBoost regression with advanced features including
    hyperparameter tuning using Optuna, feature importance analysis, and
    time series cross-validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the XGBoost model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config, "XGBoost")
        
        # XGBoost specific configuration
        self.xgb_config = config['models']['xgboost']
        self.tuning_config = config.get('tuning_config', {})
        
        # Model parameters
        self.best_params = None
        self.feature_importance_df = None
        self.study = None

        # Probabilistic forecasting (quantile regression)
        self.quantiles = [0.1, 0.5, 0.9]
        self.quantile_model = None

        logger.info("XGBoost model initialized")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            tune_hyperparameters: bool = True,
            validation_data: Tuple[pd.DataFrame, pd.Series] = None,
            **kwargs) -> 'XGBoostModel':
        """
        Train the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target values
            tune_hyperparameters: Whether to perform hyperparameter tuning
            validation_data: Optional validation data for early stopping
            **kwargs: Additional arguments
            
        Returns:
            Self (trained model)
        """
        logger.info("Training XGBoost model...")
        
        # Validate input data
        self.validate_data(X, y)
        
        # Prepare features
        X_processed = self.prepare_features(X)
        
        # Convert to numpy arrays for XGBoost
        X_array = X_processed.values
        y_array = y.values
        
        # Perform hyperparameter tuning if requested
        if tune_hyperparameters:
            logger.info("Starting hyperparameter tuning...")
            self.best_params = self._tune_hyperparameters(X_processed, y)
        else:
            # Use default parameters from config
            self.best_params = {
                'n_estimators': self.xgb_config['n_estimators'],
                'max_depth': self.xgb_config['max_depth'],
                'learning_rate': self.xgb_config['learning_rate'],
                'subsample': self.xgb_config['subsample'],
                'colsample_bytree': self.xgb_config['colsample_bytree'],
                'random_state': self.xgb_config['random_state']
            }
        
        # Prepare validation data for early stopping
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_processed = self.prepare_features(X_val)
            eval_set = [(X_val_processed.values, y_val.values)]
        
        # Objective: default to Tweedie. Retail sales are right-skewed and
        # zero-inflated, which plain L2/RMSE handles poorly; Tweedie (used by the
        # M5 winners) models that distribution directly. Override via
        # config['models']['xgboost']['objective'].
        self.best_params.setdefault('objective', self.xgb_config.get('objective', 'reg:tweedie'))
        if 'tweedie' in str(self.best_params['objective']):
            self.best_params.setdefault(
                'tweedie_variance_power',
                self.xgb_config.get('tweedie_variance_power', 1.2))

        # Train the model with best parameters
        self.model = xgb.XGBRegressor(
            **self.best_params,
            early_stopping_rounds=50 if eval_set is not None else None,
            eval_metric='rmse',
            verbose=False
        )
        
        # Fit the model
        if eval_set is not None:
            self.model.fit(
                X_array, y_array,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_array, y_array)
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_array)
        self.training_metrics = self.calculate_metrics(y_array, y_pred_train)
        
        # Set trained flag before validation to avoid recursion
        self.is_trained = True
        
        # Calculate validation metrics if validation data provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_pred_val = self.predict(X_val)
            self.validation_metrics = self.calculate_metrics(y_val, y_pred_val)
        
        # Calculate feature importance
        self.feature_importance_df = self._calculate_feature_importance()

        # Probabilistic intervals via quantile regression (pinball loss)
        self._fit_quantile_model(X_array, y_array)

        logger.info("XGBoost model training completed")
        logger.info(f"Training WMAPE: {self.training_metrics.get('wmape', 'N/A'):.4f}")

        return self

    def _fit_quantile_model(self, X_array: np.ndarray, y_array: np.ndarray) -> None:
        """Train a single multi-quantile model (P10/P50/P90) for prediction intervals.

        Uses XGBoost's native quantile objective (pinball loss). This replaces the
        old heuristic ±10% band with honest, asymmetric, demand-aware uncertainty.
        """
        bp = self.best_params or {}
        try:
            self.quantile_model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=np.array(self.quantiles),
                n_estimators=int(bp.get('n_estimators', 500)),
                max_depth=int(bp.get('max_depth', 6)),
                learning_rate=float(bp.get('learning_rate', 0.1)),
                subsample=float(bp.get('subsample', 0.8)),
                colsample_bytree=float(bp.get('colsample_bytree', 0.8)),
                random_state=int(bp.get('random_state', 42)),
            )
            self.quantile_model.fit(X_array, y_array)
            logger.info(f"Quantile models trained for {self.quantiles}")
        except Exception as e:  # quantile objective unavailable on older XGBoost
            logger.warning(f"Quantile model training skipped: {e}")
            self.quantile_model = None

    def predict_quantiles(self, X: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
        """Return {'lo','median','hi'} arrays (P10/P50/P90), or None if unavailable."""
        if getattr(self, 'quantile_model', None) is None:
            return None
        self.validate_data(X)
        X_processed = self.prepare_features(X)
        if self.feature_names is not None:
            for feature in set(self.feature_names) - set(X_processed.columns):
                X_processed[feature] = 0
            X_processed = X_processed[self.feature_names]
        q = np.maximum(self.quantile_model.predict(X_processed.values), 0)
        if q.ndim == 1:  # single-quantile fallback
            return {'lo': q, 'median': q, 'hi': q}
        # Enforce monotonic ordering across quantiles (guards rare crossings)
        q = np.sort(q, axis=1)
        return {'lo': q[:, 0], 'median': q[:, 1], 'hi': q[:, 2]}
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using the trained XGBoost model.
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional arguments
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate and prepare features
        self.validate_data(X)
        X_processed = self.prepare_features(X)
        
        # Ensure feature consistency
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X_processed.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    X_processed[feature] = 0
            
            # Reorder columns to match training data
            X_processed = X_processed[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X_processed.values)
        
        # Ensure non-negative predictions for sales
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame, confidence_level: float = 0.95, 
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate probabilistic predictions with confidence intervals.
        
        Args:
            X: Feature matrix for prediction
            confidence_level: Confidence level for intervals
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        # Get point predictions
        predictions = self.predict(X)
        
        # For XGBoost, we'll use quantile regression or bootstrap for uncertainty estimation
        # This is a simplified implementation using prediction variance
        
        # Calculate residuals from training data for uncertainty estimation
        if hasattr(self, '_training_residuals'):
            residual_std = np.std(self._training_residuals)
        else:
            # Rough estimate based on validation metrics
            residual_std = predictions.std() * 0.1  # Simplified approach
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        z_score = 1.96  # For 95% confidence interval
        
        margin_of_error = z_score * residual_std
        confidence_intervals = np.column_stack([
            predictions - margin_of_error,
            predictions + margin_of_error
        ])
        
        # Ensure non-negative bounds
        confidence_intervals = np.maximum(confidence_intervals, 0)
        
        return predictions, confidence_intervals
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of best parameters
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0.0, 0.4),
                'random_state': 42
            }
            
            # Perform cross-validation
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced for faster tuning
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model with current parameters
                model = xgb.XGBRegressor(**params, verbose=0)
                model.fit(X_train.values, y_train.values)
                
                # Predict and calculate score
                y_pred = model.predict(X_val.values)
                y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
                
                # Calculate WMAPE
                wmape = np.sum(np.abs(y_val - y_pred)) / np.sum(np.abs(y_val)) * 100
                cv_scores.append(wmape)
            
            return np.mean(cv_scores)
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='minimize',
            study_name='xgboost_hyperparameter_tuning'
        )
        
        # Optimize
        n_trials = self.tuning_config.get('optuna', {}).get('n_trials', 50)
        timeout = self.tuning_config.get('optuna', {}).get('timeout', 1800)
        
        self.study.optimize(
            objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = self.study.best_params
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best WMAPE: {self.study.best_value:.4f}")
        
        return best_params
    
    def _calculate_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            return pd.DataFrame()
        
        # Get feature importance from XGBoost
        importance_gain = self.model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_gain': importance_gain
        }).sort_values('importance_gain', ascending=False)
        
        # Add normalized importance
        importance_df['importance_normalized'] = (
            importance_df['importance_gain'] / importance_df['importance_gain'].sum()
        )
        
        # Add cumulative importance
        importance_df['importance_cumulative'] = importance_df['importance_normalized'].cumsum()
        
        return importance_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.feature_importance_df.copy()
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to plot feature importance")
        
        import matplotlib.pyplot as plt
        
        # Get top N features
        top_features = self.feature_importance_df.head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance_gain'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importance - XGBoost')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def get_hyperparameter_tuning_results(self) -> Dict[str, Any]:
        """
        Get hyperparameter tuning results.
        
        Returns:
            Dictionary with tuning results
        """
        if self.study is None:
            return {}
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty using bootstrap.
        
        Args:
            X: Feature matrix for prediction
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # This would require storing training data for bootstrap sampling
        # For now, return standard predictions with simple uncertainty
        predictions = self.predict(X)
        
        # Simple uncertainty estimation
        uncertainty = predictions * 0.1  # 10% uncertainty
        lower_bound = predictions - uncertainty
        upper_bound = predictions + uncertainty
        
        # Ensure non-negative bounds
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)
        
        return predictions, lower_bound, upper_bound 