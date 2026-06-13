"""
LightGBM model — mirrors XGBoostModel (Tweedie objective + quantile intervals).

LightGBM typically ties or slightly beats XGBoost on M5-scale tabular retail
data while training faster. Here it serves as a second base learner so the
pipeline can ensemble the two (see :mod:`src.models.ensemble`).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import lightgbm as lgb
from loguru import logger

from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM regressor with Tweedie objective and P10/P50/P90 quantile models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "LightGBM")
        # Reuse the xgboost block if no dedicated lightgbm config is present.
        self.lgb_config = config['models'].get('lightgbm', config['models'].get('xgboost', {}))
        self.best_params = None
        self.quantiles = [0.1, 0.5, 0.9]
        self.quantile_models: Dict[float, Any] = {}
        logger.info("LightGBM model initialized")

    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data=None, **kwargs) -> 'LightGBMModel':
        self.validate_data(X, y)
        X_processed = self.prepare_features(X)
        y_array = np.asarray(y)

        objective = self.lgb_config.get('objective', 'tweedie')
        params = dict(
            objective=objective,
            n_estimators=int(self.lgb_config.get('n_estimators', 1000)),
            max_depth=int(self.lgb_config.get('max_depth', 6)),
            learning_rate=float(self.lgb_config.get('learning_rate', 0.1)),
            subsample=float(self.lgb_config.get('subsample', 0.8)),
            colsample_bytree=float(self.lgb_config.get('colsample_bytree', 0.8)),
            random_state=int(self.lgb_config.get('random_state', 42)),
            verbose=-1,
        )
        if objective == 'tweedie':
            params['tweedie_variance_power'] = float(self.lgb_config.get('tweedie_variance_power', 1.2))
        self.best_params = params

        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X_processed.values, y_array)
        self.is_trained = True
        self.training_metrics = self.calculate_metrics(y_array, self.model.predict(X_processed.values))

        if validation_data is not None:
            X_val, y_val = validation_data
            self.validation_metrics = self.calculate_metrics(np.asarray(y_val), self.predict(X_val))

        self._fit_quantiles(X_processed.values, y_array, params)
        logger.info("LightGBM model training completed")
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return np.maximum(self.model.predict(self._align(X)), 0)

    def predict_proba(self, X: pd.DataFrame, confidence_level: float = 0.95, **kwargs):
        """Return (point, [[lo, hi], ...]) using the quantile models when available."""
        preds = self.predict(X)
        q = self.predict_quantiles(X)
        if q is not None:
            ci = np.column_stack([q['lo'], q['hi']])
        else:
            std = preds.std() * 0.1
            ci = np.column_stack([np.maximum(preds - 1.96 * std, 0), preds + 1.96 * std])
        return preds, ci

    def _fit_quantiles(self, X_array: np.ndarray, y_array: np.ndarray, base_params: dict) -> None:
        """One quantile model per alpha (LightGBM quantile loss is single-alpha)."""
        self.quantile_models = {}
        try:
            for q in self.quantiles:
                p = {k: v for k, v in base_params.items()
                     if k not in ('objective', 'tweedie_variance_power')}
                p.update(objective='quantile', alpha=q)
                m = lgb.LGBMRegressor(**p)
                m.fit(X_array, y_array)
                self.quantile_models[q] = m
            logger.info(f"LightGBM quantile models trained for {self.quantiles}")
        except Exception as e:
            logger.warning(f"LightGBM quantile training skipped: {e}")
            self.quantile_models = {}

    def predict_quantiles(self, X: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
        if not self.quantile_models:
            return None
        Xa = self._align(X)
        cols = np.column_stack([np.maximum(self.quantile_models[q].predict(Xa), 0)
                                for q in self.quantiles])
        cols = np.sort(cols, axis=1)  # guard against quantile crossing
        return {'lo': cols[:, 0], 'median': cols[:, 1], 'hi': cols[:, 2]}

    def _align(self, X: pd.DataFrame) -> np.ndarray:
        self.validate_data(X)
        X_processed = self.prepare_features(X)
        if self.feature_names is not None:
            for feature in set(self.feature_names) - set(X_processed.columns):
                X_processed[feature] = 0
            X_processed = X_processed[self.feature_names]
        return X_processed.values
