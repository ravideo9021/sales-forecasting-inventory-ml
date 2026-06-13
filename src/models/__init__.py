"""
Machine learning models module for the Sales Forecasting project.

This module contains implementations of various forecasting models including
XGBoost and provides a framework for ensemble methods.
"""

from .base_model import BaseModel
from .xgboost_model import XGBoostModel

# For now, only import existing models
# TODO: Add other models as they are implemented
# from .sarima_model import SARIMAModel
# from .prophet_model import ProphetModel
# from .lstm_model import LSTMModel
# from .ensemble_model import EnsembleModel

__all__ = [
    "BaseModel",
    "XGBoostModel"
] 