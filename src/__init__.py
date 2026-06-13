"""
Sales Forecasting & Inventory Optimization ML Project

A comprehensive machine learning solution for sales forecasting and inventory optimization
using real retail data from Kaggle's "Store Sales - Time Series Forecasting" competition.

This project demonstrates enterprise-level data science capabilities with predictive modeling,
optimization algorithms, and interactive dashboards suitable for Fortune 500 retail clients.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main modules for easy access
from .data import DataLoader, DataCleaner, FeatureEngineer
from .optimization import InventoryOptimizer
from .evaluation import ModelEvaluator

# Note: Dashboard and ForecastingPipeline are available in their respective modules
# from .models import XGBoostModel  # Import specific models as needed
# from .visualization import Dashboard  # Dashboard class is in app.py

__all__ = [
    "DataLoader",
    "DataCleaner", 
    "FeatureEngineer",
    "InventoryOptimizer",
    "ModelEvaluator"
] 