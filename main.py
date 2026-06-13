#!/usr/bin/env python3
"""
Main pipeline script for the Sales Forecasting & Inventory Optimization ML Project.

This script orchestrates the complete machine learning workflow from data loading
to model training, forecasting, and inventory optimization.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

from src.data import DataLoader, DataCleaner, FeatureEngineer
from src.models import XGBoostModel
from src.optimization import InventoryOptimizer
from src.evaluation import ModelEvaluator


class SalesForecastingPipeline:
    """
    Main pipeline class for sales forecasting and inventory optimization.
    
    This class orchestrates the complete workflow from data loading through
    model training, forecasting, and inventory optimization.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_loader = DataLoader(config_path)
        self.data_cleaner = DataCleaner(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.inventory_optimizer = InventoryOptimizer(self.config)
        
        # Pipeline state
        self.raw_data = {}
        self.processed_data = {}
        self.models = {}
        self.forecasts = {}
        self.optimization_results = {}
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Sales Forecasting Pipeline initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logger
        logger.add(
            "logs/pipeline.log",
            rotation="1 day",
            retention="30 days",
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', "{time} | {level} | {message}")
        )
    
    def load_data(self) -> dict:
        """
        Load all required datasets.
        
        Returns:
            Dictionary containing all loaded datasets
        """
        logger.info("Starting data loading phase...")
        
        try:
            # Check if data files exist
            raw_data_path = self.config['data']['raw_data_path']
            required_files = [
                self.config['data']['train_file'],
                self.config['data']['stores_file'],
                self.config['data']['oil_file'],
                self.config['data']['holidays_file']
            ]
            
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(raw_data_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                logger.warning(f"Missing data files: {missing_files}")
                logger.info("Please download the Kaggle dataset first:")
                logger.info("kaggle competitions download -c store-sales-time-series-forecasting")
                logger.info("unzip store-sales-time-series-forecasting.zip -d data/raw/")
                
                # Create sample data for demonstration
                logger.info("Creating sample data for demonstration...")
                self._create_sample_data()
            
            # Load all datasets
            self.raw_data = self.data_loader.load_all_data()
            
            # Get data summary
            data_summary = self.data_loader.get_data_summary()
            logger.info(f"Data loading completed. Summary: {data_summary}")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
            raise
    
    def _create_sample_data(self) -> None:
        """Create sample data for demonstration purposes."""
        logger.info("Creating sample data for demonstration...")
        
        # Create sample training data
        np.random.seed(42)
        n_stores = 10
        n_families = 5
        n_days = 365 * 3  # 3 years of data
        
        stores = [f"STORE_{i+1}" for i in range(n_stores)]
        families = ["GROCERY", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY"]
        
        # Generate date range
        start_date = datetime(2013, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Stable base sales per store-family so lag features are actually predictive
        store_family_base = {
            (s, f): np.random.gamma(4, 25)
            for s in stores for f in families
        }

        train_data = []
        prev_sales_state = {(s, f): store_family_base[(s, f)] for s in stores for f in families}

        for date in dates:
            doy = date.timetuple().tm_yday
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * doy / 365)
            weekend_factor = 1.2 if date.weekday() >= 5 else 1.0

            for store in stores:
                for family in families:
                    base = store_family_base[(store, family)]
                    promo = np.random.random() < 0.1
                    promo_factor = 1.5 if promo else 1.0

                    # Expected sales for this day
                    expected = base * seasonal_factor * weekend_factor * promo_factor

                    # Autocorrelated: 40% yesterday, 60% expected + small noise
                    sales = 0.4 * prev_sales_state[(store, family)] + 0.6 * expected
                    sales = max(0, sales + np.random.normal(0, base * 0.08))
                    prev_sales_state[(store, family)] = sales

                    train_data.append({
                        'date': date,
                        'store_nbr': int(store.split('_')[1]),
                        'family': family,
                        'sales': sales,
                        'onpromotion': 1 if promo else 0,
                    })
        
        train_df = pd.DataFrame(train_data)
        
        # Create stores data
        stores_data = []
        for i in range(n_stores):
            stores_data.append({
                'store_nbr': i + 1,
                'city': f"City_{i+1}",
                'state': f"State_{(i%3)+1}",
                'type': ['A', 'B', 'C'][i % 3],
                'cluster': (i % 5) + 1
            })
        
        stores_df = pd.DataFrame(stores_data)
        
        # Create oil data
        oil_data = []
        for date in dates:
            price = 50 + 10 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365) + np.random.normal(0, 5)
            oil_data.append({
                'date': date,
                'dcoilwtico': max(20, price)
            })
        
        oil_df = pd.DataFrame(oil_data)
        
        # Create holidays data
        holidays_data = [
            {'date': datetime(2013, 12, 25), 'type': 'Holiday', 'locale': 'National', 'transferred': False},
            {'date': datetime(2014, 12, 25), 'type': 'Holiday', 'locale': 'National', 'transferred': False},
            {'date': datetime(2015, 12, 25), 'type': 'Holiday', 'locale': 'National', 'transferred': False},
            {'date': datetime(2013, 1, 1), 'type': 'Holiday', 'locale': 'National', 'transferred': False},
            {'date': datetime(2014, 1, 1), 'type': 'Holiday', 'locale': 'National', 'transferred': False},
            {'date': datetime(2015, 1, 1), 'type': 'Holiday', 'locale': 'National', 'transferred': False},
        ]
        holidays_df = pd.DataFrame(holidays_data)
        
        # Create transactions data
        transactions_data = []
        for store in range(1, n_stores + 1):
            for date in dates:
                transactions = max(100, np.random.poisson(500) + np.random.normal(0, 50))
                transactions_data.append({
                    'date': date,
                    'store_nbr': store,
                    'transactions': transactions
                })
        
        transactions_df = pd.DataFrame(transactions_data)
        
        # Create test data (next 30 days)
        test_dates = [dates[-1] + timedelta(days=i+1) for i in range(30)]
        test_data = []
        for store in stores:
            for family in families:
                for date in test_dates:
                    test_data.append({
                        'date': date,
                        'store_nbr': int(store.split('_')[1]),
                        'family': family,
                        'onpromotion': 1 if np.random.random() < 0.1 else 0
                    })
        
        test_df = pd.DataFrame(test_data)
        
        # Save sample data
        raw_data_path = self.config['data']['raw_data_path']
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_df.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        stores_df.to_csv(os.path.join(raw_data_path, 'stores.csv'), index=False)
        oil_df.to_csv(os.path.join(raw_data_path, 'oil.csv'), index=False)
        holidays_df.to_csv(os.path.join(raw_data_path, 'holidays_events.csv'), index=False)
        transactions_df.to_csv(os.path.join(raw_data_path, 'transactions.csv'), index=False)
        
        logger.info("Sample data created successfully")
    
    def clean_data(self) -> dict:
        """
        Clean and preprocess the loaded data.
        
        Returns:
            Dictionary containing cleaned datasets
        """
        logger.info("Starting data cleaning phase...")
        
        try:
            # Clean training data
            if 'train' in self.raw_data:
                logger.info("Cleaning training data...")
                self.processed_data['train_cleaned'] = self.data_cleaner.clean_data(
                    self.raw_data['train']
                )
                
                # Generate data quality report
                quality_report = self.data_cleaner.generate_data_quality_report(
                    self.processed_data['train_cleaned']
                )
                logger.info(f"Data quality report: {quality_report}")
            
            # Store other datasets (they typically need less cleaning)
            for key in ['stores', 'oil', 'holidays', 'transactions', 'test']:
                if key in self.raw_data:
                    self.processed_data[f'{key}_cleaned'] = self.data_cleaner.validate_data_types(
                        self.raw_data[key]
                    )
            
            logger.info("Data cleaning completed successfully")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Perform feature engineering on the cleaned data.
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering phase...")
        
        try:
            # Get cleaned datasets
            train_data = self.processed_data['train_cleaned']
            oil_data = self.processed_data.get('oil_cleaned')
            holidays_data = self.processed_data.get('holidays_cleaned')
            transactions_data = self.processed_data.get('transactions_cleaned')
            
            # Perform feature engineering
            self.processed_data['features'] = self.feature_engineer.engineer_features(
                train_data, oil_data, holidays_data, transactions_data
            )
            
            # Get feature summary
            feature_summary = self.feature_engineer.get_feature_summary()
            logger.info(f"Feature engineering summary: {feature_summary}")
            
            # Save processed data
            processed_path = self.config['data']['processed_data_path']
            os.makedirs(processed_path, exist_ok=True)
            
            self.data_loader.save_processed_data(
                self.processed_data['features'], 
                'features_engineered.csv'
            )
            
            logger.info("Feature engineering completed successfully")
            return self.processed_data['features']
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            raise
    
    def _get_exclude_columns(self) -> list:
        """Return columns excluded from model features."""
        target_col = self.config['data']['target_column']
        date_col = self.config['data']['date_column']
        return [
            target_col, date_col, 'store_nbr', 'family',
            # raw oil price — derived features retained
            'dcoilwtico',
            # non-numeric / identifier columns
            'holiday_locale', 'holiday_type',
            # transactions excluded (not available at prediction time for future dates)
            'transactions', 'transactions_lag_1', 'transactions_lag_7',
            'transactions_ma_7', 'transactions_ma_30', 'transactions_growth',
        ]

    def train_models(self) -> dict:
        """
        Train forecasting models.

        Returns:
            Dictionary containing trained models
        """
        logger.info("Starting model training phase...")

        try:
            features_data = self.processed_data['features']

            target_col = self.config['data']['target_column']
            date_col = self.config['data']['date_column']

            exclude_columns = self._get_exclude_columns()

            # Select only numeric columns for modeling
            numeric_columns = features_data.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns
                             if col not in exclude_columns]
            
            X = features_data[feature_columns]
            y = features_data[target_col]
            
            # Chronological 80/20 split — sort dates and take the 80th percentile date
            sorted_dates = features_data[date_col].sort_values()
            split_date = sorted_dates.iloc[int(len(sorted_dates) * 0.8)]
            train_mask = features_data[date_col] <= split_date
            
            X_train, X_val = X[train_mask], X[~train_mask]
            y_train, y_val = y[train_mask], y[~train_mask]
            
            logger.info(f"Training set size: {len(X_train)}")
            logger.info(f"Validation set size: {len(X_val)}")
            
            # Train XGBoost model
            logger.info("Training XGBoost model...")
            xgb_model = XGBoostModel(self.config)
            xgb_model.fit(
                X_train, y_train,
                tune_hyperparameters=False,  # Set to True for full hyperparameter tuning
                validation_data=(X_val, y_val)
            )
            
            self.models['xgboost'] = xgb_model

            # LightGBM second base learner + averaging ensemble (Path-A SOTA upgrade).
            # Guarded so a missing optional dependency never breaks training.
            try:
                from src.models.lightgbm_model import LightGBMModel
                from src.models.ensemble import EnsembleForecaster
                lgb_model = LightGBMModel(self.config)
                lgb_model.fit(X_train, y_train, validation_data=(X_val, y_val))
                self.models['lightgbm'] = lgb_model
                self.models['ensemble'] = EnsembleForecaster(
                    {'xgboost': xgb_model, 'lightgbm': lgb_model})
                logger.info("LightGBM + XGBoost ensemble ready")
            except Exception as e:
                logger.warning(f"LightGBM/ensemble step skipped: {e}")

            # Rolling-origin backtest — honest multi-cutoff error estimate.
            try:
                import xgboost as _xgb
                from src.evaluation.backtest import rolling_origin_backtest
                horizon = int(self.config.get('evaluation', {})
                              .get('time_series_cv', {}).get('test_size', 30))
                factory = lambda: _xgb.XGBRegressor(
                    objective='reg:tweedie', tweedie_variance_power=1.2,
                    n_estimators=400, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)
                bt = rolling_origin_backtest(factory, X, y, features_data[date_col],
                                             n_splits=5, horizon=horizon)
                if not bt.empty:
                    Path("reports").mkdir(exist_ok=True)
                    bt.to_csv("reports/backtest_xgboost.csv", index=False)
                    logger.info(f"Backtest saved → reports/backtest_xgboost.csv\n{bt}")
            except Exception as e:
                logger.warning(f"Rolling backtest skipped: {e}")

            # Save trained models
            models_path = Path("models")
            models_path.mkdir(exist_ok=True)

            xgb_model.save_model("models/xgboost_model.joblib")

            logger.info("Model training completed successfully")
            return self.models
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def generate_forecasts(self) -> dict:
        """
        Generate sales forecasts using trained models.

        Returns:
            Dictionary containing forecasts
        """
        logger.info("Starting forecast generation phase...")

        try:
            if 'test_cleaned' not in self.processed_data:
                logger.warning("No test data available for forecasting")
                return {}

            test_data = self.processed_data['test_cleaned']
            train_data = self.processed_data['train_cleaned']
            date_col = self.config['data']['date_column']
            target_col = self.config['data']['target_column']

            # Prepend enough training history so lag/rolling features are correct for test rows.
            max_lag = max(self.config['data']['lag_features'])
            max_window = max(self.config['data']['rolling_windows'])
            history_needed = max_lag + max_window

            test_start = test_data[date_col].min()
            history_cutoff = test_start - pd.Timedelta(days=history_needed)
            train_history = train_data[train_data[date_col] >= history_cutoff].copy()

            test_with_dummy = test_data.copy()
            test_with_dummy[target_col] = 0.0

            combined_data = pd.concat([train_history, test_with_dummy], ignore_index=True)

            combined_features = self.feature_engineer.engineer_features(
                combined_data,
                self.processed_data.get('oil_cleaned'),
                self.processed_data.get('holidays_cleaned'),
                self.processed_data.get('transactions_cleaned'),
            )

            # Keep only test rows
            test_features = combined_features[combined_features[date_col] >= test_start].copy()

            # Use the same feature columns as training
            features_data = self.processed_data['features']
            exclude_columns = self._get_exclude_columns()
            numeric_columns = features_data.select_dtypes(include=[np.number]).columns
            feature_columns = [c for c in numeric_columns if c not in exclude_columns]

            X_test = test_features.reindex(columns=feature_columns, fill_value=0)

            if 'xgboost' in self.models:
                # Prefer the ensemble (XGBoost + LightGBM) when available.
                fc_model_name = 'ensemble' if 'ensemble' in self.models else 'xgboost'
                fc_model = self.models[fc_model_name]
                logger.info(f"Generating forecasts with: {fc_model_name}")
                xgb_predictions = fc_model.predict(X_test)

                forecast_df = test_data.copy()
                forecast_df['forecast'] = xgb_predictions
                forecast_df['model'] = fc_model_name

                # Probabilistic interval (P10/P90) from the quantile models
                quantiles = fc_model.predict_quantiles(X_test)
                if quantiles is not None:
                    forecast_df['forecast_lo'] = quantiles['lo']
                    forecast_df['forecast_hi'] = quantiles['hi']
                    logger.info("Added P10/P90 quantile bounds to forecasts")

                self.forecasts['xgboost'] = forecast_df
                forecast_df.to_csv('data/processed/forecasts_xgboost.csv', index=False)
                logger.info(f"Generated {len(forecast_df)} forecasts")

            logger.info("Forecast generation completed successfully")
            return self.forecasts

        except Exception as e:
            logger.error(f"Error during forecast generation: {e}")
            raise
    
    def optimize_inventory(self) -> dict:
        """
        Perform inventory optimization using forecasts.
        
        Returns:
            Dictionary containing optimization results
        """
        logger.info("Starting inventory optimization phase...")
        
        try:
            if not self.forecasts:
                logger.warning("No forecasts available for inventory optimization")
                return {}
            
            # Use XGBoost forecasts for optimization
            forecast_data = self.forecasts.get('xgboost')
            
            if forecast_data is None:
                logger.warning("No XGBoost forecasts available")
                return {}
            
            # Create sample inventory data
            inventory_data = self._create_sample_inventory_data(forecast_data)
            
            # Perform inventory optimization
            optimization_results = self.inventory_optimizer.optimize_inventory_levels(
                forecast_data, inventory_data
            )
            
            # Perform ABC analysis
            if 'item_level_results' in optimization_results:
                results_df = optimization_results['item_level_results']
                
                # Calculate annual value for ABC analysis
                results_df['annual_value'] = results_df['demand_forecast'] * 365 * 10  # Assume $10 unit cost
                
                abc_results = self.inventory_optimizer.perform_abc_analysis(
                    results_df, 'annual_value'
                )
                
                # Generate recommendations
                recommendations = self.inventory_optimizer.generate_inventory_recommendations(
                    optimization_results
                )
                
                self.optimization_results = {
                    'optimization_results': optimization_results,
                    'abc_analysis': abc_results,
                    'recommendations': recommendations
                }
                
                # Save optimization results
                recommendations.to_csv('data/processed/inventory_recommendations.csv', index=False)
                abc_results.to_csv('data/processed/abc_analysis.csv', index=False)
            
            logger.info("Inventory optimization completed successfully")
            return self.optimization_results
            
        except Exception as e:
            logger.error(f"Error during inventory optimization: {e}")
            raise
    
    def _create_sample_inventory_data(self, forecast_data: pd.DataFrame) -> pd.DataFrame:
        """Create sample inventory data for demonstration."""
        np.random.seed(42)
        
        # Create unique item IDs
        items = forecast_data[['store_nbr', 'family']].drop_duplicates()
        items['item_id'] = range(len(items))
        
        # Merge with forecast data
        forecast_with_items = forecast_data.merge(items, on=['store_nbr', 'family'])
        
        # Create inventory data
        inventory_data = []
        for item_id in items['item_id'].unique():
            inventory_data.append({
                'item_id': item_id,
                'current_inventory': max(10, np.random.poisson(100)),
                'unit_cost': np.random.uniform(5, 50),
                'lead_time': np.random.choice([7, 14, 21, 30])
            })
        
        return pd.DataFrame(inventory_data)
    
    def run_complete_pipeline(self) -> dict:
        """
        Run the complete pipeline from data loading to optimization.
        
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting complete sales forecasting pipeline...")
        
        try:
            # Phase 1: Data Loading
            self.load_data()
            
            # Phase 2: Data Cleaning
            self.clean_data()
            
            # Phase 3: Feature Engineering
            self.engineer_features()
            
            # Phase 4: Model Training
            self.train_models()
            
            # Phase 5: Forecast Generation
            self.generate_forecasts()
            
            # Phase 6: Inventory Optimization
            self.optimize_inventory()
            
            # Generate final report
            pipeline_results = {
                'data_summary': self.data_loader.get_data_summary(),
                'cleaning_summary': self.data_cleaner.get_cleaning_summary(),
                'feature_summary': self.feature_engineer.get_feature_summary(),
                'model_info': {name: model.get_model_info() for name, model in self.models.items()},
                'forecasts_summary': {name: len(forecast) for name, forecast in self.forecasts.items()},
                'optimization_summary': self.inventory_optimizer.get_optimization_summary()
            }
            
            logger.info("Complete pipeline execution finished successfully")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Sales Forecasting & Inventory Optimization Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--phase', type=str, choices=['all', 'data', 'train', 'forecast', 'optimize'],
                       default='all', help='Pipeline phase to run')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SalesForecastingPipeline(args.config)
    
    try:
        if args.phase == 'all':
            results = pipeline.run_complete_pipeline()
            logger.info("Pipeline completed successfully!")
            print("\n=== PIPELINE RESULTS ===")
            print(f"Models trained: {list(results['model_info'].keys())}")
            print(f"Forecasts generated: {sum(results['forecasts_summary'].values())}")
            print(f"Optimization completed: {bool(results['optimization_summary'])}")
            
        elif args.phase == 'data':
            pipeline.load_data()
            pipeline.clean_data()
            pipeline.engineer_features()
            logger.info("Data processing completed!")
            
        elif args.phase == 'train':
            pipeline.load_data()
            pipeline.clean_data()
            pipeline.engineer_features()
            pipeline.train_models()
            logger.info("Model training completed!")
            
        elif args.phase == 'forecast':
            pipeline.load_data()
            pipeline.clean_data()
            pipeline.engineer_features()
            pipeline.train_models()
            pipeline.generate_forecasts()
            logger.info("Forecast generation completed!")
            
        elif args.phase == 'optimize':
            pipeline.load_data()
            pipeline.clean_data()
            pipeline.engineer_features()
            pipeline.train_models()
            pipeline.generate_forecasts()
            pipeline.optimize_inventory()
            logger.info("Inventory optimization completed!")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 