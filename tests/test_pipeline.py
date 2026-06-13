"""
End-to-end and unit tests for the Sales Forecasting pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd

from src.data.data_cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
from src.models.base_model import BaseModel
from src.models.xgboost_model import XGBoostModel
from src.optimization.safety_stock import SafetyStockCalculator
from src.optimization.reorder_point import ReorderPointOptimizer
from src.optimization.inventory_optimizer import InventoryOptimizer
from src.evaluation.metrics import ModelEvaluator


# Minimal config used across tests
CONFIG = {
    'data': {
        'raw_data_path': 'data/raw/',
        'processed_data_path': 'data/processed/',
        'external_data_path': 'data/external/',
        'synthetic_data_path': 'data/synthetic/',
        'train_file': 'train.csv',
        'test_file': 'test.csv',
        'stores_file': 'stores.csv',
        'oil_file': 'oil.csv',
        'holidays_file': 'holidays_events.csv',
        'transactions_file': 'transactions.csv',
        'date_column': 'date',
        'target_column': 'sales',
        'store_column': 'store_nbr',
        'family_column': 'family',
        'lag_features': [1, 7, 14],
        'rolling_windows': [7, 14],
        'seasonal_periods': [7, 30],
    },
    'models': {
        'xgboost': {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
    },
    'optimization': {
        'safety_stock': {
            'default_service_level': 0.95,
            'lead_time_days': 14,
            'review_period_days': 7,
        },
        'eoq': {
            'holding_cost_rate': 0.25,
            'ordering_cost': 50.0,
            'annual_demand_factor': 365,
        },
        'abc_classification': {'a_threshold': 0.8, 'b_threshold': 0.95, 'c_threshold': 1.0},
    },
    'evaluation': {
        'metrics': ['mape', 'wmape', 'rmse', 'mae'],
        'cv_folds': 3,
    },
    'business': {
        'cost_of_capital': 0.12,
        'storage_cost_rate': 0.25,
        'stockout_cost_multiplier': 3.0,
        'target_service_level': 0.95,
        'critical_service_level': 0.99,
        'target_mape': 0.15,
        'target_wmape': 0.12,
        'target_inventory_turns': 8.0,
    },
}


def _make_sales_df(n_days=60, n_stores=2, families=None):
    if families is None:
        families = ['GROCERY', 'BEVERAGES']
    dates = pd.date_range('2015-01-01', periods=n_days, freq='D')
    rows = []
    rng = np.random.default_rng(42)
    for store in range(1, n_stores + 1):
        for fam in families:
            base = rng.gamma(4, 25)
            prev = base
            for d in dates:
                seasonal = 1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 365)
                expected = base * seasonal
                sales = 0.4 * prev + 0.6 * expected + rng.normal(0, base * 0.05)
                sales = max(0, sales)
                prev = sales
                rows.append({'date': d, 'store_nbr': store, 'family': fam,
                             'sales': sales, 'onpromotion': 0})
    return pd.DataFrame(rows)


# ─── DataCleaner ─────────────────────────────────────────────────────────────

class TestDataCleaner:
    def setup_method(self):
        self.cleaner = DataCleaner(CONFIG)

    def test_clean_data_preserves_shape(self):
        df = _make_sales_df()
        cleaned = self.cleaner.clean_data(df)
        assert len(cleaned) > 0
        assert 'sales' in cleaned.columns

    def test_handle_missing_values_drops_nan_sales(self):
        df = _make_sales_df(n_days=10)
        df.loc[0, 'sales'] = np.nan
        cleaned = self.cleaner.handle_missing_values(df)
        assert cleaned['sales'].isnull().sum() == 0

    def test_remove_duplicates(self):
        df = _make_sales_df(n_days=5)
        df = pd.concat([df, df], ignore_index=True)
        cleaned = self.cleaner.remove_duplicates(df)
        assert len(cleaned) == len(df) // 2

    def test_outlier_detection_caps_values(self):
        df = _make_sales_df(n_days=30)
        df.loc[0, 'sales'] = 999999.0
        cleaned = self.cleaner.detect_and_handle_outliers(df)
        assert cleaned['sales'].max() < 999999.0


# ─── FeatureEngineer ─────────────────────────────────────────────────────────

class TestFeatureEngineer:
    def setup_method(self):
        self.fe = FeatureEngineer(CONFIG)

    def test_temporal_features_created(self):
        df = _make_sales_df(n_days=10)
        result = self.fe.create_temporal_features(df)
        for col in ['year', 'month', 'day_of_week', 'is_weekend', 'month_sin', 'month_cos']:
            assert col in result.columns, f"Missing temporal feature: {col}"

    def test_lag_features_created(self):
        df = _make_sales_df(n_days=30)
        result = self.fe.create_lag_features(df)
        assert 'sales_lag_1' in result.columns
        assert 'sales_lag_7' in result.columns

    def test_lag_features_correct_values(self):
        df = _make_sales_df(n_days=20, n_stores=1, families=['GROCERY'])
        df = df.sort_values('date').reset_index(drop=True)
        result = self.fe.create_lag_features(df)
        result = result.sort_values('date').reset_index(drop=True)
        # lag_1 for row i should equal sales of row i-1
        for i in range(1, len(result)):
            assert abs(result.loc[i, 'sales_lag_1'] - result.loc[i-1, 'sales']) < 1e-6

    def test_holiday_features_no_nan_after_merge(self):
        df = _make_sales_df(n_days=30)
        df = self.fe.create_temporal_features(df)
        oil = pd.DataFrame({'date': pd.date_range('2015-01-01', periods=30, freq='D'),
                            'dcoilwtico': 50.0})
        holidays = pd.DataFrame({
            'date': pd.to_datetime(['2015-01-01', '2015-01-15']),
            'type': ['Holiday', 'Holiday'],
            'locale': ['National', 'National'],
            'transferred': [False, False],
        })
        result = self.fe.merge_external_data(df, oil_data=oil, holidays_data=holidays)
        assert result['is_holiday'].isnull().sum() == 0
        assert result['days_to_holiday'].isnull().sum() == 0
        assert result['is_holiday'].max() == 1

    def test_rolling_features_no_inf(self):
        df = _make_sales_df(n_days=20)
        result = self.fe.create_rolling_features(df)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()


# ─── XGBoostModel ────────────────────────────────────────────────────────────

class TestXGBoostModel:
    def _make_xy(self, n=200, n_features=10, seed=42):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.standard_normal((n, n_features)),
                         columns=[f'f{i}' for i in range(n_features)])
        # y has signal: f0 + f1 + noise
        y = pd.Series(X['f0'] * 10 + X['f1'] * 5 + rng.normal(0, 1, n) + 50)
        return X, y

    def test_fit_predict_shape(self):
        X, y = self._make_xy()
        model = XGBoostModel(CONFIG)
        model.fit(X, y, tune_hyperparameters=False)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_non_negative(self):
        X, y = self._make_xy()
        y = y.clip(lower=0)
        model = XGBoostModel(CONFIG)
        model.fit(X, y, tune_hyperparameters=False)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_save_load_roundtrip(self, tmp_path):
        X, y = self._make_xy()
        model = XGBoostModel(CONFIG)
        model.fit(X, y, tune_hyperparameters=False)
        path = str(tmp_path / 'test_model.joblib')
        model.save_model(path)

        model2 = XGBoostModel(CONFIG)
        model2.load_model(path)
        preds1 = model.predict(X)
        preds2 = model2.predict(X)
        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_training_metrics_computed(self):
        X, y = self._make_xy()
        model = XGBoostModel(CONFIG)
        model.fit(X, y, tune_hyperparameters=False)
        assert 'wmape' in model.training_metrics
        assert model.training_metrics['wmape'] < 100

    def test_feature_importance_shape(self):
        X, y = self._make_xy()
        model = XGBoostModel(CONFIG)
        model.fit(X, y, tune_hyperparameters=False)
        fi = model.get_feature_importance()
        assert len(fi) == X.shape[1]


# ─── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def setup_method(self):
        self.evaluator = ModelEvaluator(CONFIG)

    def test_perfect_predictions_zero_error(self):
        y = np.array([10.0, 20.0, 30.0])
        metrics = self.evaluator.calculate_all_metrics(y, y)
        assert metrics['mae'] == pytest.approx(0.0)
        assert metrics['rmse'] == pytest.approx(0.0)
        assert metrics['wmape'] == pytest.approx(0.0)

    def test_metrics_keys_present(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0])
        metrics = self.evaluator.calculate_all_metrics(y_true, y_pred)
        for key in ['mae', 'rmse', 'mape', 'wmape', 'r2']:
            assert key in metrics

    def test_handles_nan_gracefully(self):
        y_true = np.array([10.0, np.nan, 30.0])
        y_pred = np.array([11.0, 20.0, 31.0])
        metrics = self.evaluator.calculate_all_metrics(y_true, y_pred)
        assert len(metrics) > 0


# ─── SafetyStock ─────────────────────────────────────────────────────────────

class TestSafetyStock:
    def setup_method(self):
        self.calc = SafetyStockCalculator(CONFIG)

    def test_basic_safety_stock_positive(self):
        ss = self.calc.calculate_safety_stock(demand_mean=100, demand_std=20, lead_time=14)
        assert ss > 0

    def test_higher_service_level_means_more_safety_stock(self):
        ss_90 = self.calc.calculate_safety_stock(100, 20, 14, service_level=0.90)
        ss_99 = self.calc.calculate_safety_stock(100, 20, 14, service_level=0.99)
        assert ss_99 > ss_90

    def test_invalid_params_return_zero(self):
        ss = self.calc.calculate_safety_stock(demand_mean=0, demand_std=20, lead_time=14)
        assert ss == 0


# ─── ReorderPoint ────────────────────────────────────────────────────────────

class TestReorderPoint:
    def setup_method(self):
        self.rop = ReorderPointOptimizer(CONFIG)

    def test_reorder_point_exceeds_safety_stock(self):
        safety_stock = 50
        rp = self.rop.calculate_reorder_point(demand_forecast=10, lead_time=14,
                                               safety_stock=safety_stock)
        assert rp > safety_stock

    def test_higher_demand_higher_reorder_point(self):
        rp_low = self.rop.calculate_reorder_point(10, 14, 20)
        rp_high = self.rop.calculate_reorder_point(50, 14, 20)
        assert rp_high > rp_low


# ─── InventoryOptimizer ──────────────────────────────────────────────────────

class TestInventoryOptimizer:
    def setup_method(self):
        self.opt = InventoryOptimizer(CONFIG)

    def test_eoq_positive(self):
        result = self.opt.calculate_eoq(demand=1000, ordering_cost=50,
                                         holding_cost_rate=0.25, unit_cost=10)
        assert result['eoq'] > 0

    def test_abc_analysis_classes(self):
        df = pd.DataFrame({
            'item_id': range(10),
            'annual_value': [1000, 900, 800, 100, 50, 40, 30, 20, 10, 5],
        })
        result = self.opt.perform_abc_analysis(df)
        assert set(result['abc_class'].unique()).issubset({'A', 'B', 'C'})

    def test_abc_a_items_highest_value(self):
        df = pd.DataFrame({
            'item_id': range(5),
            'annual_value': [1000, 500, 100, 50, 10],
        })
        result = self.opt.perform_abc_analysis(df)
        a_items = result[result['abc_class'] == 'A']
        c_items = result[result['abc_class'] == 'C']
        if len(a_items) > 0 and len(c_items) > 0:
            assert a_items['annual_value'].mean() > c_items['annual_value'].mean()
