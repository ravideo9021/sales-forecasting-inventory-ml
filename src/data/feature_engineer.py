"""
Feature engineering module for the Sales Forecasting project.

This module handles advanced feature engineering operations including
temporal features, lag features, rolling statistics, and domain-specific features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for creating advanced features for sales forecasting.
    
    This class handles creation of temporal features, lag features, rolling statistics,
    promotional features, holiday features, and economic indicators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FeatureEngineer.
        
        Args:
            config: Configuration dictionary containing feature engineering settings
        """
        self.config = config
        self.data_config = config['data']
        self.date_column = self.data_config['date_column']
        self.target_column = self.data_config['target_column']
        self.store_column = self.data_config['store_column']
        self.family_column = self.data_config['family_column']
        
        # Feature engineering parameters
        self.lag_features = self.data_config['lag_features']
        self.rolling_windows = self.data_config['rolling_windows']
        self.seasonal_periods = self.data_config['seasonal_periods']
        
        # Feature tracking
        self.created_features = []
        self.feature_stats = {}
        
        logger.info("FeatureEngineer initialized successfully")
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from the date column.
        
        Args:
            data: Input DataFrame with date column
            
        Returns:
            DataFrame with temporal features added
        """
        logger.info("Creating temporal features...")
        
        feature_data = data.copy()
        
        if self.date_column not in feature_data.columns:
            logger.warning("Date column not found, skipping temporal features")
            return feature_data
        
        # Ensure date column is datetime
        feature_data[self.date_column] = pd.to_datetime(feature_data[self.date_column])
        
        # Basic temporal features
        feature_data['year'] = feature_data[self.date_column].dt.year
        feature_data['month'] = feature_data[self.date_column].dt.month
        feature_data['day'] = feature_data[self.date_column].dt.day
        feature_data['day_of_week'] = feature_data[self.date_column].dt.dayofweek
        feature_data['day_of_year'] = feature_data[self.date_column].dt.dayofyear
        feature_data['week_of_year'] = feature_data[self.date_column].dt.isocalendar().week
        feature_data['quarter'] = feature_data[self.date_column].dt.quarter
        
        # Boolean temporal features
        feature_data['is_weekend'] = feature_data['day_of_week'].isin([5, 6])
        feature_data['is_month_start'] = feature_data[self.date_column].dt.is_month_start
        feature_data['is_month_end'] = feature_data[self.date_column].dt.is_month_end
        feature_data['is_quarter_start'] = feature_data[self.date_column].dt.is_quarter_start
        feature_data['is_quarter_end'] = feature_data[self.date_column].dt.is_quarter_end
        feature_data['is_year_start'] = feature_data[self.date_column].dt.is_year_start
        feature_data['is_year_end'] = feature_data[self.date_column].dt.is_year_end
        
        # Cyclical encoding for temporal features
        feature_data['month_sin'] = np.sin(2 * np.pi * feature_data['month'] / 12)
        feature_data['month_cos'] = np.cos(2 * np.pi * feature_data['month'] / 12)
        feature_data['day_of_week_sin'] = np.sin(2 * np.pi * feature_data['day_of_week'] / 7)
        feature_data['day_of_week_cos'] = np.cos(2 * np.pi * feature_data['day_of_week'] / 7)
        feature_data['day_of_year_sin'] = np.sin(2 * np.pi * feature_data['day_of_year'] / 365)
        feature_data['day_of_year_cos'] = np.cos(2 * np.pi * feature_data['day_of_year'] / 365)
        
        temporal_features = [
            'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
            'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
            'is_year_start', 'is_year_end', 'month_sin', 'month_cos', 'day_of_week_sin',
            'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos'
        ]
        
        self.created_features.extend(temporal_features)
        logger.info(f"Created {len(temporal_features)} temporal features")
        
        return feature_data
    
    def create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for time series forecasting.
        
        Args:
            data: Input DataFrame with target column
            
        Returns:
            DataFrame with lag features added
        """
        logger.info("Creating lag features...")
        
        if self.target_column not in data.columns:
            logger.warning("Target column not found, skipping lag features")
            return data
        
        feature_data = data.copy()
        lag_features_created = []
        
        # Sort data by date, store, and family for proper lag calculation
        sort_columns = [self.date_column, self.store_column, self.family_column]
        available_sort_cols = [col for col in sort_columns if col in feature_data.columns]
        feature_data = feature_data.sort_values(available_sort_cols)
        
        # Create lag features for each store-family combination
        for lag in self.lag_features:
            lag_col_name = f'{self.target_column}_lag_{lag}'
            
            # Group by store and family to create proper lags
            if self.store_column in feature_data.columns and self.family_column in feature_data.columns:
                feature_data[lag_col_name] = feature_data.groupby([self.store_column, self.family_column])[self.target_column].shift(lag)
            else:
                feature_data[lag_col_name] = feature_data[self.target_column].shift(lag)
            
            lag_features_created.append(lag_col_name)
        
        # Create lag features for promotional data if available
        if 'onpromotion' in feature_data.columns:
            for lag in [1, 7, 14]:
                promo_lag_col = f'onpromotion_lag_{lag}'
                if self.store_column in feature_data.columns and self.family_column in feature_data.columns:
                    feature_data[promo_lag_col] = feature_data.groupby([self.store_column, self.family_column])['onpromotion'].shift(lag)
                else:
                    feature_data[promo_lag_col] = feature_data['onpromotion'].shift(lag)
                lag_features_created.append(promo_lag_col)
        
        self.created_features.extend(lag_features_created)
        logger.info(f"Created {len(lag_features_created)} lag features")
        
        return feature_data
    
    def create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            data: Input DataFrame with target column
            
        Returns:
            DataFrame with rolling features added
        """
        logger.info("Creating rolling features...")
        
        if self.target_column not in data.columns:
            logger.warning("Target column not found, skipping rolling features")
            return data
        
        feature_data = data.copy()
        rolling_features_created = []
        
        # Sort data properly for rolling calculations
        sort_columns = [self.date_column, self.store_column, self.family_column]
        available_sort_cols = [col for col in sort_columns if col in feature_data.columns]
        feature_data = feature_data.sort_values(available_sort_cols)
        
        # Create rolling features for each window
        for window in self.rolling_windows:
            # Rolling statistics for target variable
            if self.store_column in feature_data.columns and self.family_column in feature_data.columns:
                # Mean
                rolling_mean_col = f'{self.target_column}_rolling_mean_{window}'
                feature_data[rolling_mean_col] = feature_data.groupby([self.store_column, self.family_column])[self.target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                rolling_features_created.append(rolling_mean_col)
                
                # Standard deviation
                rolling_std_col = f'{self.target_column}_rolling_std_{window}'
                feature_data[rolling_std_col] = feature_data.groupby([self.store_column, self.family_column])[self.target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                rolling_features_created.append(rolling_std_col)
                
                # Min
                rolling_min_col = f'{self.target_column}_rolling_min_{window}'
                feature_data[rolling_min_col] = feature_data.groupby([self.store_column, self.family_column])[self.target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                rolling_features_created.append(rolling_min_col)
                
                # Max
                rolling_max_col = f'{self.target_column}_rolling_max_{window}'
                feature_data[rolling_max_col] = feature_data.groupby([self.store_column, self.family_column])[self.target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                rolling_features_created.append(rolling_max_col)
            else:
                # Simple rolling without grouping
                rolling_mean_col = f'{self.target_column}_rolling_mean_{window}'
                feature_data[rolling_mean_col] = feature_data[self.target_column].rolling(window=window, min_periods=1).mean()
                rolling_features_created.append(rolling_mean_col)
        
        for col in rolling_features_created:
            feature_data[col] = feature_data[col].ffill()
        
        self.created_features.extend(rolling_features_created)
        logger.info(f"Created {len(rolling_features_created)} rolling features")
        
        return feature_data
    
    def create_promotional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create promotional features.
        
        Args:
            data: Input DataFrame with promotional data
            
        Returns:
            DataFrame with promotional features added
        """
        logger.info("Creating promotional features...")
        
        if 'onpromotion' not in data.columns:
            logger.warning("Promotion column not found, skipping promotional features")
            return data
        
        feature_data = data.copy()
        promo_features_created = []
        
        # Sort data properly
        sort_columns = [self.date_column, self.store_column, self.family_column]
        available_sort_cols = [col for col in sort_columns if col in feature_data.columns]
        feature_data = feature_data.sort_values(available_sort_cols)
        
        # Promotion intensity features
        if self.store_column in feature_data.columns and self.family_column in feature_data.columns:
            # Rolling promotion frequency
            for window in [7, 30]:
                promo_freq_col = f'promo_frequency_{window}d'
                feature_data[promo_freq_col] = feature_data.groupby([self.store_column, self.family_column])['onpromotion'].rolling(window=window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
                promo_features_created.append(promo_freq_col)
            
            # Days since last promotion
            promo_shift = feature_data.groupby([self.store_column, self.family_column])['onpromotion'].shift(1)
            feature_data['days_since_promo'] = feature_data.groupby([self.store_column, self.family_column]).apply(
                lambda x: self._calculate_days_since_event(x['onpromotion'])
            ).reset_index(level=[0,1], drop=True)
            promo_features_created.append('days_since_promo')
            
            # Promotion streak
            feature_data['promo_streak'] = feature_data.groupby([self.store_column, self.family_column])['onpromotion'].apply(
                lambda x: self._calculate_streak(x)
            ).reset_index(level=[0,1], drop=True)
            promo_features_created.append('promo_streak')
        
        # Basic promotion features
        feature_data['promo_next_day'] = feature_data.groupby([self.store_column, self.family_column])['onpromotion'].shift(-1)
        feature_data['promo_prev_day'] = feature_data.groupby([self.store_column, self.family_column])['onpromotion'].shift(1)
        promo_features_created.extend(['promo_next_day', 'promo_prev_day'])
        
        # Fill NaN values
        for col in promo_features_created:
            if col in feature_data.columns:
                feature_data[col] = feature_data[col].fillna(0)
        
        self.created_features.extend(promo_features_created)
        logger.info(f"Created {len(promo_features_created)} promotional features")
        
        return feature_data
    
    def _calculate_days_since_event(self, series: pd.Series) -> pd.Series:
        """Calculate days since last event (promotion)."""
        result = pd.Series(index=series.index, dtype=float)
        days_since = 0
        
        for i, value in enumerate(series):
            if value == 1:
                days_since = 0
            else:
                days_since += 1
            result.iloc[i] = days_since
        
        return result
    
    def _calculate_streak(self, series: pd.Series) -> pd.Series:
        """Calculate consecutive promotion streak."""
        result = pd.Series(index=series.index, dtype=int)
        streak = 0
        
        for i, value in enumerate(series):
            if value == 1:
                streak += 1
            else:
                streak = 0
            result.iloc[i] = streak
        
        return result
    
    def merge_external_data(self, main_data: pd.DataFrame, oil_data: pd.DataFrame = None, 
                          holidays_data: pd.DataFrame = None, transactions_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Merge external data sources with the main dataset.
        
        Args:
            main_data: Main sales DataFrame
            oil_data: Oil price DataFrame
            holidays_data: Holidays DataFrame
            transactions_data: Transactions DataFrame
            
        Returns:
            DataFrame with external data merged
        """
        logger.info("Merging external data sources...")
        
        merged_data = main_data.copy()
        
        # Merge oil data
        if oil_data is not None:
            logger.info("Merging oil price data...")
            oil_processed = self._process_oil_data(oil_data)
            merged_data = pd.merge(merged_data, oil_processed, on='date', how='left')
        
        # Merge holidays data
        if holidays_data is not None:
            logger.info("Merging holidays data...")
            holidays_processed = self._process_holidays_data(holidays_data)
            merged_data = pd.merge(merged_data, holidays_processed, on='date', how='left')

            # Fill non-holiday dates
            merged_data['is_holiday'] = merged_data['is_holiday'].fillna(0).astype(int)
            merged_data['is_transferred'] = merged_data['is_transferred'].fillna(0).astype(int)
            merged_data['holiday_locale'] = merged_data['holiday_locale'].fillna('None')
            merged_data['holiday_type'] = merged_data.get('holiday_type', pd.Series('None', index=merged_data.index))
            merged_data['holiday_type'] = merged_data['holiday_type'].fillna('None')

            # Compute days_to/from_holiday for every date (vectorized over small holiday set)
            all_hd = sorted(pd.to_datetime(holidays_data['date'].unique()))
            if all_hd:
                hd_vals = np.array([h.value for h in all_hd], dtype=np.int64)
                date_vals = merged_data[self.date_column].values.astype(np.int64)
                ns_per_day = 86_400 * 10**9

                days_to = np.full(len(date_vals), 365, dtype=np.int64)
                days_from = np.full(len(date_vals), 365, dtype=np.int64)
                for hv in hd_vals:
                    future = date_vals < hv
                    if future.any():
                        diff = (hv - date_vals[future]) // ns_per_day
                        days_to[future] = np.minimum(days_to[future], diff)
                    past = date_vals > hv
                    if past.any():
                        diff = (date_vals[past] - hv) // ns_per_day
                        days_from[past] = np.minimum(days_from[past], diff)
                merged_data['days_to_holiday'] = days_to
                merged_data['days_from_holiday'] = days_from
            else:
                merged_data['days_to_holiday'] = 365
                merged_data['days_from_holiday'] = 365
        
        # Merge transactions data
        if transactions_data is not None:
            logger.info("Merging transactions data...")
            transactions_processed = self._process_transactions_data(transactions_data)
            merged_data = pd.merge(merged_data, transactions_processed, 
                                 on=['date', self.store_column], how='left')
        
        logger.info("External data merging completed")
        return merged_data
    
    def _process_oil_data(self, oil_data: pd.DataFrame) -> pd.DataFrame:
        """Process oil price data and create oil-related features."""
        oil_processed = oil_data.copy()

        oil_processed['dcoilwtico'] = oil_processed['dcoilwtico'].ffill().bfill()

        oil_processed['oil_price_lag_1'] = oil_processed['dcoilwtico'].shift(1)
        oil_processed['oil_price_lag_7'] = oil_processed['dcoilwtico'].shift(7)
        oil_processed['oil_price_change'] = oil_processed['dcoilwtico'].diff()
        oil_processed['oil_price_change_pct'] = oil_processed['dcoilwtico'].pct_change()
        oil_processed['oil_price_ma_7'] = oil_processed['dcoilwtico'].rolling(7).mean()
        oil_processed['oil_price_ma_30'] = oil_processed['dcoilwtico'].rolling(30).mean()
        oil_processed['oil_volatility_7'] = oil_processed['dcoilwtico'].rolling(7).std()

        oil_processed = oil_processed.ffill().bfill()

        return oil_processed

    def _process_holidays_data(self, holidays_data: pd.DataFrame) -> pd.DataFrame:
        """Process holidays data — returns only holiday-date rows (one per date).

        NaN filling for non-holiday dates is handled in merge_external_data after the left join.
        """
        holidays_processed = holidays_data.copy()

        holidays_agg = holidays_processed.groupby('date').agg({
            'type': lambda x: ';'.join(x.unique()),
            'locale': lambda x: ';'.join(x.unique()),
            'transferred': lambda x: int(any(x))
        }).reset_index()

        holidays_agg.columns = ['date', 'holiday_type', 'holiday_locale', 'is_transferred']
        holidays_agg['is_holiday'] = 1

        return holidays_agg
    
    def _process_transactions_data(self, transactions_data: pd.DataFrame) -> pd.DataFrame:
        """Process transactions data and create transaction-related features."""
        transactions_processed = transactions_data.copy()
        
        # Create transaction features
        transactions_processed['transactions_lag_1'] = transactions_processed.groupby(self.store_column)['transactions'].shift(1)
        transactions_processed['transactions_lag_7'] = transactions_processed.groupby(self.store_column)['transactions'].shift(7)
        
        # Rolling transaction features
        transactions_processed['transactions_ma_7'] = transactions_processed.groupby(self.store_column)['transactions'].rolling(7).mean().reset_index(level=0, drop=True)
        transactions_processed['transactions_ma_30'] = transactions_processed.groupby(self.store_column)['transactions'].rolling(30).mean().reset_index(level=0, drop=True)
        
        # Transaction growth rate
        transactions_processed['transactions_growth'] = transactions_processed.groupby(self.store_column)['transactions'].pct_change()
        
        transactions_processed = transactions_processed.ffill().fillna(0)
        
        return transactions_processed
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features...")
        
        feature_data = data.copy()
        interaction_features = []
        
        # Promotion and temporal interactions
        if 'onpromotion' in feature_data.columns:
            feature_data['promo_weekend'] = feature_data['onpromotion'] * feature_data['is_weekend']
            feature_data['promo_month_end'] = feature_data['onpromotion'] * feature_data['is_month_end']
            feature_data['promo_holiday'] = feature_data['onpromotion'] * feature_data.get('is_holiday', 0)
            interaction_features.extend(['promo_weekend', 'promo_month_end', 'promo_holiday'])
        
        # Oil price and temporal interactions
        if 'dcoilwtico' in feature_data.columns:
            feature_data['oil_weekend'] = feature_data['dcoilwtico'] * feature_data['is_weekend']
            feature_data['oil_month'] = feature_data['dcoilwtico'] * feature_data['month']
            interaction_features.extend(['oil_weekend', 'oil_month'])
        
        # Store and family specific features
        if self.store_column in feature_data.columns and self.family_column in feature_data.columns:
            # Store-family mean sales (target encoding)
            store_family_mean = feature_data.groupby([self.store_column, self.family_column])[self.target_column].mean()
            feature_data['store_family_mean_sales'] = feature_data.set_index([self.store_column, self.family_column]).index.map(store_family_mean)
            interaction_features.append('store_family_mean_sales')
        
        self.created_features.extend(interaction_features)
        logger.info(f"Created {len(interaction_features)} interaction features")
        
        return feature_data
    
    def engineer_features(self, main_data: pd.DataFrame, oil_data: pd.DataFrame = None,
                         holidays_data: pd.DataFrame = None, transactions_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Perform complete feature engineering pipeline.
        
        Args:
            main_data: Main sales DataFrame
            oil_data: Oil price DataFrame
            holidays_data: Holidays DataFrame
            transactions_data: Transactions DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting complete feature engineering pipeline...")
        
        # Reset feature tracking
        self.created_features = []
        self.feature_stats = {}
        
        # Step 1: Create temporal features
        engineered_data = self.create_temporal_features(main_data)
        
        # Step 2: Merge external data
        engineered_data = self.merge_external_data(engineered_data, oil_data, holidays_data, transactions_data)
        
        # Step 3: Create lag features
        engineered_data = self.create_lag_features(engineered_data)
        
        # Step 4: Create rolling features
        engineered_data = self.create_rolling_features(engineered_data)
        
        # Step 5: Create promotional features
        engineered_data = self.create_promotional_features(engineered_data)
        
        # Step 6: Create interaction features
        engineered_data = self.create_interaction_features(engineered_data)
        
        # Final statistics
        self.feature_stats['total_features_created'] = len(self.created_features)
        self.feature_stats['final_shape'] = engineered_data.shape
        self.feature_stats['feature_list'] = self.created_features
        
        logger.info("Complete feature engineering pipeline finished")
        logger.info(f"Total features created: {len(self.created_features)}")
        logger.info(f"Final dataset shape: {engineered_data.shape}")
        
        return engineered_data
    
    def get_feature_importance(self, data: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Calculate basic feature importance using correlation.
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with feature importance scores
        """
        if target_col is None:
            target_col = self.target_column
        
        if target_col not in data.columns:
            logger.warning("Target column not found for feature importance calculation")
            return pd.DataFrame()
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if target_col not in numeric_data.columns:
            logger.warning("Target column is not numeric")
            return pd.DataFrame()
        
        # Calculate correlations
        correlations = numeric_data.corr()[target_col].abs().sort_values(ascending=False)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': correlations.index,
            'importance': correlations.values
        })
        
        # Remove target column itself
        feature_importance = feature_importance[feature_importance['feature'] != target_col]
        
        return feature_importance
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get a summary of feature engineering operations.
        
        Returns:
            Dictionary containing feature engineering statistics
        """
        return self.feature_stats 