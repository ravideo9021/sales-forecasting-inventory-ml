"""
Data cleaner module for the Sales Forecasting project.

This module handles data cleaning, validation, and preprocessing operations
for the retail sales dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from loguru import logger


class DataCleaner:
    """
    Data cleaner class for preprocessing the retail sales dataset.
    
    This class handles data cleaning, validation, outlier detection,
    missing value imputation, and data quality checks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataCleaner.
        
        Args:
            config: Configuration dictionary containing data processing settings
        """
        self.config = config
        self.data_config = config['data']
        self.date_column = self.data_config['date_column']
        self.target_column = self.data_config['target_column']
        self.store_column = self.data_config['store_column']
        self.family_column = self.data_config['family_column']
        
        # Cleaning statistics
        self.cleaning_stats = {}
        
        logger.info("DataCleaner initialized successfully")
    
    def validate_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and convert data types for the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with corrected data types
        """
        logger.info("Validating data types...")
        
        # Create a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Convert date column
        if self.date_column in cleaned_data.columns:
            cleaned_data[self.date_column] = pd.to_datetime(
                cleaned_data[self.date_column], errors='coerce'
            )
        
        # Convert numeric columns
        numeric_columns = ['sales', 'onpromotion']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        
        # Keep categorical columns as object type to avoid issues with feature engineering
        categorical_columns = [self.store_column, self.family_column]
        for col in categorical_columns:
            if col in cleaned_data.columns:
                # Convert to object instead of categorical to avoid category issues
                cleaned_data[col] = cleaned_data[col].astype('object')
        
        logger.info("Data type validation completed")
        return cleaned_data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        cleaned_data = data.copy()
        missing_stats = {}
        
        # Check for missing values
        missing_counts = cleaned_data.isnull().sum()
        missing_stats['initial_missing'] = missing_counts.to_dict()
        
        # Handle missing values in sales (target variable)
        if self.target_column in cleaned_data.columns:
            # For sales, we might want to drop rows with missing values
            # or use forward fill for time series
            initial_missing_sales = cleaned_data[self.target_column].isnull().sum()
            cleaned_data = cleaned_data.dropna(subset=[self.target_column])
            final_missing_sales = cleaned_data[self.target_column].isnull().sum()
            
            missing_stats['sales_missing_handled'] = {
                'initial': initial_missing_sales,
                'final': final_missing_sales,
                'dropped': initial_missing_sales - final_missing_sales
            }
        
        # Handle missing values in onpromotion (fill with 0)
        if 'onpromotion' in cleaned_data.columns:
            initial_missing_promo = cleaned_data['onpromotion'].isnull().sum()
            cleaned_data['onpromotion'] = cleaned_data['onpromotion'].fillna(0)
            final_missing_promo = cleaned_data['onpromotion'].isnull().sum()
            
            missing_stats['promotion_missing_handled'] = {
                'initial': initial_missing_promo,
                'final': final_missing_promo,
                'filled_with_zero': initial_missing_promo - final_missing_promo
            }
        
        # Handle missing values in date column
        if self.date_column in cleaned_data.columns:
            initial_missing_date = cleaned_data[self.date_column].isnull().sum()
            cleaned_data = cleaned_data.dropna(subset=[self.date_column])
            final_missing_date = cleaned_data[self.date_column].isnull().sum()
            
            missing_stats['date_missing_handled'] = {
                'initial': initial_missing_date,
                'final': final_missing_date,
                'dropped': initial_missing_date - final_missing_date
            }
        
        # Final missing value check
        missing_stats['final_missing'] = cleaned_data.isnull().sum().to_dict()
        
        self.cleaning_stats['missing_values'] = missing_stats
        logger.info("Missing value handling completed")
        
        return cleaned_data
    
    def detect_and_handle_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and handle outliers in the dataset.
        
        Args:
            data: Input DataFrame
            method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        cleaned_data = data.copy()
        outlier_stats = {}
        
        if self.target_column not in cleaned_data.columns:
            logger.warning("Target column not found, skipping outlier detection")
            return cleaned_data
        
        # Group by store and family for outlier detection
        groups = cleaned_data.groupby([self.store_column, self.family_column])
        outlier_mask = pd.Series(False, index=cleaned_data.index)
        
        for (store, family), group in groups:
            if method == 'iqr':
                outlier_mask_group = self._detect_outliers_iqr(group[self.target_column])
            elif method == 'zscore':
                outlier_mask_group = self._detect_outliers_zscore(group[self.target_column])
            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                continue
            
            outlier_mask.loc[group.index] = outlier_mask_group
        
        # Calculate outlier statistics
        total_outliers = outlier_mask.sum()
        outlier_percentage = (total_outliers / len(cleaned_data)) * 100
        
        outlier_stats['detection_method'] = method
        outlier_stats['total_outliers'] = int(total_outliers)
        outlier_stats['outlier_percentage'] = float(outlier_percentage)
        
        # Handle outliers (cap at 99th percentile)
        if total_outliers > 0:
            # Calculate 99th percentile for each store-family combination
            percentile_99 = groups[self.target_column].quantile(0.99)
            
            # Cap outliers at 99th percentile
            for (store, family), group in groups:
                group_mask = (cleaned_data[self.store_column] == store) & \
                           (cleaned_data[self.family_column] == family)
                cap_value = percentile_99.loc[(store, family)]
                
                outlier_in_group = outlier_mask & group_mask
                cleaned_data.loc[outlier_in_group, self.target_column] = cap_value
            
            outlier_stats['handling_method'] = 'capped_at_99th_percentile'
        
        self.cleaning_stats['outliers'] = outlier_stats
        logger.info(f"Outlier detection and handling completed: {total_outliers} outliers found ({outlier_percentage:.2f}%)")
        
        return cleaned_data
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        return z_scores > 3
    
    def validate_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data consistency and quality.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating data consistency...")
        
        validation_results = {}
        
        # Check for negative sales
        if self.target_column in data.columns:
            negative_sales = (data[self.target_column] < 0).sum()
            validation_results['negative_sales'] = {
                'count': int(negative_sales),
                'percentage': float((negative_sales / len(data)) * 100)
            }
        
        # Check for duplicate records
        duplicates = data.duplicated().sum()
        validation_results['duplicates'] = {
            'count': int(duplicates),
            'percentage': float((duplicates / len(data)) * 100)
        }
        
        # Check date range consistency
        if self.date_column in data.columns:
            date_range = {
                'min': data[self.date_column].min().strftime('%Y-%m-%d'),
                'max': data[self.date_column].max().strftime('%Y-%m-%d'),
                'total_days': (data[self.date_column].max() - data[self.date_column].min()).days
            }
            validation_results['date_range'] = date_range
        
        # Check for data gaps
        if self.date_column in data.columns:
            expected_dates = pd.date_range(
                start=data[self.date_column].min(),
                end=data[self.date_column].max(),
                freq='D'
            )
            actual_dates = data[self.date_column].unique()
            missing_dates = set(expected_dates) - set(actual_dates)
            
            validation_results['data_gaps'] = {
                'missing_dates_count': len(missing_dates),
                'missing_dates_percentage': float((len(missing_dates) / len(expected_dates)) * 100)
            }
        
        # Check store and family consistency
        if self.store_column in data.columns:
            validation_results['stores'] = {
                'unique_count': int(data[self.store_column].nunique()),
                'values': list(data[self.store_column].unique())
            }
        
        if self.family_column in data.columns:
            validation_results['families'] = {
                'unique_count': int(data[self.family_column].nunique()),
                'values': list(data[self.family_column].unique())
            }
        
        self.cleaning_stats['validation'] = validation_results
        logger.info("Data consistency validation completed")
        
        return validation_results
    
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records from the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicate records...")
        
        initial_count = len(data)
        cleaned_data = data.drop_duplicates()
        final_count = len(cleaned_data)
        removed_count = initial_count - final_count
        
        self.cleaning_stats['duplicates_removed'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_count': removed_count
        }
        
        logger.info(f"Duplicate removal completed: {removed_count} duplicates removed")
        return cleaned_data
    
    def sort_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sort data by date and other relevant columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Sorted DataFrame
        """
        logger.info("Sorting data...")
        
        sort_columns = [self.date_column, self.store_column, self.family_column]
        available_columns = [col for col in sort_columns if col in data.columns]
        
        cleaned_data = data.sort_values(available_columns).reset_index(drop=True)
        
        logger.info("Data sorting completed")
        return cleaned_data
    
    def clean_data(self, data: pd.DataFrame, outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        Perform complete data cleaning pipeline.
        
        Args:
            data: Input DataFrame
            outlier_method: Method for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting complete data cleaning pipeline...")
        
        # Reset cleaning statistics
        self.cleaning_stats = {}
        
        # Step 1: Validate data types
        cleaned_data = self.validate_data_types(data)
        
        # Step 2: Handle missing values
        cleaned_data = self.handle_missing_values(cleaned_data)
        
        # Step 3: Remove duplicates
        cleaned_data = self.remove_duplicates(cleaned_data)
        
        # Step 4: Detect and handle outliers
        cleaned_data = self.detect_and_handle_outliers(cleaned_data, outlier_method)
        
        # Step 5: Validate data consistency
        self.validate_data_consistency(cleaned_data)
        
        # Step 6: Sort data
        cleaned_data = self.sort_data(cleaned_data)
        
        # Final statistics
        self.cleaning_stats['final_shape'] = cleaned_data.shape
        self.cleaning_stats['cleaning_completed'] = True
        
        logger.info("Complete data cleaning pipeline finished")
        logger.info(f"Final dataset shape: {cleaned_data.shape}")
        
        return cleaned_data
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the cleaning operations performed.
        
        Returns:
            Dictionary containing cleaning statistics
        """
        return self.cleaning_stats
    
    def generate_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing data quality metrics
        """
        logger.info("Generating data quality report...")
        
        quality_report = {
            'basic_info': {
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum(),
                'dtypes': data.dtypes.to_dict()
            },
            'missing_values': {
                'total_missing': data.isnull().sum().sum(),
                'missing_percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
                'missing_by_column': data.isnull().sum().to_dict()
            },
            'duplicates': {
                'total_duplicates': data.duplicated().sum(),
                'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100
            }
        }
        
        # Add target variable statistics
        if self.target_column in data.columns:
            target_stats = data[self.target_column].describe()
            quality_report['target_variable'] = {
                'mean': float(target_stats['mean']),
                'std': float(target_stats['std']),
                'min': float(target_stats['min']),
                'max': float(target_stats['max']),
                'median': float(target_stats['50%']),
                'skewness': float(data[self.target_column].skew()),
                'kurtosis': float(data[self.target_column].kurtosis())
            }
        
        # Add categorical variable statistics
        categorical_columns = [self.store_column, self.family_column]
        for col in categorical_columns:
            if col in data.columns:
                quality_report[f'{col}_stats'] = {
                    'unique_count': int(data[col].nunique()),
                    'most_common': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'most_common_count': int(data[col].value_counts().iloc[0]) if not data[col].value_counts().empty else 0
                }
        
        logger.info("Data quality report generated")
        return quality_report 