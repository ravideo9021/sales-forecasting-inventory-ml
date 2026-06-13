"""
Data loader module for the Sales Forecasting project.

This module handles loading and initial processing of the Kaggle dataset
for the Store Sales - Time Series Forecasting competition.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import yaml
from loguru import logger


class DataLoader:
    """
    Data loader class for managing the Kaggle dataset.
    
    This class handles loading, validation, and initial processing of all
    data files from the Store Sales - Time Series Forecasting competition.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the DataLoader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.data_paths = self.config['data']
        self._validate_paths()
        
        # Initialize data containers
        self.train_data = None
        self.test_data = None
        self.stores_data = None
        self.oil_data = None
        self.holidays_data = None
        self.transactions_data = None
        
        logger.info("DataLoader initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _validate_paths(self) -> None:
        """Validate that all required data paths exist."""
        required_paths = [
            self.data_paths['raw_data_path'],
            self.data_paths['processed_data_path'],
            self.data_paths['external_data_path'],
            self.data_paths['synthetic_data_path']
        ]
        
        for path in required_paths:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load the training data.
        
        Returns:
            DataFrame containing the training data
        """
        file_path = os.path.join(
            self.data_paths['raw_data_path'], 
            self.data_paths['train_file']
        )
        
        try:
            self.train_data = pd.read_csv(file_path)
            self.train_data[self.data_paths['date_column']] = pd.to_datetime(
                self.train_data[self.data_paths['date_column']]
            )
            
            logger.info(f"Training data loaded: {self.train_data.shape}")
            logger.info(f"Date range: {self.train_data[self.data_paths['date_column']].min()} to "
                       f"{self.train_data[self.data_paths['date_column']].max()}")
            
            return self.train_data
            
        except FileNotFoundError:
            logger.error(f"Training data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load the test data.
        
        Returns:
            DataFrame containing the test data
        """
        file_path = os.path.join(
            self.data_paths['raw_data_path'], 
            self.data_paths['test_file']
        )
        
        try:
            self.test_data = pd.read_csv(file_path)
            self.test_data[self.data_paths['date_column']] = pd.to_datetime(
                self.test_data[self.data_paths['date_column']]
            )
            
            logger.info(f"Test data loaded: {self.test_data.shape}")
            logger.info(f"Date range: {self.test_data[self.data_paths['date_column']].min()} to "
                       f"{self.test_data[self.data_paths['date_column']].max()}")
            
            return self.test_data
            
        except FileNotFoundError:
            logger.error(f"Test data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def load_stores_data(self) -> pd.DataFrame:
        """
        Load the stores metadata.
        
        Returns:
            DataFrame containing the stores data
        """
        file_path = os.path.join(
            self.data_paths['raw_data_path'], 
            self.data_paths['stores_file']
        )
        
        try:
            self.stores_data = pd.read_csv(file_path)
            logger.info(f"Stores data loaded: {self.stores_data.shape}")
            return self.stores_data
            
        except FileNotFoundError:
            logger.error(f"Stores data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading stores data: {e}")
            raise
    
    def load_oil_data(self) -> pd.DataFrame:
        """
        Load the oil price data.
        
        Returns:
            DataFrame containing the oil price data
        """
        file_path = os.path.join(
            self.data_paths['raw_data_path'], 
            self.data_paths['oil_file']
        )
        
        try:
            self.oil_data = pd.read_csv(file_path)
            self.oil_data['date'] = pd.to_datetime(self.oil_data['date'])
            self.oil_data = self.oil_data.sort_values('date')
            
            logger.info(f"Oil data loaded: {self.oil_data.shape}")
            logger.info(f"Date range: {self.oil_data['date'].min()} to {self.oil_data['date'].max()}")
            
            return self.oil_data
            
        except FileNotFoundError:
            logger.error(f"Oil data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading oil data: {e}")
            raise
    
    def load_holidays_data(self) -> pd.DataFrame:
        """
        Load the holidays and events data.
        
        Returns:
            DataFrame containing the holidays data
        """
        file_path = os.path.join(
            self.data_paths['raw_data_path'], 
            self.data_paths['holidays_file']
        )
        
        try:
            self.holidays_data = pd.read_csv(file_path)
            self.holidays_data['date'] = pd.to_datetime(self.holidays_data['date'])
            self.holidays_data = self.holidays_data.sort_values('date')
            
            logger.info(f"Holidays data loaded: {self.holidays_data.shape}")
            logger.info(f"Date range: {self.holidays_data['date'].min()} to {self.holidays_data['date'].max()}")
            
            return self.holidays_data
            
        except FileNotFoundError:
            logger.error(f"Holidays data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading holidays data: {e}")
            raise
    
    def load_transactions_data(self) -> pd.DataFrame:
        """
        Load the transactions data.
        
        Returns:
            DataFrame containing the transactions data
        """
        file_path = os.path.join(
            self.data_paths['raw_data_path'], 
            self.data_paths['transactions_file']
        )
        
        try:
            self.transactions_data = pd.read_csv(file_path)
            self.transactions_data['date'] = pd.to_datetime(self.transactions_data['date'])
            self.transactions_data = self.transactions_data.sort_values('date')
            
            logger.info(f"Transactions data loaded: {self.transactions_data.shape}")
            logger.info(f"Date range: {self.transactions_data['date'].min()} to {self.transactions_data['date'].max()}")
            
            return self.transactions_data
            
        except FileNotFoundError:
            logger.error(f"Transactions data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading transactions data: {e}")
            raise
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data files.
        
        Returns:
            Dictionary containing all loaded datasets
        """
        logger.info("Loading all data files...")
        
        data_dict = {
            'train': self.load_train_data(),
            'test': self.load_test_data(),
            'stores': self.load_stores_data(),
            'oil': self.load_oil_data(),
            'holidays': self.load_holidays_data(),
            'transactions': self.load_transactions_data()
        }
        
        logger.info("All data files loaded successfully")
        return data_dict
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of all loaded datasets.
        
        Returns:
            Dictionary containing summary statistics for each dataset
        """
        summary = {}
        
        datasets = {
            'train': self.train_data,
            'test': self.test_data,
            'stores': self.stores_data,
            'oil': self.oil_data,
            'holidays': self.holidays_data,
            'transactions': self.transactions_data
        }
        
        for name, data in datasets.items():
            if data is not None:
                summary[name] = {
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict(),
                    'missing_values': data.isnull().sum().to_dict(),
                    'memory_usage': data.memory_usage(deep=True).sum()
                }
                
                # Add date range for time series data
                if 'date' in data.columns:
                    summary[name]['date_range'] = {
                        'min': data['date'].min().strftime('%Y-%m-%d'),
                        'max': data['date'].max().strftime('%Y-%m-%d')
                    }
        
        return summary
    
    def save_processed_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to the processed data directory.
        
        Args:
            data: DataFrame to save
            filename: Name of the file to save
        """
        file_path = os.path.join(self.data_paths['processed_data_path'], filename)
        
        try:
            data.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from the processed data directory.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame containing the processed data
        """
        file_path = os.path.join(self.data_paths['processed_data_path'], filename)
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Processed data loaded from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Processed data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise