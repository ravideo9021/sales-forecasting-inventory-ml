"""
Safety stock calculation module for the Sales Forecasting project.

This module implements various methods for calculating optimal safety stock levels
based on demand variability, lead time, and service level requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from loguru import logger


class SafetyStockCalculator:
    """
    Safety stock calculator for inventory optimization.
    
    This class implements various methods for calculating safety stock
    including demand variability, lead time variability, and service level optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SafetyStockCalculator.
        
        Args:
            config: Configuration dictionary containing safety stock parameters
        """
        self.config = config
        self.safety_stock_config = config['optimization']['safety_stock']
        
        # Default parameters
        self.default_service_level = self.safety_stock_config['default_service_level']
        self.default_lead_time = self.safety_stock_config['lead_time_days']
        self.review_period = self.safety_stock_config['review_period_days']
        
        logger.info("SafetyStockCalculator initialized successfully")
    
    def calculate_safety_stock(self, demand_mean: float, demand_std: float, 
                             lead_time: float, service_level: float = None) -> float:
        """
        Calculate safety stock using demand variability.
        
        Args:
            demand_mean: Mean daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            service_level: Desired service level (between 0 and 1)
            
        Returns:
            Safety stock quantity
        """
        if service_level is None:
            service_level = self.default_service_level
        
        if demand_mean <= 0 or demand_std < 0 or lead_time <= 0:
            logger.warning("Invalid parameters for safety stock calculation")
            return 0
        
        # Z-score for the desired service level
        z_score = stats.norm.ppf(service_level)
        
        # Safety stock formula: Z * σ_demand * √(lead_time)
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        
        return max(0, safety_stock)
    
    def calculate_safety_stock_with_lead_time_variability(self, 
                                                        demand_mean: float, 
                                                        demand_std: float,
                                                        lead_time_mean: float, 
                                                        lead_time_std: float,
                                                        service_level: float = None) -> float:
        """
        Calculate safety stock considering both demand and lead time variability.
        
        Args:
            demand_mean: Mean daily demand
            demand_std: Standard deviation of daily demand
            lead_time_mean: Mean lead time in days
            lead_time_std: Standard deviation of lead time
            service_level: Desired service level
            
        Returns:
            Safety stock quantity
        """
        if service_level is None:
            service_level = self.default_service_level
        
        if any(x <= 0 for x in [demand_mean, lead_time_mean]) or any(x < 0 for x in [demand_std, lead_time_std]):
            logger.warning("Invalid parameters for safety stock calculation")
            return 0
        
        # Z-score for the desired service level
        z_score = stats.norm.ppf(service_level)
        
        # Combined variance considering both demand and lead time variability
        # Var(demand during lead time) = LT_mean * Var(demand) + demand_mean² * Var(LT)
        demand_variance_component = lead_time_mean * (demand_std ** 2)
        lead_time_variance_component = (demand_mean ** 2) * (lead_time_std ** 2)
        
        total_variance = demand_variance_component + lead_time_variance_component
        total_std = np.sqrt(total_variance)
        
        safety_stock = z_score * total_std
        
        return max(0, safety_stock)
    
    def calculate_safety_stock_periodic_review(self, demand_mean: float, demand_std: float,
                                             lead_time: float, review_period: float = None,
                                             service_level: float = None) -> float:
        """
        Calculate safety stock for periodic review systems.
        
        Args:
            demand_mean: Mean daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            review_period: Review period in days
            service_level: Desired service level
            
        Returns:
            Safety stock quantity
        """
        if service_level is None:
            service_level = self.default_service_level
        
        if review_period is None:
            review_period = self.review_period
        
        if demand_mean <= 0 or demand_std < 0 or lead_time <= 0 or review_period <= 0:
            logger.warning("Invalid parameters for periodic review safety stock calculation")
            return 0
        
        # Z-score for the desired service level
        z_score = stats.norm.ppf(service_level)
        
        # For periodic review, we need to cover demand during lead time + review period
        protection_period = lead_time + review_period
        
        safety_stock = z_score * demand_std * np.sqrt(protection_period)
        
        return max(0, safety_stock)
    
    def calculate_safety_stock_seasonal(self, historical_demand: pd.Series, 
                                      lead_time: float, service_level: float = None,
                                      seasonal_periods: int = 365) -> Dict[str, float]:
        """
        Calculate seasonal safety stock adjustments.
        
        Args:
            historical_demand: Time series of historical demand
            lead_time: Lead time in days
            service_level: Desired service level
            seasonal_periods: Number of periods in a season (e.g., 365 for yearly)
            
        Returns:
            Dictionary with seasonal safety stock multipliers
        """
        if service_level is None:
            service_level = self.default_service_level
        
        if len(historical_demand) < seasonal_periods * 2:
            logger.warning("Insufficient data for seasonal analysis")
            return {'base_safety_stock': self.calculate_safety_stock(
                historical_demand.mean(), historical_demand.std(), lead_time, service_level
            )}
        
        # Calculate seasonal indices
        seasonal_demand = {}
        
        for period in range(seasonal_periods):
            # Get demand for this period across all years
            period_demand = []
            for i in range(period, len(historical_demand), seasonal_periods):
                if i < len(historical_demand):
                    period_demand.append(historical_demand.iloc[i])
            
            if period_demand:
                period_mean = np.mean(period_demand)
                period_std = np.std(period_demand)
                
                safety_stock = self.calculate_safety_stock(
                    period_mean, period_std, lead_time, service_level
                )
                
                seasonal_demand[f'period_{period}'] = safety_stock
        
        return seasonal_demand
    
    def calculate_abc_based_safety_stock(self, annual_value: float, demand_mean: float,
                                       demand_std: float, lead_time: float,
                                       abc_class: str = 'B') -> float:
        """
        Calculate safety stock based on ABC classification.
        
        Args:
            annual_value: Annual value of the item
            demand_mean: Mean daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            abc_class: ABC classification ('A', 'B', or 'C')
            
        Returns:
            Safety stock quantity
        """
        # Different service levels for different ABC classes
        service_levels = {
            'A': 0.99,  # High service level for A items
            'B': 0.95,  # Medium service level for B items
            'C': 0.90   # Lower service level for C items
        }
        
        service_level = service_levels.get(abc_class, 0.95)
        
        safety_stock = self.calculate_safety_stock(
            demand_mean, demand_std, lead_time, service_level
        )
        
        return safety_stock
    
    def optimize_safety_stock_cost(self, demand_mean: float, demand_std: float,
                                 lead_time: float, holding_cost_rate: float,
                                 stockout_cost: float, unit_cost: float) -> Dict[str, float]:
        """
        Optimize safety stock considering holding costs and stockout costs.
        
        Args:
            demand_mean: Mean daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            holding_cost_rate: Annual holding cost rate
            stockout_cost: Cost per stockout incident
            unit_cost: Unit cost of the item
            
        Returns:
            Dictionary with optimal safety stock and costs
        """
        # Range of service levels to evaluate
        service_levels = np.arange(0.50, 0.999, 0.01)
        best_cost = float('inf')
        optimal_service_level = 0.95
        optimal_safety_stock = 0
        
        results = []
        
        for sl in service_levels:
            # Calculate safety stock for this service level
            safety_stock = self.calculate_safety_stock(demand_mean, demand_std, lead_time, sl)
            
            # Calculate holding cost
            holding_cost = safety_stock * unit_cost * holding_cost_rate
            
            # Calculate expected stockout cost
            # Simplified: assumes one stockout event per lead time cycle
            stockout_probability = 1 - sl
            cycles_per_year = 365 / lead_time
            expected_stockout_cost = stockout_probability * stockout_cost * cycles_per_year
            
            total_cost = holding_cost + expected_stockout_cost
            
            results.append({
                'service_level': sl,
                'safety_stock': safety_stock,
                'holding_cost': holding_cost,
                'stockout_cost': expected_stockout_cost,
                'total_cost': total_cost
            })
            
            if total_cost < best_cost:
                best_cost = total_cost
                optimal_service_level = sl
                optimal_safety_stock = safety_stock
        
        return {
            'optimal_safety_stock': optimal_safety_stock,
            'optimal_service_level': optimal_service_level,
            'optimal_total_cost': best_cost,
            'cost_breakdown': results
        }
    
    def calculate_safety_stock_shortage_gaming(self, base_demand_mean: float,
                                             demand_std: float, lead_time: float,
                                             gaming_factor: float = 1.2) -> float:
        """
        Calculate safety stock accounting for demand gaming/shortage situations.
        
        Args:
            base_demand_mean: Base mean daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            gaming_factor: Factor to account for demand inflation during shortages
            
        Returns:
            Safety stock quantity
        """
        # Adjust demand for gaming effect
        adjusted_demand_mean = base_demand_mean * gaming_factor
        adjusted_demand_std = demand_std * gaming_factor
        
        # Use higher service level during shortage periods
        shortage_service_level = 0.99
        
        safety_stock = self.calculate_safety_stock(
            adjusted_demand_mean, adjusted_demand_std, lead_time, shortage_service_level
        )
        
        return safety_stock
    
    def calculate_dynamic_safety_stock(self, demand_forecast: List[float],
                                     forecast_errors: List[float],
                                     lead_time: float) -> List[float]:
        """
        Calculate dynamic safety stock that adjusts based on forecast accuracy.
        
        Args:
            demand_forecast: List of demand forecasts
            forecast_errors: List of historical forecast errors
            lead_time: Lead time in days
            
        Returns:
            List of dynamic safety stock values
        """
        dynamic_safety_stocks = []
        
        for i, forecast in enumerate(demand_forecast):
            # Use recent forecast errors to estimate current uncertainty
            window_size = min(30, len(forecast_errors))  # Use last 30 periods or available data
            recent_errors = forecast_errors[max(0, i-window_size):i] if i > 0 else [0]
            
            if not recent_errors:
                recent_errors = [forecast * 0.1]  # Default 10% error
            
            error_std = np.std(recent_errors) if len(recent_errors) > 1 else abs(recent_errors[0])
            
            # Adjust service level based on forecast accuracy
            if error_std / max(forecast, 1) < 0.1:  # Low error
                service_level = 0.90
            elif error_std / max(forecast, 1) < 0.2:  # Medium error
                service_level = 0.95
            else:  # High error
                service_level = 0.99
            
            safety_stock = self.calculate_safety_stock(
                forecast, error_std, lead_time, service_level
            )
            
            dynamic_safety_stocks.append(safety_stock)
        
        return dynamic_safety_stocks
    
    def generate_safety_stock_report(self, item_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive safety stock analysis report.
        
        Args:
            item_data: DataFrame with item information
            
        Returns:
            Dictionary containing safety stock analysis
        """
        report = {
            'summary': {},
            'item_analysis': [],
            'recommendations': []
        }
        
        total_items = len(item_data)
        total_safety_stock_value = 0
        
        for idx, item in item_data.iterrows():
            try:
                demand_mean = item.get('demand_mean', 10)
                demand_std = item.get('demand_std', 3)
                lead_time = item.get('lead_time', self.default_lead_time)
                unit_cost = item.get('unit_cost', 1.0)
                
                # Calculate different safety stock methods
                basic_ss = self.calculate_safety_stock(demand_mean, demand_std, lead_time)
                
                abc_class = item.get('abc_class', 'B')
                abc_ss = self.calculate_abc_based_safety_stock(
                    item.get('annual_value', 1000), demand_mean, demand_std, lead_time, abc_class
                )
                
                # Cost optimization if cost data available
                if all(col in item for col in ['holding_cost_rate', 'stockout_cost']):
                    cost_opt = self.optimize_safety_stock_cost(
                        demand_mean, demand_std, lead_time,
                        item['holding_cost_rate'], item['stockout_cost'], unit_cost
                    )
                    optimal_ss = cost_opt['optimal_safety_stock']
                else:
                    optimal_ss = basic_ss
                
                item_analysis = {
                    'item_id': item.get('item_id', idx),
                    'basic_safety_stock': basic_ss,
                    'abc_safety_stock': abc_ss,
                    'optimal_safety_stock': optimal_ss,
                    'current_safety_stock': item.get('current_safety_stock', basic_ss),
                    'safety_stock_value': optimal_ss * unit_cost,
                    'service_level': self.default_service_level
                }
                
                report['item_analysis'].append(item_analysis)
                total_safety_stock_value += item_analysis['safety_stock_value']
                
            except Exception as e:
                logger.warning(f"Error analyzing item {idx}: {e}")
                continue
        
        # Summary statistics
        if report['item_analysis']:
            analysis_df = pd.DataFrame(report['item_analysis'])
            
            report['summary'] = {
                'total_items': total_items,
                'total_safety_stock_value': total_safety_stock_value,
                'average_safety_stock': analysis_df['optimal_safety_stock'].mean(),
                'safety_stock_range': {
                    'min': analysis_df['optimal_safety_stock'].min(),
                    'max': analysis_df['optimal_safety_stock'].max()
                }
            }
            
            # Generate recommendations
            high_ss_items = analysis_df[analysis_df['optimal_safety_stock'] > analysis_df['optimal_safety_stock'].quantile(0.9)]
            low_turnover_items = analysis_df[analysis_df['safety_stock_value'] > analysis_df['safety_stock_value'].quantile(0.8)]
            
            if not high_ss_items.empty:
                report['recommendations'].append({
                    'type': 'high_safety_stock',
                    'description': f'{len(high_ss_items)} items have unusually high safety stock requirements',
                    'items': high_ss_items['item_id'].tolist()
                })
            
            if not low_turnover_items.empty:
                report['recommendations'].append({
                    'type': 'high_value_safety_stock',
                    'description': f'{len(low_turnover_items)} items have high-value safety stock',
                    'items': low_turnover_items['item_id'].tolist()
                })
        
        return report 