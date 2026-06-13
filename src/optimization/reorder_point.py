"""
Reorder point optimization module for the Sales Forecasting project.

This module implements reorder point calculation and optimization methods
for inventory management systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from loguru import logger


class ReorderPointOptimizer:
    """
    Reorder point optimizer for inventory management.
    
    This class calculates optimal reorder points based on demand forecasts,
    lead times, and safety stock requirements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ReorderPointOptimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_config = config['optimization']
        
        logger.info("ReorderPointOptimizer initialized successfully")
    
    def calculate_reorder_point(self, demand_forecast: float, lead_time: float,
                              safety_stock: float) -> float:
        """
        Calculate basic reorder point.
        
        Args:
            demand_forecast: Forecasted daily demand
            lead_time: Lead time in days
            safety_stock: Safety stock quantity
            
        Returns:
            Reorder point quantity
        """
        if demand_forecast <= 0 or lead_time <= 0:
            logger.warning("Invalid parameters for reorder point calculation")
            return safety_stock
        
        # ROP = (Average demand × Lead time) + Safety stock
        average_demand_during_lead_time = demand_forecast * lead_time
        reorder_point = average_demand_during_lead_time + safety_stock
        
        return max(0, reorder_point)
    
    def calculate_dynamic_reorder_point(self, demand_forecasts: List[float],
                                      lead_time: float, safety_stock: float) -> float:
        """
        Calculate dynamic reorder point using forecast trend.
        
        Args:
            demand_forecasts: List of demand forecasts for lead time period
            lead_time: Lead time in days
            safety_stock: Safety stock quantity
            
        Returns:
            Dynamic reorder point quantity
        """
        if not demand_forecasts or lead_time <= 0:
            logger.warning("Invalid parameters for dynamic reorder point calculation")
            return safety_stock
        
        # Use forecasts for each day during lead time
        forecast_length = min(len(demand_forecasts), int(lead_time))
        total_demand_during_lead_time = sum(demand_forecasts[:forecast_length])
        
        # If we have fewer forecasts than lead time days, extrapolate
        if forecast_length < lead_time:
            avg_forecast = np.mean(demand_forecasts[:forecast_length])
            remaining_days = lead_time - forecast_length
            total_demand_during_lead_time += avg_forecast * remaining_days
        
        reorder_point = total_demand_during_lead_time + safety_stock
        
        return max(0, reorder_point) 