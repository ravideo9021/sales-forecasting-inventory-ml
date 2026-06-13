"""
Inventory optimization engine for the Sales Forecasting project.

This module implements comprehensive inventory optimization including EOQ calculation,
safety stock optimization, ABC/XYZ classification, and multi-objective optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.optimize import minimize
import pulp
from loguru import logger

from .safety_stock import SafetyStockCalculator
from .reorder_point import ReorderPointOptimizer


class InventoryOptimizer:
    """
    Comprehensive inventory optimization engine.
    
    This class implements various inventory optimization techniques including
    Economic Order Quantity (EOQ), safety stock calculation, ABC/XYZ classification,
    and multi-objective optimization for inventory management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the InventoryOptimizer.
        
        Args:
            config: Configuration dictionary containing optimization parameters
        """
        self.config = config
        self.optimization_config = config['optimization']
        self.business_config = config['business']
        
        # Initialize sub-optimizers
        self.safety_stock_calculator = SafetyStockCalculator(config)
        self.reorder_point_optimizer = ReorderPointOptimizer(config)
        
        # Optimization results
        self.optimization_results = {}
        self.abc_classification = None
        self.xyz_classification = None
        
        logger.info("InventoryOptimizer initialized successfully")
    
    def calculate_eoq(self, demand: float, ordering_cost: float, holding_cost_rate: float,
                     unit_cost: float) -> Dict[str, float]:
        """
        Calculate Economic Order Quantity (EOQ).
        
        Args:
            demand: Annual demand
            ordering_cost: Cost per order
            holding_cost_rate: Holding cost rate (fraction of unit cost)
            unit_cost: Unit cost of the item
            
        Returns:
            Dictionary containing EOQ results
        """
        if demand <= 0 or ordering_cost <= 0 or holding_cost_rate <= 0 or unit_cost <= 0:
            logger.warning("Invalid parameters for EOQ calculation")
            return {}
        
        # Calculate holding cost per unit
        holding_cost_per_unit = holding_cost_rate * unit_cost
        
        # Classic EOQ formula
        eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost_per_unit)
        
        # Calculate related metrics
        total_cost_per_year = (demand / eoq) * ordering_cost + (eoq / 2) * holding_cost_per_unit
        ordering_cost_per_year = (demand / eoq) * ordering_cost
        holding_cost_per_year = (eoq / 2) * holding_cost_per_unit
        orders_per_year = demand / eoq
        cycle_time_days = 365 / orders_per_year
        
        return {
            'eoq': float(eoq),
            'total_cost_per_year': float(total_cost_per_year),
            'ordering_cost_per_year': float(ordering_cost_per_year),
            'holding_cost_per_year': float(holding_cost_per_year),
            'orders_per_year': float(orders_per_year),
            'cycle_time_days': float(cycle_time_days)
        }
    
    def calculate_eoq_with_quantity_discounts(self, demand: float, ordering_cost: float,
                                            holding_cost_rate: float, 
                                            discount_schedule: List[Dict]) -> Dict[str, Any]:
        """
        Calculate EOQ with quantity discounts.
        
        Args:
            demand: Annual demand
            ordering_cost: Cost per order
            holding_cost_rate: Holding cost rate
            discount_schedule: List of dictionaries with 'quantity' and 'unit_cost'
            
        Returns:
            Dictionary containing optimal order quantity and costs
        """
        if not discount_schedule:
            logger.warning("No discount schedule provided")
            return {}
        
        # Sort discount schedule by quantity
        discount_schedule = sorted(discount_schedule, key=lambda x: x['quantity'])
        
        best_option = None
        best_total_cost = float('inf')
        
        for discount in discount_schedule:
            quantity_threshold = discount['quantity']
            unit_cost = discount['unit_cost']
            
            # Calculate EOQ for this price level
            eoq_result = self.calculate_eoq(demand, ordering_cost, holding_cost_rate, unit_cost)
            
            if not eoq_result:
                continue
            
            eoq = eoq_result['eoq']
            
            # If EOQ is less than quantity threshold, use threshold
            order_quantity = max(eoq, quantity_threshold)
            
            # Calculate total cost including purchase cost
            holding_cost_per_unit = holding_cost_rate * unit_cost
            total_cost = (demand * unit_cost + 
                         (demand / order_quantity) * ordering_cost + 
                         (order_quantity / 2) * holding_cost_per_unit)
            
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_option = {
                    'optimal_order_quantity': float(order_quantity),
                    'unit_cost': float(unit_cost),
                    'total_annual_cost': float(total_cost),
                    'purchase_cost': float(demand * unit_cost),
                    'ordering_cost': float((demand / order_quantity) * ordering_cost),
                    'holding_cost': float((order_quantity / 2) * holding_cost_per_unit),
                    'discount_tier': discount
                }
        
        return best_option
    
    def perform_abc_analysis(self, data: pd.DataFrame, value_column: str = 'annual_value',
                           a_threshold: float = 0.8, b_threshold: float = 0.95) -> pd.DataFrame:
        """
        Perform ABC analysis on inventory items.
        
        Args:
            data: DataFrame with inventory data
            value_column: Column name for annual value
            a_threshold: Cumulative percentage threshold for A items
            b_threshold: Cumulative percentage threshold for B items
            
        Returns:
            DataFrame with ABC classification
        """
        if value_column not in data.columns:
            logger.error(f"Column '{value_column}' not found in data")
            return data
        
        # Sort by value in descending order
        sorted_data = data.sort_values(value_column, ascending=False).copy()
        
        # Calculate cumulative percentage
        total_value = sorted_data[value_column].sum()
        sorted_data['cumulative_value'] = sorted_data[value_column].cumsum()
        sorted_data['cumulative_percentage'] = sorted_data['cumulative_value'] / total_value
        
        # Assign ABC classes
        conditions = [
            sorted_data['cumulative_percentage'] <= a_threshold,
            sorted_data['cumulative_percentage'] <= b_threshold
        ]
        choices = ['A', 'B']
        sorted_data['abc_class'] = np.select(conditions, choices, default='C')
        
        # Calculate statistics
        abc_stats = sorted_data.groupby('abc_class').agg({
            value_column: ['count', 'sum'],
            'cumulative_percentage': 'max'
        }).round(4)
        
        self.abc_classification = {
            'data': sorted_data,
            'statistics': abc_stats
        }
        
        logger.info("ABC analysis completed")
        logger.info(f"A items: {(sorted_data['abc_class'] == 'A').sum()} "
                   f"({(sorted_data['abc_class'] == 'A').mean()*100:.1f}%)")
        logger.info(f"B items: {(sorted_data['abc_class'] == 'B').sum()} "
                   f"({(sorted_data['abc_class'] == 'B').mean()*100:.1f}%)")
        logger.info(f"C items: {(sorted_data['abc_class'] == 'C').sum()} "
                   f"({(sorted_data['abc_class'] == 'C').mean()*100:.1f}%)")
        
        return sorted_data
    
    def perform_xyz_analysis(self, data: pd.DataFrame, demand_column: str = 'demand',
                           x_cv_threshold: float = 0.2, y_cv_threshold: float = 0.5) -> pd.DataFrame:
        """
        Perform XYZ analysis based on demand variability.
        
        Args:
            data: DataFrame with demand data
            demand_column: Column name for demand values
            x_cv_threshold: CV threshold for X items (low variability)
            y_cv_threshold: CV threshold for Y items (medium variability)
            
        Returns:
            DataFrame with XYZ classification
        """
        if demand_column not in data.columns:
            logger.error(f"Column '{demand_column}' not found in data")
            return data
        
        result_data = data.copy()
        
        # Calculate coefficient of variation (CV) for each item
        # This assumes demand_column contains time series data or we need to group by item
        # For simplicity, we'll calculate CV assuming data is already aggregated
        
        if 'demand_std' not in result_data.columns:
            # Calculate standard deviation if not provided
            result_data['demand_std'] = result_data[demand_column] * 0.3  # Simplified assumption
        
        # Calculate coefficient of variation
        result_data['cv'] = result_data['demand_std'] / result_data[demand_column]
        result_data['cv'] = result_data['cv'].fillna(0)
        
        # Assign XYZ classes
        conditions = [
            result_data['cv'] <= x_cv_threshold,
            result_data['cv'] <= y_cv_threshold
        ]
        choices = ['X', 'Y']
        result_data['xyz_class'] = np.select(conditions, choices, default='Z')
        
        # Calculate statistics
        xyz_stats = result_data.groupby('xyz_class').agg({
            'cv': ['count', 'mean', 'std'],
            demand_column: ['mean', 'sum']
        }).round(4)
        
        self.xyz_classification = {
            'data': result_data,
            'statistics': xyz_stats
        }
        
        logger.info("XYZ analysis completed")
        logger.info(f"X items: {(result_data['xyz_class'] == 'X').sum()} "
                   f"({(result_data['xyz_class'] == 'X').mean()*100:.1f}%)")
        logger.info(f"Y items: {(result_data['xyz_class'] == 'Y').sum()} "
                   f"({(result_data['xyz_class'] == 'Y').mean()*100:.1f}%)")
        logger.info(f"Z items: {(result_data['xyz_class'] == 'Z').sum()} "
                   f"({(result_data['xyz_class'] == 'Z').mean()*100:.1f}%)")
        
        return result_data
    
    def optimize_inventory_levels(self, forecast_data: pd.DataFrame,
                                inventory_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize inventory levels using forecasts and current inventory data.
        
        Args:
            forecast_data: DataFrame with demand forecasts
            inventory_data: DataFrame with current inventory information
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info("Starting inventory level optimization...")
        
        # Merge forecast and inventory data
        if 'item_id' in forecast_data.columns and 'item_id' in inventory_data.columns:
            merged_data = pd.merge(forecast_data, inventory_data, on='item_id', how='inner')
        else:
            logger.warning("No common item identifier found, using index-based merge")
            merged_data = pd.concat([forecast_data, inventory_data], axis=1)
        
        optimization_results = []
        
        for idx, row in merged_data.iterrows():
            try:
                # Extract necessary parameters
                demand_forecast = row.get('forecast', row.get('demand', 0))
                demand_std = row.get('forecast_std', demand_forecast * 0.2)
                current_inventory = row.get('current_inventory', 0)
                unit_cost = row.get('unit_cost', 1.0)
                lead_time = row.get('lead_time', self.optimization_config['safety_stock']['lead_time_days'])
                
                # Calculate safety stock
                safety_stock = self.safety_stock_calculator.calculate_safety_stock(
                    demand_forecast, demand_std, lead_time
                )
                
                # Calculate reorder point
                reorder_point = self.reorder_point_optimizer.calculate_reorder_point(
                    demand_forecast, lead_time, safety_stock
                )
                
                # Calculate EOQ
                ordering_cost = self.optimization_config['eoq']['ordering_cost']
                holding_cost_rate = self.optimization_config['eoq']['holding_cost_rate']
                
                eoq_result = self.calculate_eoq(
                    demand_forecast * 365,  # Annualize demand
                    ordering_cost,
                    holding_cost_rate,
                    unit_cost
                )
                
                # Calculate recommended order quantity
                if current_inventory <= reorder_point:
                    recommended_order = eoq_result.get('eoq', 0)
                else:
                    recommended_order = 0
                
                # Calculate inventory metrics
                inventory_turns = (demand_forecast * 365) / max(current_inventory, 1)
                service_level = self.optimization_config['safety_stock']['default_service_level']
                
                optimization_results.append({
                    'item_id': row.get('item_id', idx),
                    'current_inventory': current_inventory,
                    'demand_forecast': demand_forecast,
                    'demand_std': demand_std,
                    'safety_stock': safety_stock,
                    'reorder_point': reorder_point,
                    'eoq': eoq_result.get('eoq', 0),
                    'recommended_order': recommended_order,
                    'inventory_turns': inventory_turns,
                    'service_level': service_level,
                    'total_cost': eoq_result.get('total_cost_per_year', 0)
                })
                
            except Exception as e:
                logger.warning(f"Error optimizing item {idx}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(optimization_results)
        
        total_recommended_orders = results_df['recommended_order'].sum()
        unit_cost_col = merged_data['unit_cost'] if 'unit_cost' in merged_data.columns else pd.Series(1.0, index=merged_data.index)
        total_inventory_value = (results_df['current_inventory'].values * unit_cost_col.values[:len(results_df)]).sum()
        average_inventory_turns = results_df['inventory_turns'].mean()
        
        self.optimization_results = {
            'item_level_results': results_df,
            'aggregate_metrics': {
                'total_recommended_orders': total_recommended_orders,
                'total_inventory_value': total_inventory_value,
                'average_inventory_turns': average_inventory_turns,
                'items_needing_reorder': (results_df['recommended_order'] > 0).sum()
            }
        }
        
        logger.info("Inventory optimization completed")
        logger.info(f"Items needing reorder: {(results_df['recommended_order'] > 0).sum()}")
        logger.info(f"Total recommended order value: {total_recommended_orders:.2f}")
        
        return self.optimization_results
    
    def multi_objective_optimization(self, data: pd.DataFrame, 
                                   objectives: List[str] = ['minimize_cost', 'maximize_service_level'],
                                   constraints: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform multi-objective optimization for inventory management.
        
        Args:
            data: DataFrame with inventory data
            objectives: List of optimization objectives
            constraints: Dictionary of constraints
            
        Returns:
            Dictionary containing Pareto optimal solutions
        """
        logger.info("Starting multi-objective optimization...")
        
        if constraints is None:
            constraints = {
                'max_total_inventory_value': 1000000,  # $1M
                'min_service_level': 0.95,
                'max_storage_capacity': 10000
            }
        
        # Create optimization problem
        prob = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)
        
        # Decision variables
        n_items = len(data)
        order_quantities = pulp.LpVariable.dicts("OrderQty", range(n_items), lowBound=0)
        
        # Objective function (simplified to cost minimization)
        total_cost = pulp.lpSum([
            order_quantities[i] * data.iloc[i].get('unit_cost', 1.0) +
            (order_quantities[i] / 2) * data.iloc[i].get('holding_cost', 0.25)
            for i in range(n_items)
        ])
        
        prob += total_cost, "Total_Cost"
        
        # Constraints
        # Budget constraint
        if 'max_total_inventory_value' in constraints:
            prob += pulp.lpSum([
                order_quantities[i] * data.iloc[i].get('unit_cost', 1.0)
                for i in range(n_items)
            ]) <= constraints['max_total_inventory_value'], "Budget_Constraint"
        
        # Storage capacity constraint
        if 'max_storage_capacity' in constraints:
            prob += pulp.lpSum([order_quantities[i] for i in range(n_items)]) <= constraints['max_storage_capacity'], "Storage_Constraint"
        
        # Service level constraints (simplified)
        for i in range(n_items):
            min_order_qty = data.iloc[i].get('safety_stock', 0) + data.iloc[i].get('reorder_point', 0)
            prob += order_quantities[i] >= min_order_qty, f"Service_Level_Constraint_{i}"
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        if prob.status == pulp.LpStatusOptimal:
            optimal_quantities = [order_quantities[i].varValue for i in range(n_items)]
            optimal_cost = pulp.value(prob.objective)
            
            results = {
                'status': 'optimal',
                'optimal_order_quantities': optimal_quantities,
                'optimal_total_cost': optimal_cost,
                'decision_variables': {f'item_{i}': optimal_quantities[i] for i in range(n_items)}
            }
            
            logger.info(f"Multi-objective optimization completed successfully")
            logger.info(f"Optimal total cost: {optimal_cost:.2f}")
            
        else:
            results = {
                'status': 'infeasible',
                'message': 'No feasible solution found'
            }
            logger.warning("Multi-objective optimization failed - no feasible solution")
        
        return results
    
    def generate_inventory_recommendations(self, optimization_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate actionable inventory recommendations.
        
        Args:
            optimization_results: Results from inventory optimization
            
        Returns:
            DataFrame with prioritized recommendations
        """
        if 'item_level_results' not in optimization_results:
            logger.error("No item-level results found in optimization results")
            return pd.DataFrame()
        
        results_df = optimization_results['item_level_results'].copy()
        
        # Generate recommendations
        recommendations = []
        
        for idx, row in results_df.iterrows():
            recommendation = {
                'item_id': row['item_id'],
                'current_inventory': row['current_inventory'],
                'reorder_point': row['reorder_point'],
                'recommended_order': row['recommended_order'],
                'priority': 'Medium',
                'action': 'Monitor',
                'reason': 'Normal inventory levels'
            }
            
            # Determine priority and action
            if row['current_inventory'] <= row['safety_stock']:
                recommendation['priority'] = 'Critical'
                recommendation['action'] = 'Emergency Order'
                recommendation['reason'] = 'Below safety stock'
            elif row['current_inventory'] <= row['reorder_point']:
                recommendation['priority'] = 'High'
                recommendation['action'] = 'Place Order'
                recommendation['reason'] = 'Below reorder point'
            elif row['inventory_turns'] < 4:
                recommendation['priority'] = 'Low'
                recommendation['action'] = 'Reduce Stock'
                recommendation['reason'] = 'Low inventory turnover'
            
            recommendations.append(recommendation)
        
        recommendations_df = pd.DataFrame(recommendations)
        
        # Sort by priority
        priority_order = {'Critical': 1, 'High': 2, 'Medium': 3, 'Low': 4}
        recommendations_df['priority_rank'] = recommendations_df['priority'].map(priority_order)
        recommendations_df = recommendations_df.sort_values('priority_rank')
        
        logger.info(f"Generated {len(recommendations_df)} inventory recommendations")
        
        return recommendations_df
    
    def calculate_inventory_kpis(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key inventory performance indicators.
        
        Args:
            data: DataFrame with inventory data
            
        Returns:
            Dictionary of KPIs
        """
        kpis = {}
        
        try:
            # Inventory turnover
            if 'annual_demand' in data.columns and 'average_inventory' in data.columns:
                total_demand = data['annual_demand'].sum()
                total_inventory = data['average_inventory'].sum()
                kpis['inventory_turnover'] = total_demand / max(total_inventory, 1)
            
            # Fill rate / Service level
            if 'stockouts' in data.columns and 'total_orders' in data.columns:
                total_orders = data['total_orders'].sum()
                total_stockouts = data['stockouts'].sum()
                kpis['fill_rate'] = (total_orders - total_stockouts) / max(total_orders, 1)
            
            # Carrying cost
            if 'inventory_value' in data.columns:
                total_inventory_value = data['inventory_value'].sum()
                carrying_cost_rate = self.optimization_config['eoq']['holding_cost_rate']
                kpis['total_carrying_cost'] = total_inventory_value * carrying_cost_rate
            
            # ABC distribution
            if 'abc_class' in data.columns:
                abc_counts = data['abc_class'].value_counts()
                total_items = len(data)
                kpis['abc_a_percentage'] = abc_counts.get('A', 0) / total_items * 100
                kpis['abc_b_percentage'] = abc_counts.get('B', 0) / total_items * 100
                kpis['abc_c_percentage'] = abc_counts.get('C', 0) / total_items * 100
            
            logger.info("Inventory KPIs calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating inventory KPIs: {e}")
        
        return kpis
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all optimization results.
        
        Returns:
            Dictionary containing optimization summary
        """
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'optimization_results': self.optimization_results,
            'abc_classification': self.abc_classification,
            'xyz_classification': self.xyz_classification
        }
        
        return summary 