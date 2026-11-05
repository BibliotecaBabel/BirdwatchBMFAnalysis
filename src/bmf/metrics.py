# mechanism_metrics.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy.stats import kendalltau
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from scipy.sparse import csr_matrix
from numba import jit, prange

@dataclass
class SelectionInfo:
    """Dataclass to store selection information for a specific k."""
    k: int
    selected_indices: List[int]
    selected_item_ids: List[Any]
    mechanism_ratings: List[float]


class MetricCategory(Enum):
    """Categories of metrics."""
    PRIMARY = "primary"  # Selection-based: OQ, SW, MS
    RANKING = "ranking"  # Ranking-based: Kendall's tau, Precision@k
    ERROR = "error"      # Error metrics: MSE, RMSE, MAE
    MECHANISM = "mechanism"  # Mechanism-level: loss, risk, regularization
    DESCRIPTIVE = "descriptive"  # Descriptive statistics: mean, median, std


# ============================================================================
# SELECTOR
# ============================================================================

class Selector:
    """
    Handles item selection logic for mechanism.
    
    This class encapsulates the mechanism's selection logic based on
    estimated ratings. Oracle selections are handled by individual metrics.
    """
    
    def __init__(self, estimated_col: str = 'estimated_item_quality',
                 true_col: str = 'true_item_quality'):
        """
        Initialize selector.
        
        Args:
            estimated_col: Column name for mechanism's estimated values
            true_col: Column name for ground truth values
        """
        self.estimated_col = estimated_col
        self.true_col = true_col
    
    def get_selection_info(self, items_df: pd.DataFrame, k: int) -> SelectionInfo:
        """
        Get mechanism selection information for a specific k.
        
        Args:
            items_df: DataFrame with item information
            k: Number of items to select
            
        Returns:
            SelectionInfo object with mechanism selections
        """
        mechanism_ratings = items_df[self.estimated_col].values
        n_items = len(items_df)
        
        if k > n_items:
            k = n_items
            warnings.warn(f"Requested k={k} exceeds n_items={n_items}. Using k={n_items}")
        
        # Mechanism selection based on estimated ratings
        sorted_indices = np.argsort(-mechanism_ratings)
        selected_indices = sorted_indices[:k].tolist()
        selected_item_ids = items_df.iloc[selected_indices]['item_id'].values.tolist()
        
        return SelectionInfo(
            k=k,
            selected_indices=selected_indices,
            selected_item_ids=selected_item_ids,
            mechanism_ratings=mechanism_ratings[selected_indices].tolist()
        )


# ============================================================================
# ORACLE SELECTORS
# ============================================================================

class OQOracleSelector:
    """Selector that picks top-k items by true quality (OQ oracle)."""

    def __init__(self, true_col: str = 'true_item_quality'):
        self.true_col = true_col
        self.estimated_col = true_col  # metrics that read selector.estimated_col get truth

    def get_selection_info(self, items_df: pd.DataFrame, k: int) -> SelectionInfo:
        true_qualities = items_df[self.true_col].values
        n_items = len(items_df)
        if k > n_items:
            k = n_items
        oracle_indices = np.argsort(-true_qualities)[:k]
        selected_item_ids = items_df.iloc[oracle_indices]['item_id'].values.tolist()
        return SelectionInfo(
            k=k,
            selected_indices=oracle_indices.tolist(),
            selected_item_ids=selected_item_ids,
            mechanism_ratings=true_qualities[oracle_indices].tolist()
        )


# ============================================================================
# BASE METRIC
# ============================================================================

class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, name: str, category: MetricCategory, requires_k: bool = False):
        self.name = name
        self.category = category
        self.requires_k = requires_k
    
    @abstractmethod
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame, 
                  selection_info: Optional[SelectionInfo] = None, 
                  selector: Optional[Selector] = None, **kwargs) -> Dict[str, Any]:
        """Calculate the metric."""
        pass
    
    def _calculate_precision(self, selected_item_ids: List[Any], 
                           oracle_item_ids: List[Any]) -> Dict[str, Any]:
        """
        Calculate precision (overlap) between mechanism and oracle selections.
        
        Precision = |intersection| / |mechanism_selection|
        
        Args:
            selected_item_ids: Item IDs selected by mechanism
            oracle_item_ids: Item IDs selected by oracle
            
        Returns:
            Dictionary with precision metrics
        """
        selected_set = set(selected_item_ids)
        oracle_set = set(oracle_item_ids)
        
        intersection = selected_set & oracle_set
        n_overlap = len(intersection)
        n_selected = len(selected_set)
        
        precision = n_overlap / n_selected if n_selected > 0 else 0.0
        
        return {
            'precision': float(precision),
            'n_overlap': n_overlap,
            'overlapping_items': list(intersection)
        }


# ============================================================================
# PRIMARY METRICS (Selection-based)
# ============================================================================

class OQMetric(BaseMetric):
    """Objective Quality metric."""
    
    def __init__(self):
        super().__init__("OQ", MetricCategory.PRIMARY, requires_k=True)
    
    def _get_quality_oracle_selection(self, items_df: pd.DataFrame, k: int,
                                     true_col: str = 'true_item_quality') -> Tuple[List[int], pd.Series]:
        """
        Get quality oracle selection for OQ metric.
        
        The quality oracle selects the top-k items by true quality.
        
        Args:
            items_df: DataFrame with item information
            k: Number of items to select
            true_col: Column name for true quality values
            
        Returns:
            Tuple of (oracle_indices, oracle_qualities as Series indexed by item_id)
        """
        true_qualities = items_df[true_col].values
        oracle_indices = np.argsort(-true_qualities)[:k]
        oracle_item_ids = items_df.iloc[oracle_indices]['item_id'].values
        
        # Return as Series indexed by item_id
        oracle_qualities = pd.Series(
            true_qualities[oracle_indices],
            index=oracle_item_ids,
            name='true_quality'
        )
        
        return oracle_indices.tolist(), oracle_qualities
    
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  selection_info: SelectionInfo, selector: Optional[Selector] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate Objective Quality metric.
        
        Args:
            items_df: DataFrame with item information
            ratings_df: DataFrame with ratings
            selection_info: Selection information from mechanism
            selector: Selector instance (optional, for accessing column names)
            
        Returns:
            Dictionary with OQ metrics including per-user utilities and precision
        """
        # Get true quality column name
        true_col = selector.true_col if selector else 'true_item_quality'
        
        # Mechanism's selected items
        mech_selected = selection_info.selected_indices
        selected_item_ids = items_df.iloc[mech_selected]['item_id'].values
        oq_mech = items_df.iloc[mech_selected][true_col].sum()
        
        # Calculate per-user utilities for mechanism's selection
        mech_ratings = ratings_df[ratings_df['item_id'].isin(selected_item_ids)]
        user_utilities_mech = mech_ratings.groupby('user_id')['user_utility'].sum()
        
        # Quality Oracle: selects items with highest true quality
        oracle_indices, oracle_qualities = self._get_quality_oracle_selection(
            items_df, selection_info.k, true_col
        )
        oracle_item_ids = items_df.iloc[oracle_indices]['item_id'].values
        oq_oracle = oracle_qualities.sum()
        
        # Calculate per-user utilities for oracle's selection
        oracle_ratings = ratings_df[ratings_df['item_id'].isin(oracle_item_ids)]
        user_utilities_oracle = oracle_ratings.groupby('user_id')['user_utility'].sum()
        
        # Calculate gap
        oq_gap = 1.0 - oq_mech / oq_oracle if oq_oracle != 0 else 0.0
        
        # Get mechanism's selected qualities as Series indexed by item_id
        mech_qualities = pd.Series(
            items_df.iloc[mech_selected][true_col].values,
            index=selected_item_ids,
            name='true_quality'
        )
        
        # Calculate precision
        precision_metrics = self._calculate_precision(
            selected_item_ids.tolist(), 
            oracle_item_ids.tolist()
        )
        
        return {
            'value': float(oq_mech),
            'oracle_value': float(oq_oracle),
            'gap': float(oq_gap),
            'selected_qualities': mech_qualities.to_dict(),
            'oracle_qualities': oracle_qualities.to_dict(),
            'oracle_indices': oracle_indices,
            'user_utilities_mech': user_utilities_mech.to_dict(),
            'user_utilities_oracle': user_utilities_oracle.to_dict(),
            **precision_metrics
        }


class SWMetric(BaseMetric):
    """Social Welfare metric."""
    
    def __init__(self):
        super().__init__("SW", MetricCategory.PRIMARY, requires_k=True)
    
    def _get_average_utility_oracle_selection(self, items_df: pd.DataFrame, 
                                            ratings_df: pd.DataFrame, 
                                            k: int) -> Tuple[List[int], pd.Series]:
        """Get selection by Average Utility Oracle: top-k items by average utility."""
        # Calculate average utility for each item (vectorized)
        avg_utilities = ratings_df.groupby('item_id')['user_utility'].mean()
        
        # Map to item indices using merge (vectorized)
        item_avg_df = items_df[['item_id']].merge(
            avg_utilities.rename('avg_utility'), 
            left_on='item_id', 
            right_index=True, 
            how='left'
        ).fillna(0)
        
        avg_utility_array = item_avg_df['avg_utility'].values
        
        # Select top-k by average utility
        sorted_indices = np.argsort(-avg_utility_array)[:k]
        oracle_item_ids = items_df.iloc[sorted_indices]['item_id'].values
        
        # Return as Series indexed by item_id for clarity
        oracle_avg_utils = pd.Series(
            avg_utility_array[sorted_indices],
            index=oracle_item_ids,
            name='avg_utility'
        )
        
        return sorted_indices.tolist(), oracle_avg_utils
    
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  selection_info: SelectionInfo, selector: Optional[Selector] = None,
                  **kwargs) -> Dict[str, Any]:
        selected_item_ids = selection_info.selected_item_ids
        
        # Filter ratings for mechanism's selection
        mech_ratings = ratings_df[ratings_df['item_id'].isin(selected_item_ids)]
        sw_mech = mech_ratings['user_utility'].sum()
        
        # Calculate per-user utilities for mechanism's selection (vectorized)
        user_utilities_mech = mech_ratings.groupby('user_id')['user_utility'].sum()
        
        # Calculate social welfare for Average Utility Oracle
        oracle_selected, oracle_avg_utils = self._get_average_utility_oracle_selection(
            items_df, ratings_df, selection_info.k
        )
        oracle_item_ids = items_df.iloc[oracle_selected]['item_id'].values
        oracle_ratings = ratings_df[ratings_df['item_id'].isin(oracle_item_ids)]
        sw_oracle = oracle_ratings['user_utility'].sum()
        
        # Calculate per-user utilities for oracle selection (vectorized)
        user_utilities_oracle = oracle_ratings.groupby('user_id')['user_utility'].sum()
        
        sw_gap = 1.0 - sw_mech/sw_oracle if sw_oracle != 0 else 0.0
        
        # Get average utilities for mechanism's selection (indexed by item_id)
        avg_utilities = ratings_df.groupby('item_id')['user_utility'].mean()
        mech_avg_utils = avg_utilities.reindex(selected_item_ids, fill_value=0)
        mech_avg_utils.name = 'avg_utility'
        
        # Calculate precision
        precision_metrics = self._calculate_precision(
            selected_item_ids, 
            oracle_item_ids.tolist()
        )
        
        return {
            'value': float(sw_mech),
            'oracle_value': float(sw_oracle),
            'gap': float(sw_gap),
            'selected_avg_utilities': mech_avg_utils.to_dict(),
            'oracle_avg_utilities': oracle_avg_utils.to_dict(),
            'n_users': ratings_df['user_id'].nunique(),
            'user_utilities_mech': user_utilities_mech.to_dict(),
            'user_utilities_oracle': user_utilities_oracle.to_dict(),
            **precision_metrics
        }


class CCMetric(BaseMetric):
    """Simplified and optimized Condorcet Compliance metric."""
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Args:
            tolerance: Tolerance for floating point comparisons
        """
        super().__init__("CC", MetricCategory.PRIMARY, requires_k=True)
        self.tolerance = tolerance
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _compute_cc_numba(utility_matrix: np.ndarray,
                          selected_indices: np.ndarray,
                          non_selected_indices: np.ndarray,
                          tolerance: float) -> float:
        """
        Numba-optimized CC computation with parallel processing.
        
        Args:
            utility_matrix: (n_users, n_items) array, NaN for missing
            selected_indices: Array of selected item indices
            non_selected_indices: Array of non-selected item indices
            tolerance: Tolerance for preference comparison
            
        Returns:
            CC score (0-1)
        """
        n_users = utility_matrix.shape[0]
        s = len(selected_indices)
        ns = len(non_selected_indices)
        
        if s == 0 or ns == 0:
            return 1.0 if ns == 0 else 0.0
        
        n_satisfied = 0
        n_total_pairs = s * ns
        
        # Extract columns once for cache efficiency
        selected_utils = utility_matrix[:, selected_indices]
        non_selected_utils = utility_matrix[:, non_selected_indices]
        
        # Parallel loop over selected items
        for i in prange(s):
            for j in range(ns):
                count_both = 0
                count_prefer = 0
                
                # Vectorized user loop (Numba handles this efficiently)
                for u in range(n_users):
                    u_sel = selected_utils[u, i]
                    u_non = non_selected_utils[u, j]
                    
                    # Check if both ratings exist
                    if not (np.isnan(u_sel) or np.isnan(u_non)):
                        count_both += 1
                        if u_sel - u_non >= -tolerance:  # Weak preference with tolerance
                            count_prefer += 1
                
                # Check majority condition
                if count_both > 0 and count_prefer >= count_both / 2:  # Avoid float division
                    n_satisfied += 1
        
        return n_satisfied / n_total_pairs if n_total_pairs > 0 else 0.0
    
    def _build_utility_matrix(self, items_df: pd.DataFrame, 
                             ratings_df: pd.DataFrame) -> Tuple[np.ndarray, Dict, pd.Index]:
        """
        Build dense utility matrix in one pass.
        
        Returns:
            utility_matrix: (n_users, n_items) array with NaN for missing
            item_id_to_idx: Mapping from item_id to column index
            user_ids: Index of user_ids (for mapping back to users)
        """
        # Get unique users and items
        user_ids = ratings_df['user_id'].unique()
        item_ids = items_df['item_id'].unique()
        
        # Create mappings
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        
        # Initialize matrix with NaN
        n_users = len(user_ids)
        n_items = len(item_ids)
        utility_matrix = np.full((n_users, n_items), np.nan, dtype=np.float32)
        
        # Fill matrix using numpy indexing (fast)
        user_indices = ratings_df['user_id'].map(user_id_to_idx).values
        item_indices = ratings_df['item_id'].map(item_id_to_idx).values
        
        # Use efficient assignment
        utility_matrix[user_indices, item_indices] = ratings_df['user_utility'].values.astype(np.float32)
        
        return utility_matrix, item_id_to_idx, pd.Index(user_ids)
    
    def _get_median_oracle_selection(self, items_df: pd.DataFrame,
                                   ratings_df: pd.DataFrame, k: int) -> Tuple[np.ndarray, pd.Series]:
        """
        Get Median Utility Oracle selection (fast).
        
        Returns:
            oracle_indices: Array of selected item indices
            oracle_medians: Series of median utilities indexed by item_id
        """
        # Compute median utilities using pandas (already optimized in C)
        median_series = ratings_df.groupby('item_id')['user_utility'].median()
        
        # Create array with -inf for missing items
        median_array = np.full(len(items_df), -np.inf)
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(items_df['item_id'])}
        
        # Fill array efficiently
        for item_id, median_val in median_series.items():
            if item_id in item_id_to_idx:
                median_array[item_id_to_idx[item_id]] = median_val
        
        # Get top-k indices
        oracle_indices = np.argsort(-median_array)[:k]
        oracle_item_ids = items_df.iloc[oracle_indices]['item_id'].values
        
        # Return as Series indexed by item_id
        oracle_medians = pd.Series(
            median_array[oracle_indices],
            index=oracle_item_ids,
            name='median_utility'
        )
        
        return oracle_indices, oracle_medians
    
    def _compute_user_utilities(self, utility_matrix: np.ndarray,
                               selected_indices: np.ndarray,
                               user_ids: pd.Index) -> pd.Series:
        """
        Compute per-user utilities from utility matrix.
        
        Args:
            utility_matrix: (n_users, n_items) array
            selected_indices: Array of selected item indices
            user_ids: Index of user IDs
            
        Returns:
            Series of user utilities indexed by user_id
        """
        # Sum utilities across selected items for each user (ignoring NaN)
        user_utils = np.nansum(utility_matrix[:, selected_indices], axis=1)
        
        return pd.Series(user_utils, index=user_ids, name='user_utility')
    
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  selection_info: SelectionInfo, selector: Optional[Selector] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Simplified CC calculation with per-user utilities and precision.
        
        Returns:
            Dictionary with value, oracle_value, gap, per-user utilities, and precision
        """
        # 1. Build utility matrix
        utility_matrix, item_id_to_idx, user_ids = self._build_utility_matrix(items_df, ratings_df)
        
        # 2. Convert selections to indices
        selected_indices = np.array([
            item_id_to_idx[item_id] for item_id in selection_info.selected_item_ids
            if item_id in item_id_to_idx
        ], dtype=np.int32)
        
        # All item indices
        all_indices = np.arange(len(items_df), dtype=np.int32)
        mask = np.isin(all_indices, selected_indices, invert=True)
        non_selected_indices = all_indices[mask]
        
        # 3. Compute per-user utilities for mechanism's selection
        user_utilities_mech = self._compute_user_utilities(
            utility_matrix, selected_indices, user_ids
        )
        
        # 4. Compute CC for mechanism
        if len(selected_indices) == 0 or len(non_selected_indices) == 0:
            cc_mech = 1.0 if len(non_selected_indices) == 0 else 0.0
        else:
            cc_mech = self._compute_cc_numba(
                utility_matrix, selected_indices, non_selected_indices, self.tolerance
            )
        
        # 5. Compute CC for Median Utility Oracle
        median_oracle_indices, oracle_medians = self._get_median_oracle_selection(
            items_df, ratings_df, selection_info.k
        )
        
        # 6. Compute per-user utilities for oracle's selection
        user_utilities_oracle = self._compute_user_utilities(
            utility_matrix, median_oracle_indices, user_ids
        )
        
        if len(median_oracle_indices) == 0 or len(non_selected_indices) == 0:
            cc_oracle = 1.0 if len(non_selected_indices) == 0 else 0.0
        else:
            cc_oracle = self._compute_cc_numba(
                utility_matrix, median_oracle_indices, non_selected_indices, self.tolerance
            )
        
        # 7. Store oracle indices for reference
        selection_info.median_utility_oracle_indices = median_oracle_indices.tolist()
        
        # 8. Get median utilities for mechanism's selection
        median_series = ratings_df.groupby('item_id')['user_utility'].median()
        mech_medians = median_series.reindex(selection_info.selected_item_ids, fill_value=0)
        mech_medians.name = 'median_utility'
        
        # 9. Get oracle item IDs for precision calculation
        oracle_item_ids = items_df.iloc[median_oracle_indices]['item_id'].values
        
        # 10. Calculate precision
        precision_metrics = self._calculate_precision(
            selection_info.selected_item_ids,
            oracle_item_ids.tolist()
        )
        
        # 11. Return results with utilities and precision
        return {
            'value': float(cc_mech),
            'oracle_value': float(cc_oracle),
            'gap': float(1.0 - cc_mech),
            'n_selected': len(selected_indices),
            'n_non_selected': len(non_selected_indices),
            'n_users': utility_matrix.shape[0],
            'selected_median_utilities': mech_medians.to_dict(),
            'oracle_median_utilities': oracle_medians.to_dict(),
            'user_utilities_mech': user_utilities_mech.to_dict(),
            'user_utilities_oracle': user_utilities_oracle.to_dict(),
            **precision_metrics
        }


# ============================================================================
# RANKING METRICS
# ============================================================================

class KendallTauMetric(BaseMetric):
    """Kendall's Tau correlation metric."""
    
    def __init__(self, estimated_col: str = 'estimated_item_quality', 
                 true_col: str = 'true_item_quality'):
        super().__init__("kendalls_tau", MetricCategory.RANKING, requires_k=False)
        self.estimated_col = estimated_col
        self.true_col = true_col
    
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  selection_info: Optional[SelectionInfo] = None,
                  selector: Optional[Selector] = None, **kwargs) -> Dict[str, Any]:
        mechanism_ratings = items_df[self.estimated_col].values
        true_qualities = items_df[self.true_col].values
        
        tau, p_value = kendalltau(mechanism_ratings, true_qualities)
        if np.isnan(tau):
            tau = 0.0
            warnings.warn("Kendall's tau calculation returned NaN, setting to 0")
        
        return {
            'value': float(tau),
            'p_value': float(p_value) if not np.isnan(p_value) else None,
        }


class PrecisionAtKMetric(BaseMetric):
    """Precision@k metric."""
    
    def __init__(self, estimated_col: str = 'estimated_item_quality',
                 true_col: str = 'true_item_quality'):
        super().__init__(f"Pk", MetricCategory.RANKING, requires_k=True)
        self.estimated_col = estimated_col
        self.true_col = true_col
    
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  selection_info: SelectionInfo,
                  selector: Optional[Selector] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate Precision@k using the selection_info.
        
        Args:
            items_df: DataFrame with item information
            ratings_df: DataFrame with ratings (not used)
            selection_info: Selection information containing k and mechanism selections
            selector: Selector instance (optional)
            
        Returns:
            Dictionary with precision value and details
        """
        k = selection_info.k
        
        # Get column names from selector if available
        estimated_col = selector.estimated_col if selector else self.estimated_col
        true_col = selector.true_col if selector else self.true_col
        
        true_qualities = items_df[true_col].values
        mechanism_ratings = items_df[estimated_col].values
        
        # Get top-k items by true quality (ground truth)
        true_top_k = set(np.argsort(-true_qualities)[:k])
        
        # Get top-k items by mechanism ratings (already in selection_info)
        mech_top_k = set(selection_info.selected_indices)
        
        # Calculate precision
        precision = len(mech_top_k.intersection(true_top_k)) / k if k > 0 else 0.0
        
        return {
            'value': float(precision),
            'k': k,
            'true_top_k': list(true_top_k),
            'mech_top_k': list(mech_top_k),
            'intersection': list(mech_top_k.intersection(true_top_k)),
            'n_correct': len(mech_top_k.intersection(true_top_k))
        }


# ============================================================================
# ERROR METRICS
# ============================================================================

class ErrorMetrics(BaseMetric):
    """Calculate error metrics (MSE, RMSE, MAE) for specified columns."""
    
    def __init__(self, target_cols: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize error metrics calculator.
        
        Args:
            target_cols: List of (estimated_col, true_col) tuples.
                        Default: [('estimated_item_quality', 'true_item_quality')]
        """
        super().__init__("error_metrics", MetricCategory.ERROR, requires_k=False)
        
        if target_cols is None:
            self.target_cols = [('estimated_item_quality', 'true_item_quality')]
        else:
            self.target_cols = target_cols
    
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  selection_info: Optional[SelectionInfo] = None,
                  selector: Optional[Selector] = None,
                  users_df: Optional[pd.DataFrame] = None,
                  **kwargs) -> Dict[str, Any]:
        """Calculate MSE, RMSE, and MAE for all target columns."""
        results = {}
        
        # Item-level errors
        for estimated_col, true_col in self.target_cols:
            if estimated_col in items_df.columns and true_col in items_df.columns:
                estimated = items_df[estimated_col].values
                true = items_df[true_col].values
                
                errors = estimated - true
                mse = float(np.mean(errors ** 2))
                rmse = float(np.sqrt(mse))
                mae = float(np.mean(np.abs(errors)))
                
                base_name = true_col.replace('true_', '')
                results[f'mse_{base_name}'] = mse
                results[f'rmse_{base_name}'] = rmse
                results[f'mae_{base_name}'] = mae
        
        # User-level errors (if users_df provided)
        if users_df is not None:
            user_error_cols = [
                ('estimated_user_ideology', 'true_user_ideology'),
            ]
            for estimated_col, true_col in user_error_cols:
                if estimated_col in users_df.columns and true_col in users_df.columns:
                    estimated = users_df[estimated_col].values
                    true = users_df[true_col].values
                    
                    errors = estimated - true
                    mse = float(np.mean(errors ** 2))
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(errors)))
                    
                    base_name = true_col.replace('true_', '')
                    results[f'mse_{base_name}'] = mse
                    results[f'rmse_{base_name}'] = rmse
                    results[f'mae_{base_name}'] = mae
        
        return results


# ============================================================================
# DESCRIPTIVE STATISTICS METRICS
# ============================================================================

class DescriptiveStatistics(BaseMetric):
    """Calculate descriptive statistics (mean, median) for user and item attributes."""
    
    def __init__(self, 
                 user_cols: Optional[List[str]] = None,
                 item_cols: Optional[List[str]] = None):
        """
        Initialize descriptive statistics calculator.
        
        Args:
            user_cols: List of user column names to compute stats for.
                      Default: ['true_user_ideology', 'reported_user_ideology', 'estimated_user_ideology']
            item_cols: List of item column names to compute stats for.
                      Default: ['true_item_quality', 'estimated_item_quality', 
                               'true_item_ideology', 'estimated_item_ideology']
        """
        super().__init__("descriptive_stats", MetricCategory.DESCRIPTIVE, requires_k=False)
        
        if user_cols is None:
            self.user_cols = ['true_user_ideology', 'reported_user_ideology', 'estimated_user_ideology']
        else:
            self.user_cols = user_cols
        
        if item_cols is None:
            self.item_cols = [
                'true_item_quality', 'estimated_item_quality',
                'true_item_ideology', 'estimated_item_ideology'
            ]
        else:
            self.item_cols = item_cols
    
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  selection_info: Optional[SelectionInfo] = None,
                  selector: Optional[Selector] = None,
                  users_df: Optional[pd.DataFrame] = None,
                  **kwargs) -> Dict[str, Any]:
        """Calculate mean and median for all specified columns."""
        results = {}
        
        # User-level statistics
        if users_df is not None:
            for col in self.user_cols:
                if col in users_df.columns:
                    values = users_df[col].dropna()
                    if len(values) > 0:
                        results[f'{col}_mean'] = float(values.mean())
                        results[f'{col}_median'] = float(values.median())
                        results[f'{col}_std'] = float(values.std())
                        # results[f'{col}_count'] = len(values)
        
        # Item-level statistics
        for col in self.item_cols:
            if col in items_df.columns:
                values = items_df[col].dropna()
                if len(values) > 0:
                    results[f'{col}_mean'] = float(values.mean())
                    results[f'{col}_median'] = float(values.median())
                    results[f'{col}_std'] = float(values.std())
                    # results[f'{col}_count'] = len(values)
        
        return results


# ============================================================================
# MECHANISM METRICS
# ============================================================================

class MechanismLevelMetrics(BaseMetric):
    """Extract mechanism-level metrics from dataframe."""
    
    def __init__(self, metric_names: Optional[List[str]] = None):
        """
        Initialize mechanism-level metrics extractor.
        
        Args:
            metric_names: List of column names to extract.
                         Default: ['loss', 'risk', 'regularization', 'total_loss']
        """
        super().__init__("mechanism_level", MetricCategory.MECHANISM, requires_k=False)
        if metric_names is None:
            self.metric_names = ['loss', 'risk', 'regularization', 'total_loss']
        else:
            self.metric_names = metric_names
    
    def calculate(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  selection_info: Optional[SelectionInfo] = None,
                  selector: Optional[Selector] = None, **kwargs) -> Dict[str, Any]:
        metrics = {}
        for metric in self.metric_names:
            if metric in items_df.columns:
                value = items_df[metric].dropna()
                if len(value) > 0:
                    metrics[metric] = float(value.iloc[0])
        return metrics


# ============================================================================
# MAIN CALCULATOR
# ============================================================================

class MechanismMetricsCalculator:
    """
    Elegant, modular calculator for mechanism evaluation metrics.
    
    Usage:
        # Initialize with metric instances
        calculator = MechanismMetricsCalculator(
            metrics=[
                OQMetric(),
                SWMetric(),
                MAMetric(approval_threshold=0.5),
                ErrorMetrics(target_cols=[
                    ('estimated_item_quality', 'true_item_quality'),
                    ('estimated_item_ideology', 'true_item_ideology')
                ]),
                DescriptiveStatistics()
            ],
            selector=Selector(estimated_col='my_estimates', true_col='my_truth'),
            k_values=[5, 10]
        )
        
        results = calculator.calculate(users_df, items_df, ratings_df)
    """
    
    def __init__(self,
                 metrics: List[BaseMetric],
                 selector: Optional[Selector] = None,
                 k_values: Optional[List[int]] = None,
                 oracle_selectors: Optional[Dict[str, Any]] = None):
        """
        Initialize calculator with metric instances.

        Args:
            metrics: List of instantiated metric objects
            selector: Selector instance for item selection (default: creates one with defaults)
            k_values: List of k values for selection metrics (default: [5, 10, 20])
            oracle_selectors: Dict mapping summary-key prefix -> selector-like object.
                              Each oracle selector re-evaluates all k-dependent metrics
                              under its selection. Keys become suffixes in summary, e.g.
                              {'oq_oracle': OQOracleSelector()} produces 'OQ_oq_oracle@k'.
        """
        self.metrics = {m.name: m for m in metrics}
        self.selector = selector or Selector()
        self.k_values = sorted(k_values) if k_values else [5, 10, 20]
        self.oracle_selectors = oracle_selectors or {}
    
    @classmethod
    def from_names(cls,
                   metric_names: List[str],
                   selector: Optional[Selector] = None,
                   k_values: Optional[List[int]] = None,
                   **metric_kwargs) -> 'MechanismMetricsCalculator':
        """
        Create calculator from metric names with default parameters.
        
        Args:
            metric_names: List of metric names ('OQ', 'SW', 'MA', etc.)
            selector: Selector instance
            k_values: List of k values
            **metric_kwargs: Additional arguments for specific metrics
                - ms_approval_threshold: float for MA metric
                - error_target_cols: List[Tuple[str, str]] for ErrorMetrics
                - mechanism_metric_names: List[str] for MechanismLevelMetrics
                - kendall_estimated_col, kendall_true_col: for KendallTauMetric
                - precision_estimated_col, precision_true_col: for PrecisionAtKMetric
                - descriptive_user_cols, descriptive_item_cols: for DescriptiveStatistics
        
        Example:
            calculator = MechanismMetricsCalculator.from_names(
                ['OQ', 'SW', 'MA', 'error_metrics', 'descriptive_stats'],
                ms_approval_threshold=0.5,
                error_target_cols=[
                    ('estimated_item_quality', 'true_item_quality'),
                    ('estimated_item_ideology', 'true_item_ideology')
                ]
            )
        """
        registry = {
            'OQ': lambda: OQMetric(),
            'SW': lambda: SWMetric(),
            'CC': lambda: CCMetric(),
            'kendalls_tau': lambda: KendallTauMetric(
                metric_kwargs.get('kendall_estimated_col', 'estimated_item_quality'),
                metric_kwargs.get('kendall_true_col', 'true_item_quality')
            ),
            'Pk': lambda: PrecisionAtKMetric(
                metric_kwargs.get('precision_estimated_col', 'estimated_item_quality'),
                metric_kwargs.get('precision_true_col', 'true_item_quality')
            ),
            'error_metrics': lambda: ErrorMetrics(metric_kwargs.get('error_target_cols')),
            'mechanism_level': lambda: MechanismLevelMetrics(
                metric_kwargs.get('mechanism_metric_names')
            ),
            'descriptive_stats': lambda: DescriptiveStatistics(
                metric_kwargs.get('descriptive_user_cols'),
                metric_kwargs.get('descriptive_item_cols')
            ),
        }
        
        metrics = []
        for name in metric_names:
            if name in registry:
                metrics.append(registry[name]())
            else:
                raise ValueError(f"Unknown metric name: {name}")
        
        return cls(metrics=metrics, selector=selector, k_values=k_values)
    
    @classmethod
    def with_categories(cls,
                       categories: List[MetricCategory],
                       selector: Optional[Selector] = None,
                       k_values: Optional[List[int]] = None,
                       **metric_kwargs) -> 'MechanismMetricsCalculator':
        """
        Create calculator with all metrics from specified categories.
        
        Args:
            categories: List of MetricCategory enums
            selector: Selector instance
            k_values: List of k values
            **metric_kwargs: Additional arguments for specific metrics
        
        Example:
            calculator = MechanismMetricsCalculator.with_categories(
                [MetricCategory.PRIMARY, MetricCategory.ERROR, MetricCategory.DESCRIPTIVE],
                ms_approval_threshold=0.5
            )
        """
        metric_map = {
            MetricCategory.PRIMARY: ['OQ', 'SW', 'CC'],
            MetricCategory.RANKING: ['kendalls_tau', 'Pk'],
            MetricCategory.ERROR: ['error_metrics'],
            MetricCategory.MECHANISM: ['mechanism_level'],
            MetricCategory.DESCRIPTIVE: ['descriptive_stats'],
        }
        
        metric_names = []
        for category in categories:
            metric_names.extend(metric_map.get(category, []))
        
        return cls.from_names(metric_names, selector=selector, k_values=k_values, **metric_kwargs)
    
    @classmethod
    def all_metrics(cls,
                   selector: Optional[Selector] = None,
                   k_values: Optional[List[int]] = None,
                   **metric_kwargs) -> 'MechanismMetricsCalculator':
        """Create calculator with all available metrics."""
        return cls.with_categories(
            [MetricCategory.PRIMARY, MetricCategory.RANKING,
             MetricCategory.ERROR, MetricCategory.MECHANISM,
             MetricCategory.DESCRIPTIVE],
            selector=selector,
            k_values=k_values,
            **metric_kwargs
        )
    
    def calculate(self,
                users_df: pd.DataFrame,
                items_df: pd.DataFrame,
                ratings_df: pd.DataFrame,
                include_summary: bool = True,
                include_selection_details: bool = True) -> Dict[str, Any]:
        """
        Calculate all configured metrics.
        
        Args:
            users_df: DataFrame with user information
            items_df: DataFrame with item information
            ratings_df: DataFrame with ratings and utilities
            include_summary: Whether to include flattened summary dict (default: True)
            include_selection_details: Whether to include selection details by k (default: True)
        
        Returns:
            Dictionary with results organized by category and k value
        """
        # Organize metrics by whether they need k
        k_dependent = {name: m for name, m in self.metrics.items() if m.requires_k}
        k_independent = {name: m for name, m in self.metrics.items() if not m.requires_k}
        
        results = {
            'by_k': {},
            'aggregate': {},
        }
        
        # Calculate k-dependent metrics
        for k in self.k_values:
            selection_info = self.selector.get_selection_info(items_df, k)
            k_results = {'k': k, 'metrics': {}}
            
            # Add selection details if requested
            if include_selection_details:
                k_results['selection'] = {
                    # 'selected_indices': selection_info.selected_indices,
                    'selected_item_ids': selection_info.selected_item_ids,
                    'mechanism_ratings': selection_info.mechanism_ratings,
                }
            
            for metric_name, metric in k_dependent.items():
                result = metric.calculate(items_df, ratings_df, selection_info,
                                        selector=self.selector)
                k_results['metrics'][metric_name] = result

                # Add to summary with k suffix
                if include_summary:
                    if 'summary' not in results:
                        results['summary'] = {}
                    results['summary'][f'{metric_name}@{k}'] = result['value']
                    if 'gap' in result:
                        results['summary'][f'{metric_name}_gap@{k}'] = result['gap']
                    if 'precision' in result:
                        results['summary'][f'{metric_name}_pk@{k}'] = result['precision']

            # Evaluate k-dependent metrics under each oracle selector
            if self.oracle_selectors:
                k_results['oracle_metrics'] = {}
                for prefix, oracle_sel in self.oracle_selectors.items():
                    oracle_selection_info = oracle_sel.get_selection_info(items_df, k)
                    oracle_metric_results = {}
                    for metric_name, metric in k_dependent.items():
                        result = metric.calculate(items_df, ratings_df, oracle_selection_info,
                                                  selector=oracle_sel)
                        oracle_metric_results[metric_name] = result
                        if include_summary:
                            if 'summary' not in results:
                                results['summary'] = {}
                            results['summary'][f'{metric_name}_{prefix}@{k}'] = result['value']
                            if 'gap' in result:
                                results['summary'][f'{metric_name}_{prefix}_gap@{k}'] = result['gap']
                            if 'precision' in result:
                                results['summary'][f'{metric_name}_{prefix}_pk@{k}'] = result['precision']
                    k_results['oracle_metrics'][prefix] = oracle_metric_results

            results['by_k'][k] = k_results
        
        # Calculate k-independent metrics
        for metric_name, metric in k_independent.items():
            result = metric.calculate(items_df, ratings_df, 
                                    selector=self.selector, users_df=users_df)
            
            # Mechanism level metrics, error metrics, and descriptive stats return a dict of values
            if isinstance(result, dict) and 'value' not in result:
                results['aggregate'][metric_name] = result
                if include_summary:
                    if 'summary' not in results:
                        results['summary'] = {}
                    results['summary'].update(result)
            else:
                results['aggregate'][metric_name] = result
                if include_summary:
                    if 'summary' not in results:
                        results['summary'] = {}
                    results['summary'][metric_name] = result['value']
        
        # Add metadata
        results['metadata'] = {
            'k_values': self.k_values,
            'metrics_calculated': list(self.metrics.keys()),
            'n_items': items_df['item_id'].nunique(),
            'n_users': users_df['user_id'].nunique(),
            'selector': {
                'estimated_col': self.selector.estimated_col,
                'true_col': self.selector.true_col
            }
        }
        
        return results
    
    @staticmethod
    def register_metric(metric_class: type):
        """
        Register a custom metric class.
        Note: This is less necessary now that you pass metric instances directly,
        but kept for backwards compatibility.
        """
        pass  # No longer needed with instance-based approach


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_evaluate(users_df: pd.DataFrame,
                  items_df: pd.DataFrame,
                  ratings_df: pd.DataFrame,
                  metric_names: List[str] = None,
                  k_values: List[int] = None,
                  **metric_kwargs) -> pd.Series:
    """
    Quick evaluation returning just the summary metrics as a Series.
    
    Example:
        summary = quick_evaluate(
            users_df, items_df, ratings_df, 
            metric_names=['OQ', 'SW', 'descriptive_stats'],
            k_values=[10],
            ms_approval_threshold=0.5
        )
    """
    calculator = MechanismMetricsCalculator.from_names(
        metric_names or ['OQ', 'SW'],
        k_values=k_values,
        **metric_kwargs
    )
    results = calculator.calculate(users_df, items_df, ratings_df)
    return pd.Series(results['summary'])

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
# 1. Most composable: Pass metric instances directly
calculator = MechanismMetricsCalculator(
    metrics=[
        OQMetric(),
        SWMetric(),
        MSMetric(approval_threshold=0.5),
        KendallTauMetric(estimated_col='my_estimates', true_col='my_truth'),
        ErrorMetrics(target_cols=[
            ('estimated_item_quality', 'true_item_quality'),
            ('estimated_item_ideology', 'true_item_ideology')
        ]),
        MechanismLevelMetrics(metric_names=['loss', 'risk'])
    ],
    selector=Selector(estimated_col='my_estimates', true_col='my_truth'),
    k_values=[5, 10]
)
results = calculator.calculate(users_df, items_df, ratings_df)

# 2. From names (convenience method)
calculator = MechanismMetricsCalculator.from_names(
    ['OQ', 'SW', 'MS', 'error_metrics'],
    selector=Selector(estimated_col='my_estimates'),
    k_values=[5, 10],
    ms_approval_threshold=0.5,
    error_target_cols=[
        ('estimated_item_quality', 'true_item_quality'),
        ('estimated_item_ideology', 'true_item_ideology')
    ]
)
results = calculator.calculate(users_df, items_df, ratings_df)

# 3. By category
calculator = MechanismMetricsCalculator.with_categories(
    [MetricCategory.PRIMARY, MetricCategory.ERROR],
    ms_approval_threshold=0.5
)
results = calculator.calculate(users_df, items_df, ratings_df)

# 4. All metrics
calculator = MechanismMetricsCalculator.all_metrics(
    k_values=[5, 10, 20],
    ms_approval_threshold=0.5
)
results = calculator.calculate(users_df, items_df, ratings_df, include_summary=False)

# 5. Custom selector for different columns
selector = Selector(
    estimated_col='my_custom_estimates',
    true_col='my_custom_ground_truth'
)
calculator = MechanismMetricsCalculator(
    metrics=[OQMetric(), SWMetric()],
    selector=selector,
    k_values=[10]
)
results = calculator.calculate(users_df, items_df, ratings_df)

# 6. Multiple error targets with custom selector
calculator = MechanismMetricsCalculator(
    metrics=[
        ErrorMetrics(target_cols=[
            ('estimated_item_quality', 'true_item_quality'),
            ('estimated_item_ideology', 'true_item_ideology'),
        ])
    ]
)
results = calculator.calculate(users_df, items_df, ratings_df)
print(results['summary'])  
# mse_item_quality, rmse_item_quality, mae_item_quality, 
# mse_item_ideology, rmse_item_ideology, mae_item_ideology

# 7. Custom metric
"""