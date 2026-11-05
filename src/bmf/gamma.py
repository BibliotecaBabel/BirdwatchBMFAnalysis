import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ============================================================================
# GAMMA CALCULATION FUNCTIONS
# ============================================================================

def calculate_user_gamma_regression(group):
    """Calculate gamma using user regression: intercept of u_reported ~ u_hat.

    The anchor is a property of the fit relative to the data it was fit on —
    the *reports*. The fit recenters ideologies as u_hat ~ u_reported - gamma,
    so regressing reported ideology on u_hat recovers gamma as the intercept
    (the gauge check; agrees with the item-based estimator under correct
    specification). Regressing *true* ideology instead conflates the anchor
    with the misreport shift (it estimates gamma minus the coalition's offset)
    and points the wrong way under manipulation. Under truthful reporting the
    two coincide; falls back to true ideology if reports were not saved.
    """
    u_col = ('reported_user_ideology' if 'reported_user_ideology' in group
             else 'true_user_ideology')
    u_rep = group[u_col].values
    u_hat = group['estimated_user_ideology'].values

    if len(u_rep) < 2:
        return np.nan

    # Fit: u_reported = β₀ + β₁ * u_hat
    beta = np.linalg.lstsq(
        np.column_stack([np.ones_like(u_hat), u_hat]),
        u_rep,
        rcond=None
    )[0]
    return beta[0]  # β₀ is gamma


def calculate_item_gamma_regression(group):
    """Calculate gamma using item regression: slope of q_hat ~ v_true"""
    v_true = group['true_item_ideology'].values
    q_hat = group['estimated_item_quality'].values
    
    if len(v_true) < 2:
        return np.nan
    
    # Fit: q_hat = β₀ + β₁ * v_true
    beta = np.linalg.lstsq(
        np.column_stack([np.ones_like(v_true), v_true]), 
        q_hat, 
        rcond=None
    )[0]
    return beta[1]  # β₁ is gamma


def calculate_neutral_gamma(group):
    """Calculate gamma using neutral agent method.

    The agent the mechanism treats as neutral (u_hat ~ 0) is the one whose
    *report* sits at the anchor, so their reported ideology estimates gamma.
    (Under manipulation that agent can be an extremizer; their true ideology
    would instead be gamma minus their misreport shift.) Falls back to true
    ideology if reports were not saved — identical under truthful reporting.
    """
    u_col = ('reported_user_ideology' if 'reported_user_ideology' in group
             else 'true_user_ideology')
    u_rep = group[u_col].values
    u_hat = group['estimated_user_ideology'].values

    if len(u_hat) == 0:
        return np.nan

    neutral_idx = np.argmin(np.abs(u_hat))
    return u_rep[neutral_idx]

# OLDER VERSION OF THE FUNCTIONS

# def calculate_user_gamma(group):
#     """Calculate gamma using user method: mean(u_true - u_hat)"""
#     u_true = group['true_user_ideology']
#     u_hat = group['estimated_user_ideology']
#     return np.mean(u_true - u_hat) if len(u_true) > 0 else np.nan

# def calculate_item_gamma(group):
#     """Calculate gamma using item method with true v"""
#     v_true = group['true_item_ideology']
#     q_true = group['true_item_quality']
#     q_hat = group['estimated_item_quality']
    
#     if len(v_true) == 0:
#         return np.nan
    
#     denominator = np.sum(v_true**2)
#     if denominator < 1e-10:
#         return np.nan
    
#     numerator = np.sum(v_true * (q_hat - q_true))
#     return numerator / denominator


# ============================================================================
# MAIN PROCESSING (USING GROUPBY.APPLY)
# ============================================================================

def gamma_estimates(users_df, items_df):
    """
    Compute all three gamma estimates for a single group of data.
    
    Parameters:
    - users_df: DataFrame containing users for this group
    - items_df: DataFrame containing items for this group
    
    Returns dict with:
    - gamma_user, gamma_item, gamma_neutral
    - mu_u_true, and other metadata
    """
    
    # Calculate gamma estimates
    gamma_user = calculate_user_gamma_regression(users_df)
    gamma_neutral = calculate_neutral_gamma(users_df)
    gamma_item = calculate_item_gamma_regression(items_df)
    
    # Get mu_u_true
    # TODO change for the generalized version, this applies m=n, \lambda_u = \lambda_q
    gamma_ideal_l2 = users_df['true_user_ideology'].mean() /2
    # TODO change for the generalized version, this applies m>>n, \lambda_u = \lambda_q
    gamma_ideal_l1 = users_df['true_user_ideology'].median()/2
    
    # Get metadata (take first row)
    result = {
        'gamma_user': gamma_user,
        'gamma_neutral': gamma_neutral,
        'gamma_item': gamma_item,
        'gamma_ideal_l2': gamma_ideal_l2,  
        'gamma_ideal_l1': gamma_ideal_l1, 
    }
    
    # Add discrepancy columns
    for method in ['gamma_user', 'gamma_item', 'gamma_neutral']:
        result[f'{method}_diff'] = result[method] - gamma_ideal_l2
        result[f'{method}_abs_diff'] = np.abs(result[f'{method}_diff'])
    
    return pd.Series(result)


def gamma_estimates_grouped(users_df, items_df, group_cols=['experiment_name', 'replication']):
    """
    Compute gamma estimates for grouped data.
    
    Parameters:
    - users_df: Full users DataFrame
    - items_df: Full items DataFrame  
    - group_cols: Columns to group by
    
    Returns DataFrame with gamma estimates for each group
    """
    results = []
    
    for group_keys, user_group in users_df.groupby(group_cols):
        # Create filter for matching items
        if isinstance(group_keys, tuple):
            item_filter = pd.Series(True, index=items_df.index)
            for col, key in zip(group_cols, group_keys):
                item_filter &= (items_df[col] == key)
        else:
            item_filter = (items_df[group_cols[0]] == group_keys)
        
        item_group = items_df[item_filter]
        
        # Compute estimates
        result = gamma_estimates(user_group, item_group)
        
        # Add group keys
        if isinstance(group_keys, tuple):
            for col, key in zip(group_cols, group_keys):
                result[col] = key
        else:
            result[group_cols[0]] = group_keys
            
        results.append(result)
    
    return pd.DataFrame(results)
