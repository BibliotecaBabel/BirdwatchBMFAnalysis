import json
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import numpy as np

# Each entry maps an alias column name to either:
#   - a source column name (str), or
#   - a dict with 'source' (str) and optional 'condition': (col, value)
#     meaning: only populate alias where condition_col == condition_value
DEFAULT_COLUMN_ALIASES = {
    'dg.agent_mean': {
        'source': 'dg.user_distribution.mean',
        'condition': ('dg.user_distribution.name', 'gaussian'),
    },
    'dg.item_mean': {
        'source': 'dg.item_ideology_distribution.mean',
        'condition': ('dg.item_ideology_distribution.name', 'gaussian'),
    },
}


def _apply_column_aliases(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    """
    Add alias columns to df by copying from renamed source columns.

    For each alias:
    - If the alias column already exists, skip with a warning (data already uses the old name).
    - If the source column is absent, skip silently.
    - If a condition (col, value) is given, only rows where condition_col == value get the alias;
      this guards against aliasing a parameter that has the same key but different semantics under
      a different distribution type (e.g. 'mean' of a gaussian vs. a uniform).
    """
    import warnings
    df = df.copy()
    for alias, spec in aliases.items():
        if isinstance(spec, str):
            source, condition = spec, None
        else:
            source, condition = spec['source'], spec.get('condition')

        if alias in df.columns:
            warnings.warn(
                f"Column alias '{alias}' already exists in params_df — skipping.",
                stacklevel=3,
            )
            continue

        if source not in df.columns:
            continue

        if condition is not None:
            cond_col, cond_val = condition
            if cond_col not in df.columns:
                continue  # condition column absent — skip rather than fill blindly
            mask = df[cond_col] == cond_val
            df.loc[mask, alias] = df.loc[mask, source]
        else:
            df[alias] = df[source]

    return df


def load_project_data(project_names, base_data_path='./data', column_aliases=None):
    """
    Load dataframes for specific project names only.

    Parameters
    ----------
    project_names : list of str
        Project directory names to load (e.g., ['agent-mean-sweep']).
    base_data_path : str
        Base path where project directories live (default: './data').
    column_aliases : dict or None
        Alias definitions applied to params_df after loading.  Pass an empty
        dict ``{}`` to disable all aliases.  ``None`` uses DEFAULT_COLUMN_ALIASES,
        which back-fills 'dg.agent_mean' from 'dg.user_distribution.mean' and
        'dg.item_mean' from 'dg.item_ideology_distribution.mean' (gaussian only).
        Custom dicts are *merged* with the defaults; pass explicit ``None`` values
        to override a specific default entry.

    Returns
    -------
    tuple of (params_df, metrics_df, items_df, users_df)
    """
    effective_aliases = dict(DEFAULT_COLUMN_ALIASES)
    if column_aliases is not None:
        effective_aliases.update(column_aliases)
    # Remove entries explicitly set to None by the caller
    effective_aliases = {k: v for k, v in effective_aliases.items() if v is not None}

    # Build full paths for selected projects only
    results_paths = [f'{base_data_path}/{project}' for project in project_names]
    
    # --- Load config files ---
    config_files = [f for rp in results_paths for f in Path(rp).glob('config_exp_*.json')]
    config_data = [json.loads(f.read_text()) for f in config_files]
    params_df = pd.json_normalize(config_data, sep='.')
    params_df = _apply_column_aliases(params_df, effective_aliases)

    # --- Load summaries ---
    summary_files = [f for rp in results_paths for f in Path(rp).glob('summaries_exp_*.json')]
    
    data = []
    for f in summary_files:
        exp_name = f.stem.replace('summaries_', '')
        replications = json.loads(f.read_text())
        
        for replication in replications:
            filtered = {k: replication[k] for k in ['summary', 'training_metrics', 'general_metrics', 'replication'] if k in replication}
            filtered['experiment_name'] = exp_name
            data.append(filtered)
    
    metrics_df = pd.json_normalize(data, sep='.', max_level=2)
    
    # --- Load CSV files ---
    # Simply load all CSVs from the selected project directories
    item_files = [str(f) for rp in results_paths for f in Path(rp).glob('item_*.csv')]
    user_files = [str(f) for rp in results_paths for f in Path(rp).glob('user_*.csv')]
    
    # Load only the selected files
    if item_files:
        items_df = dd.read_csv(item_files, include_path_column='filename').compute()
        # items_df = items_df.merge(params_df, on='experiment_name').drop(columns='filename')
        items_df = items_df.query("experiment_name in @params_df.experiment_name")
        # items_df['majority_support'] = (np.sign(items_df['dg.agent_mean']) == np.sign(items_df['true_item_ideology'])).astype(int)
    else:
        items_df = pd.DataFrame()
    
    if user_files:
        users_df = dd.read_csv(user_files, include_path_column='filename').compute()
        # users_df = users_df.merge(params_df, on='experiment_name').drop(columns='filename')
        users_df = users_df.query("experiment_name in @params_df.experiment_name")
    else:
        users_df = pd.DataFrame()
    
    return params_df, metrics_df, items_df, users_df

