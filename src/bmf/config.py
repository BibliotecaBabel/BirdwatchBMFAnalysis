# config.py
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Any
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Flatten function for your nested structure
def flatten_config(config):
    """Flatten nested JSON config into a single-level dict"""
    flat = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            # Flatten nested dicts with dot notation
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    # Convert lists to strings or handle as needed
                    flat[f'{key}.{sub_key}'] = str(sub_value)
                else:
                    flat[f'{key}.{sub_key}'] = sub_value
        elif isinstance(value, list):
            # Convert lists to strings
            flat[key] = str(value)
        else:
            flat[key] = value
    
    return flat


@dataclass
class ModelConfig:
    reg_lambda: float = 0.03
    reg_intercept_multiplier: float = 5.0
    regularization_type: str = "l2"
    loss_type: str = 'mse'
    d_factor: int = 1
    ctr_lambda_multiplier: Optional[float] = 1.0
    constraint: Any = MISSING


@dataclass
class DGConfig:
    # Distribution configs (will be populated by Hydra from YAML)
    user_distribution: Any = MISSING
    item_ideology_distribution: Any = MISSING
    item_quality_distribution: Any = MISSING

    # Rating interaction term g(u, v); None defaults to bilinear (well-specified)
    interaction: Any = None

    # Unmeasured confounder: a second latent factor pair adding w_i * z_j to the
    # rating. Both stds default to 0.0, which disables it and leaves the
    # single-factor ground truth unchanged.
    confounder_user_std: float = 0.0
    confounder_item_std: float = 0.0

    # Core parameters
    num_users: int = 100
    num_items: int = 100
    
    # Rating assignment
    weight: float = 0.0
    weight_mode: str = 'multiplicative'
    extremize: str = 'single'
    extremize_percentile: float = 1.0
    
    # Selection parameters
    select_same: float = 1.0
    select_opp: float = 1.0
    select_mode: str = 'single'
    
    # Random rating config
    random_rating: bool = False
    random_n_items: int = 10

    # Ratings noise
    noise_std: float = 0.1
    binary_ratings: bool = False


@dataclass
class TrainingConfig:
    # Training parameters (only these 4 as requested)
    max_epochs: int = 10
    batch_size: int = -1  # -1 means full batch
    learning_rate: float = 0.1 
    log_every_n_steps: int = 10
    min_delta: float = 1e-5
    patience: int = 5


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dg: DGConfig = field(default_factory=DGConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    project_name: str = "gaussian-exp"
    save_folder: str = "data/gaussian-exp"
    replications: int = 3
    random_seed: int = 42
    # Scope of the data-seed hash: 'full' hashes the whole config (legacy;
    # every swept parameter gets its own draws), 'dg' hashes only the dg
    # config so arms differing solely in mechanism config (e.g. constraint
    # sweeps) draw identical populations and can be compared paired.
    seed_scope: str = "full"
    debug_mode: bool = False
    k_values: List[int] = field(default_factory=lambda: [5, 10, 50])
    approval_threshold: float = 0.0
    comet_api_key: Optional[str] = None
    comet_workspace: Optional[str] = None
    log_to_comet: Optional[bool] = False
    log_summary_experiment: Optional[bool] = False


# Register configs with Hydra
cs = ConfigStore.instance()

# Store base config
cs.store(name="base_config", node=ExperimentConfig)
