# mechanism.py

import logging
from typing import Dict, Optional, Union
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np

from bmf.constraints import BaseConstraint, NoConstraint


sys_logger = logging.getLogger(__name__)


class BiasedMatrixFactorization(pl.LightningModule):
    def __init__(
        self,
        n: int,
        m: int,
        d_factor: int = 1,
        reg_lambda: float = 0.03,
        reg_intercept_multiplier: int = 5.0,
        learning_rate: float = 0.2,
        regularization_type: str = 'l2',
        loss_type: str = 'mse',
        constraint: Optional[BaseConstraint] = None,
        ctr_lambda_multiplier: float = 1.0,
        init_seed: Optional[int] = None,
    ):
        super().__init__()
        # self.save_hyperparameters(ignore=['constraint'])
        self.save_hyperparameters()
        
        # Model parameters
        self.n = n
        self.m = m
        self.d_factor = d_factor
        self.reg_lambda = reg_lambda
        self.reg_lambda_intercept = reg_lambda * reg_intercept_multiplier
        self.learning_rate = learning_rate
        self.regularization_type = regularization_type
        self.loss_type = loss_type.lower()
        self.ctr_lambda_multiplier = ctr_lambda_multiplier
        self.constraint = constraint
        self.init_seed = init_seed

        self.ctr_lambda = self.reg_lambda * self.ctr_lambda_multiplier

        # Embeddings
        self.user_ideology = torch.nn.Embedding(n, d_factor, sparse=False)
        self.item_ideology = torch.nn.Embedding(m, d_factor, sparse=False)
        self.user_quality = torch.nn.Embedding(n, 1, sparse=False)
        self.item_quality = torch.nn.Embedding(m, 1, sparse=False)
        self.global_intercept = torch.nn.parameter.Parameter(torch.zeros(1, 1))
        
        # Initialize with explicit seed if provided
        self._initialize_parameters()

        # Set constraint (default to NoConstraint if None)
        if constraint is None:
            self.constraint = NoConstraint()
        else:
            self.constraint = constraint

        # Loss
        if self.loss_type == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif self.loss_type == 'log_loss':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
    
    def _initialize_parameters(self):
        """
        Initialize model parameters with optional explicit seeding.
        
        Note: This only controls the initialization, not training randomness
        (batch shuffling, dropout, etc.) which are controlled by PyTorch Lightning's
        seed_everything().
        """
        if self.init_seed is not None:
            # Create a generator for deterministic initialization
            generator = torch.Generator()
            generator.manual_seed(self.init_seed)
            
            torch.nn.init.xavier_uniform_(self.user_ideology.weight, generator=generator)
            torch.nn.init.xavier_uniform_(self.item_ideology.weight, generator=generator)
            self.user_quality.weight.data.fill_(0.0)
            self.item_quality.weight.data.fill_(0.0)
        else:
            # Default initialization (uses PyTorch's current RNG state)
            torch.nn.init.xavier_uniform_(self.user_ideology.weight)
            torch.nn.init.xavier_uniform_(self.item_ideology.weight)
            self.user_quality.weight.data.fill_(0.0)
            self.item_quality.weight.data.fill_(0.0)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def forward(self, user, item):
        prediction = self.global_intercept + self.user_quality(user) + self.item_quality(item)
        prediction += (
            self.user_ideology(user) * self.item_ideology(item)
        ).sum(dim=1, keepdim=True)
        return prediction.squeeze()
    
    def _compute_regularization_norms(self):
        """Compute regularization norms (reused in training_step and metrics)."""
        if self.regularization_type == 'l2':
            intercept_norm = (
                (self.user_quality.weight**2).mean() + 
                (self.item_quality.weight**2).mean() +
                (self.global_intercept**2).mean()
            )
            factor_norm = (
                (self.user_ideology.weight**2).mean() + 
                (self.item_ideology.weight**2).mean()
            )
        else:  # L1
            intercept_norm = (
                torch.abs(self.user_quality.weight).mean() + 
                torch.abs(self.item_quality.weight).mean() +
                torch.abs(self.global_intercept).mean()
            )
            factor_norm = (
                torch.abs(self.user_ideology.weight).mean() + 
                torch.abs(self.item_ideology.weight).mean()
            )
        return intercept_norm, factor_norm
    
    def _compute_constraint_penalty(self):
        """Compute constraint penalty using the constraint object."""
        # Get user ideology embeddings
        user_embeddings = self.user_ideology.weight
        
        # Handle multi-dimensional case
        if user_embeddings.ndim > 1:
            # Flatten for penalty computation
            user_embeddings = user_embeddings.view(-1)
        
        return self.constraint.penalty(user_embeddings)
    
    def _compute_weighted_regularization(self, intercept_norm, factor_norm):
        """Compute weighted regularization using current lambdas."""
        return (
            self.reg_lambda_intercept * intercept_norm + 
            self.reg_lambda * factor_norm
        )
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        y_pred = self(user_ids, item_ids)
        
        if self.loss_type == 'log_loss':
            ratings = ratings.float()
        
        # Risk (prediction error)
        risk = self.criterion(y_pred, ratings)
        
        # Regularization using helper methods
        intercept_norm, factor_norm = self._compute_regularization_norms()
        reg_loss = self._compute_weighted_regularization(intercept_norm, factor_norm)
        reg_norm = intercept_norm + factor_norm  # Unweighted
        
        # Constraint penalty using helper
        constraint_penalty = self._compute_constraint_penalty()
        
        # Total loss
        total_loss = risk + reg_loss
        if self.constraint.use_penalty:
            total_loss += self.ctr_lambda * constraint_penalty
        
        # Logging
        self.log('train/loss', total_loss)
        self.log('train/risk', risk)
        self.log('train/reg_norm', reg_norm)
        self.log('train/weighted_reg', reg_loss)
        self.log('train/constraint_penalty', constraint_penalty)
        
        # Log median statistics for monitoring
        with torch.no_grad():
            user_ideology = self.user_ideology.weight
            if user_ideology.ndim > 1:
                user_ideology = user_ideology.view(-1)
            median_val = torch.median(user_ideology).item()
            self.log('train/user_median', median_val)
        
        # Store batch values (optional, for debug)
        self.last_batch_metrics = {
            'risk': risk.item(),
            'reg_norm': reg_norm.item(),
            'weighted_reg': reg_loss.item(),
            'constraint_penalty': constraint_penalty.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss
    
    def on_train_epoch_end(self):
        """Call constraint epoch callback if defined."""
        if self.constraint:
            self.constraint.epoch_callback(self)

    def compute_metrics(self, dataloader):
        """
        Compute accurate metrics on entire dataset.
        Reuses the same helper methods as training_step.
        """
        self.eval()
        device = next(self.parameters()).device
        
        total_risk = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                y_pred = self(user_ids, item_ids)
                if self.loss_type == 'log_loss':
                    ratings = ratings.float()
                
                batch_risk = self.criterion(y_pred, ratings)
                batch_size = user_ids.size(0)
                total_risk += batch_risk.item() * batch_size
                total_samples += batch_size
        
        avg_risk = total_risk / total_samples if total_samples > 0 else 0.0
        
        # REUSE the same helpers as training_step
        intercept_norm, factor_norm = self._compute_regularization_norms()
        reg_norm = intercept_norm.item() + factor_norm.item()
        weighted_reg = self._compute_weighted_regularization(intercept_norm, factor_norm).item()
        constraint_penalty = self._compute_constraint_penalty().item()
        
        # Calculate total loss - matching training_step logic
        total_loss = avg_risk + weighted_reg
        if self.constraint and self.constraint.use_penalty:
            total_loss += self.ctr_lambda * constraint_penalty
        
        return {
            'risk': avg_risk,
            'reg_intercept_norm': intercept_norm.item(),
            'reg_factor_norm': factor_norm.item(),
            'reg_norm': reg_norm,
            'weighted_reg': weighted_reg,
            'constraint_penalty': constraint_penalty,
            'total_loss': total_loss,
        }


class MFDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        ratings: pd.DataFrame, 
        batch_size: int, 
        loss_type: str, 
        num_workers: int = 0,
        shuffle_seed: Optional[int] = None  # NEW: explicit shuffle seed
    ):
        super().__init__()
        self.ratings = ratings
        self.batch_size = batch_size if batch_size != -1 else len(ratings)
        self.loss_type = loss_type.lower()
        self.num_workers = num_workers
        self.shuffle_seed = shuffle_seed
    
    def setup(self, stage=None):
        ratings = self.ratings.dropna(subset=['rating'])
        
        user_ids = torch.LongTensor(ratings['user_id'].values)
        item_ids = torch.LongTensor(ratings['item_id'].values)
        ratings_tensor = torch.FloatTensor(ratings['rating'].values)
        
        # input should be 0, 1
        # if self.loss_type == 'log_loss':
        #     # this transform from -1, 1 to 0, 1
        #     ratings_tensor = (ratings_tensor + 1) / 2
        
        self.train_dataset = TensorDataset(user_ids, item_ids, ratings_tensor)
    
    def train_dataloader(self):
        # Create generator for shuffling if seed provided
        generator = None
        if self.shuffle_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.shuffle_seed)
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=generator,
        )


def create_constraint(constraint_cfg):
    """
    Create constraint object from config.
    
    Supports both:
    - Hydra instantiated objects (when using structured configs)
    - Legacy configs (for backward compatibility)
    """
    from hydra.utils import instantiate
    
    # If constraint_cfg is None, return NoConstraint
    if constraint_cfg is None:
        return NoConstraint()
    
    # If it's already a constraint object (from Hydra), use it
    if isinstance(constraint_cfg, BaseConstraint):
        return constraint_cfg
    
    # If it's a DictConfig with _target_ (from Hydra), instantiate it
    if hasattr(constraint_cfg, '_target_') or isinstance(constraint_cfg, dict):
        return instantiate(constraint_cfg)
    
    sys_logger.warning(f"Failed to create constraint from config:{constraint_cfg} Using NoConstraint.")
    return NoConstraint()


def run_bmf(
    ratings: pd.DataFrame,
    dg_config,
    model_config,
    training_config,
    comet_logger=None,
    mechanism_seed: Optional[int] = None  # NEW: explicit mechanism seed
):
    """
    Runs the matrix factorization model with explicit seed control.
    
    Args:
        ratings: Training data
        dg_config: Data generation config
        model_config: Model config
        training_config: Training config
        comet_logger: Optional Comet logger
        mechanism_seed: Optional seed for model initialization and training randomness.
                       If None, uses PyTorch Lightning's global seed (set via seed_everything).
    
    Notes on randomness:
        - Model initialization: Controlled by init_seed parameter
        - Batch shuffling: Controlled by shuffle_seed in DataModule
        - Training ops (dropout, etc.): Controlled by PyTorch Lightning's seed_everything
        
        If mechanism_seed is provided, it's used for both initialization and shuffling.
        Otherwise, relies on global PyTorch state set by pl.seed_everything().
    """
    n_agents = dg_config.num_users
    m_items = dg_config.num_items

    # After loading ratings
    is_binary = ratings['rating'].isin([0, 1]).all()

    if is_binary and model_config.loss_type == 'mse':
        sys_logger.warning("Binary ratings with MSE loss - consider using log_loss")
        
    if not is_binary and model_config.loss_type == 'log_loss':
        sys_logger.warning("Continuous ratings with log_loss - consider using mse")
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='train/loss',
        min_delta=training_config.min_delta,
        patience=training_config.patience,
        verbose=False,
        mode='min'
    )

    # Create constraint
    constraint = create_constraint(model_config.constraint)
    
    # Create model with explicit initialization seed
    mf_model = BiasedMatrixFactorization(
        n=n_agents,
        m=m_items,
        d_factor=model_config.d_factor,
        reg_lambda=model_config.reg_lambda,
        reg_intercept_multiplier=model_config.reg_intercept_multiplier,
        learning_rate=training_config.learning_rate,
        regularization_type=model_config.regularization_type,
        loss_type=model_config.loss_type,
        constraint=constraint,
        ctr_lambda_multiplier=model_config.ctr_lambda_multiplier,
        init_seed=mechanism_seed, 
    )
    
    # Data module with explicit shuffle seed
    datamodule = MFDataModule(
        ratings=ratings,
        batch_size=training_config.batch_size,
        loss_type=model_config.loss_type,
        shuffle_seed=mechanism_seed
    )
    
    # Trainer. Without a Comet logger, logging is switched off: Lightning's
    # default CSV logger writes to a shared lightning_logs/version_N folder,
    # and parallel jobs race for the same version number and corrupt each
    # other's metrics file. Early stopping reads the callback metrics, which
    # do not need a logger.
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        logger=comet_logger if comet_logger is not None else False,
        log_every_n_steps=training_config.log_every_n_steps,
        callbacks=[early_stopping],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    
    trainer.fit(mf_model, datamodule=datamodule)
    
    # Apply final projection if constraint supports it
    if constraint and hasattr(constraint, 'finalize'):
        constraint.finalize(mf_model)
    
    return mf_model, datamodule, trainer


def format_results(mf_model: BiasedMatrixFactorization, ratings: pd.DataFrame, datamodule: MFDataModule):
    """
    Formats model results. Only includes essential constraint diagnostics.
    """
    user_ids = torch.LongTensor(ratings['user_id'].unique())
    item_ids = torch.LongTensor(ratings['item_id'].unique())
    
    # User results with actual median
    with torch.no_grad():
        user_ideology = mf_model.user_ideology.weight
        if user_ideology.ndim > 1:
            user_ideology = user_ideology.view(-1)
        user_median = torch.median(user_ideology).item()
        
        # Compute mean_sign, only if temp is part of the constraint
        if mf_model.constraint and hasattr(mf_model.constraint, 'temp'):
            smooth_sign = torch.tanh(user_ideology / mf_model.constraint.temp)
            mean_sign = torch.mean(smooth_sign).item()
        else:
            mean_sign = torch.mean(torch.sign(user_ideology)).item()
    
    # User results dataframe
    user_results = pd.DataFrame({
        "user_id": user_ids.cpu().numpy(),
        "estimated_user_quality": mf_model.user_quality(user_ids).cpu().detach().numpy().flatten(),
        "estimated_user_ideology": mf_model.user_ideology(user_ids).cpu().detach().numpy().flatten(),
    }).drop_duplicates().reset_index(drop=True)
    
    # Item results dataframe
    item_results = pd.DataFrame({
        "item_id": item_ids.cpu().numpy(),
        "estimated_item_quality": mf_model.item_quality(item_ids).cpu().detach().numpy().flatten(),
        "estimated_item_ideology": mf_model.item_ideology(item_ids).cpu().detach().numpy().flatten(),
    }).drop_duplicates().sort_values(by=['item_id']).reset_index(drop=True)
    
    # Device
    device = next(mf_model.parameters()).device
    
    # Predictions
    user_ids_tensor = torch.LongTensor(ratings['user_id'].values).to(device)
    item_ids_tensor = torch.LongTensor(ratings['item_id'].values).to(device)
    
    with torch.no_grad():
        estimated_ratings = mf_model(user_ids_tensor, item_ids_tensor).cpu().numpy()
    
    ratings_w_results = ratings.copy()
    ratings_w_results['estimated_rating'] = estimated_ratings
    ratings_w_results['global_intercept'] = mf_model.global_intercept.item()

    # Final model metrics
    dataloader = datamodule.train_dataloader()
    model_metrics = mf_model.compute_metrics(dataloader)
    
    # Only essential constraint diagnostics - actual achieved metrics
    model_metrics.update({
        'user_ideology_median': user_median,
        'mean_sign': mean_sign,
    })
    
    return user_results, item_results, ratings_w_results, model_metrics