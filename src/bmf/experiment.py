# experiment.py
import os
import logging
import hashlib
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from pytorch_lightning.loggers import CometLogger
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from dotenv import load_dotenv
import json
from omegaconf import OmegaConf
import torch

# Add these imports if needed, adjust based on your actual structure
from bmf.data_generators import DataGenerator
from bmf.mechanism import run_bmf, format_results
from bmf.config import ExperimentConfig
from bmf.metrics import (
    CCMetric,
    MechanismMetricsCalculator,
    OQMetric,
    OQOracleSelector,
    PrecisionAtKMetric,
    SWMetric,
    KendallTauMetric,
    Selector,
    ErrorMetrics,
    DescriptiveStatistics)

sys_logger = logging.getLogger(__name__)
load_dotenv()

# Seed offsets of experiments run before the seed stopped depending on the
# logging credentials, keyed by experiment name (see _compute_data_seed).
RECORDED_SEED_OFFSETS: Dict[str, int] = json.loads(
    Path(__file__).with_name("seed_offsets.json").read_text()
)


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = OmegaConf.to_object(config)

        # Get k_values from config or use default
        k_values = getattr(self.config, 'k_values', [20, 50])

        self.metrics_calculator = MechanismMetricsCalculator(
            metrics=[
                OQMetric(),
                SWMetric(),
                CCMetric(),
                PrecisionAtKMetric(
                    'estimated_item_quality',
                    'true_item_quality'
                ),
                KendallTauMetric(estimated_col='estimated_item_quality', true_col='true_item_quality'),
                ErrorMetrics(target_cols=[
                    ('estimated_item_quality', 'true_item_quality'),
                    ('estimated_item_ideology', 'true_item_ideology'),
                ]),
                DescriptiveStatistics(),
                # MechanismLevelMetrics(metric_names=['loss', 'risk'])
            ],
            selector=Selector(estimated_col='estimated_item_quality', true_col='true_item_quality'),
            k_values=k_values,
            oracle_selectors={'oq_oracle': OQOracleSelector(true_col='true_item_quality')}
        )
        
        # Create save folder using training config
        Path(self.config.save_folder).mkdir(parents=True, exist_ok=True)
    
    def _align_estimated_signs(self, model, true_users: pd.DataFrame, true_items: pd.DataFrame):
        """
        Post-process model to align signs with true parameters.
        
        Since we have access to true values in experiments, we can determine
        whether to flip the signs of estimated ideology vectors.
        
        Args:
            model: Trained BiasedMatrixFactorization model
            true_users: DataFrame with true user parameters (must have 'ideology' column)
            true_items: DataFrame with true item parameters (must have 'ideology' column)
        """
        device = next(model.parameters()).device
        
        # Sort true values by ID to match model ordering
        true_users = true_users.sort_values('user_id')
        true_items = true_items.sort_values('item_id')
        
        # Convert true ideologies to tensor
        # Assuming ideology is stored as a list or array in the 'ideology' column
        if isinstance(true_users['true_user_ideology'].iloc[0], (list, np.ndarray)):
            true_user_ideology = torch.tensor(
                np.stack(true_users['true_user_ideology'].values), 
                device=device, dtype=torch.float32
            )
        else:
            # If single-dimensional ideology
            true_user_ideology = torch.tensor(
                true_users['true_user_ideology'].values.reshape(-1, 1), 
                device=device, dtype=torch.float32
            )
        
        if isinstance(true_items['true_item_ideology'].iloc[0], (list, np.ndarray)):
            true_item_ideology = torch.tensor(
                np.stack(true_items['true_item_ideology'].values), 
                device=device, dtype=torch.float32
            )
        else:
            true_item_ideology = torch.tensor(
                true_items['true_item_ideology'].values.reshape(-1, 1), 
                device=device, dtype=torch.float32
            )
        
        # Get estimated ideologies
        user_ids = torch.arange(model.n, device=device)
        item_ids = torch.arange(model.m, device=device)
        
        est_user_ideology = model.user_ideology(user_ids)
        est_item_ideology = model.item_ideology(item_ids)
        
        # Check if dimensions match
        if est_user_ideology.shape[1] != true_user_ideology.shape[1]:
            raise ValueError(f"Dimension mismatch: estimated user ideology has {est_user_ideology.shape[1]} dims, "
                           f"but true user ideology has {true_user_ideology.shape[1]} dims")
        
        # For each dimension, decide whether to flip sign
        for d in range(est_user_ideology.shape[1]):
            # Compute correlation between true and estimated for this dimension
            true_user_dim = true_user_ideology[:, d]
            est_user_dim = est_user_ideology[:, d]
            
            true_item_dim = true_item_ideology[:, d]
            est_item_dim = est_item_ideology[:, d]
            
            # Compute correlations using torch
            def compute_correlation(x, y):
                # Pearson correlation
                x_centered = x - x.mean()
                y_centered = y - y.mean()
                numerator = (x_centered * y_centered).sum()
                denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
                return numerator / (denominator + 1e-8)
            
            user_corr = compute_correlation(true_user_dim, est_user_dim)
            item_corr = compute_correlation(true_item_dim, est_item_dim)
            
            # Decide whether to flip this dimension
            # We flip if the average correlation is negative
            avg_corr = (user_corr + item_corr) / 2
            
            if avg_corr < 0:
                # Flip sign for this dimension
                with torch.no_grad():
                    # Flip user ideology for this dimension
                    model.user_ideology.weight.data[:, d] *= -1
                    
                    # Flip item ideology for this dimension
                    model.item_ideology.weight.data[:, d] *= -1
        
        return model
    
    # def _compute_alignment_metrics(self, model, true_users: pd.DataFrame, true_items: pd.DataFrame):
    #     """
    #     Compute metrics to evaluate how well signs are aligned.
    #     Returns metrics that help understand the sign recovery.
    #     """
    #     device = next(model.parameters()).device
        
    #     # Sort true values
    #     true_users = true_users.sort_values('user_id')
    #     true_items = true_items.sort_values('item_id')
        
    #     # Convert to tensors
    #     if isinstance(true_users['ideology'].iloc[0], (list, np.ndarray)):
    #         true_user_ideology = torch.tensor(
    #             np.stack(true_users['ideology'].values), 
    #             device=device
    #         )
    #         true_user_quality = torch.tensor(
    #             true_users['quality'].values, 
    #             device=device
    #         ).unsqueeze(1)
    #     else:
    #         true_user_ideology = torch.tensor(
    #             true_users['ideology'].values, 
    #             device=device
    #         ).unsqueeze(1)
    #         true_user_quality = torch.tensor(
    #             true_users['quality'].values, 
    #             device=device
    #         ).unsqueeze(1)
        
    #     if isinstance(true_items['ideology'].iloc[0], (list, np.ndarray)):
    #         true_item_ideology = torch.tensor(
    #             np.stack(true_items['ideology'].values), 
    #             device=device
    #         )
    #         true_item_quality = torch.tensor(
    #             true_items['quality'].values, 
    #             device=device
    #         ).unsqueeze(1)
    #     else:
    #         true_item_ideology = torch.tensor(
    #             true_items['ideology'].values, 
    #             device=device
    #         ).unsqueeze(1)
    #         true_item_quality = torch.tensor(
    #             true_items['quality'].values, 
    #             device=device
    #         ).unsqueeze(1)
        
    #     # Get estimated parameters
    #     user_ids = torch.arange(model.n, device=device)
    #     item_ids = torch.arange(model.m, device=device)
        
    #     est_user_ideology = model.user_ideology(user_ids)
    #     est_user_quality = model.user_quality(user_ids)
    #     est_item_ideology = model.item_ideology(item_ids)
    #     est_item_quality = model.item_quality(item_ids)
        
    #     # Compute correlations
    #     def compute_corr_matrix(true, est):
    #         """Compute correlation between true and estimated vectors"""
    #         # Flatten if multi-dimensional
    #         if true.dim() > 1 and true.shape[1] > 1:
    #             true_flat = true.flatten()
    #             est_flat = est.flatten()
    #         else:
    #             true_flat = true.squeeze()
    #             est_flat = est.squeeze()
            
    #         # Compute Pearson correlation
    #         true_centered = true_flat - true_flat.mean()
    #         est_centered = est_flat - est_flat.mean()
    #         numerator = (true_centered * est_centered).sum()
    #         denominator = torch.sqrt((true_centered**2).sum() * (est_centered**2).sum())
    #         return numerator / (denominator + 1e-8)
        
    #     # Compute metrics
    #     metrics = {
    #         'user_ideology_corr': compute_corr_matrix(true_user_ideology, est_user_ideology).item(),
    #         'item_ideology_corr': compute_corr_matrix(true_item_ideology, est_item_ideology).item(),
    #         'user_quality_corr': compute_corr_matrix(true_user_quality, est_user_quality).item(),
    #         'item_quality_corr': compute_corr_matrix(true_item_quality, est_item_quality).item(),
    #     }
        
    #     # Compute sign recovery rate (for binary classification of sign)
    #     def compute_sign_recovery_rate(true, est):
    #         true_sign = torch.sign(true.flatten())
    #         est_sign = torch.sign(est.flatten())
    #         agreement = (true_sign == est_sign).float().mean()
    #         return agreement.item()
        
    #     metrics.update({
    #         'user_ideology_sign_recovery': compute_sign_recovery_rate(true_user_ideology, est_user_ideology),
    #         'item_ideology_sign_recovery': compute_sign_recovery_rate(true_item_ideology, est_item_ideology),
    #     })
        
    #     return metrics
    
    def _compute_data_seed(self, replication_idx: int) -> int:
        """
        Base seed of one replication:

            random_seed + replication_idx * 100000 + offset

        where the offset, in [0, 100000), is specific to the experiment, so
        that different configs and different replications draw independently.

        The offset is a hash of the config without the logging credentials.
        seed_scope 'full' hashes the whole config, so every swept parameter
        gets its own draws. seed_scope 'dg' hashes only the data-generation
        config, so arms that differ solely in the mechanism config draw
        identical populations per replication and can be compared paired.

        Experiments run before this hash excluded the credentials (it covered
        comet_api_key and comet_workspace, so it cannot be recomputed from the
        public config) have their offset recorded in seed_offsets.json, which
        reproduces their populations exactly.
        """
        offset = RECORDED_SEED_OFFSETS.get(self._build_experiment_name())
        if offset is None:
            config_dict = asdict(self.config)
            config_dict.pop("comet_api_key", None)
            config_dict.pop("comet_workspace", None)
            self._drop_inactive_dg_fields(config_dict)
            hashed = (config_dict['dg'] if self.config.seed_scope == 'dg'
                      else config_dict)
            hashed_str = json.dumps(hashed, sort_keys=True, default=str)
            offset = int(hashlib.md5(hashed_str.encode()).hexdigest()[:8], 16) % 100000

        return self.config.random_seed + replication_idx * 100000 + offset

    def _compute_component_seeds(self, replication_idx: int) -> Dict[str, int]:
        """Generate independent seeds for data and mechanism"""
        base_seed = self._compute_data_seed(replication_idx)
        ss = np.random.SeedSequence(base_seed)
        child_seeds = ss.spawn(2)
        
        return {
            'data': int(child_seeds[0].generate_state(1)[0]),
            'mechanism': int(child_seeds[1].generate_state(1)[0])
        }

    
    def run_single_replication(self, replication_idx: int, replication_logger: CometLogger) -> tuple:
        """Run a single replication of the experiment with its own logger"""
        
        # seed = self.config.random_seed + replication_idx
        # seed = self._compute_data_seed(replication_idx)

        seeds = self._compute_component_seeds(replication_idx)

        # Generate data with replication-specific seed
        data_gen = DataGenerator(self.config.dg)
        generated_data = data_gen.generate(seed=seeds['data'])
        
        # Extract components
        ratings = generated_data.ratings
        true_users = generated_data.users
        true_items = generated_data.items

        filtered_ratings = ratings.query("keep")
        
        # Run with this replication's logger
        pl.seed_everything(seed=seeds['mechanism'], workers=True)

        model, datamodule, trainer = run_bmf(
            ratings=filtered_ratings,
            dg_config=self.config.dg,
            model_config=self.config.model,
            training_config=self.config.training,
            comet_logger=replication_logger,
            mechanism_seed=seeds['mechanism']
        )
        
        # POST-PROCESS: Align signs with true parameters
        model = self._align_estimated_signs(model, true_users, true_items)
        
        # Compute alignment metrics (for debugging/analysis)
        # alignment_metrics = self._compute_alignment_metrics(model, true_users, true_items)
        
        # Format results (estimated values) - now returns 4 values
        u_res, i_res, r_res, model_metrics = format_results(model, ratings, datamodule)
        
        # Add alignment metrics to model_metrics
        # model_metrics.update({
        #     'alignment': alignment_metrics
        # })
        
        # Merge ground truth into results
        u_res = u_res.merge(true_users, on="user_id", how="left", suffixes=('', '_true'))
        i_res = i_res.merge(true_items, on="item_id", how="left", suffixes=('', '_true'))
        
        # Add replication metadata
        u_res["replication"] = i_res["replication"] = r_res["replication"] = replication_idx
        # u_res["seed"] = i_res["seed"] = r_res["seed"] = seed
        
        return seeds, u_res, i_res, r_res, model, datamodule, trainer, model_metrics
    
    def _log_replication_metrics(self, replication_logger: CometLogger, metrics: Dict[str, Any]) -> None:
        """
        Log only the summary metrics to Comet ML.
        
        Args:
            replication_logger: Comet logger for this replication
            metrics: Dictionary returned by calculate_all_metrics
        """
        if replication_logger is None:
            return
            
        summary = metrics.get('summary', {})
        
        # Log each summary metric
        for metric_name, metric_value in summary.items():
            try:
                # Convert numpy types to Python native types for JSON serialization
                if isinstance(metric_value, (np.integer, np.floating)):
                    metric_value = metric_value.item()
                elif isinstance(metric_value, np.ndarray):
                    # For arrays, log as a list or skip
                    if metric_value.size == 1:
                        metric_value = metric_value.item()
                    else:
                        # Skip logging arrays, or log a summary statistic
                        continue
                
                # Log the metric
                replication_logger.experiment.log_metric(metric_name, metric_value)
                
            except Exception as e:
                sys_logger.warning(f"Failed to log metric {metric_name}: {e}")
        
        # Optionally log some important non-summary metrics
        # if 'general_metrics' in metrics:
        #     for metric_name, metric_value in metrics['general_metrics'].items():
        #         try:
        #             replication_logger.experiment.log_metric(f"general/{metric_name}", metric_value)
        #         except Exception as e:
        #             sys_logger.debug(f"Failed to log general metric {metric_name}: {e}")
        
        # if 'training_metrics' in metrics['aggregate']:
        #     for metric_name, metric_value in metrics['aggregate']['training_metrics'].items():
        #         try:
        #             replication_logger.experiment.log_metric(f"training/{metric_name}", metric_value)
        #         except Exception as e:
        #             sys_logger.debug(f"Failed to log training metric {metric_name}: {e}")
        
        # Log metadata about the metrics calculation
        if 'metadata' in metrics:
            metadata = metrics['metadata']
            # Log key metadata as parameters
            for key in ['k_values', 'approval_threshold', 'primary_metrics', 'n_items', 'n_users']:
                if key in metadata:
                    value = metadata[key]
                    replication_logger.experiment.log_parameter(f"metrics_{key}", value)
        
        sys_logger.debug(f"Logged {len(summary)} summary metrics to Comet")
    
    def run_experiment(self, comet_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run full experiment with multiple replications, each as separate Comet experiment.
        
        Returns:
            Dictionary with experiment status and local paths to results.
        """
        exp_name = self._build_experiment_name()
        
        # Check if already run
        if self._already_done(exp_name):
            sys_logger.info(f"Skipping {exp_name} (results exist)")
            return self._get_existing_results_dict(exp_name)
        
        # Store results from all replications
        all_user, all_item, all_ratings = [], [], []
        replication_summaries = []
        
        for k in range(self.config.replications):
            sys_logger.info(f"Running replication {k+1}/{self.config.replications}")

            # Create a dedicated logger for this replication
            replication_logger = self._create_replication_logger(k)
            
            # Run replication (with its own Comet experiment)
            rep_results = self.run_single_replication(
                k, replication_logger=replication_logger)
            
            rep_seeds, u_res, i_res, r_res, model, datamodule, trainer, model_metrics = rep_results

            # Calculate and store summary using general metrics calculator       
            rep_metrics = self.metrics_calculator.calculate(
                users_df=u_res,
                items_df=i_res,
                ratings_df=r_res
            )

            # Add model metrics to the results
            if model_metrics:
                rep_metrics['aggregate']['training_metrics'] = model_metrics
                rep_metrics['summary'].update(model_metrics)
            
            # Add replication-specific metadata
            rep_metrics["replication"] = k
            rep_metrics["seeds"] = rep_seeds
            rep_metrics['experiment_name'] = exp_name
            
            replication_summaries.append(rep_metrics)

            # Log summary for this replication
            # made a change so that it will only log the first one. To save space in comet
            if replication_logger and k==0:
                self._log_replication_metrics(replication_logger, rep_metrics)
                replication_logger.experiment.end()
            
            # Store results
            all_user.append(u_res)
            all_item.append(i_res)
            all_ratings.append(r_res)
        
        # Save all results locally
        self._save_local_results(exp_name, all_user, all_item, all_ratings, replication_summaries)
        
        return {
            "status": "completed",
            "name": exp_name,
            "user_path": f"{self.config.save_folder}/user_{exp_name}.csv",
            "item_path": f"{self.config.save_folder}/item_{exp_name}.csv",
            "ratings_path": f"{self.config.save_folder}/ratings_{exp_name}.csv",
            "config_path": f"{self.config.save_folder}/config_{exp_name}.json",
            "summary_path": f"{self.config.save_folder}/summaries_{exp_name}.json",
            "num_replications": self.config.replications,
            "replication_summaries": replication_summaries
        }
    
    # ────────────────────────────────────────────────────────────
    # Helper Methods
    # ────────────────────────────────────────────────────────────
    
    @staticmethod
    def _drop_inactive_dg_fields(config_dict: Dict[str, Any]) -> None:
        """
        Remove dg fields that are absent from pre-existing configs when they are
        left at their disabling default, mutating config_dict in place.

        Both the data seed and the experiment name are hashes of the whole
        config, so a newly added dg field would otherwise shift the seeds and
        the identity of every experiment that predates it. Dropping a field only
        when it is inactive keeps those runs byte-identical while still giving
        activated ones their own hash.

        Currently: a bilinear (or unset) interaction, and a disabled confounder,
        both of which reproduce the original single-factor bilinear ground truth;
        also the top-level seed_scope when at its legacy default 'full'.
        """
        if config_dict.get('seed_scope', 'full') == 'full':
            config_dict.pop('seed_scope', None)

        dg = config_dict.get('dg')
        if not isinstance(dg, dict):
            return

        interaction = dg.get('interaction')
        interaction_name = (
            interaction.get('name') if isinstance(interaction, dict)
            else getattr(interaction, 'name', None)
        )
        if interaction is None or interaction_name == 'bilinear':
            dg.pop('interaction', None)

        if not (dg.get('confounder_user_std') and dg.get('confounder_item_std')):
            dg.pop('confounder_user_std', None)
            dg.pop('confounder_item_std', None)

    def _build_experiment_name(self) -> str:
        """Build experiment name from configuration"""
        config_dict = asdict(self.config)

        # Remove project-specific fields from name generation
        if "training" in config_dict:
            config_dict["training"].pop("project_name", None)
            config_dict["training"].pop("save_folder", None)
        config_dict.pop("comet_api_key", None)
        config_dict.pop("comet_workspace", None)
        self._drop_inactive_dg_fields(config_dict)

        # Create a deterministic hash
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"exp_{config_hash}"
    
    def _already_done(self, exp_name: str) -> bool:
        """Check if experiment has already been run"""
        user_path = Path(f"{self.config.save_folder}/user_{exp_name}.csv")
        config_path = Path(f"{self.config.save_folder}/config_{exp_name}.json")
        return user_path.exists() and config_path.exists()
    
    def _get_existing_results_dict(self, exp_name: str) -> Dict[str, Any]:
        """Return dict for existing results"""
        return {
            "status": "skipped", 
            "name": exp_name,
            "user_path": f"{self.config.save_folder}/user_{exp_name}.csv",
            "item_path": f"{self.config.save_folder}/item_{exp_name}.csv",
            "ratings_path": f"{self.config.save_folder}/ratings_{exp_name}.csv",
            "config_path": f"{self.config.save_folder}/config_{exp_name}.json",
            "summary_path": f"{self.config.save_folder}/summaries_{exp_name}.json"
        }
    
    def _create_replication_logger(self, replication_idx: int) -> Optional[CometLogger]:
        """Create a Comet logger for a specific replication"""
        
        if not self.config.log_to_comet:
            return None

        if not hasattr(self.config, 'comet_api_key') or not self.config.comet_api_key:
            return None
        
        # only logging the first rep
        if replication_idx!=0:
            return None
        
        exp_name = self._build_experiment_name()
        rep_exp_name = f"{exp_name}_rep{replication_idx}"
        
        try:
            logger = CometLogger(
                api_key=self.config.comet_api_key,
                project_name=self.config.project_name,
                workspace=getattr(self.config, 'comet_workspace', None),
                experiment_name=rep_exp_name,
                auto_output_logging='simple',
                log_code=False,
                log_graph=False,
            )
            
            # Log configuration for this replication
            config_dict = {
                "experiment_name": exp_name,
                "replication": replication_idx,
                "random_seed": self.config.random_seed + replication_idx,
                "total_replications": self.config.replications,
                "model": asdict(self.config.model),
                "dg": asdict(self.config.dg),
                "training": asdict(self.config.training),
            }
            
            logger.experiment.log_parameters(config_dict)
            
            # Add tags for grouping
            tags = [
                f"exp:{exp_name}",
                f"replication:{replication_idx}",
                "replication"
            ]
            
            # Add debug tag if in debug mode
            if getattr(self.config, 'debug_mode', False):
                tags.append("debug")
            
            for tag in tags:
                logger.experiment.add_tag(tag)
            
            return logger
            
        except Exception as e:
            sys_logger.error(f"Failed to create Comet logger for replication {replication_idx}: {e}")
            return None
    
    def _save_local_results(self, 
                            exp_name: str, 
                            users: List[pd.DataFrame], 
                           items: List[pd.DataFrame], ratings: List[pd.DataFrame],
                           replication_summaries: List[Dict[str, Any]]) -> None:
        """Save all results to local disk"""
        save_dir = Path(self.config.save_folder)
        
        # 1. Save detailed results
        if users:
            user_df = pd.concat(users, ignore_index=True)
            user_df.insert(0, "experiment_name", exp_name)
            user_df.to_csv(save_dir / f"user_{exp_name}.csv", index=False)
        
        if items:
            item_df = pd.concat(items, ignore_index=True)
            item_df.insert(0, "experiment_name", exp_name)
            item_df.to_csv(save_dir / f"item_{exp_name}.csv", index=False)
        
        if ratings:
            ratings_df = pd.concat(ratings, ignore_index=True)
            ratings_df.insert(0, "experiment_name", exp_name)
            ratings_df.to_csv(save_dir / f"ratings_{exp_name}.csv", index=False)
        
        # 2. Save configuration
        config_summary = asdict(self.config)
        config_summary['timestamp'] = datetime.now().isoformat()
        config_summary['experiment_name'] = exp_name

        config_summary.pop('comet_api_key', None)
        config_summary.pop('comet_workspace', None)

        
        config_path = save_dir / f"config_{exp_name}.json"
        with open(config_path, "w") as f:
            json.dump(config_summary, f, indent=2, default=str)
        
        # 3. Save replication summaries
        if replication_summaries:
            # Save as JSON
            summary_path = save_dir / f"summaries_{exp_name}.json"
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            serializable_summaries = convert_numpy_types(replication_summaries)
            
            with open(summary_path, "w") as f:
                json.dump(serializable_summaries, f, indent=2, default=str)