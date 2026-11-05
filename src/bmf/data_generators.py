# data_generators.py
from bmf.item_model import ItemModel
from bmf.config import DGConfig
from bmf.user_model import UserModel

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


@dataclass
class GeneratedData:
    """Container for all generated data including ground truth"""
    ratings: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame


class DataGenerator:
    """Handles all data generation, separate from experiment logic"""

    def __init__(self, dg_config: DGConfig):
        self.dg_config = dg_config
        self.agent_model = None
        self.item_model = None
        self.confounder_w = None
        self.confounder_z = None

    def _create_user_distribution(self):
        """
        Create user distribution object from config.
        
        Supports both:
        - Hydra instantiated objects (when using structured configs)
        - String-based configs (for backward compatibility / sweeps)
        """
        from hydra.utils import instantiate
        # from bmf.distributions import (
        #     GaussianDistribution, 
        #     GaussianMixtureDistribution, 
        #     GridDistribution
        # )
        
        user_dist_config = self.dg_config.user_distribution
        
        # If it's already a distribution object (from Hydra), use it
        if hasattr(user_dist_config, 'sample'):
            return user_dist_config
        
        # If it's a DictConfig/dict (from Hydra with _target_), instantiate it
        if hasattr(user_dist_config, '_target_') or isinstance(user_dist_config, dict):
            return instantiate(user_dist_config)
        
        raise ValueError(f"Unknown user distribution: {user_dist_config}")
        
        # Otherwise, it's a string - use factory pattern for backward compatibility
        # if user_dist_config == "gaussian":
        #     return GaussianDistribution(
        #         mean=self.dg_config.agent_mean,
        #         std=self.dg_config.agent_std
        #     )
        # elif user_dist_config == "gaussian_mixture":
        #     return GaussianMixtureDistribution(
        #         means=self.dg_config.gmm_means,
        #         stds=self.dg_config.gmm_std,
        #         weights=(1 - self.dg_config.gmm_weight_r, self.dg_config.gmm_weight_r)
        #     )
        # elif user_dist_config == "grid":
        #     return GridDistribution(min_val=-2.0, max_val=2.0)
        # else:
        #     raise ValueError(f"Unknown user distribution: {user_dist_config}")
    
    def _create_item_ideology_distribution(self):
        """
        Create item ideology distribution from config.
        
        Supports both:
        - Hydra instantiated objects (when using structured configs)
        - String-based configs (for backward compatibility / sweeps)
        """
        from hydra.utils import instantiate
        # from bmf.distributions import (
        #     GaussianDistribution, 
        #     GaussianMixtureDistribution, 
        #     GridDistribution
        # )
        
        # Handle new naming (item_ideology_distribution) or old (item_distribution)
        if hasattr(self.dg_config, 'item_ideology_distribution'):
            dist_config = self.dg_config.item_ideology_distribution
        else:
            # Backward compatibility: old name
            dist_config = self.dg_config.item_distribution
        
        # If it's already a distribution object, use it
        if hasattr(dist_config, 'sample'):
            return dist_config
        
        # If it's a DictConfig/dict (from Hydra with _target_), instantiate it
        if hasattr(dist_config, '_target_') or isinstance(dist_config, dict):
            return instantiate(dist_config)
        
        raise ValueError(f"Unknown item ideology distribution: {dist_config}")
        
        # Backward compatibility: string-based
        # if dist_config == "gaussian":
        #     return GaussianDistribution(
        #         mean=self.dg_config.item_mean,
        #         std=self.dg_config.item_std
        #     )
        # elif dist_config == "gaussian_mixture":
        #     return GaussianMixtureDistribution(
        #         means=self.dg_config.gmm_means,
        #         stds=self.dg_config.gmm_std,
        #         weights=(1 - self.dg_config.gmm_weight_r, self.dg_config.gmm_weight_r)
        #     )
        # elif dist_config == "grid":
        #     return GridDistribution(min_val=-2.0, max_val=2.0)
        # else:
        #     raise ValueError(f"Unknown item ideology distribution: {dist_config}")
    
    def _create_item_quality_distribution(self):
        """
        Create item quality distribution from config.
        
        Returns None if not specified (ItemModel will default to N(0,1))
        """
        from hydra.utils import instantiate
        # from bmf.distributions import GaussianDistribution
        
        # If not specified, return None (ItemModel will default to N(0,1))
        if not hasattr(self.dg_config, 'item_quality_distribution'):
            return None
        
        dist_config = self.dg_config.item_quality_distribution
        
        # If None, return None
        if dist_config is None:
            return None
        
        # If it's already a distribution object, use it
        if hasattr(dist_config, 'sample'):
            return dist_config
        
        # If it's a DictConfig with _target_, instantiate it
        if hasattr(dist_config, '_target_') or isinstance(dist_config, dict):
            return instantiate(dist_config)
        
        raise ValueError(f"Unknown item quality distribution: {dist_config}")
        
        # String-based (unlikely for quality, but for completeness)
        # if dist_config == "gaussian":
        #     return GaussianDistribution(mean=0.0, std=1.0)
        # else:
        #     raise ValueError(f"Unknown item quality distribution: {dist_config}")

    def _confounder_enabled(self) -> bool:
        """True when both confounder stds are configured non-zero."""
        user_std = getattr(self.dg_config, 'confounder_user_std', 0.0) or 0.0
        item_std = getattr(self.dg_config, 'confounder_item_std', 0.0) or 0.0
        return user_std > 0.0 and item_std > 0.0

    def _sample_confounder(self, rng: np.random.Generator):
        """
        Draw the unmeasured confounder factors w (per user) and z (per item).

        The ground truth becomes r_ij = q_j + g(u_i, v_j) + w_i * z_j + noise,
        i.e. two latent factors, while BMF is fitted with one. Sampled once per
        generate() so the utility and reported-rating paths share the draw.

        Sets self.confounder_w / self.confounder_z to None when disabled, which
        keeps the RNG stream identical to a run without a confounder.
        """
        if not self._confounder_enabled():
            self.confounder_w = None
            self.confounder_z = None
            return

        self.confounder_w = rng.normal(
            0.0, self.dg_config.confounder_user_std, self.dg_config.num_users
        )
        self.confounder_z = rng.normal(
            0.0, self.dg_config.confounder_item_std, self.dg_config.num_items
        )

    def _confounder_term(self, agent_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """Per-observation w_i * z_j, or zeros when the confounder is disabled."""
        if self.confounder_w is None or self.confounder_z is None:
            return np.zeros(agent_indices.shape, dtype=float)
        return self.confounder_w[agent_indices] * self.confounder_z[item_indices]

    def _create_interaction(self):
        """
        Create the rating interaction term g(u, v) from config.

        Defaults to BilinearInteraction (the well-specified case) when not
        specified, so existing configs keep their current behaviour.
        """
        from hydra.utils import instantiate
        from bmf.interactions import BilinearInteraction

        interaction_config = getattr(self.dg_config, 'interaction', None)

        # If not specified, default to the well-specified bilinear interaction
        if interaction_config is None:
            return BilinearInteraction()

        # If it's already an interaction object (from Hydra), use it
        if callable(interaction_config) and not isinstance(interaction_config, dict):
            return interaction_config

        # If it's a DictConfig/dict with _target_, instantiate it
        if hasattr(interaction_config, '_target_') or isinstance(interaction_config, dict):
            return instantiate(interaction_config)

        raise ValueError(f"Unknown interaction: {interaction_config}")

    def generate(self, seed: int = 42) -> GeneratedData:
        """
        Generate complete dataset with ground truth using explicit seeding.
        
        Args:
            seed: Random seed for reproducible data generation
        
        Returns:
            GeneratedData object containing:
                - ratings: DataFrame with ratings and metadata
                - users: DataFrame with user_id, ground truth, and reported user attributes
                - items: DataFrame with item_id and ground truth item attributes
        """
        # Create RNG for this generation (no global state pollution)
        rng = np.random.default_rng(seed)

        # Create distribution objects
        user_distribution = self._create_user_distribution()
        item_ideology_distribution = self._create_item_ideology_distribution()
        item_quality_distribution = self._create_item_quality_distribution()

        # Generate agents with explicit RNG
        self.agent_model = UserModel(
            n=self.dg_config.num_users,
            distribution=user_distribution
        )
        self.agent_model.generate_agents(rng=rng)

        # Generate items with explicit RNG
        self.item_model = ItemModel(
            m=self.dg_config.num_items,
            ideology_distribution=item_ideology_distribution,
            quality_distribution=item_quality_distribution
        )
        self.item_model.generate_items(rng=rng)

        # Sample the unmeasured confounder once, so the utility and reported-rating
        # paths below see the same draw
        self._sample_confounder(rng=rng)

        # Generate ratings with explicit RNG
        ratings = self._generate_ratings(rng=rng)
        
        # Extract ground truth data (deterministic, no randomness)
        users = self._extract_user_ground_truth()
        items = self._extract_item_ground_truth()

        return GeneratedData(ratings=ratings, users=users, items=items)

    def _compute_agent_weights(self, agent_ideology: np.ndarray) -> np.ndarray:
        """
        Compute agent weights based on extremizer status.
        
        Args:
            agent_ideology: Array of agent ideologies
            
        Returns:
            Array of weights (0 for non-extremizers, weight for extremizers)
        """
        weight = self.dg_config.weight
        
        if weight == 0.0:
            return np.zeros_like(agent_ideology)
        
        extremizer_mask = self._get_extremizer_mask(agent_ideology)
        return np.where(extremizer_mask, weight, 0.0)
    
    def _get_extremizer_mask(self, agent_ideology: np.ndarray) -> np.ndarray:
        """
        Determine which agents apply exaggeration weight based on percentile.
        
        Args:
            agent_ideology: Array of agent ideologies
            
        Returns:
            Boolean mask where True means agent applies exaggeration
        
        Note:
            extremize_percentile meaning depends on mode:
            - 'both' mode: fraction of total population, split between tails
              * 1.0 means top 50% and bottom 50% (everyone)
              * 0.2 means top 10% and bottom 10% (20% total)
            - 'single' mode: fraction of positive ideology users only
              * 1.0 means all positive users (50% of total if symmetric)
              * 0.2 means top 20% of positive users (10% of total if symmetric)
        """
        extremize_percentile = self.dg_config.extremize_percentile
        extremize_mode = self.dg_config.extremize
        
        if extremize_mode == 'both':
            # extremize_percentile is total fraction, split equally between tails
            tail_percentile = extremize_percentile / 2
            top_threshold = np.percentile(agent_ideology, (1 - tail_percentile) * 100)
            bottom_threshold = np.percentile(agent_ideology, tail_percentile * 100)
            extremizer_mask = (agent_ideology >= top_threshold) | (agent_ideology <= bottom_threshold)
        elif extremize_mode == 'single':
            # extremize_percentile applies only to positive ideology users
            positive_mask = agent_ideology > 0
            if positive_mask.any():
                positive_ideology = agent_ideology[positive_mask]
                top_threshold = np.percentile(positive_ideology, (1 - extremize_percentile) * 100)
                extremizer_mask = agent_ideology >= top_threshold
            else:
                extremizer_mask = np.zeros_like(agent_ideology, dtype=bool)
        else:
            # No extremizing
            extremizer_mask = np.zeros_like(agent_ideology, dtype=bool)
        
        return extremizer_mask

    def _generate_ratings(self, rng: np.random.Generator) -> pd.DataFrame:
        """
        Generate ratings based on configuration with explicit RNG.
        
        Args:
            rng: NumPy random generator for all random operations
            
        Returns:
            DataFrame with ratings and metadata
        """
        agents = self.agent_model.agents
        items = self.item_model.items
        n_agents = agents.shape[0]
        n_items = items.shape[0]
        
        # Generate agent-item pairs
        if self.dg_config.random_rating:
            # Random subset of items per agent
            item_ids = items[:, 0].astype(int)
            
            # Vectorized random choice for all agents at once
            selected_item_ids = np.array([
                rng.choice(item_ids, size=self.dg_config.random_n_items, replace=False)
                for _ in range(n_agents)
            ])
            
            agent_indices = np.repeat(np.arange(n_agents), self.dg_config.random_n_items)
            item_indices = selected_item_ids.flatten()
        else:
            # Full cartesian product
            agent_indices = np.repeat(np.arange(n_agents), n_items)
            item_indices = np.tile(np.arange(n_items), n_agents)

        # Extract attributes (vectorized indexing)
        agent_ideology_all = agents[agent_indices, 1]
        item_ideology_all = items[item_indices, 2]
        item_quality = items[item_indices, 1]

        # Compute masks (vectorized operations)
        same_sign_mask = (agent_ideology_all * item_ideology_all) > 0
        opposite_sign_mask = (agent_ideology_all * item_ideology_all) < 0

        # Selection logic
        if self.dg_config.select_mode == 'both':
            subject_mask = np.ones_like(agent_ideology_all, dtype=bool)
        elif self.dg_config.select_mode == 'single':
            subject_mask = agent_ideology_all > 0
        else:
            raise ValueError(f"Invalid select_mode: {self.dg_config.select_mode}")

        # Inclusion probabilities (vectorized)
        prob = np.ones_like(agent_ideology_all, dtype=float)
        prob[subject_mask & same_sign_mask] = self.dg_config.select_same
        prob[subject_mask & opposite_sign_mask] = self.dg_config.select_opp

        # Random keep mask using explicit RNG
        keep_mask = rng.random(prob.shape) < prob

        # Compute agent weights for all agents (do this once, then index)
        unique_agent_ideology = agents[:, 1]
        agent_weight_base = self._compute_agent_weights(unique_agent_ideology)
        agent_weight = agent_weight_base[agent_indices]

        # Interaction term g(u, v): bilinear when well-specified, otherwise misspecified
        interaction = self._create_interaction()

        # Unmeasured confounder w_i * z_j, zeros when disabled. It is a separate
        # latent channel from ideology, so exaggeration does not scale it: an
        # extremizer overstates their ideological reaction, not the confounder.
        confounder = self._confounder_term(agent_indices, item_indices)

        # Compute utilities (true preferences)
        utilities = item_quality + interaction(agent_ideology_all, item_ideology_all) + confounder

        # Compute reported ratings (may differ from utilities due to exaggeration).
        # 'multiplicative' scales the interaction; 'additive' extremizes the
        # reported ideology before it enters the interaction.
        if self.dg_config.weight_mode == 'multiplicative':
            ratings = item_quality + (1 + agent_weight) * interaction(agent_ideology_all, item_ideology_all) + confounder
        elif self.dg_config.weight_mode == 'additive':
            reported_ideology = agent_ideology_all + np.sign(agent_ideology_all) * agent_weight
            ratings = item_quality + interaction(reported_ideology, item_ideology_all) + confounder
        else:
            raise ValueError(f"Invalid weight_mode: {self.dg_config.weight_mode}")

        # Add noise to ratings using explicit RNG
        noise_std = self.dg_config.noise_std
        if noise_std > 0:
            noise = rng.normal(0, noise_std, size=ratings.shape)
            ratings_noisy = ratings + noise
        else:
            ratings_noisy = ratings.copy()

        # Store continuous ratings
        ratings_continuous = ratings_noisy.copy()

        # Compute logistic probabilities
        rating_probabilities = 1 / (1 + np.exp(-ratings_noisy))

        # Generate binary ratings if configured
        if self.dg_config.binary_ratings:
            ratings_binary = rng.binomial(1, rating_probabilities)
        else:
            ratings_binary = None

        # Choose which rating to use as primary
        primary_rating = ratings_binary if self.dg_config.binary_ratings else ratings_continuous

        # Create DataFrame efficiently (build dict then DataFrame once)
        user_ids = agents[agent_indices, 0].astype(int)
        item_ids = items[item_indices, 0].astype(int)

        ratings_df = pd.DataFrame({
            "user_id": user_ids,
            "item_id": item_ids,
            "rating": primary_rating,  # Binary or continuous based on config
            "rating_continuous": ratings_continuous,  # Always store continuous
            "rating_probability": rating_probabilities,  # Logistic probability
            "rating_no_noise": ratings,  # Existing: continuous without noise
            "user_utility": utilities,
            "inclusion_prob": prob,
            "keep": keep_mask
        })
        
        return ratings_df

    def _extract_user_ground_truth(self) -> pd.DataFrame:
        """
        Extract ground truth user attributes including reported ideology.
        
        Returns:
            DataFrame with user ground truth data
        """
        if self.agent_model is None or self.agent_model.agents is None:
            return pd.DataFrame(
                columns=["user_id", "true_user_ideology", "reported_user_ideology",
                        "agent_weight", "is_extremizer", "true_user_confounder"]
            )
        
        agents = self.agent_model.agents
        agent_ideology = agents[:, 1]
        
        # Compute agent weights (handles weight=0 case internally)
        agent_weight = self._compute_agent_weights(agent_ideology)
        extremizer_mask = agent_weight != 0
        
        # Calculate reported ideology based on weight_mode
        if self.dg_config.weight_mode == 'multiplicative':
            reported_ideology = agent_ideology * (1 + agent_weight)
        elif self.dg_config.weight_mode == 'additive':
            reported_ideology = agent_ideology + np.sign(agent_ideology) * agent_weight
        else:
            reported_ideology = agent_ideology.copy()
        
        # Build DataFrame efficiently
        return pd.DataFrame({
            "user_id": agents[:, 0].astype(int),
            "true_user_ideology": agent_ideology,
            "reported_user_ideology": reported_ideology,
            "agent_weight": agent_weight,
            "is_extremizer": extremizer_mask,
            "true_user_confounder": (
                self.confounder_w if self.confounder_w is not None
                else np.zeros_like(agent_ideology)
            ),
        })

    def _extract_item_ground_truth(self) -> pd.DataFrame:
        """
        Extract ground truth item attributes.
        
        Returns:
            DataFrame with item ground truth data
        """
        if self.item_model is None or self.item_model.items is None:
            return pd.DataFrame(
                columns=["item_id", "true_item_quality", "true_item_ideology",
                         "true_item_confounder"]
            )

        items = self.item_model.items

        # Build DataFrame efficiently (direct array indexing)
        return pd.DataFrame({
            "item_id": items[:, 0].astype(int),
            "true_item_quality": items[:, 1],
            "true_item_ideology": items[:, 2],
            "true_item_confounder": (
                self.confounder_z if self.confounder_z is not None
                else np.zeros(items.shape[0])
            ),
        })