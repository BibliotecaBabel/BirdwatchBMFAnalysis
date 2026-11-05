# item_model.py
import numpy as np
import matplotlib.pyplot as plt
from bmf.distributions import EntityDistribution, GaussianDistribution


class ItemModel:
    """
    Model for generating items with quality and ideology attributes.
    
    Items have two attributes:
    - Quality: sampled from quality_distribution (default: N(0,1))
    - Ideology: sampled from ideology_distribution
    """
    
    def __init__(
        self,
        m: int,
        ideology_distribution: EntityDistribution,
        quality_distribution: EntityDistribution = None
    ):
        """
        Initialize the item model.
        
        Args:
            m: Number of items
            ideology_distribution: Distribution for item ideology
            quality_distribution: Distribution for item quality. 
                                 If None, defaults to GaussianDistribution(0, 1)
        """
        self.m = m
        self.ideology_distribution = ideology_distribution
        
        # Default to N(0,1) if not specified
        if quality_distribution is None:
            self.quality_distribution = GaussianDistribution(mean=0.0, std=1.0)
        else:
            self.quality_distribution = quality_distribution
        
        self.items = None

    def generate_items(self, rng: np.random.Generator = None):
        """
        Generate m items with quality and ideology.
        
        Args:
            rng: NumPy random generator. If None, uses default_rng()
        
        Returns:
            None (sets self.items as m x 3 array: [id, quality, ideology])
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Pre-allocate array: [id, quality, ideology]
        self.items = np.empty((self.m, 3), dtype=np.float64)
        
        # Column 0: Item IDs
        self.items[:, 0] = np.arange(self.m)
        
        # Column 1: Quality from distribution
        self.items[:, 1] = self.quality_distribution.sample(self.m, rng)
        
        # Column 2: Ideology from distribution
        self.items[:, 2] = self.ideology_distribution.sample(self.m, rng)

    def get_distribution_figure(self, color="blue"):
        """
        Plots the distribution of items in quality vs. ideology space.
        
        Returns:
            matplotlib axis object
        """
        if self.items is None:
            raise ValueError(
                "Items have not been generated yet. Call generate_items() first."
            )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            self.items[:, 2],  # ideology
            self.items[:, 1],  # quality
            alpha=0.5, 
            s=10, 
            color=color
        )

        ax.set_xlabel("Ideology")
        ax.set_ylabel("Quality")
        ax.set_title("Item Distribution")

        plt.tight_layout()
        return ax


if __name__ == "__main__":
    from bmf.distributions import GaussianMixtureDistribution, GaussianDistribution
    
    # Demonstrate usage with explicit seeding
    rng = np.random.default_rng(42)
    
    # Example 1: Default quality (N(0,1))
    model1 = ItemModel(
        m=1000,
        ideology_distribution=GaussianMixtureDistribution(
            means=(-0.7, 0.7),
            stds=(0.3, 0.3),
            weights=(0.3, 0.7)
        )
    )
    model1.generate_items(rng=rng)
    
    # Example 2: Custom quality distribution
    model2 = ItemModel(
        m=1000,
        ideology_distribution=GaussianDistribution(mean=0.0, std=1.0),
        quality_distribution=GaussianDistribution(mean=5.0, std=2.0)
    )
    model2.generate_items(rng=rng)
    
    model1.get_distribution_figure()
    plt.show()