# user_model.py
import numpy as np
import matplotlib.pyplot as plt
from bmf.distributions import EntityDistribution


class UserModel:
    """
    Model for generating users with ideology values.
    """
    
    def __init__(self, n: int, distribution: EntityDistribution):
        """
        Initialize the user model.
        
        Args:
            n: Number of users to generate
            distribution: EntityDistribution object defining the ideology distribution
        """
        self.n = n
        self.distribution = distribution
        self.agents = None

    def generate_agents(self, rng: np.random.Generator = None):
        """
        Generate n agents with ideology values.
        
        Args:
            rng: NumPy random generator. If None, uses default_rng()
            
        Returns:
            None (sets self.agents as n x 2 array with [user_id, ideology])
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Pre-allocate array for better performance
        self.agents = np.empty((self.n, 2), dtype=np.float64)
        
        # Column 0: User IDs
        self.agents[:, 0] = np.arange(self.n)
        
        # Column 1: Ideology from distribution
        self.agents[:, 1] = self.distribution.sample(self.n, rng)

    def rating(
        self,
        agent_ideology: float,
        item_quality: float,
        item_ideology: float,
        weight: float = 0.0,
    ) -> float:
        """
        Compute rating for a single agent-item pair.
        
        Args:
            agent_ideology: Agent's ideology value
            item_quality: Item's quality
            item_ideology: Item's ideology
            weight: Exaggeration weight (only applied for positive ideology)
            
        Returns:
            Rating value
        """
        # Only add weight for positive ideology
        w = 1 if agent_ideology <= 0 else (1 + weight)
        return item_quality + w * agent_ideology * item_ideology

    def get_distribution_figure(self, color="blue"):
        """
        Plot the distribution of ideology values.
        
        Returns:
            matplotlib axis object
        """
        if self.agents is None:
            raise ValueError(
                "Agents have not been generated yet. Call generate_agents() first."
            )

        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Use numpy histogram for better performance with large data
        ax.hist(
            self.agents[:, 1],  # ideology column
            bins=30, 
            color=color, 
            alpha=0.7, 
            edgecolor='black'
        )

        ax.set_xlabel("Ideology")
        ax.set_ylabel("Frequency")
        ax.set_title("User Ideology Distribution")

        plt.tight_layout()
        return ax


if __name__ == "__main__":
    from bmf.distributions import GaussianDistribution, GaussianMixtureDistribution
    
    # Demonstrate usage with different distributions
    rng = np.random.default_rng(42)
    
    print("Testing Gaussian Distribution:")
    model1 = UserModel(
        n=1000, 
        distribution=GaussianDistribution(mean=0.0, std=1.0)
    )
    model1.generate_agents(rng=rng)
    print(f"Mean: {model1.agents[:, 1].mean():.3f}")
    print(f"Std: {model1.agents[:, 1].std():.3f}")
    
    print("\nTesting Gaussian Mixture Distribution:")
    model2 = UserModel(
        n=1000, 
        distribution=GaussianMixtureDistribution(
            means=(-0.7, 0.7),
            stds=(0.3, 0.3),
            weights=(0.3, 0.7)
        )
    )
    model2.generate_agents(rng=rng)
    print(f"Mean: {model2.agents[:, 1].mean():.3f}")
    print(f"Std: {model2.agents[:, 1].std():.3f}")
    
    model1.get_distribution_figure(color="blue")
    plt.show()