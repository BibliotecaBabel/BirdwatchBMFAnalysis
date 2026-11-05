# distributions.py
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


class EntityDistribution(ABC):
    """Base class for entity attribute distributions"""
    name: str
    
    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate n samples from the distribution.
        
        Args:
            n: Number of samples to generate
            rng: NumPy random generator
            
        Returns:
            Array of n sampled values
        """
        pass


@dataclass
class GaussianDistribution(EntityDistribution):
    """Normal distribution"""
    name: str = 'gaussian'
    mean: float = 0.0
    std: float = 1.0
    
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, n)


@dataclass
class GaussianMixtureDistribution(EntityDistribution):
    """Gaussian Mixture Model for bimodal/multimodal distributions"""
    name: str = 'gmm'
    means: tuple = (-0.7, 0.7)
    stds: tuple = (1.0, 1.0)
    weights: tuple = (0.5, 0.5)
    
    def __post_init__(self):
        """Validate that means, stds, and weights have the same length"""
        if not (len(self.means) == len(self.stds) == len(self.weights)):
            raise ValueError("means, stds, and weights must have the same length")
        
        # Convert to numpy arrays for easier computation
        self.means = np.array(self.means)
        self.stds = np.array(self.stds)
        self.weights = np.array(self.weights)
        
        # Normalize weights to sum to 1
        self.weights = self.weights / self.weights.sum()
    
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from Gaussian Mixture Model.
        
        This is more performant than using sklearn.mixture.GaussianMixture
        since we're just sampling, not fitting.
        """
        # Sample component assignments based on weights
        components = rng.choice(
            len(self.weights), 
            size=n, 
            p=self.weights
        )
        
        # Pre-allocate result
        samples = np.empty(n, dtype=np.float64)
        
        # Sample from each component
        for i, (mean, std) in enumerate(zip(self.means, self.stds)):
            mask = components == i
            n_samples = np.sum(mask)
            if n_samples > 0:
                samples[mask] = rng.normal(mean, std, n_samples)
        
        return samples
    

@dataclass
class SkewTDistribution(EntityDistribution):
    """Skew-t distribution for asymmetric distributions with heavy tails"""
    name: str = 'skewt'
    location: float = 0.0    # ξ: shifts distribution left/right
    scale: float = 1.0       # ω: stretches the distribution
    skewness: float = 0.0    # α: controls skewness
    df: float = 3.0          # ν: degrees of freedom (lower = heavier tails)
    
    def __post_init__(self):
        if self.df <= 0:
            raise ValueError("Degrees of freedom (df) must be positive")
    
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from skew-t distribution using a standard algorithm.
        
        Uses: if X ~ skew-normal and W ~ χ²(ν)/ν independent, then
        Y = X/√W has a skew-t distribution
        """
        delta = self.skewness / np.sqrt(1 + self.skewness**2)
        
        # Generate skew-normal samples
        z0 = rng.normal(0, 1, n)
        z1 = rng.normal(0, 1, n)
        skew_normal_samples = delta * np.abs(z0) + np.sqrt(1 - delta**2) * z1
        
        # Generate chi-squared samples and scale
        chi_sq = rng.chisquare(self.df, n)
        w = np.sqrt(chi_sq / self.df)
        
        # Apply t-distribution transformation
        samples = skew_normal_samples / w
        
        # Apply location and scale
        return self.location + self.scale * samples


@dataclass
class ShiftedGammaDistribution(EntityDistribution):
    """Shifted gamma distribution for strong right-skewness spanning negative/positive values"""
    name: str = 'shifted_gamma'
    shape: float = 2.0       # k: shape parameter (smaller = more skewed)
    scale: float = 1.0       # θ: scale parameter
    shift: float = 0.0       # c: shift to get negative values (Y = X - c)
    
    def __post_init__(self):
        if self.shape <= 0:
            raise ValueError("Shape parameter must be positive")
        if self.scale <= 0:
            raise ValueError("Scale parameter must be positive")
    
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from shifted gamma distribution.
        
        The gamma distribution is only defined for positive values,
        so we shift it by subtracting a constant to get negative values.
        """
        # Sample from gamma distribution
        samples = rng.gamma(self.shape, self.scale, n)
        
        # Apply shift
        return samples - self.shift


@dataclass
class GridDistribution(EntityDistribution):
    """Uniform grid distribution for deterministic testing"""
    name: str = 'grid'
    min_val: float = -2.0
    max_val: float = 2.0
    
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Note: rng is ignored for grid distribution as it's deterministic.
        """
        return np.linspace(self.min_val, self.max_val, n)
