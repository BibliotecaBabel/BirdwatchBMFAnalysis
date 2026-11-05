# interactions.py
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Interaction(ABC):
    """Base class for the user-item interaction term in the rating model.

    Ratings are generated as `q_j + g(u_i, v_j) + noise`, where `g` is the
    interaction. BMF can only represent the bilinear `u_i * v_j`, so any other
    interaction makes the fitted mechanism misspecified: whatever `g` contains
    beyond the bilinear part becomes omitted-variable bias.
    """
    name: str

    @abstractmethod
    def __call__(self, user_ideology: np.ndarray, item_ideology: np.ndarray) -> np.ndarray:
        """
        Compute the interaction term elementwise over rating observations.

        Args:
            user_ideology: Per-observation user ideology (already indexed)
            item_ideology: Per-observation item ideology (already indexed)

        Returns:
            Array of interaction values, same shape as the inputs
        """
        pass


@dataclass
class BilinearInteraction(Interaction):
    """Dot-product preferences: g(u, v) = u * v.

    The well-specified case — matches what BMF fits.
    """
    name: str = 'bilinear'

    def __call__(self, user_ideology: np.ndarray, item_ideology: np.ndarray) -> np.ndarray:
        return user_ideology * item_ideology


@dataclass
class EuclideanInteraction(Interaction):
    """Ideological distance preferences: g(u, v) = -(u - v)².

    Users prefer items closer to them in ideology, so the term is negative
    squared distance. Expanding gives -u² + 2*u*v - v²: BMF captures the
    2*u*v part and omits both quadratic terms.
    """
    name: str = 'euclidean'

    def __call__(self, user_ideology: np.ndarray, item_ideology: np.ndarray) -> np.ndarray:
        return -((user_ideology - item_ideology) ** 2)


@dataclass
class HigherOrderInteraction(Interaction):
    """Polynomial preferences: g(u, v) = u*v + phi * u² * v².

    Adds an amplified (polarizing) component on top of the bilinear term.
    BMF fits the bilinear part; the quadratic product is omitted.
    `phi = 0` reduces exactly to bilinear.
    """
    name: str = 'higher_order'
    phi: float = 0.1

    def __call__(self, user_ideology: np.ndarray, item_ideology: np.ndarray) -> np.ndarray:
        bilinear = user_ideology * item_ideology
        return bilinear + self.phi * (bilinear ** 2)
