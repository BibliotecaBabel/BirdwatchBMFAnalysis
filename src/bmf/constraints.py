from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


class BaseConstraint(ABC):
    """Base class for constraints."""
    
    name: str
    
    @abstractmethod
    def penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute constraint penalty for loss."""
        pass
    
    def project(self, model):
        """Project model parameters to satisfy constraint (if applicable)."""
        pass
    
    def step_callback(self, model):
        """Called after each optimization step (if applicable)."""
        pass
    
    def epoch_callback(self, model):
        """Called at the end of each epoch (if applicable)."""
        pass
    
    def finalize(self, model):
        """Called at the end of training (if applicable)."""
        pass
    
    @property
    def use_penalty(self) -> bool:
        """Whether this constraint adds a penalty to the loss."""
        return False


@dataclass
class NoConstraint(BaseConstraint):
    """No constraint applied."""
    name: str = 'none'
    
    def penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=embeddings.device)


@dataclass
class MedianPenaltyConstraint(BaseConstraint):
    """Median constraint via sum-of-signs penalty."""
    
    name: str = 'penalty'
    temp: float = 1.0
    anneal_rate: float = 0.95
    
    def __post_init__(self):
        self._use_penalty = True
    
    @property
    def use_penalty(self) -> bool:
        return self._use_penalty
    
    def penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute penalty: (mean of tanh(embeddings/temp))^2"""
        if embeddings.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        smooth_sign = torch.tanh(embeddings / self.temp)
        mean_sign = torch.mean(smooth_sign)
        return mean_sign ** 2
    
    def anneal(self):
        """Gradually reduce temperature for sharper constraint."""
        self.temp = max(0.01, self.temp * self.anneal_rate)
    
    def epoch_callback(self, model):
        """Anneal temperature after each epoch."""
        self.anneal()


@dataclass
class MedianProjectionConstraint(BaseConstraint):
    """
    Median constraint via gauge transformation projection.
    
    Applies the gauge transformation to enforce median(user_ideology) = 0.
    Transformation:
      user_ideology -> user_ideology - gamma
      item_quality -> item_quality + gamma * item_ideology
    
    Args:
        warmup_epochs: Number of epochs to train without projection, if < 0 -> does not apply epochwise
        project_every: How often to project (every N epochs after warmup)
        final_projection: Whether to apply final projection at end of training
        projection_method: 'median' or 'mean' (for comparison)
    """
    
    name: str = 'projection'
    warmup_epochs: int = 10
    project_every: int = 1
    final_projection: bool = True
    projection_method: str = 'median'
    
    def __post_init__(self):
        self.epoch_count = 0
    
    def penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Projection method doesn't add penalty to loss."""
        return torch.tensor(0.0, device=embeddings.device)
    
    def _compute_center(self, tensor: torch.Tensor):
        """Compute center (median or mean)."""
        if self.projection_method == 'median':
            if tensor.ndim > 1:
                # For multi-dimensional, compute median per dimension
                return torch.median(tensor, dim=0).values
            else:
                return torch.median(tensor)
        elif self.projection_method == 'mean':
            if tensor.ndim > 1:
                return torch.mean(tensor, dim=0)
            else:
                return torch.mean(tensor)
        else:
            raise ValueError(f"Unknown projection method: {self.projection_method}")
    
    def project(self, model):
        """
        Project model parameters to satisfy median(user_ideology) = 0.
        Applies the gauge transformation:
          user_ideology -> user_ideology - gamma
          item_quality -> item_quality + gamma * item_ideology
        """
        with torch.no_grad():
            # Compute the shift needed to make center zero
            user_ideology = model.user_ideology.weight
            
            if user_ideology.ndim > 1:
                # Multi-dimensional case (d_factor > 1)
                gamma = self._compute_center(user_ideology)
                
                # Update user ideology: u_i -> u_i - gamma
                model.user_ideology.weight -= gamma
                
                # Update item quality: q_j -> q_j + gamma · v_j
                item_ideology = model.item_ideology.weight  # shape: (m, d_factor)
                if gamma.ndim == 0:
                    model.item_quality.weight += gamma * item_ideology
                else:
                    # Vector gamma: q_j += gamma · v_j
                    delta_q = torch.matmul(item_ideology, gamma.unsqueeze(-1)).squeeze()
                    model.item_quality.weight[:, 0] += delta_q
            else:
                # Scalar case (d_factor = 1)
                gamma = self._compute_center(user_ideology)
                
                # Update user ideology
                model.user_ideology.weight -= gamma
                
                # Update item quality: q_j -> q_j + gamma * v_j
                model.item_quality.weight += gamma * model.item_ideology.weight
    
    def epoch_callback(self, model):
        """Apply projection at end of epoch based on warmup and frequency."""
        # Conditions to not apply projection on epochs
        if self.warmup_epochs < 0:
            return
        
        self.epoch_count += 1
        
        # Check if we're past warmup period
        if self.epoch_count <= self.warmup_epochs:
            return
        
        # Check if it's time to project (based on frequency)
        epoch_since_warmup = self.epoch_count - self.warmup_epochs
        if epoch_since_warmup % self.project_every == 0:
            self.project(model)
    
    def finalize(self, model):
        """Apply final projection at end of training if configured."""
        if self.final_projection:
            self.project(model)


@dataclass
class HybridConstraint(BaseConstraint):
    """Combines penalty during optimization with final projection."""
    
    name: str = 'hybrid'
    temp: float = 1.0
    anneal_rate: float = 0.95
    final_projection: bool = True
    projection_method: str = 'median'
    
    def __post_init__(self):
        self.penalty_constraint = MedianPenaltyConstraint(
            temp=self.temp,
            anneal_rate=self.anneal_rate
        )
        self.projection_constraint = MedianProjectionConstraint(
            warmup_epochs=-1,  # Never project during training
            project_every=1000,
            final_projection=self.final_projection,
            projection_method=self.projection_method
        )
    
    @property
    def use_penalty(self) -> bool:
        return True
    
    def penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.penalty_constraint.penalty(embeddings)
    
    def project(self, model):
        self.projection_constraint.project(model)
    
    def epoch_callback(self, model):
        self.penalty_constraint.epoch_callback(model)
    
    def finalize(self, model):
        self.projection_constraint.finalize(model)