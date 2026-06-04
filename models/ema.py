"""
Exponential Moving Average (EMA) Model.

Provides a smoothed version of model weights for more stable predictions
in semi-supervised learning.
"""

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn


class EMAModel:
    """
    Exponential Moving Average model wrapper.

    Maintains a shadow copy of model weights that are updated as:
        shadow = decay * shadow + (1 - decay) * model_weights.

    This provides smoother, more stable predictions compared to the
    rapidly changing training model.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize EMA model.

        Args:
            model: The model to track.
            decay: EMA decay rate (higher = smoother, slower updates).
            device: Device to store shadow weights.
        """
        self.decay = decay
        self.device = device

        # Create shadow model (deep copy).
        self.shadow = deepcopy(model)
        self.shadow.eval()

        # Disable gradients for shadow model.
        for param in self.shadow.parameters():
            param.requires_grad = False

        if device is not None:
            self.shadow.to(device)

        # Track number of updates.
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update shadow weights with current model weights.

        Args:
            model: The training model with updated weights.
        """
        self.num_updates += 1

        # Optionally use warmup decay (start with lower decay, increase over time).
        decay = min(
            self.decay, (1 + self.num_updates) / (10 + self.num_updates)
        )

        # Update each parameter.
        for shadow_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            if self.device is not None:
                model_param_data = model_param.data.to(self.device)
            else:
                model_param_data = model_param.data

            shadow_param.data.mul_(decay).add_(
                model_param_data, alpha=1 - decay
            )

        # Update buffers (e.g., BatchNorm running stats).
        for shadow_buffer, model_buffer in zip(
            self.shadow.buffers(), model.buffers()
        ):
            shadow_buffer.copy_(model_buffer)

    def forward(self, *args, **kwargs):
        """
        Forward pass through shadow model.
        """
        return self.shadow(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Alias for forward.
        """
        return self.forward(*args, **kwargs)

    def eval(self):
        """
        Set shadow model to eval mode.
        """
        self.shadow.eval()
        return self

    def train(self):
        """
        EMA model should always be in eval mode.
        """
        self.shadow.eval()
        return self

    def state_dict(self):
        """
        Get state dict of shadow model.
        """
        return {
            "shadow": self.shadow.state_dict(),
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict):
        """
        Load state dict into shadow model.
        """
        self.shadow.load_state_dict(state_dict["shadow"])
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]

    def to(self, device):
        """
        Move shadow model to device.
        """
        self.shadow.to(device)
        self.device = device
        return self


class EMAModelWithRampup(EMAModel):
    """
    EMA model with decay ramp-up schedule.

    Starts with a lower decay (faster updates) and gradually increases
    to the target decay. This helps the EMA model track the training
    model better during early training when weights change rapidly.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        rampup_steps: int = 1000,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: The model to track.
            decay: Final EMA decay rate.
            rampup_steps: Number of steps to ramp up from 0.5 to decay.
            device: Device to store shadow weights.
        """
        super().__init__(model, decay, device)
        self.target_decay = decay
        self.rampup_steps = rampup_steps
        self.initial_decay = 0.5

    def get_current_decay(self) -> float:
        """
        Get the current decay value based on ramp-up schedule.
        """
        if self.num_updates >= self.rampup_steps:
            return self.target_decay

        # Linear ramp-up from initial_decay to target_decay.
        progress = self.num_updates / self.rampup_steps
        return (
            self.initial_decay
            + (self.target_decay - self.initial_decay) * progress
        )

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update with ramped decay.
        """
        self.num_updates += 1
        decay = self.get_current_decay()

        for shadow_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            if self.device is not None:
                model_param_data = model_param.data.to(self.device)
            else:
                model_param_data = model_param.data

            shadow_param.data.mul_(decay).add_(
                model_param_data, alpha=1 - decay
            )

        for shadow_buffer, model_buffer in zip(
            self.shadow.buffers(), model.buffers()
        ):
            shadow_buffer.copy_(model_buffer)


def create_ema_model(
    model: nn.Module,
    decay: float = 0.999,
    use_rampup: bool = True,
    rampup_steps: int = 1000,
    device: Optional[torch.device] = None,
) -> EMAModel:
    """
    Factory function to create an EMA model.

    Args:
        model: The model to track.
        decay: EMA decay rate.
        use_rampup: Whether to use decay ramp-up.
        rampup_steps: Number of ramp-up steps (if use_rampup=True).
        device: Device for shadow model.

    Returns:
        EMAModel or EMAModelWithRampup instance
    """
    if use_rampup:
        return EMAModelWithRampup(model, decay, rampup_steps, device)
    else:
        return EMAModel(model, decay, device)


# =============================================================================
# Testing
# =============================================================================


def test_ema_model():
    """
    Test EMA model functionality.
    """

    # Simple test model.
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )

    # Create EMA.
    ema = create_ema_model(
        model, decay=0.99, use_rampup=True, rampup_steps=100
    )

    # Simulate training updates.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for step in range(200):
        # Fake training step.
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update EMA.
        ema.update(model)

        if step % 50 == 0:
            # Compare outputs.
            with torch.no_grad():
                test_x = torch.randn(1, 10)
                model_out = model(test_x)
                ema_out = ema(test_x)

                if isinstance(ema, EMAModelWithRampup):
                    print(
                        f"Step {step}: decay={ema.get_current_decay():.4f}, "
                        f"model={model_out.item():.3f}, ema={ema_out.item():.3f}"
                    )
                else:
                    print(
                        f"Step {step}: model={model_out.item():.3f}, "
                        f"ema={ema_out.item():.3f}"
                    )

    # Test save/load.
    state = ema.state_dict()
    ema2 = create_ema_model(model, decay=0.99)
    ema2.load_state_dict(state)

    print("\nEMA model test passed!")


if __name__ == "__main__":
    test_ema_model()
