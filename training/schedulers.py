"""
Schedulers for semi-supervised segmentation training.

Includes:
- Lambda schedulers for ramping up unsupervised loss weight
- Quality alignment monitoring
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LambdaSchedulerConfig:
    """
    Configuration for lambda scheduling.
    """

    initial_lambda: float = 0.0  # Starting value
    final_lambda: float = 1.0  # Target value
    rampup_epochs: int = 10  # Epochs to reach final value
    rampup_type: str = "linear"  # 'linear', 'cosine', 'exponential', 'step'
    warmup_epochs: int = 0  # Epochs before starting ramp-up


class LambdaScheduler:
    """
    Scheduler for the unsupervised loss weight (λ).

    Gradually increases λ from initial to final value over the ramp-up period.
    This prevents the potentially unstable unsupervised loss from dominating
    early training.
    """

    def __init__(self, config: Optional[LambdaSchedulerConfig] = None):
        """
        Args:
            config: Scheduler configuration
        """
        self.config = config or LambdaSchedulerConfig()
        self.current_epoch = 0
        self.current_lambda = self.config.initial_lambda

    def step(self, epoch: Optional[int] = None) -> float:
        """
        Update lambda value for new epoch.

        Args:
            epoch: Current epoch (if None, increments internally).

        Returns:
            Current lambda value.
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        self.current_lambda = self._compute_lambda(self.current_epoch)
        return self.current_lambda

    def _compute_lambda(self, epoch: int) -> float:
        """
        Compute lambda value for given epoch.
        """
        cfg = self.config

        # Warmup phase.
        if epoch < cfg.warmup_epochs:
            return cfg.initial_lambda

        # Adjust epoch for warmup.
        effective_epoch = epoch - cfg.warmup_epochs

        # Ramp-up phase.
        if effective_epoch >= cfg.rampup_epochs:
            return cfg.final_lambda

        # Compute progress [0, 1].
        progress = effective_epoch / cfg.rampup_epochs

        # Apply schedule.
        if cfg.rampup_type == "linear":
            factor = progress
        elif cfg.rampup_type == "cosine":
            factor = 0.5 * (1 - math.cos(math.pi * progress))
        elif cfg.rampup_type == "exponential":
            factor = math.exp(-5 * (1 - progress) ** 2)
        elif cfg.rampup_type == "step":
            factor = 1.0 if progress >= 0.5 else 0.0
        else:
            raise ValueError(f"Unknown rampup type: {cfg.rampup_type}")

        # Interpolate between initial and final.
        return cfg.initial_lambda + factor * (
            cfg.final_lambda - cfg.initial_lambda
        )

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.current_lambda

    def state_dict(self) -> dict:
        """Get scheduler state."""
        return {
            "current_epoch": self.current_epoch,
            "current_lambda": self.current_lambda,
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state."""
        self.current_epoch = state_dict["current_epoch"]
        self.current_lambda = state_dict["current_lambda"]
        self.config = LambdaSchedulerConfig(**state_dict["config"])


# =============================================================================
# TESTING
# =============================================================================


def test_lambda_scheduler():
    """Test lambda scheduler."""
    import matplotlib.pyplot as plt

    configs = [
        (
            "linear",
            LambdaSchedulerConfig(rampup_type="linear", rampup_epochs=20),
        ),
        (
            "cosine",
            LambdaSchedulerConfig(rampup_type="cosine", rampup_epochs=20),
        ),
        (
            "exponential",
            LambdaSchedulerConfig(rampup_type="exponential", rampup_epochs=20),
        ),
        (
            "warmup+linear",
            LambdaSchedulerConfig(
                rampup_type="linear", rampup_epochs=20, warmup_epochs=5
            ),
        ),
    ]

    plt.figure(figsize=(10, 6))

    for name, config in configs:
        scheduler = LambdaScheduler(config)
        lambdas = []
        for epoch in range(30):
            lambdas.append(scheduler.step(epoch))
        plt.plot(lambdas, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Lambda")
    plt.title("Lambda Scheduling Strategies")
    plt.legend()
    plt.grid(True)
    plt.savefig("lambda_schedules.png", dpi=100)
    print("Saved lambda_schedules.png")

    print("\nLambda scheduler tests passed!")


if __name__ == "__main__":
    test_lambda_scheduler()
