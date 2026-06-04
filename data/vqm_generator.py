"""
Variable Quality Mask (VQM) Generator.

This module provides on-the-fly generation of corrupted masks with computed
quality scores for training the Segmentation Quality Predictor (g_φ).

Key features:
- Configurable corruption distribution.
- Cross-image swap for forcing image-dependence.
- Chained corruptions for complex degradations.
- Thread-safe random number generation.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .corruption_ops import (
    CATASTROPHIC_CORRUPTION_REGISTRY,
    CORRUPTION_REGISTRY,
    HEAVY_CORRUPTION_REGISTRY,
    apply_corruption,
    identity,
)


def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        pred: Predicted/corrupted mask.
        target: Ground truth mask.

    Returns:
        Dice coefficient in [0, 1].
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()

    if union == 0:
        return 1.0  # Both empty

    return 2.0 * intersection / union


@dataclass
class CorruptionConfig:
    """
    Configuration for corruption sampling distribution.
    """

    # Probability of each corruption category.
    identity_prob: float = 0.12  # Dice = 1.0 anchor.
    single_transform_prob: float = 0.28  # Single corruption.
    chained_transform_prob: float = 0.18  # 2-3 chained corruptions.
    cross_image_prob: float = 0.18  # Mask from different image.
    heavy_prob: float = 0.12  # Heavy corruption.
    catastrophic_prob: float = 0.05  # Very severe corruption.
    blob_prob: float = 0.07  # Blob approximations.

    # Chaining parameters.
    chain_length_range: Tuple[int, int] = (2, 3)

    # Cross-image swap parameters.
    prefer_similar_swap: bool = True  # Prefer masks with similar area.

    def __post_init__(self):
        """
        Validate that probabilities sum to 1.
        """
        total = (
            self.identity_prob
            + self.single_transform_prob
            + self.chained_transform_prob
            + self.cross_image_prob
            + self.heavy_prob
            + self.catastrophic_prob
            + self.blob_prob
        )

        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Corruption probabilities must sum to 1.0, got {total}"
            )


class VQMGenerator:
    """
    On-the-fly Variable Quality Mask generator.

    This class generates corrupted versions of ground truth masks along with
    their quality scores (Dice coefficient) for training g_φ.
    """

    def __init__(
        self,
        config: Optional[CorruptionConfig] = None,
        gt_masks: Optional[List[np.ndarray]] = None,
        gt_mask_areas: Optional[np.ndarray] = None,
    ):
        """
        Initialize the VQM generator.

        Args:
            config: Corruption distribution configuration.
            gt_masks: List of all ground truth masks (for cross-image swap).
            gt_mask_areas: Precomputed areas of GT masks (for efficient similar-swap).
        """
        self.config = config or CorruptionConfig()
        self.gt_masks = gt_masks
        self.gt_mask_areas = gt_mask_areas

        # Build corruption pools.
        self._build_corruption_pools()

        # Build category sampling distribution.
        self._build_category_distribution()

    def _build_corruption_pools(self):
        """
        Build lists of corruption names for each category.
        """
        # Single transforms (moderate).
        self.single_pool = list(CORRUPTION_REGISTRY.keys())

        # Heavy transforms.
        self.heavy_pool = list(HEAVY_CORRUPTION_REGISTRY.keys())

        # Catastrophic transforms.
        self.catastrophic_pool = list(CATASTROPHIC_CORRUPTION_REGISTRY.keys())

        # Blob approximations (subset of single).
        self.blob_pool = [
            "gaussian_blob",
            "gaussian_blob_large",
            "circular_blob",
            "convex_hull",
            "bounding_box",
        ]

        # Transforms suitable for chaining (avoid catastrophic).
        self.chainable_pool = [
            name
            for name in self.single_pool
            if name not in self.blob_pool  # Blobs shouldn't be chained.
        ]

    def _build_category_distribution(self):
        """
        Build the categorical distribution for sampling corruption types.
        """
        self.categories = [
            "identity",
            "single",
            "chained",
            "cross_image",
            "heavy",
            "catastrophic",
            "blob",
        ]

        self.category_probs = np.array(
            [
                self.config.identity_prob,
                self.config.single_transform_prob,
                self.config.chained_transform_prob,
                self.config.cross_image_prob,
                self.config.heavy_prob,
                self.config.catastrophic_prob,
                self.config.blob_prob,
            ]
        )

    def set_gt_masks(self, gt_masks: List[np.ndarray]):
        """
        Set the ground truth masks for cross-image swap.

        This should be called after initialization if gt_masks weren't provided,
        or to update the mask pool.
        """
        self.gt_masks = gt_masks
        self.gt_mask_areas = np.array([m.sum() for m in gt_masks])

    def _sample_category(self, rng: np.random.RandomState) -> str:
        """
        Sample a corruption category.
        """
        return rng.choice(self.categories, p=self.category_probs)

    def _sample_single_corruption(self, rng: np.random.RandomState) -> str:
        """
        Sample a single corruption name.
        """
        # Weight-based sampling.
        weights = np.array(
            [CORRUPTION_REGISTRY[name]["weight"] for name in self.single_pool]
        )
        weights = weights / weights.sum()
        return rng.choice(self.single_pool, p=weights)

    def _sample_heavy_corruption(self, rng: np.random.RandomState) -> str:
        """
        Sample a heavy corruption name.
        """
        weights = np.array(
            [
                HEAVY_CORRUPTION_REGISTRY[name]["weight"]
                for name in self.heavy_pool
            ]
        )
        weights = weights / weights.sum()
        return rng.choice(self.heavy_pool, p=weights)

    def _sample_catastrophic_corruption(
        self, rng: np.random.RandomState
    ) -> str:
        """
        Sample a catastrophic corruption name.
        """
        weights = np.array(
            [
                CATASTROPHIC_CORRUPTION_REGISTRY[name]["weight"]
                for name in self.catastrophic_pool
            ]
        )
        weights = weights / weights.sum()
        return rng.choice(self.catastrophic_pool, p=weights)

    def _sample_blob_corruption(self, rng: np.random.RandomState) -> str:
        """
        Sample a blob approximation corruption."""
        return rng.choice(self.blob_pool)

    def _apply_chained_corruptions(
        self, mask: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, str]:
        """
        Apply a chain of 2-3 corruptions.
        """
        chain_length = rng.randint(
            self.config.chain_length_range[0],
            self.config.chain_length_range[1] + 1,
        )

        # Sample unique corruptions for the chain.
        corruption_names = rng.choice(
            self.chainable_pool,
            size=min(chain_length, len(self.chainable_pool)),
            replace=False,
        ).tolist()

        # Apply corruptions sequentially.
        result = mask.copy()
        for name in corruption_names:
            result = apply_corruption(name, result, rng)

        chain_name = "chain:" + "-".join(corruption_names)
        return result, chain_name

    def _apply_cross_image_swap(
        self,
        current_idx: int,
        current_mask: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, str, float]:
        """
        Swap with a mask from a different image.

        Args:
            current_idx: Index of the current mask.
            current_mask: Current mask.
            rng: Random state.

        Returns:
            Tuple of (swapped_mask, corruption_name, quality_score)
        """
        if self.gt_masks is None or len(self.gt_masks) < 2:
            # Fallback to identity if no masks available.
            return current_mask.copy(), "identity_fallback", 1.0

        n_masks = len(self.gt_masks)

        if self.config.prefer_similar_swap and self.gt_mask_areas is not None:
            # Prefer masks with similar area (harder to distinguish).
            current_area = current_mask.sum()
            area_diff = np.abs(
                self.gt_mask_areas.astype(np.float64) - current_area
            )

            # Exclude self.
            area_diff[current_idx] = np.inf

            # Sample from top 20% most similar (by area).
            k = max(1, n_masks // 5)
            similar_indices = np.argsort(area_diff)[:k]
            swap_idx = rng.choice(similar_indices)
        else:
            # Random swap (excluding self).
            valid_indices = [i for i in range(n_masks) if i != current_idx]
            swap_idx = rng.choice(valid_indices)

        swapped_mask = self.gt_masks[swap_idx].copy()

        # Resize if shapes don't match.
        if swapped_mask.shape != current_mask.shape:
            import cv2

            swapped_mask = cv2.resize(
                swapped_mask.astype(np.uint8),
                (current_mask.shape[1], current_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Compute quality (Dice between original GT and swapped mask).
        quality = dice_coefficient(swapped_mask, current_mask)

        return swapped_mask, f"cross_swap_idx{swap_idx}", quality

    def __call__(
        self,
        gt_mask: np.ndarray,
        idx: int,
        rng: Optional[np.random.RandomState] = None,
        force_category: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Generate a corrupted mask with quality score.

        Args:
            gt_mask: Ground truth binary mask.
            idx: Index of current sample (for cross-image exclusion).
            rng: Random state for reproducibility.
            force_category: Force a specific corruption category (for testing).

        Returns:
            Tuple of:
                - corrupted_mask: The degraded mask (np.ndarray)
                - quality_score: Dice(gt_mask, corrupted_mask) (float)
                - corruption_name: Identifier for the corruption applied (str)
        """
        if rng is None:
            rng = np.random.RandomState()

        # Sample corruption category.
        category = force_category or self._sample_category(rng)

        # Apply corruption based on category.
        if category == "identity":
            corrupted_mask = identity(gt_mask, rng)
            corruption_name = "identity"

        elif category == "single":
            corruption_name = self._sample_single_corruption(rng)
            corrupted_mask = apply_corruption(corruption_name, gt_mask, rng)

        elif category == "chained":
            corrupted_mask, corruption_name = self._apply_chained_corruptions(
                gt_mask, rng
            )

        elif category == "cross_image":
            corrupted_mask, corruption_name, quality = (
                self._apply_cross_image_swap(idx, gt_mask, rng)
            )
            # Return early since quality is already computed.
            return corrupted_mask, quality, corruption_name

        elif category == "heavy":
            corruption_name = self._sample_heavy_corruption(rng)
            corrupted_mask = apply_corruption(corruption_name, gt_mask, rng)

        elif category == "catastrophic":
            corruption_name = self._sample_catastrophic_corruption(rng)
            corrupted_mask = apply_corruption(corruption_name, gt_mask, rng)

        elif category == "blob":
            corruption_name = self._sample_blob_corruption(rng)
            corrupted_mask = apply_corruption(corruption_name, gt_mask, rng)

        else:
            raise ValueError(f"Unknown category: {category}")

        # Compute quality score.
        quality_score = dice_coefficient(corrupted_mask, gt_mask)

        return corrupted_mask, quality_score, corruption_name

    def generate_multiple(
        self,
        gt_mask: np.ndarray,
        idx: int,
        n_samples: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate multiple corrupted versions of a mask.

        Useful for creating diverse training batches from a single GT mask.

        Args:
            gt_mask: Ground truth binary mask.
            idx: Index of current sample (for cross-image exclusion).
            n_samples: Number of corrupted masks to generate.
            rng: Random state for reproducibility.

        Returns:
            List of tuples: (corrupted_mask, quality_score, corruption_name)
        """
        if rng is None:
            rng = np.random.RandomState()

        return [self(gt_mask, idx, rng) for _ in range(n_samples)]


def test_vqm_generator():
    """
    Test function to verify the VQM generator works correctly.
    """
    import matplotlib.pyplot as plt

    # Create a simple test mask.
    mask = np.zeros((224, 224), dtype=np.uint8)
    cv2.circle(mask, (112, 112), 50, 1, -1)

    # Create generator with dummy GT masks.
    gt_masks = [mask.copy() for _ in range(10)]
    # Create some variation in GT masks.
    for i, m in enumerate(gt_masks):
        gt_masks[i] = np.roll(m, i * 10, axis=0)

    config = CorruptionConfig()
    generator = VQMGenerator(config, gt_masks)

    # Generate samples and collect quality distribution.
    rng = np.random.RandomState(42)
    qualities = []
    categories_used = []

    for i in range(500):
        _, quality, name = generator(mask, idx=0, rng=rng)
        qualities.append(quality)
        # Extract category from corruption name.
        if name == "identity":
            categories_used.append("identity")
        elif name.startswith("chain:"):
            categories_used.append("chained")
        elif name.startswith("cross_swap"):
            categories_used.append("cross_image")
        else:
            categories_used.append("single/heavy/catastrophic/blob")

    # Print statistics.
    print(f"Quality distribution:")
    print(f"  Mean: {np.mean(qualities):.3f}")
    print(f"  Std:  {np.std(qualities):.3f}")
    print(f"  Min:  {np.min(qualities):.3f}")
    print(f"  Max:  {np.max(qualities):.3f}")

    # Histogram of qualities
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(qualities, bins=20, edgecolor="black")
    plt.xlabel("Quality Score (Dice)")
    plt.ylabel("Count")
    plt.title("Quality Distribution (500 samples)")

    plt.subplot(1, 2, 2)
    from collections import Counter

    cat_counts = Counter(categories_used)
    plt.bar(cat_counts.keys(), cat_counts.values())
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Corruption Category Distribution")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("vqm_test_distribution.png", dpi=100)
    print("Saved distribution plot to vqm_test_distribution.png")


if __name__ == "__main__":
    import cv2

    test_vqm_generator()
