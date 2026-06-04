"""
Corruption Operations for Variable Quality Mask Generation.

This module contains all corruption functions that degrade ground truth masks
to create training data for the Segmentation Quality Predictor (g_φ).

Each corruption function follows the signature:
    func(mask: np.ndarray, rng: np.random.RandomState, **kwargs) -> np.ndarray

Using a RandomState object (not global np.random) ensures reproducibility
and thread-safety for parallel data loading.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

# =============================================================================
# MORPHOLOGICAL OPERATIONS
# =============================================================================


def erosion(
    mask: np.ndarray, rng: np.random.RandomState, kernel_size: int = 7
) -> np.ndarray:
    """
    Erode the mask, simulating under-segmentation.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        kernel_size: Size of the erosion kernel.

    Returns:
        Eroded mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def dilation(
    mask: np.ndarray, rng: np.random.RandomState, kernel_size: int = 7
) -> np.ndarray:
    """
    Dilate the mask, simulating over-segmentation.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        kernel_size: Size of the dilation kernel.

    Returns:
        Dilated mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def random_morphology(
    mask: np.ndarray,
    rng: np.random.RandomState,
    kernel_range: Tuple[int, int] = (5, 11),
) -> np.ndarray:
    """
    Apply random erosion or dilation with random kernel size.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        kernel_range: Range of kernel sizes to choose from.

    Returns:
        Randomly eroded or dilated mask.
    """
    kernel_size = rng.randint(kernel_range[0], kernel_range[1] + 1)
    # Make kernel size odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    if rng.random() < 0.5:
        return erosion(mask, rng, kernel_size)
    else:
        return dilation(mask, rng, kernel_size)


def opening(
    mask: np.ndarray, rng: np.random.RandomState, kernel_size: int = 5
) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation).

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        kernel_size: Size of the opening kernel.

    Returns:
        Opened mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)


def closing(
    mask: np.ndarray, rng: np.random.RandomState, kernel_size: int = 5
) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion).

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        kernel_size: Size of the closing kernel.

    Returns:
        Closed mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


# =============================================================================
# SHAPE APPROXIMATIONS
# =============================================================================


def ellipse_approximation(
    mask: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """
    Approximate the mask with a fitted ellipse.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.

    Returns:
        Elliptical approximation of the mask.
    """
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros_like(mask)

    # Get the largest contour.
    largest_contour = max(contours, key=cv2.contourArea)

    # If the largest contour has fewer than 5 points, return an empty mask.
    if len(largest_contour) < 5:
        return np.zeros_like(mask)

    try:
        ellipse = cv2.fitEllipse(largest_contour)
        result = np.zeros_like(mask)
        cv2.ellipse(result, ellipse, 1, -1)
        return result
    except cv2.error:
        # fitEllipse can fail on degenerate cases.
        return np.zeros_like(mask)


def polygon_approximation(
    mask: np.ndarray, rng: np.random.RandomState, n_sides: int = 5
) -> np.ndarray:
    """
    Approximate the mask with an n-sided polygon.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        n_sides: Number of sides for the polygon approximation.

    Returns:
        Polygon approximation of the mask.
    """
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros_like(mask)

    # Get the largest contour.
    largest_contour = max(contours, key=cv2.contourArea)

    # Binary search for epsilon that gives approximately n_sides.
    eps_min, eps_max = 0, cv2.arcLength(largest_contour, True)
    best_poly = largest_contour

    # Perform binary search for the optimal epsilon.
    for _ in range(20):
        eps_mid = (eps_min + eps_max) / 2
        poly = cv2.approxPolyDP(largest_contour, eps_mid, True)

        if len(poly) > n_sides:
            eps_min = eps_mid
        elif len(poly) < n_sides:
            eps_max = eps_mid
        else:
            best_poly = poly
            break

        if abs(len(poly) - n_sides) < abs(len(best_poly) - n_sides):
            best_poly = poly

    result = np.zeros_like(mask)
    cv2.fillPoly(result, [best_poly], 1)
    return result


def convex_hull(mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Replace mask with its convex hull.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.

    Returns:
        Convex hull of the mask.
    """
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros_like(mask)

    # Combine all contours.
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)

    result = np.zeros_like(mask)
    cv2.fillPoly(result, [hull], 1)
    return result


def bounding_box_fill(
    mask: np.ndarray, rng: np.random.RandomState, padding_ratio: float = 0.0
) -> np.ndarray:
    """
    Replace mask with its bounding box (optionally with padding).

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        padding_ratio: Ratio of padding to add to the bounding box.

    Returns:
        Bounding box of the mask.
    """
    coords = np.where(mask > 0)

    if len(coords[0]) == 0:
        return np.zeros_like(mask)

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Apply padding.
    h, w = mask.shape
    pad_y = int((y_max - y_min) * padding_ratio)
    pad_x = int((x_max - x_min) * padding_ratio)

    y_min = max(0, y_min - pad_y)
    y_max = min(h - 1, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w - 1, x_max + pad_x)

    result = np.zeros_like(mask)
    result[y_min : y_max + 1, x_min : x_max + 1] = 1
    return result


# =============================================================================
# BOUNDARY PERTURBATIONS
# =============================================================================


def jagged_boundary(
    mask: np.ndarray,
    rng: np.random.RandomState,
    kernel_size: int = 5,
    noise_prob: float = 0.5,
) -> np.ndarray:
    """
    Add noise to the boundary of the mask.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        kernel_size: Size of the kernel used to detect the boundary.
        noise_prob: Probability of flipping a boundary pixel to 0.

    Returns:
        Mask with noisy boundaries.
    """
    mask_uint8 = mask.astype(np.uint8)

    # Find the boundary.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    boundary = dilated - eroded

    # Generate noise for boundary pixels.
    noise = (rng.random(mask.shape) > noise_prob).astype(np.uint8)

    result = mask_uint8.copy()
    result[boundary > 0] = noise[boundary > 0]
    return result


def smooth_boundary(
    mask: np.ndarray,
    rng: np.random.RandomState,
    blur_size: int = 15,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Smooth the mask boundary using Gaussian blur.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        blur_size: Size of the Gaussian kernel.
        threshold: Threshold to apply after blurring.

    Returns:
        Mask with smoothed boundaries.
    """
    mask_float = mask.astype(np.float32)

    # Ensure blur_size is odd.
    if blur_size % 2 == 0:
        blur_size += 1

    blurred = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    return (blurred > threshold).astype(np.uint8)


def elastic_deformation(
    mask: np.ndarray,
    rng: np.random.RandomState,
    alpha: float = 50,
    sigma: float = 5,
) -> np.ndarray:
    """
    Apply elastic deformation to the mask.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        alpha: Scale of the displacement fields.
        sigma: Smoothness of the displacement fields.

    Returns:
        Mask with elastic deformation.
    """
    shape = mask.shape

    # Generate random displacement fields.
    dx = rng.randn(*shape) * alpha
    dy = rng.randn(*shape) * alpha

    # Smooth the displacement fields.
    dx = cv2.GaussianBlur(dx.astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(dy.astype(np.float32), (0, 0), sigma)

    # Create coordinate grids.
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")

    # Apply displacement.
    new_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    new_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)

    # Remap.
    result = cv2.remap(
        mask.astype(np.float32), new_x, new_y, interpolation=cv2.INTER_NEAREST
    )
    return (result > 0.5).astype(np.uint8)


def grid_distortion(
    mask: np.ndarray,
    rng: np.random.RandomState,
    num_steps: int = 5,
    distort_limit: float = 0.3,
) -> np.ndarray:
    """
    Apply grid-based distortion to the mask.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        num_steps: Number of steps for the grid distortion.
        distort_limit: Limit of the distortion.

    Returns:
        Mask with grid-based distortion.
    """
    try:
        import albumentations as A

        transform = A.GridDistortion(
            p=1.0,
            num_steps=num_steps,
            distort_limit=distort_limit,
            border_mode=cv2.BORDER_CONSTANT,
        )
        # Albumentations needs HWC format.
        mask_3d = np.expand_dims(mask, axis=-1)
        result = transform(image=mask_3d)["image"]
        return result.squeeze()
    except ImportError:
        # Fallback to elastic deformation if albumentations not available.
        return elastic_deformation(mask, rng)


# =============================================================================
# STRUCTURAL CORRUPTIONS
# =============================================================================


def add_holes(
    mask: np.ndarray,
    rng: np.random.RandomState,
    num_holes: int = 3,
    radius_range: Tuple[int, int] = (5, 15),
) -> np.ndarray:
    """
    Add random circular holes to the foreground.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        num_holes: Number of holes to add.
        radius_range: Range of radii for the holes.

    Returns:
        Mask with added holes.
    """
    result = mask.astype(np.uint8).copy()

    # Get foreground pixel coordinates.
    fg_coords = np.where(mask > 0)
    if len(fg_coords[0]) == 0:
        return result

    for _ in range(num_holes):
        # Random foreground position.
        idx = rng.randint(0, len(fg_coords[0]))
        cy, cx = fg_coords[0][idx], fg_coords[1][idx]

        # Random radius.
        radius = rng.randint(radius_range[0], radius_range[1] + 1)

        # Draw black circle (hole).
        cv2.circle(result, (cx, cy), radius, 0, -1)

    return result


def add_islands(
    mask: np.ndarray,
    rng: np.random.RandomState,
    num_islands: int = 3,
    radius_range: Tuple[int, int] = (5, 15),
) -> np.ndarray:
    """
    Add random circular islands to the background.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        num_islands: Number of islands to add.
        radius_range: Range of radii for the islands.

    Returns:
        Mask with added islands.
    """
    result = mask.astype(np.uint8).copy()

    # Get background pixel coordinates.
    bg_coords = np.where(mask == 0)
    if len(bg_coords[0]) == 0:
        return result

    for _ in range(num_islands):
        # Random background position.
        idx = rng.randint(0, len(bg_coords[0]))
        cy, cx = bg_coords[0][idx], bg_coords[1][idx]

        # Random radius.
        radius = rng.randint(radius_range[0], radius_range[1] + 1)

        # Draw white circle (island).
        cv2.circle(result, (cx, cy), radius, 1, -1)

    return result


def holes_and_islands(
    mask: np.ndarray,
    rng: np.random.RandomState,
    num_holes: int = 3,
    num_islands: int = 3,
    radius_range: Tuple[int, int] = (5, 15),
) -> np.ndarray:
    """
    Add both holes and islands to the mask.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        num_holes: Number of holes to add.
        num_islands: Number of islands to add.
        radius_range: Range of radii for holes and islands.

    Returns:
        Mask with both holes and islands.
    """
    result = add_holes(mask, rng, num_holes, radius_range)
    result = add_islands(result, rng, num_islands, radius_range)
    return result


def keep_largest_component(
    mask: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """
    Keep only the largest connected component.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.

    Returns:
        Mask containing only the largest connected component.
    """
    mask_uint8 = mask.astype(np.uint8)

    # Find connected components.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

    if num_labels <= 1:
        return mask_uint8

    # Find largest component (excluding background at label 0).
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    return (labels == largest_label).astype(np.uint8)


def drop_random_component(
    mask: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """
    Randomly drop one connected component.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.

    Returns:
        Mask with one random component dropped.
    """
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

    if num_labels <= 2:  # Only background + 1 component.
        return mask_uint8

    # Randomly select a component to drop (excluding background).
    drop_label = rng.randint(1, num_labels)

    result = mask_uint8.copy()
    result[labels == drop_label] = 0
    return result


def random_rectangle_dropout(
    mask: np.ndarray,
    rng: np.random.RandomState,
    num_rects: int = 2,
    size_range: Tuple[float, float] = (0.1, 0.3),
) -> np.ndarray:
    """
    Drop random rectangular regions from the mask.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        num_rects: Number of rectangles to drop.
        size_range: Range of rectangle sizes (fraction of image).

    Returns:
        Mask with random rectangular regions dropped.
    """
    h, w = mask.shape
    result = mask.astype(np.uint8).copy()

    for _ in range(num_rects):
        # Random rectangle size (as fraction of image).
        rect_h = int(h * rng.uniform(size_range[0], size_range[1]))
        rect_w = int(w * rng.uniform(size_range[0], size_range[1]))

        # Random position.
        y = rng.randint(0, max(1, h - rect_h))
        x = rng.randint(0, max(1, w - rect_w))

        result[y : y + rect_h, x : x + rect_w] = 0

    return result


# =============================================================================
# GEOMETRIC CORRUPTIONS
# =============================================================================


def shift_mask(
    mask: np.ndarray, rng: np.random.RandomState, max_shift_ratio: float = 0.2
) -> np.ndarray:
    """
    Shift the mask by a random amount.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        max_shift_ratio: Maximum shift as a fraction of image size.

    Returns:
        Mask with random shift.
    """
    h, w = mask.shape

    shift_y = int(h * max_shift_ratio * (rng.random() * 2 - 1))
    shift_x = int(w * max_shift_ratio * (rng.random() * 2 - 1))

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(mask.astype(np.float32), M, (w, h)).astype(np.uint8)


def scale_mask(
    mask: np.ndarray,
    rng: np.random.RandomState,
    scale_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    Scale the mask from its center.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        scale_range: Range of scale factors.

    Returns:
        Mask with random scaling.
    """
    h, w = mask.shape

    scale = rng.uniform(scale_range[0], scale_range[1])

    # Get center of mass.
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return mask.astype(np.uint8)

    cy = coords[0].mean()
    cx = coords[1].mean()

    # Create transformation matrix (scale around center).
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)

    return cv2.warpAffine(mask.astype(np.float32), M, (w, h)).astype(np.uint8)


def rotate_mask(
    mask: np.ndarray,
    rng: np.random.RandomState,
    angle_range: Tuple[float, float] = (-30, 30),
) -> np.ndarray:
    """
    Rotate the mask around its center (without rotating the image).

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        angle_range: Range of rotation angles in degrees.

    Returns:
        Mask with random rotation.
    """
    h, w = mask.shape

    angle = rng.uniform(angle_range[0], angle_range[1])

    # Get center of mass.
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return mask.astype(np.uint8)

    cy = coords[0].mean()
    cx = coords[1].mean()

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    return cv2.warpAffine(mask.astype(np.float32), M, (w, h)).astype(np.uint8)


def random_affine(
    mask: np.ndarray,
    rng: np.random.RandomState,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    shift_range: float = 0.1,
    rotation_range: float = 15,
) -> np.ndarray:
    """
    Apply random affine transformation (scale, shift, rotation).

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        scale_range: Range of scale factors.
        shift_range: Maximum shift as a fraction of image size.
        rotation_range: Range of rotation angles in degrees.

    Returns:
        Mask with random affine transformation.
    """
    h, w = mask.shape

    # Random parameters.
    scale = rng.uniform(scale_range[0], scale_range[1])
    shift_y = int(h * shift_range * (rng.random() * 2 - 1))
    shift_x = int(w * shift_range * (rng.random() * 2 - 1))
    angle = rng.uniform(-rotation_range, rotation_range)

    # Get center.
    cy, cx = h / 2, w / 2

    # Combined transformation.
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += shift_x
    M[1, 2] += shift_y

    return cv2.warpAffine(mask.astype(np.float32), M, (w, h)).astype(np.uint8)


# =============================================================================
# BLOB APPROXIMATIONS (Simulates early f_θ predictions)
# =============================================================================


def gaussian_blob(
    mask: np.ndarray, rng: np.random.RandomState, size_factor: float = 1.0
) -> np.ndarray:
    """
    Replace mask with a Gaussian blob at the centroid.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        size_factor: Factor to control the size of the Gaussian blob.

    Returns:
        Mask approximated as a Gaussian blob.
    """
    h, w = mask.shape

    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return np.zeros_like(mask)

    # Centroid.
    cy = coords[0].mean()
    cx = coords[1].mean()

    # Estimate size from mask.
    std_y = coords[0].std() * size_factor
    std_x = coords[1].std() * size_factor

    # Create Gaussian blob.
    y_grid, x_grid = np.ogrid[:h, :w]
    gaussian = np.exp(
        -(
            (y_grid - cy) ** 2 / (2 * std_y**2 + 1e-6)
            + (x_grid - cx) ** 2 / (2 * std_x**2 + 1e-6)
        )
    )

    # Threshold to match approximate area.
    original_area = mask.sum()
    threshold = np.percentile(gaussian, 100 * (1 - original_area / (h * w)))

    return (gaussian > threshold).astype(np.uint8)


def circular_blob(
    mask: np.ndarray, rng: np.random.RandomState, size_factor: float = 1.0
) -> np.ndarray:
    """
    Replace mask with a circular blob at the centroid.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        size_factor: Factor to control the size of the circular blob.

    Returns:
        Mask approximated as a circular blob.
    """
    h, w = mask.shape

    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return np.zeros_like(mask)

    # Centroid.
    cy = int(coords[0].mean())
    cx = int(coords[1].mean())

    # Estimate radius from mask area.
    area = mask.sum()
    radius = int(np.sqrt(area / np.pi) * size_factor)

    result = np.zeros_like(mask)
    cv2.circle(result, (cx, cy), radius, 1, -1)
    return result


# =============================================================================
# CATASTROPHIC CORRUPTIONS
# =============================================================================


def invert_mask(mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Invert the mask (foreground <-> background).

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.

    Returns:
        Mask with inverted foreground and background.
    """
    return 1 - mask.astype(np.uint8)


def empty_mask(mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Return an empty mask (all zeros).

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.

    Returns:
        Empty mask.
    """
    return np.zeros_like(mask, dtype=np.uint8)


def full_mask(mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Return a full mask (all ones).

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.

    Returns:
        Full mask (all ones).
    """
    return np.ones_like(mask, dtype=np.uint8)


def random_mask(
    mask: np.ndarray, rng: np.random.RandomState, density: float = 0.5
) -> np.ndarray:
    """
    Return a random binary mask.

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.
        density: Density of the random mask.

    Returns:
        Random binary mask.
    """
    return (rng.random(mask.shape) < density).astype(np.uint8)


# =============================================================================
# IDENTITY (No corruption)
# =============================================================================


def identity(mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Return the mask unchanged. Dice = 1.0

    Args:
        mask: Ground truth mask (binary).
        rng: Random state generator.

    Returns:
        Original mask.
    """
    return mask.astype(np.uint8).copy()


# =============================================================================
# CORRUPTION REGISTRY
# =============================================================================

# All single corruptions with their default parameters and relative weights.
CORRUPTION_REGISTRY = {
    # Morphological (moderate quality degradation).
    "erosion_5": {
        "func": erosion,
        "params": {"kernel_size": 5},
        "weight": 1.0,
    },
    "erosion_7": {
        "func": erosion,
        "params": {"kernel_size": 7},
        "weight": 1.0,
    },
    "erosion_9": {
        "func": erosion,
        "params": {"kernel_size": 9},
        "weight": 0.8,
    },
    "dilation_5": {
        "func": dilation,
        "params": {"kernel_size": 5},
        "weight": 1.0,
    },
    "dilation_7": {
        "func": dilation,
        "params": {"kernel_size": 7},
        "weight": 1.0,
    },
    "dilation_9": {
        "func": dilation,
        "params": {"kernel_size": 9},
        "weight": 0.8,
    },
    "opening_5": {
        "func": opening,
        "params": {"kernel_size": 5},
        "weight": 0.5,
    },
    "closing_5": {
        "func": closing,
        "params": {"kernel_size": 5},
        "weight": 0.5,
    },
    # Shape approximations (variable quality).
    "ellipse": {"func": ellipse_approximation, "params": {}, "weight": 1.0},
    "polygon_3": {
        "func": polygon_approximation,
        "params": {"n_sides": 3},
        "weight": 0.5,
    },
    "polygon_5": {
        "func": polygon_approximation,
        "params": {"n_sides": 5},
        "weight": 0.8,
    },
    "polygon_8": {
        "func": polygon_approximation,
        "params": {"n_sides": 8},
        "weight": 0.8,
    },
    "convex_hull": {"func": convex_hull, "params": {}, "weight": 1.0},
    "bounding_box": {"func": bounding_box_fill, "params": {}, "weight": 0.5},
    # Boundary perturbations.
    "jagged_3": {
        "func": jagged_boundary,
        "params": {"kernel_size": 3},
        "weight": 1.0,
    },
    "jagged_5": {
        "func": jagged_boundary,
        "params": {"kernel_size": 5},
        "weight": 1.0,
    },
    "smooth_11": {
        "func": smooth_boundary,
        "params": {"blur_size": 11},
        "weight": 0.8,
    },
    "smooth_15": {
        "func": smooth_boundary,
        "params": {"blur_size": 15},
        "weight": 0.8,
    },
    "elastic_light": {
        "func": elastic_deformation,
        "params": {"alpha": 30, "sigma": 4},
        "weight": 1.0,
    },
    "elastic_heavy": {
        "func": elastic_deformation,
        "params": {"alpha": 60, "sigma": 5},
        "weight": 0.8,
    },
    "grid_light": {
        "func": grid_distortion,
        "params": {"num_steps": 5, "distort_limit": 0.2},
        "weight": 1.0,
    },
    "grid_heavy": {
        "func": grid_distortion,
        "params": {"num_steps": 8, "distort_limit": 0.4},
        "weight": 0.7,
    },
    # Structural corruptions.
    "holes_small": {
        "func": add_holes,
        "params": {"num_holes": 2, "radius_range": (3, 8)},
        "weight": 1.0,
    },
    "holes_large": {
        "func": add_holes,
        "params": {"num_holes": 3, "radius_range": (8, 15)},
        "weight": 0.8,
    },
    "islands_small": {
        "func": add_islands,
        "params": {"num_islands": 2, "radius_range": (3, 8)},
        "weight": 1.0,
    },
    "holes_and_islands": {
        "func": holes_and_islands,
        "params": {"num_holes": 2, "num_islands": 2},
        "weight": 1.0,
    },
    "largest_only": {
        "func": keep_largest_component,
        "params": {},
        "weight": 0.8,
    },
    "drop_component": {
        "func": drop_random_component,
        "params": {},
        "weight": 0.5,
    },
    "rect_dropout": {
        "func": random_rectangle_dropout,
        "params": {"num_rects": 1},
        "weight": 0.8,
    },
    # Geometric corruptions.
    "shift_small": {
        "func": shift_mask,
        "params": {"max_shift_ratio": 0.1},
        "weight": 1.0,
    },
    "shift_medium": {
        "func": shift_mask,
        "params": {"max_shift_ratio": 0.2},
        "weight": 0.8,
    },
    "shift_large": {
        "func": shift_mask,
        "params": {"max_shift_ratio": 0.3},
        "weight": 0.5,
    },
    "scale_down": {
        "func": scale_mask,
        "params": {"scale_range": (0.75, 0.9)},
        "weight": 1.0,
    },
    "scale_up": {
        "func": scale_mask,
        "params": {"scale_range": (1.1, 1.25)},
        "weight": 1.0,
    },
    "rotate_small": {
        "func": rotate_mask,
        "params": {"angle_range": (-15, 15)},
        "weight": 1.0,
    },
    "rotate_large": {
        "func": rotate_mask,
        "params": {"angle_range": (-30, 30)},
        "weight": 0.7,
    },
    "random_affine": {"func": random_affine, "params": {}, "weight": 0.8},
    # Blob approximations (simulates early f_θ).
    "gaussian_blob": {
        "func": gaussian_blob,
        "params": {"size_factor": 1.0},
        "weight": 1.0,
    },
    "gaussian_blob_large": {
        "func": gaussian_blob,
        "params": {"size_factor": 1.2},
        "weight": 0.7,
    },
    "circular_blob": {
        "func": circular_blob,
        "params": {"size_factor": 1.0},
        "weight": 0.8,
    },
}

# Heavy corruptions (lower quality results).
HEAVY_CORRUPTION_REGISTRY = {
    "erosion_11": {
        "func": erosion,
        "params": {"kernel_size": 11},
        "weight": 1.0,
    },
    "dilation_11": {
        "func": dilation,
        "params": {"kernel_size": 11},
        "weight": 1.0,
    },
    "shift_very_large": {
        "func": shift_mask,
        "params": {"max_shift_ratio": 0.4},
        "weight": 1.0,
    },
    "scale_very_small": {
        "func": scale_mask,
        "params": {"scale_range": (0.5, 0.7)},
        "weight": 1.0,
    },
    "scale_very_large": {
        "func": scale_mask,
        "params": {"scale_range": (1.4, 1.6)},
        "weight": 1.0,
    },
    "grid_very_heavy": {
        "func": grid_distortion,
        "params": {"num_steps": 12, "distort_limit": 0.6},
        "weight": 1.0,
    },
    "holes_many": {
        "func": add_holes,
        "params": {"num_holes": 5, "radius_range": (10, 20)},
        "weight": 1.0,
    },
}

# Catastrophic corruptions (very low or zero quality).
CATASTROPHIC_CORRUPTION_REGISTRY = {
    "invert": {"func": invert_mask, "params": {}, "weight": 1.0},
    "empty": {"func": empty_mask, "params": {}, "weight": 0.5},
    "full": {"func": full_mask, "params": {}, "weight": 0.5},
    "random_50": {
        "func": random_mask,
        "params": {"density": 0.5},
        "weight": 0.3,
    },
}


def apply_corruption(
    corruption_name: str, mask: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """Apply a named corruption from the registry."""
    # Check all registries.
    for registry in [
        CORRUPTION_REGISTRY,
        HEAVY_CORRUPTION_REGISTRY,
        CATASTROPHIC_CORRUPTION_REGISTRY,
    ]:
        if corruption_name in registry:
            entry = registry[corruption_name]
            return entry["func"](mask, rng, **entry["params"])

    # Special case: identity.
    if corruption_name == "identity":
        return identity(mask, rng)

    raise ValueError(f"Unknown corruption: {corruption_name}")


def get_all_corruption_names() -> list:
    """Get names of all available corruptions."""
    names = ["identity"]
    names.extend(CORRUPTION_REGISTRY.keys())
    names.extend(HEAVY_CORRUPTION_REGISTRY.keys())
    names.extend(CATASTROPHIC_CORRUPTION_REGISTRY.keys())
    return names
