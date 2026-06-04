import os

import pandas as pd
from PIL import Image


def resize_image_and_mask(
    img_path: os.PathLike,
    mask_path: os.PathLike,
    size: int,
    src_dataset_dir: os.PathLike,
    img_out_dir: os.PathLike,
    mask_out_dir: os.PathLike,
):
    """
    Resize an image and its mask to a fixed size.

    Args:
        img_path: Path to the image file.
        mask_path: Path to the mask file.
        size: The size to resize the image and mask to.
        src_dataset_dir: The directory containing the source dataset.
        img_out_dir: The directory to save the resized image to.
        mask_out_dir: The directory to save the resized mask to.

    Returns:
        A pandas Series containing the paths to the resized image and mask.
    """
    img = Image.open(src_dataset_dir / img_path)
    mask = Image.open(src_dataset_dir / mask_path)

    img = img.resize((size, size))
    mask = mask.resize((size, size))

    img.save(img_out_dir / img_path.name)
    mask.save(mask_out_dir / mask_path.name)

    return pd.Series(
        [img_out_dir / img_path.name, mask_out_dir / mask_path.name]
    )
