"""
Resize unlabaled colonoscopy images to 224x224, and write them to disk.
"""

import concurrent.futures
import os
from pathlib import Path

import cv2

# ------------------- Config -------------------
RESIZE_DIM = (224, 224)

SRC_DATASET_DIR = Path(
    "<PATH_TO_DATASET>/all_polyp_images_combined/"
)

DEST_DATASET_DIR = Path(
    f"<PATH_TO_DESTINATION>/colonoscopy_unlabeled_resized_{224}/"
)

# ------------------- Main -------------------

# Create the destination directory if it doesn't exist.
DEST_DATASET_DIR.mkdir(exist_ok=True, parents=True)


# Function to resize an image and write to disk.
def resize_and_write_image(img_path: os.PathLike, save_dir: os.PathLike):
    # Read the image.
    img = cv2.imread(str(img_path))

    if img is None:
        raise ValueError(f"Image {img_path} not found.")
        return

    # Resize the image.
    resized_img = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_CUBIC)

    # Save the image.
    save_path = save_dir / img_path.name
    cv2.imwrite(str(save_path), resized_img)


# Function to process all images in a directory.
def process_images_in_dir(src_dir: os.PathLike, dest_dir: os.PathLike):
    # Get all image paths in the directory.
    img_paths = list(src_dir.glob("**/*.jpg"))

    # Process all images in the directory in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(resize_and_write_image, img_path, dest_dir)
            for img_path in img_paths
        ]
        # Wait for all futures to complete.
        concurrent.futures.wait(futures)


def main():
    process_images_in_dir(SRC_DATASET_DIR, DEST_DATASET_DIR)


if __name__ == "__main__":
    main()
