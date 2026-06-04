import os
from pathlib import Path
from typing import List

import pandas as pd

UNLABELED_DATASET_DIR = Path(
    "<PATH_TO_DATASET>/colonoscopy_unlabeled_resized_224"
)
METADATA_PATH_PREFIX = "/dev/shm/myramdisk/colonoscopy_unlabeled_resized_224"

METADATA_FILE_PATH = "./colonoscopy_unlabeled_metadata.csv"


def list_image_paths(unlabeled_dataset_dir: Path) -> List[Path]:
    return list(unlabeled_dataset_dir.glob("**/*.jpg"))


def create_metadata(image_paths: List[Path]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "img_path": image_paths,
        }
    )
    df["img_path"] = df["img_path"].apply(
        lambda x: str(x).replace(
            str(UNLABELED_DATASET_DIR), METADATA_PATH_PREFIX
        )
    )
    df.to_csv(METADATA_FILE_PATH, index=False)
    return df


def main():
    image_paths = list_image_paths(UNLABELED_DATASET_DIR)
    df = create_metadata(image_paths)
    print(df.head())


if __name__ == "__main__":
    main()
