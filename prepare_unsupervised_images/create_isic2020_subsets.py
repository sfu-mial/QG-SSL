"""
Create subsets of ISIC2020 (train) images: 1k, 5k, 10k, 20k, 30k, stratified by malignancy.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------- Config -------------------
SUBSET_TRAIN_SIZES = {
    "1k": 1000,
    "5k": 5000,
    "10k": 10000,
    "20k": 20000,
    "30k": 30000,
    # Total before removing duplicates: 33126,
    # Total after removing duplicates: 32701,
}

SUBSET_VAL_SIZE = 1000
SUBSET_TEST_SIZE = 1701

ISIC2020_DATA_DIR = Path("<PATH_TO_DATASET>")

ISIC2020_TRAIN_METADATA_FILE = (
    ISIC2020_DATA_DIR / "ISIC_2020_Training_GroundTruth.csv"
)
# Duplicates file also sourced from ISIC: 
# https://challenge.isic-archive.com/data/#2020
ISIC2020_TRAIN_DUPLICATES_FILE = (
    ISIC2020_DATA_DIR / "ISIC_2020_Training_Duplicates.csv"
)

DEST_DIR = Path("./ISIC2020_train_subsets/")

TARGET_FILE_DIR = "/dev/shm/myramdisk/ISIC2020_resized_224/train/"

METADATA_STRATIFY_BY = "benign_malignant"

SEED = 8888


# ------------------- Main -------------------
def main():
    # Read the metadata.
    metadata_df = pd.read_csv(
        ISIC2020_TRAIN_METADATA_FILE, sep=",", header="infer"
    )
    print(f"Read {len(metadata_df)} rows of metadata.")

    # Read the duplicates.
    duplicates_df = pd.read_csv(
        ISIC2020_TRAIN_DUPLICATES_FILE, sep=",", header="infer"
    )
    print(f"Read {len(duplicates_df)} rows of duplicates.")

    # Remove the duplicates from the original metadata.
    # The `duplicates` file has 2 columns: `image_name_1` and `image_name_2`.
    # We will simply remove the rows from the metadata file where
    # `image_name` matches `image_2`.
    metadata_df = metadata_df[
        ~metadata_df["image_name"].isin(duplicates_df["image_name_2"])
    ]
    print(f"Removed {len(duplicates_df)} duplicates.")
    print(f"Remaining {len(metadata_df)} rows of metadata.")

    eval_df, remainder = train_test_split(
        metadata_df,
        train_size=SUBSET_VAL_SIZE + SUBSET_TEST_SIZE,
        stratify=metadata_df[METADATA_STRATIFY_BY],
        random_state=SEED,
    )

    val_df, test_df = train_test_split(
        eval_df,
        train_size=SUBSET_VAL_SIZE,
        stratify=eval_df[METADATA_STRATIFY_BY],
        random_state=SEED,
    )
    val_df.to_csv(DEST_DIR / "val.csv", index=False)
    test_df.to_csv(DEST_DIR / "test.csv", index=False)

    # Create subsets stratified by malignancy.
    for subset_size in SUBSET_TRAIN_SIZES:
        # From the remainder set, create a train set.
        if subset_size == "30k":
            train_df = remainder
        else:
            train_df, _ = train_test_split(
                remainder,
                train_size=SUBSET_TRAIN_SIZES[subset_size],
                stratify=remainder[METADATA_STRATIFY_BY],
                random_state=SEED,
            )
        # Append a column `img_path` to the dataframe.
        train_df["img_path"] = train_df["image_name"].apply(
            lambda x: f"{TARGET_FILE_DIR}{x}.jpg"
        )
        train_df.to_csv(
            DEST_DIR / f"train_subset_{subset_size}.csv", index=False
        )

    return


if __name__ == "__main__":
    # Create the destination directory if it doesn't exist.
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    main()
