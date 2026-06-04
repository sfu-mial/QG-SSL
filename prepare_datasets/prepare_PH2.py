"""
This script prepares the segmentations metadata for the PH2 dataset.
We read the train-val-test splits from the prediction split files.
These split files only contain the image names (and diag labels, but those
aren't used for this project).
After that, we resize all the images and masks to a fixed size and create
new directories for the resized images and masks.
"""

from pathlib import Path

import pandas as pd
from utils import resize_image_and_mask

# ------------------- Config -------------------
IMG_SIZE = 224

DATASET_NAME = "PH2"
SRC_DATASET_DIR = Path(
    "<PATH_TO_PH2_DATASET>/PH2_Dataset_images/"
)
SRC_SEGS_METADATA_FILE = Path(
    "<PATH_TO_PH2_DATASET>/PH2_Segmentations.csv"
)
SRC_PRED_SPLIT_FILES = {
    "train": Path(
        "<PATH_TO_PH2_DATASET>/DiagnosisLabels/train_GT.csv"
    ),
    "val": Path(
        "<PATH_TO_PH2_DATASET>/DiagnosisLabels/val_GT.csv"
    ),
    "test": Path(
        "<PATH_TO_PH2_DATASET>/DiagnosisLabels/test_GT.csv"
    ),
}


DST_META_OUTPUT_DIR = Path(f"./{DATASET_NAME}_segs_metadata")
DST_IMG_OUTPUT_DIR = Path(
    f"<PATH_TO_DESTINATION>/{DATASET_NAME}_resized_{IMG_SIZE}/imgs"
)
DST_SEG_OUTPUT_DIR = Path(
    f"<PATH_TO_DESTINATION>/{DATASET_NAME}_resized_{IMG_SIZE}/segs"
)

DST_META_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DST_IMG_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DST_SEG_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main():
    # Load the segmentations metadata.
    segs_df = pd.read_csv(SRC_SEGS_METADATA_FILE, header="infer", sep=",")

    # Load the splits from the prediction split files.
    pred_splits_df = {
        "train": pd.read_csv(
            SRC_PRED_SPLIT_FILES["train"], header=None, sep=","
        ),
        "val": pd.read_csv(SRC_PRED_SPLIT_FILES["val"], header=None, sep=","),
        "test": pd.read_csv(
            SRC_PRED_SPLIT_FILES["test"], header=None, sep=","
        ),
    }

    # Add column headers to the predictions' splits.
    for split in pred_splits_df:
        pred_splits_df[split].columns = ["img", "diag"]

    # Merge the predictions' splits with the segmentations metadata.
    merged_df = {}
    for split in pred_splits_df:
        merged_df[split] = pred_splits_df[split].merge(
            segs_df, left_on="img", right_on="img", how="left"
        )

        # Now, add back the "PH2_Dataset_images" prefix to the images.
        merged_df[split] = merged_df[split].assign(
            img=lambda x: SRC_DATASET_DIR / x["img"],
            seg=lambda x: SRC_DATASET_DIR / x["seg"],
        )

        # Resize the images and masks.
        merged_df[split][["img", "seg"]] = merged_df[split].apply(
            lambda x: resize_image_and_mask(
                x["img"],
                x["seg"],
                IMG_SIZE,
                SRC_DATASET_DIR,
                DST_IMG_OUTPUT_DIR,
                DST_SEG_OUTPUT_DIR,
            ),
            axis=1,
        )

    # Save the merged dataframes. Exclude the "diag" column.
    for split in merged_df:
        merged_df[split][["img", "seg"]] = merged_df[split][
            ["img", "seg"]
        ].apply(
            lambda x: x.astype(str).str.replace(
                "<PATH_TO_DESTINATION>/",
                "/dev/shm/myramdisk/",
                regex=False,
            )
        )
        merged_df[split] = merged_df[split].drop(columns=["diag"])
        # Rename columns to "image_path" and "mask_path".
        merged_df[split].columns = ["image_path", "mask_path"]
        merged_df[split].to_csv(
            DST_META_OUTPUT_DIR / f"{split}.csv", index=False
        )


if __name__ == "__main__":
    main()
