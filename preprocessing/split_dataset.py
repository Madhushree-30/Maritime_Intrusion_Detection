
import random
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    "/content/drive/MyDrive/ResearchProject"
)


SOURCE_IMAGES = (
    PROJECT
    / "dataset"
    / "images"
)

SOURCE_LABELS = (
    PROJECT
    / "dataset"
    / "labels"
)


# ============================================================
# SPLIT SETTINGS
# ============================================================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.20

TEST_RATIO = 0.10

RANDOM_SEED = 42


random.seed(
    RANDOM_SEED
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    (
        SOURCE_IMAGES / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    (
        SOURCE_LABELS / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FIND IMAGES
# ============================================================

valid_extensions = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG"

}


image_files = [

    file

    for file in SOURCE_IMAGES.iterdir()

    if file.is_file()

    and file.suffix in valid_extensions

]


print(
    "Total images:",
    len(image_files)
)


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    image_files
)


total = len(
    image_files
)


train_end = int(
    total * TRAIN_RATIO
)

val_end = (
    train_end
    + int(total * VAL_RATIO)
)


train_files = (
    image_files[:train_end]
)

val_files = (
    image_files[
        train_end:val_end
    ]
)

test_files = (
    image_files[val_end:]
)


splits = {

    "train": train_files,

    "val": val_files,

    "test": test_files

}


# ============================================================
# COPY IMAGES + LABELS
# ============================================================

for split, files in splits.items():


    print(
        f"\nProcessing {split}..."
    )


    for image_file in files:


        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = (

            SOURCE_IMAGES
            / split
            / image_file.name

        )


        shutil.copy2(

            image_file,
            destination_image

        )


        # ----------------------------------------------------
        # Corresponding label
        # ----------------------------------------------------

        label_file = (

            SOURCE_LABELS
            / (
                image_file.stem
                + ".txt"
            )

        )


        if not label_file.exists():

            print(

                "WARNING: Missing label:",
                image_file.name

            )

            continue


        destination_label = (

            SOURCE_LABELS
            / split
            / label_file.name

        )


        shutil.copy2(

            label_file,
            destination_label

        )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "DATASET SPLIT COMPLETED"
)

print(
    "=" * 60
)


for split, files in splits.items():


    image_count = len(

        list(

            (
                SOURCE_IMAGES
                / split
            ).glob("*")

        )

    )


    label_count = len(

        list(

            (
                SOURCE_LABELS
                / split
            ).glob("*.txt")

        )

    )


    print(

        f"{split.upper():10} "
        f"Images: {image_count:4} | "
        f"Labels: {label_count:4}"

    )


print(
    "\nRandom seed:",
    RANDOM_SEED
)
