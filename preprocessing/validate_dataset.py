
import os
from pathlib import Path
from PIL import Image


PROJECT = Path(
    "/content/drive/MyDrive/ResearchProject"
)

IMAGE_DIR = (
    PROJECT
    / "dataset"
    / "images"
)

LABEL_DIR = (
    PROJECT
    / "dataset"
    / "labels"
)


valid_extensions = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG"

}


images = [

    f

    for f in IMAGE_DIR.iterdir()

    if f.is_file()
    and f.suffix in valid_extensions

]


print("=" * 60)

print(
    "DATASET VALIDATION"
)

print("=" * 60)


print(
    "Total images:",
    len(images)
)


missing_labels = []

invalid_labels = []

corrupted_images = []

total_boxes = 0


for image_path in images:


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    try:

        with Image.open(
            image_path
        ) as img:

            img.verify()


    except Exception:

        corrupted_images.append(
            image_path.name
        )

        continue


    # ========================================================
    # FIND LABEL
    # ========================================================

    label_path = (

        LABEL_DIR
        / (
            image_path.stem
            + ".txt"
        )

    )


    if not label_path.exists():

        missing_labels.append(
            image_path.name
        )

        continue


    # ========================================================
    # VALIDATE YOLO LABEL
    # ========================================================

    with open(
        label_path,
        "r"
    ) as f:

        lines = f.readlines()


    for line_number, line in enumerate(
        lines,
        1
    ):


        values = (
            line.strip().split()
        )


        # YOLO format = 5 values

        if len(values) != 5:

            invalid_labels.append(

                f"{image_path.name}: "
                f"line {line_number}"

            )

            continue


        try:

            class_id = int(
                values[0]
            )

            x = float(
                values[1]
            )

            y = float(
                values[2]
            )

            width = float(
                values[3]
            )

            height = float(
                values[4]
            )


            # Coordinates must be normalized

            if not (

                0 <= x <= 1
                and
                0 <= y <= 1
                and
                0 <= width <= 1
                and
                0 <= height <= 1

            ):

                invalid_labels.append(

                    f"{image_path.name}: "
                    f"line {line_number}"

                )

                continue


            total_boxes += 1


        except ValueError:

            invalid_labels.append(

                f"{image_path.name}: "
                f"line {line_number}"

            )


# ============================================================
# RESULTS
# ============================================================

print(
    "\nImages:",
    len(images)
)

print(
    "Total bounding boxes:",
    total_boxes
)


print(
    "\nMissing labels:",
    len(missing_labels)
)

print(
    "Invalid labels:",
    len(invalid_labels)
)

print(
    "Corrupted images:",
    len(corrupted_images)
)


# ============================================================
# DETAILS
# ============================================================

if missing_labels:

    print(
        "\nMissing label examples:"
    )

    for item in missing_labels[:10]:

        print(
            " ",
            item
        )


if invalid_labels:

    print(
        "\nInvalid label examples:"
    )

    for item in invalid_labels[:10]:

        print(
            " ",
            item
        )


if corrupted_images:

    print(
        "\nCorrupted image examples:"
    )

    for item in corrupted_images[:10]:

        print(
            " ",
            item
        )


# ============================================================
# FINAL STATUS
# ============================================================

if (

    len(missing_labels) == 0
    and
    len(invalid_labels) == 0
    and
    len(corrupted_images) == 0

):

    print(
        "\n✅ DATASET VALIDATION PASSED"
    )

else:

    print(
        "\n⚠️ DATASET HAS PROBLEMS"
    )
