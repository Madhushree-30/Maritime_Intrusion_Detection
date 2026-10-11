
import os
import json
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT = Path("/content/drive/MyDrive/ResearchProject")

# Dataset remains in KaggleHub cache
RAW_DIR = Path(
    "/root/.cache/kagglehub/datasets/"
    "sarribere99/high-resolution-sar-images-dataset-hrsid/"
    "versions/1"
)

OUTPUT_IMAGES = PROJECT / "dataset" / "images"
OUTPUT_LABELS = PROJECT / "dataset" / "labels"


# Create output directories

OUTPUT_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_LABELS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND COCO ANNOTATION FILE
# ============================================================

json_files = list(
    RAW_DIR.rglob("*.json")
)

if not json_files:

    raise FileNotFoundError(
        "No COCO JSON annotation file found "
        "inside downloaded dataset."
    )


print("JSON files found:")

for file in json_files:

    print("  ", file)


# ============================================================
# USE FIRST JSON FILE
# ============================================================

ANNOTATION_FILE = json_files[0]

print("\nUsing annotation file:")
print(ANNOTATION_FILE)


# ============================================================
# LOAD COCO DATA
# ============================================================

with open(
    ANNOTATION_FILE,
    "r"
) as f:

    coco = json.load(f)


print("\nCOCO dataset information")
print("------------------------")

print(
    "Images:",
    len(coco.get("images", []))
)

print(
    "Annotations:",
    len(coco.get("annotations", []))
)

print(
    "Categories:",
    len(coco.get("categories", []))
)


# ============================================================
# CATEGORY MAPPING
# ============================================================

categories = sorted(
    coco["categories"],
    key=lambda x: x["id"]
)


category_mapping = {

    category["id"]: index

    for index, category
    in enumerate(categories)

}


print("\nCategory Mapping")
print("----------------")


for category in categories:

    print(

        f"COCO ID {category['id']} "
        f"-> YOLO ID "
        f"{category_mapping[category['id']]} "
        f"-> {category['name']}"

    )


# ============================================================
# IMAGE INFORMATION
# ============================================================

images = {

    image["id"]: image

    for image in coco["images"]

}


# ============================================================
# GROUP ANNOTATIONS BY IMAGE
# ============================================================

annotations_by_image = {}


for annotation in coco["annotations"]:

    image_id = annotation["image_id"]


    if image_id not in annotations_by_image:

        annotations_by_image[image_id] = []


    annotations_by_image[
        image_id
    ].append(annotation)


# ============================================================
# FIND ALL IMAGES
# ============================================================

image_files = {}


valid_extensions = (

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG"

)


for image_path in RAW_DIR.rglob("*"):

    if image_path.is_file():

        if image_path.suffix in valid_extensions:

            image_files[
                image_path.name
            ] = image_path


print(
    "\nImages found in downloaded dataset:",
    len(image_files)
)


# ============================================================
# COCO → YOLO CONVERSION
# ============================================================

converted = 0

missing_images = 0

total_boxes = 0


for image_id, image_info in images.items():


    filename = os.path.basename(
        image_info["file_name"]
    )


    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    if filename not in image_files:

        print(
            "Missing image:",
            filename
        )

        missing_images += 1

        continue


    source_image = image_files[
        filename
    ]


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    image_width = image_info[
        "width"
    ]

    image_height = image_info[
        "height"
    ]


    # --------------------------------------------------------
    # Copy processed image to Drive
    # --------------------------------------------------------

    destination_image = (
        OUTPUT_IMAGES / filename
    )


    shutil.copy2(

        source_image,
        destination_image

    )


    # --------------------------------------------------------
    # Create YOLO label file
    # --------------------------------------------------------

    label_filename = (

        Path(filename).stem
        + ".txt"

    )


    destination_label = (

        OUTPUT_LABELS
        / label_filename

    )


    with open(

        destination_label,
        "w"

    ) as label_file:


        annotations = (

            annotations_by_image.get(
                image_id,
                []
            )

        )


        for annotation in annotations:


            category_id = (
                annotation["category_id"]
            )


            if category_id not in category_mapping:

                continue


            class_id = (
                category_mapping[
                    category_id
                ]
            )


            # ------------------------------------------------
            # COCO bounding box
            #
            # x_min
            # y_min
            # width
            # height
            # ------------------------------------------------

            (
                x_min,
                y_min,
                box_width,
                box_height
            ) = annotation["bbox"]


            # ------------------------------------------------
            # Convert to center coordinates
            # ------------------------------------------------

            center_x = (

                x_min
                + box_width / 2

            )


            center_y = (

                y_min
                + box_height / 2

            )


            # ------------------------------------------------
            # Normalize coordinates
            # ------------------------------------------------

            center_x = (
                center_x
                / image_width
            )

            center_y = (
                center_y
                / image_height
            )

            box_width = (
                box_width
                / image_width
            )

            box_height = (
                box_height
                / image_height
            )


            # ------------------------------------------------
            # Keep values within valid range
            # ------------------------------------------------

            center_x = max(
                0,
                min(1, center_x)
            )

            center_y = max(
                0,
                min(1, center_y)
            )

            box_width = max(
                0,
                min(1, box_width)
            )

            box_height = max(
                0,
                min(1, box_height)
            )


            # ------------------------------------------------
            # Write YOLO format
            # ------------------------------------------------

            label_file.write(

                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"

            )


            total_boxes += 1


    converted += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n")

print("=" * 60)

print(
    "COCO → YOLO PREPROCESSING COMPLETED"
)

print("=" * 60)


print(
    "Images converted :",
    converted
)

print(
    "Images missing   :",
    missing_images
)

print(
    "Bounding boxes   :",
    total_boxes
)


print("\nOutput images:")

print(
    OUTPUT_IMAGES
)


print("\nOutput labels:")

print(
    OUTPUT_LABELS
)


print(
    "\nRaw dataset was NOT saved to Google Drive."
)

print("\nDone!")
