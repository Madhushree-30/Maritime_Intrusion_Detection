
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF


# ============================================================
# PATHS
# ============================================================

PROJECT = Path("/content/drive/MyDrive/ResearchProject")

DATASET = PROJECT / "dataset"

TRAIN_IMAGES = DATASET / "images" / "train"
TRAIN_LABELS = DATASET / "labels" / "train"

VAL_IMAGES = DATASET / "images" / "val"
VAL_LABELS = DATASET / "labels" / "val"

RESULTS = PROJECT / "results" / "Faster-RCNN"
MODEL_DIR = PROJECT / "models" / "faster_rcnn"

RESULTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

class YOLODataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.images = []

        valid_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]

        for file in self.image_dir.iterdir():

            if file.suffix.lower() in valid_extensions:
                self.images.append(file)

        self.images.sort()

        print(
            f"Found {len(self.images)} images in {self.image_dir}"
        )


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        image_path = self.images[index]

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        label_path = (
            self.label_dir /
            f"{image_path.stem}.txt"
        )

        boxes = []
        labels = []

        if label_path.exists():

            with open(label_path, "r") as f:

                for line in f:

                    values = line.strip().split()

                    if len(values) != 5:
                        continue

                    class_id = int(values[0])

                    x_center = float(values[1])
                    y_center = float(values[2])
                    box_width = float(values[3])
                    box_height = float(values[4])

                    # YOLO normalized coordinates
                    x_center *= width
                    y_center *= height

                    box_width *= width
                    box_height *= height

                    xmin = x_center - box_width / 2
                    ymin = y_center - box_height / 2

                    xmax = x_center + box_width / 2
                    ymax = y_center + box_height / 2

                    # Keep coordinates inside image
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)

                    xmax = min(width, xmax)
                    ymax = min(height, ymax)

                    if xmax > xmin and ymax > ymin:

                        boxes.append(
                            [xmin, ymin, xmax, ymax]
                        )

                        # Faster R-CNN:
                        # 0 = background
                        # 1 = ship
                        labels.append(class_id + 1)


        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        if len(boxes) == 0:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )


        area = (
            (boxes[:, 3] - boxes[:, 1]) *
            (boxes[:, 2] - boxes[:, 0])
        )

        iscrowd = torch.zeros(
            (len(boxes),),
            dtype=torch.int64
        )


        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": torch.tensor(
                [index]
            ),

            "area": area,

            "iscrowd": iscrowd
        }


        image = TF.to_tensor(image)

        return image, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    return tuple(zip(*batch))


# ============================================================
# MODEL
# ============================================================

def create_model(num_classes):

    weights = (
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    in_features = (
        model.roi_heads.box_predictor.cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            num_classes
        )
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = YOLODataset(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    val_dataset = YOLODataset(
        VAL_IMAGES,
        VAL_LABELS
    )


    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    num_classes = 2

    model = create_model(
        num_classes
    )

    model.to(device)


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )


    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    epochs = 10

    best_loss = float("inf")


    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        print(
            f"\nEpoch "
            f"{epoch + 1}/{epochs}"
        )


        for batch_idx, (
            images,
            targets
        ) in enumerate(train_loader):

            images = [
                image.to(device)
                for image in images
            ]

            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                }
                for target in targets
            ]


            loss_dict = model(
                images,
                targets
            )

            losses = sum(
                loss
                for loss in loss_dict.values()
            )


            optimizer.zero_grad()

            losses.backward()

            optimizer.step()


            total_loss += losses.item()


            if (
                batch_idx + 1
            ) % 10 == 0:

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"Loss: "
                    f"{losses.item():.4f}"
                )


        epoch_loss = (
            total_loss /
            len(train_loader)
        )


        print(
            f"Epoch {epoch + 1} "
            f"Training Loss: "
            f"{epoch_loss:.4f}"
        )


        scheduler.step()


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            best_path = (
                MODEL_DIR /
                "best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "loss": epoch_loss
                },
                best_path
            )

            print(
                "✅ Best model saved:"
            )

            print(best_path)


        # ----------------------------------------------------
        # SAVE CHECKPOINT
        # ----------------------------------------------------

        checkpoint_path = (
            RESULTS /
            f"checkpoint_epoch_{epoch + 1}.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )


    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    final_path = (
        MODEL_DIR /
        "final.pth"
    )

    torch.save(
        model.state_dict(),
        final_path
    )

    print(
        "\n✅ Training completed!"
    )

    print(
        "Best model:",
        MODEL_DIR / "best.pth"
    )

    print(
        "Final model:",
        final_path
    )


if __name__ == "__main__":
    main()
