
from ultralytics import YOLO

def main():
    # Load YOLOv11 Nano model
    model = YOLO("yolo11n.pt")

    model.train(
        data="/content/drive/MyDrive/ResearchProject/dataset/data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        pretrained=True,
        project="/content/drive/MyDrive/ResearchProject/results",
        name="YOLOv11",
        save=True,
        plots=True
    )

if __name__ == "__main__":
    main()
