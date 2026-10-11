
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="/content/drive/MyDrive/ResearchProject/dataset/data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        workers=2,
        project="/content/drive/MyDrive/ResearchProject/results",
        name="YOLOv8",
        optimizer="AdamW",
        lr0=0.001,
        pretrained=True,
        save=True,
        plots=True
    )

if __name__ == "__main__":
    main()
