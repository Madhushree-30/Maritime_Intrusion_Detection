
from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")

    model.train(
        data="/content/drive/MyDrive/ResearchProject/dataset/data.yaml",

        epochs=10,
        imgsz=640,
        batch=8,

        optimizer="AdamW",
        lr0=0.0001,

        project="/content/drive/MyDrive/ResearchProject/results",
        name="RT-DETR",

        pretrained=True,
        save=True,
        plots=True
    )

if __name__ == "__main__":
    main()
