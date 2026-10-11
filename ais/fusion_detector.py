
import os
import glob
import pandas as pd
from ultralytics import YOLO


class FusionDetector:
    def __init__(self, radar_model_path, visual_model_path):
        self.radar_model = YOLO(radar_model_path)
        self.visual_model = YOLO(visual_model_path)

    def detect_folder(self, radar_folder, camera_folder, output_csv):
        detections = []

        # Read all radar images
        radar_images = sorted(
            glob.glob(os.path.join(radar_folder, '*.png')) +
            glob.glob(os.path.join(radar_folder, '*.jpg')) +
            glob.glob(os.path.join(radar_folder, '*.jpeg'))
        )

        # Read all camera images
        camera_images = sorted(
            glob.glob(os.path.join(camera_folder, '*.png')) +
            glob.glob(os.path.join(camera_folder, '*.jpg')) +
            glob.glob(os.path.join(camera_folder, '*.jpeg'))
        )

        total_frames = max(len(radar_images), len(camera_images))

        for frame in range(total_frames):

            # -----------------------------
            # Radar detections (HRSID)
            # -----------------------------
            if frame < len(radar_images):
                radar_results = self.radar_model.predict(
                    source=radar_images[frame],
                    conf=0.25,
                    verbose=False
                )

                for r in radar_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        detections.append({
                            'frame': frame,
                            'class': 'ship',
                            'source': 'radar',
                            'confidence': float(box.conf[0]),
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'cx': (x1 + x2) / 2,
                            'cy': (y1 + y2) / 2
                        })

            # -----------------------------
            # Camera detections (SeaDronesSee)
            # -----------------------------
            if frame < len(camera_images):
                visual_results = self.visual_model.predict(
                    source=camera_images[frame],
                    conf=0.25,
                    verbose=False
                )

                for r in visual_results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        class_name = r.names[cls]

                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        detections.append({
                            'frame': frame,
                            'class': class_name,
                            'source': 'camera',
                            'confidence': float(box.conf[0]),
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'cx': (x1 + x2) / 2,
                            'cy': (y1 + y2) / 2
                        })

        df = pd.DataFrame(detections)

        if len(df) == 0:
            print('No detections found')
            return df

        # Tracker-compatible format
        df = df[
            [
                'frame',
                'class',
                'source',
                'confidence',
                'x1',
                'y1',
                'x2',
                'y2',
                'cx',
                'cy'
            ]
        ]

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

        print('Fusion completed successfully')
        print(f'Total detections: {len(df)}')
        print(f'Output saved to: {output_csv}')

        return df


# -------------------------------------------------------
# Los Angeles configuration
# -------------------------------------------------------

RADAR_MODEL = '/content/drive/MyDrive/ResearchProject/models/best_model/best_ship_detector.pt'
VISUAL_MODEL = '/content/drive/MyDrive/ResearchProject/models/seedrones/best_seedrones_detector.pt'

RADAR_FOLDER = "/content/drive/MyDrive/ResearchProject/dataset/images/val"
CAMERA_FOLDER = "/root/.cache/kagglehub/datasets/ubiratanfilho/sds-dataset/versions/1/compressed/images/val"

OUTPUT = '/content/drive/MyDrive/ResearchProject/ais/fused_detections.csv'


if __name__ == '__main__':
    detector = FusionDetector(
        RADAR_MODEL,
        VISUAL_MODEL
    )

    result = detector.detect_folder(
        RADAR_FOLDER,
        CAMERA_FOLDER,
        OUTPUT
    )

    print(result.head())
