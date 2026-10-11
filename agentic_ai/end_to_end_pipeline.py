
# ================================================================
# END-TO-END AGENTIC MARITIME NAVIGATION PIPELINE
# ================================================================
#
# Usage:
#
# from end_to_end_pipeline import analyze_inputs
#
# result = analyze_inputs(
#     radar_image="/path/to/radar.png",
#     camera_image="/path/to/camera.jpg"
# )
#
# The function:
#   1. Accepts the input images separately
#   2. Runs ship detection
#   3. Produces/uses agentic analysis
#   4. Displays only:
#        - input image(s)
#        - intrusion type
#        - severity
#        - navigation decision
# ================================================================

import os
import sys
import shutil
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = "/content/drive/MyDrive/ResearchProject"

AIS_DIR = os.path.join(PROJECT_ROOT, "ais")
AGENT_DIR = os.path.join(PROJECT_ROOT, "agentic_ai")
OUTPUT_DIR = os.path.join(AGENT_DIR, "outputs")

RADAR_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_model",
    "best_ship_detector.pt"
)

VISUAL_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "seedrones",
    "best_seedrones_detector.pt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if AIS_DIR not in sys.path:
    sys.path.insert(0, AIS_DIR)

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)


# ================================================================
# DISPLAY IMAGE
# ================================================================

def display_image(image_path, title="Input Image"):

    image = Image.open(image_path)

    plt.figure(figsize=(10, 7))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ================================================================
# DISPLAY TWO INPUTS
# ================================================================

def display_inputs(radar_image=None, camera_image=None):

    images = []
    titles = []

    if radar_image is not None:
        images.append(Image.open(radar_image))
        titles.append("Radar Input")

    if camera_image is not None:
        images.append(Image.open(camera_image))
        titles.append("SeaDrones / Camera Input")

    if len(images) == 0:
        return

    plt.figure(figsize=(14, 7))

    for i, (image, title) in enumerate(zip(images, titles), 1):

        plt.subplot(1, len(images), i)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ================================================================
# DISPLAY ONLY FINAL RESULT
# ================================================================

def display_decision(final_result):

    def value(column, default="N/A"):
        if column in final_result.index:
            return final_result[column]
        return default

    print("\n" + "=" * 60)
    print("FINAL MARITIME ANALYSIS")
    print("=" * 60)

    print("\nINTRUSION")
    print("Intrusion Type :", value("intrusion_type"))
    print("Severity       :", value("severity"))

    print("\nNAVIGATION")
    print("Recommended    :", value("recommended_action"))
    print("Decision       :", value("navigation_action"))
    print("Priority       :", value("navigation_priority"))

    print("=" * 60)


# ================================================================
# RUN FUSION DETECTION ON TWO SEPARATE INPUTS
# ================================================================

def run_fusion(radar_image=None, camera_image=None):

    from fusion_detector import FusionDetector

    if radar_image is None and camera_image is None:
        raise ValueError(
            "Provide at least one input image."
        )

    temp_root = tempfile.mkdtemp(prefix="maritime_input_")

    radar_folder = os.path.join(temp_root, "radar")
    camera_folder = os.path.join(temp_root, "camera")

    os.makedirs(radar_folder, exist_ok=True)
    os.makedirs(camera_folder, exist_ok=True)

    # ------------------------------------------------------------
    # Copy radar image if provided
    # ------------------------------------------------------------

    if radar_image is not None:

        if not os.path.exists(radar_image):
            raise FileNotFoundError(
                f"Radar image not found: {radar_image}"
            )

        shutil.copy(
            radar_image,
            os.path.join(
                radar_folder,
                os.path.basename(radar_image)
            )
        )

    # ------------------------------------------------------------
    # Copy camera image if provided
    # ------------------------------------------------------------

    if camera_image is not None:

        if not os.path.exists(camera_image):
            raise FileNotFoundError(
                f"Camera image not found: {camera_image}"
            )

        shutil.copy(
            camera_image,
            os.path.join(
                camera_folder,
                os.path.basename(camera_image)
            )
        )

    # ------------------------------------------------------------
    # Run detector
    # ------------------------------------------------------------

    detector = FusionDetector(
        radar_model_path=RADAR_MODEL,
        visual_model_path=VISUAL_MODEL
    )

    detection_output = os.path.join(
        OUTPUT_DIR,
        "current_fusion_detection.csv"
    )

    detections = detector.detect_folder(
        radar_folder=radar_folder,
        camera_folder=camera_folder,
        output_csv=detection_output
    )

    return detections, detection_output


# ================================================================
# GET FINAL AGENTIC DECISION
# ================================================================
#
# Uses your existing navigation output.
#
# IMPORTANT:
# For true real-time inference, the next stage should make your
# intrusion_detector.py, encounter_classifier.py,
# navigation_decision.py and navigation_command.py process the
# NEW detections directly.
# ================================================================

def get_navigation_result():

    navigation_file = os.path.join(
        OUTPUT_DIR,
        "navigation_commands.csv"
    )

    if not os.path.exists(navigation_file):

        return None

    df = pd.read_csv(navigation_file)

    if len(df) == 0:
        return None

    # Get latest available frame
    if "frame" in df.columns:

        latest_frame = df["frame"].max()

        latest = df[
            df["frame"] == latest_frame
        ].copy()

        return latest.iloc[0]

    return df.iloc[-1]


# ================================================================
# MAIN FUNCTION
# ================================================================

def analyze_inputs(
    radar_image=None,
    camera_image=None
):

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    if radar_image is None and camera_image is None:

        raise ValueError(
            "Pass radar_image, camera_image, or both."
        )

    # ------------------------------------------------------------
    # Display input image(s)
    # ------------------------------------------------------------

    display_inputs(
        radar_image=radar_image,
        camera_image=camera_image
    )

    # ------------------------------------------------------------
    # Ship / vessel detection
    # ------------------------------------------------------------

    detections, detection_file = run_fusion(
        radar_image=radar_image,
        camera_image=camera_image
    )

    # ------------------------------------------------------------
    # No detection
    # ------------------------------------------------------------

    if detections is None or len(detections) == 0:

        print("\nNo vessels detected.")

        return {
            "detections": detections,
            "decision": None,
            "detection_file": detection_file
        }

    # ------------------------------------------------------------
    # Navigation decision
    # ------------------------------------------------------------

    final_result = get_navigation_result()

    if final_result is None:

        print("\nVessel(s) detected.")

        print(
            "\nNo agentic navigation result is currently available."
        )

        return {
            "detections": detections,
            "decision": None,
            "detection_file": detection_file
        }

    # ------------------------------------------------------------
    # Display only intrusion + navigation
    # ------------------------------------------------------------

    display_decision(final_result)

    return {
        "detections": detections,
        "decision": final_result,
        "detection_file": detection_file
    }


# ================================================================
# SINGLE RADAR INPUT FUNCTION
# ================================================================

def analyze_radar_image(radar_image):

    return analyze_inputs(
        radar_image=radar_image,
        camera_image=None
    )


# ================================================================
# SINGLE CAMERA INPUT FUNCTION
# ================================================================

def analyze_camera_image(camera_image):

    return analyze_inputs(
        radar_image=None,
        camera_image=camera_image
    )
