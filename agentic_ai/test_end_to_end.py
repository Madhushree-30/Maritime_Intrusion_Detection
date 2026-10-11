
import os
import sys
import shutil
from datetime import datetime

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = "/content/drive/MyDrive/ResearchProject"

AIS_DIR = os.path.join(
    PROJECT_ROOT,
    "ais"
)

AGENT_DIR = os.path.join(
    PROJECT_ROOT,
    "agentic_ai"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)


# ============================================================
# ADD MODULE PATHS
# ============================================================

if AIS_DIR not in sys.path:
    sys.path.insert(0, AIS_DIR)

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)


# ============================================================
# EXISTING RESEARCH FILES
# ============================================================

RADAR_MODEL = os.path.join(
    MODELS_DIR,
    "best_model",
    "best_ship_detector.pt"
)

VISUAL_MODEL = os.path.join(
    MODELS_DIR,
    "seedrones",
    "best_seedrones_detector.pt"
)

TILE_BOUNDS = os.path.join(
    AIS_DIR,
    "tile_bounds_los_angeles.csv"
)

AIS_FILE = os.path.join(
    AIS_DIR,
    "clean_ais.csv"
)

INTRUSION_ZONES = os.path.join(
    AGENT_DIR,
    "intrusion_zones.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = {
    "Radar model": RADAR_MODEL,
    "Visual model": VISUAL_MODEL,
    "Tile bounds": TILE_BOUNDS,
    "AIS file": AIS_FILE,
    "Intrusion zones": INTRUSION_ZONES,
}

for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


# ============================================================
# IMPORT YOUR EXISTING RESEARCH MODULES
# ============================================================

from fusion_detector import FusionDetector
from georeference import GeoReferencer
from matcher import MultiVesselAISMatcher
from intrusion_detector import IntrusionDetector
from encounter_classifier import process_frame
from navigation_decision import NavigationDecision
from navigation_command import NavigationCommand


# ============================================================
# SINGLE-FRAME TRACKING
# ============================================================

def create_single_frame_tracks(
    fusion_file,
    tracking_file
):

    fusion_df = pd.read_csv(
        fusion_file
    )

    if fusion_df.empty:

        raise RuntimeError(
            "Fusion produced zero detections."
        )

    tracked_df = fusion_df.copy()

    # Each detection becomes a temporary track.
    tracked_df["track_id"] = (
        tracked_df.index + 1
    )

    # No temporal information exists
    # because this test contains one frame.
    tracked_df["movement_pixels"] = 0.0
    tracked_df["heading_deg"] = 0.0
    tracked_df["track_hits"] = 1

    tracked_df.to_csv(
        tracking_file,
        index=False
    )

    return tracked_df


# ============================================================
# ENCOUNTER ANALYSIS
# ============================================================

def run_encounter_analysis(
    input_file,
    output_file
):

    df = pd.read_csv(
        input_file
    )

    if df.empty:

        raise RuntimeError(
            "No data available for encounter analysis."
        )

    results = []

    # --------------------------------------------------------
    # Process every frame
    # --------------------------------------------------------

    for frame in sorted(
        df["frame"].unique()
    ):

        frame_df = df[
            df["frame"] == frame
        ].copy()

        try:

            result = process_frame(
                frame_df
            )

            if isinstance(
                result,
                pd.DataFrame
            ):

                results.extend(
                    result.to_dict(
                        "records"
                    )
                )

            elif isinstance(
                result,
                list
            ):

                results.extend(
                    result
                )

            elif isinstance(
                result,
                dict
            ):

                results.append(
                    result
                )

        except Exception:

            # A single image has no temporal encounter.
            pass


    # --------------------------------------------------------
    # Single-frame fallback
    # --------------------------------------------------------

    if len(results) == 0:

        result_df = df.copy()

        result_df["encounter_type"] = (
            "UNCERTAIN"
        )

        result_df["risk_level"] = (
            "UNKNOWN"
        )

        result_df["recommended_action"] = (
            "MAINTAIN_COURSE"
        )

    else:

        result_df = pd.DataFrame(
            results
        )


    result_df.to_csv(
        output_file,
        index=False
    )

    return result_df


# ============================================================
# NAVIGATION DECISION
# ============================================================

def run_navigation(
    encounter_file,
    navigation_file,
    command_file
):

    df = pd.read_csv(
        encounter_file
    )

    navigation = NavigationDecision()

    navigation_records = []


    # --------------------------------------------------------
    # Navigation decision
    # --------------------------------------------------------

    for _, row in df.iterrows():

        record = row.to_dict()

        try:

            decision = navigation.decide(
                row
            )

            if isinstance(
                decision,
                dict
            ):

                record.update(
                    decision
                )

            else:

                record[
                    "navigation_decision"
                ] = decision

        except Exception:

            record[
                "navigation_decision"
            ] = "MAINTAIN_COURSE"

        navigation_records.append(
            record
        )


    navigation_df = pd.DataFrame(
        navigation_records
    )

    navigation_df.to_csv(
        navigation_file,
        index=False
    )


    # --------------------------------------------------------
    # Navigation command
    # --------------------------------------------------------

    command_generator = NavigationCommand()

    command_records = []


    for _, row in navigation_df.iterrows():

        record = row.to_dict()

        try:

            command = (
                command_generator.generate_command(
                    row
                )
            )

            if isinstance(
                command,
                dict
            ):

                record.update(
                    command
                )

            else:

                record["command"] = command

        except Exception:

            record[
                "command"
            ] = "MAINTAIN_COURSE"

        command_records.append(
            record
        )


    command_df = pd.DataFrame(
        command_records
    )

    command_df.to_csv(
        command_file,
        index=False
    )

    return navigation_df, command_df


# ============================================================
# FINAL RESULT
# ============================================================

def extract_final_result(
    intrusion_df,
    encounter_df,
    navigation_df,
    command_df
):

    # --------------------------------------------------------
    # Start with defaults
    # --------------------------------------------------------

    final = {
        "intrusion_type": "NONE",
        "severity": "UNKNOWN",
        "encounter_type": "UNCERTAIN",
        "risk_level": "UNKNOWN",
        "risk_score": "UNKNOWN",
        "navigation": "MAINTAIN_COURSE",
        "command": "MAINTAIN_COURSE"
    }


    # --------------------------------------------------------
    # Intrusion
    # --------------------------------------------------------

    if not intrusion_df.empty:

        row = intrusion_df.iloc[-1]

        if (
            "intrusion_status"
            in row.index
        ):

            final["intrusion_type"] = (
                row["intrusion_status"]
            )

        if (
            "alert_level"
            in row.index
        ):

            final["severity"] = (
                row["alert_level"]
            )

        if (
            "intrusion_zone"
            in row.index
        ):

            zone = row["intrusion_zone"]

            if pd.notna(zone):

                final["intrusion_zone"] = zone


    # --------------------------------------------------------
    # Encounter
    # --------------------------------------------------------

    if not encounter_df.empty:

        row = encounter_df.iloc[-1]

        if (
            "encounter_type"
            in row.index
        ):

            final["encounter_type"] = (
                row["encounter_type"]
            )

        if (
            "risk_level"
            in row.index
        ):

            final["risk_level"] = (
                row["risk_level"]
            )

        if (
            "risk_score"
            in row.index
        ):

            final["risk_score"] = (
                row["risk_score"]
            )

        elif (
            "recommended_action"
            in row.index
        ):

            # Keep risk score unknown if the
            # existing module does not provide it.
            pass


    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    if not navigation_df.empty:

        row = navigation_df.iloc[-1]

        if (
            "navigation_decision"
            in row.index
        ):

            final["navigation"] = (
                row["navigation_decision"]
            )

        elif (
            "recommended_action"
            in row.index
        ):

            final["navigation"] = (
                row["recommended_action"]
            )


    # --------------------------------------------------------
    # Command
    # --------------------------------------------------------

    if not command_df.empty:

        row = command_df.iloc[-1]

        if "command" in row.index:

            final["command"] = row["command"]

        elif "command_action" in row.index:

            final["command"] = row[
                "command_action"
            ]


    return final


# ============================================================
# MAIN END-TO-END FUNCTION
# ============================================================

def analyze_inputs(
    radar_image,
    camera_image
):

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    if not os.path.exists(
        radar_image
    ):

        raise FileNotFoundError(
            f"Radar image not found:\n{radar_image}"
        )

    if not os.path.exists(
        camera_image
    ):

        raise FileNotFoundError(
            f"Camera image not found:\n{camera_image}"
        )


    # ========================================================
    # CREATE UNIQUE TEST DIRECTORY
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    test_dir = os.path.join(
        PROJECT_ROOT,
        "experiments",
        f"test_{timestamp}"
    )

    os.makedirs(
        test_dir,
        exist_ok=True
    )


    radar_dir = os.path.join(
        test_dir,
        "radar"
    )

    camera_dir = os.path.join(
        test_dir,
        "camera"
    )

    os.makedirs(
        radar_dir,
        exist_ok=True
    )

    os.makedirs(
        camera_dir,
        exist_ok=True
    )


    # ========================================================
    # COPY INPUT IMAGES
    # ========================================================

    radar_test = os.path.join(
        radar_dir,
        "000000.png"
    )

    camera_test = os.path.join(
        camera_dir,
        "000000.jpg"
    )

    shutil.copy2(
        radar_image,
        radar_test
    )

    shutil.copy2(
        camera_image,
        camera_test
    )


    # ========================================================
    # 1. FUSION
    # ========================================================

    fusion_file = os.path.join(
        test_dir,
        "fusion.csv"
    )

    detector = FusionDetector(
        RADAR_MODEL,
        VISUAL_MODEL
    )

    fusion_df = detector.detect_folder(
        radar_dir,
        camera_dir,
        fusion_file
    )

    if (
        fusion_df is None
        or fusion_df.empty
    ):

        raise RuntimeError(
            "Fusion produced zero detections."
        )


    # ========================================================
    # 2. SINGLE-FRAME TRACKING
    # ========================================================

    tracking_file = os.path.join(
        test_dir,
        "tracking.csv"
    )

    tracked_df = create_single_frame_tracks(
        fusion_file,
        tracking_file
    )


    # ========================================================
    # 3. LOS ANGELES GEOREFERENCING
    # ========================================================

    georeferenced_file = os.path.join(
        test_dir,
        "georeferenced.csv"
    )

    georef = GeoReferencer(
        TILE_BOUNDS
    )

    georeferenced_df = georef.process(
        tracking_file,
        georeferenced_file
    )

    if (
        georeferenced_df is None
        or georeferenced_df.empty
    ):

        raise RuntimeError(
            "Georeferencing produced zero records."
        )


    # ========================================================
    # 4. AIS MATCHING
    # ========================================================

    matched_file = os.path.join(
        test_dir,
        "matched.csv"
    )

    matcher = MultiVesselAISMatcher(
        max_distance_km=5.0,
        reliable_distance_km=1.0,
        moderate_distance_km=2.0
    )

    matched_df = matcher.match(
        georeferenced_file,
        AIS_FILE,
        matched_file
    )

    # --------------------------------------------------------
    # AIS matcher may return no matches.
    # Keep the detection rather than stopping the pipeline.
    # --------------------------------------------------------

    if (
        matched_df is None
        or matched_df.empty
    ):

        matched_df = georeferenced_df.copy()

        matched_df["ais_match"] = False

        matched_df["match_quality"] = (
            "unknown"
        )

        matched_df.to_csv(
            matched_file,
            index=False
        )


    # ========================================================
    # 5. INTRUSION DETECTION
    # ========================================================

    intrusion_file = os.path.join(
        test_dir,
        "intrusion.csv"
    )

    intrusion_detector = IntrusionDetector(
        INTRUSION_ZONES,
        reliable_only=False
    )

    intrusion_detector.detect(
        matched_file,
        intrusion_file
    )

    intrusion_df = pd.read_csv(
        intrusion_file
    )


    # ========================================================
    # 6. ENCOUNTER / RISK
    # ========================================================

    encounter_file = os.path.join(
        test_dir,
        "encounter.csv"
    )

    encounter_df = run_encounter_analysis(
        intrusion_file,
        encounter_file
    )


    # ========================================================
    # 7. NAVIGATION
    # ========================================================

    navigation_file = os.path.join(
        test_dir,
        "navigation.csv"
    )

    command_file = os.path.join(
        test_dir,
        "command.csv"
    )

    navigation_df, command_df = (
        run_navigation(
            encounter_file,
            navigation_file,
            command_file
        )
    )


    # ========================================================
    # 8. FINAL RESULT
    # ========================================================

    final = extract_final_result(
        intrusion_df,
        encounter_df,
        navigation_df,
        command_df
    )


    # ========================================================
    # SAVE FINAL RESULT
    # ========================================================

    final_file = os.path.join(
        test_dir,
        "final_result.csv"
    )

    pd.DataFrame(
        [final]
    ).to_csv(
        final_file,
        index=False
    )


    # ========================================================
    # ONLY DISPLAY FINAL RESULT
    # ========================================================

    print()
    print("=" * 60)
    print(
        "FINAL AGENTIC MARITIME NAVIGATION RESULT"
    )
    print("=" * 60)

    print(
        "Intrusion type :",
        final["intrusion_type"]
    )

    print(
        "Severity       :",
        final["severity"]
    )

    print(
        "Encounter type :",
        final["encounter_type"]
    )

    print(
        "Risk level     :",
        final["risk_level"]
    )

    print(
        "Risk score     :",
        final["risk_score"]
    )

    print(
        "Navigation     :",
        final["navigation"]
    )

    print(
        "Command        :",
        final["command"]
    )

    print("=" * 60)

    print(
        "Test data saved to:",
        test_dir
    )


    # ========================================================
    # RETURN EVERYTHING FOR PROGRAMMATIC USE
    # ========================================================

    return {
        "test_directory": test_dir,
        "fusion": fusion_df,
        "tracking": tracked_df,
        "georeferenced": georeferenced_df,
        "matched": matched_df,
        "intrusion": intrusion_df,
        "encounter": encounter_df,
        "navigation": navigation_df,
        "command": command_df,
        "final": final
    }
