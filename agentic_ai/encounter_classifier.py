
import os
import math
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "georeferenced_detections.csv"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/ResearchProject/agentic_ai/"
    "outputs"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "encounter_classification.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Distances are in kilometers.
# These are simulation/research thresholds, not real vessel
# navigation-control limits.

SAFE_DISTANCE_KM = 0.20
MODERATE_DISTANCE_KM = 0.10
HIGH_DISTANCE_KM = 0.05
CRITICAL_DISTANCE_KM = 0.02

# Bearing sectors relative to own vessel heading.
#
# 0°   = directly ahead
# 90°  = starboard/right
# 180° = directly behind
# 270° = port/left

HEAD_ON_BEARING = 30

STARBOARD_MIN = 30
STARBOARD_MAX = 150

PORT_MIN = 210
PORT_MAX = 330

REAR_MIN = 150
REAR_MAX = 210

# Heading difference
OPPOSITE_HEADING_MIN = 120
SIMILAR_HEADING_MAX = 30


# ============================================================
# GEOGRAPHIC DISTANCE
# ============================================================

def distance_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2
        * R
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )


# ============================================================
# TRUE BEARING FROM OWN VESSEL TO TARGET
# ============================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlon = math.radians(
        lon2 - lon1
    )

    x = (
        math.sin(dlon)
        * math.cos(lat2_rad)
    )

    y = (
        math.cos(lat1_rad)
        * math.sin(lat2_rad)
        -
        math.sin(lat1_rad)
        * math.cos(lat2_rad)
        * math.cos(dlon)
    )

    bearing = math.degrees(
        math.atan2(x, y)
    )

    return (bearing + 360) % 360


# ============================================================
# ANGLE DIFFERENCE
# ============================================================

def angle_difference(
    angle1,
    angle2
):

    diff = abs(
        angle1 - angle2
    ) % 360

    return min(
        diff,
        360 - diff
    )


# ============================================================
# RELATIVE BEARING
# ============================================================

def relative_bearing(
    own_heading,
    target_bearing
):

    return (
        target_bearing
        - own_heading
    ) % 360


# ============================================================
# ENCOUNTER CLASSIFICATION
# ============================================================

def classify_encounter(
    own_heading,
    target_heading,
    rel_bearing
):

    heading_diff = angle_difference(
        own_heading,
        target_heading
    )

    # --------------------------------------------------------
    # 1. HEAD-ON
    #
    # Target approximately ahead and headings are opposite.
    # --------------------------------------------------------

    if (
        rel_bearing <= HEAD_ON_BEARING
        or rel_bearing >= 360 - HEAD_ON_BEARING
    ):

        if heading_diff >= OPPOSITE_HEADING_MIN:

            return "HEAD_ON"


    # --------------------------------------------------------
    # 2. OVERTAKING
    #
    # Similar headings and target is generally behind.
    # --------------------------------------------------------

    if (
        REAR_MIN <= rel_bearing <= REAR_MAX
        and heading_diff <= SIMILAR_HEADING_MAX
    ):

        return "OVERTAKING"


    # --------------------------------------------------------
    # 3. CROSSING STARBOARD
    #
    # Target is on the right side.
    # --------------------------------------------------------

    if (
        STARBOARD_MIN
        < rel_bearing
        < STARBOARD_MAX
    ):

        return "CROSSING_STARBOARD"


    # --------------------------------------------------------
    # 4. CROSSING PORT
    #
    # Target is on the left side.
    # --------------------------------------------------------

    if (
        PORT_MIN
        < rel_bearing
        < PORT_MAX
    ):

        return "CROSSING_PORT"


    # --------------------------------------------------------
    # 5. UNCERTAIN
    # --------------------------------------------------------

    return "UNCERTAIN"


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(
    distance,
    encounter_type
):

    # --------------------------------------------------------
    # SAFE
    # --------------------------------------------------------

    if distance > SAFE_DISTANCE_KM:

        return (
            "LOW",
            10
        )


    # --------------------------------------------------------
    # MODERATE
    # --------------------------------------------------------

    if distance > MODERATE_DISTANCE_KM:

        return (
            "MODERATE",
            40
        )


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if distance > HIGH_DISTANCE_KM:

        return (
            "HIGH",
            65
        )


    # --------------------------------------------------------
    # CRITICAL
    #
    # Very close vessels.
    # --------------------------------------------------------

    if distance <= CRITICAL_DISTANCE_KM:

        return (
            "CRITICAL",
            95
        )


    return (
        "HIGH",
        75
    )


# ============================================================
# ACTION SELECTION
# ============================================================

def recommend_action(
    encounter_type,
    risk_level,
    relative_bearing
):

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    if risk_level == "LOW":

        return "MAINTAIN_COURSE"


    # --------------------------------------------------------
    # MODERATE RISK
    # --------------------------------------------------------

    if risk_level == "MODERATE":

        if encounter_type == "CROSSING_STARBOARD":

            return "TURN_LEFT"

        if encounter_type == "CROSSING_PORT":

            return "TURN_RIGHT"

        if encounter_type == "HEAD_ON":

            return "SLOW_DOWN"

        if encounter_type == "OVERTAKING":

            return "SLOW_DOWN"

        return "SLOW_DOWN"


    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if risk_level == "HIGH":

        if encounter_type == "HEAD_ON":

            return "TURN_RIGHT"

        if encounter_type == "CROSSING_STARBOARD":

            return "TURN_LEFT"

        if encounter_type == "CROSSING_PORT":

            return "TURN_RIGHT"

        if encounter_type == "OVERTAKING":

            return "SLOW_DOWN"

        # Uncertain high-risk situation
        return "SLOW_DOWN"


    # --------------------------------------------------------
    # CRITICAL RISK
    #
    # IMPORTANT:
    # Critical does NOT automatically mean STOP.
    # --------------------------------------------------------

    if risk_level == "CRITICAL":

        if encounter_type == "HEAD_ON":

            return "STOP"

        if encounter_type == "CROSSING_STARBOARD":

            return "TURN_LEFT"

        if encounter_type == "CROSSING_PORT":

            return "TURN_RIGHT"

        if encounter_type == "OVERTAKING":

            return "STOP"

        # If geometry is uncertain, stop rather than
        # making an arbitrary turn.
        return "STOP"


    return "SLOW_DOWN"


# ============================================================
# PROCESS ONE VESSEL AGAINST ALL OTHER VESSELS
# ============================================================

def process_frame(
    frame_df
):

    results = []

    for own_index, own in frame_df.iterrows():

        candidates = []

        for target_index, target in frame_df.iterrows():

            if own_index == target_index:
                continue

            # ------------------------------------------------
            # Distance
            # ------------------------------------------------

            distance = distance_km(
                own["latitude"],
                own["longitude"],
                target["latitude"],
                target["longitude"]
            )

            # ------------------------------------------------
            # Ignore vessels outside the interaction area.
            # ------------------------------------------------

            if distance > SAFE_DISTANCE_KM:

                continue

            # ------------------------------------------------
            # Geographic bearing
            # ------------------------------------------------

            target_bearing = calculate_bearing(
                own["latitude"],
                own["longitude"],
                target["latitude"],
                target["longitude"]
            )

            # ------------------------------------------------
            # Bearing relative to own vessel
            # ------------------------------------------------

            rel_bearing = relative_bearing(
                own["heading_deg"],
                target_bearing
            )

            # ------------------------------------------------
            # Encounter
            # ------------------------------------------------

            encounter = classify_encounter(
                own["heading_deg"],
                target["heading_deg"],
                rel_bearing
            )

            # ------------------------------------------------
            # Risk
            # ------------------------------------------------

            risk_level, risk_score = calculate_risk(
                distance,
                encounter
            )

            # ------------------------------------------------
            # Action
            # ------------------------------------------------

            action = recommend_action(
                encounter,
                risk_level,
                rel_bearing
            )

            candidates.append({

                "track_id":
                    own["track_id"],

                "target_track_id":
                    target["track_id"],

                "distance_km":
                    distance,

                "target_bearing_deg":
                    target_bearing,

                "relative_bearing_deg":
                    rel_bearing,

                "heading_difference_deg":
                    angle_difference(
                        own["heading_deg"],
                        target["heading_deg"]
                    ),

                "encounter_type":
                    encounter,

                "risk_level":
                    risk_level,

                "risk_score":
                    risk_score,

                "recommended_action":
                    action
            })


        # ====================================================
        # NO NEARBY VESSEL
        # ====================================================

        if not candidates:

            results.append({

                "track_id":
                    own["track_id"],

                "target_track_id":
                    None,

                "distance_km":
                    None,

                "target_bearing_deg":
                    None,

                "relative_bearing_deg":
                    None,

                "heading_difference_deg":
                    None,

                "encounter_type":
                    "SAFE",

                "risk_level":
                    "LOW",

                "risk_score":
                    0,

                "recommended_action":
                    "MAINTAIN_COURSE"
            })

            continue


        # ====================================================
        # SELECT MOST IMPORTANT TARGET
        #
        # Risk first, distance second.
        # ====================================================

        best = sorted(
            candidates,
            key=lambda x: (
                -x["risk_score"],
                x["distance_km"]
            )
        )[0]

        results.append(best)


    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("ENCOUNTER CLASSIFIER")
print("=" * 60)

print("\nLoading:")

print(INPUT_FILE)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "\nTotal detections:",
    len(df)
)

print(
    "Unique frames:",
    df["frame"].nunique()
)

print(
    "Unique tracks:",
    df["track_id"].nunique()
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "frame",
    "track_id",
    "heading_deg",
    "latitude",
    "longitude"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# REMOVE INVALID GEOGRAPHIC VALUES
# ============================================================

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "heading_deg"
    ]
).copy()


# ============================================================
# PROCESS FRAME BY FRAME
# ============================================================

all_results = []

total_frames = df["frame"].nunique()

print(
    "\nProcessing",
    total_frames,
    "frames..."
)

for frame_number, frame_df in df.groupby(
    "frame"
):

    frame_result = process_frame(
        frame_df
    )

    frame_result["frame"] = frame_number

    all_results.append(
        frame_result
    )


# ============================================================
# COMBINE
# ============================================================

if all_results:

    result = pd.concat(
        all_results,
        ignore_index=True
    )

else:

    result = pd.DataFrame()


# ============================================================
# SORT
# ============================================================

if not result.empty:

    result = result.sort_values(
        [
            "frame",
            "track_id"
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 60)
print("ENCOUNTER CLASSIFICATION COMPLETED")
print("=" * 60)

print(
    "\nOutput records:",
    len(result)
)


if not result.empty:

    print("\n")
    print("ENCOUNTER DISTRIBUTION")
    print("-" * 40)

    print(
        result[
            "encounter_type"
        ].value_counts()
    )


    print("\n")
    print("RISK DISTRIBUTION")
    print("-" * 40)

    print(
        result[
            "risk_level"
        ].value_counts()
    )


    print("\n")
    print("ACTION DISTRIBUTION")
    print("-" * 40)

    print(
        result[
            "recommended_action"
        ].value_counts()
    )


    print("\n")
    print("ACTION PERCENTAGES")
    print("-" * 40)

    print(
        (
            result[
                "recommended_action"
            ]
            .value_counts(
                normalize=True
            )
            * 100
        )
        .round(2)
    )


    print("\n")
    print("CRITICAL SITUATIONS")
    print("-" * 40)

    critical = result[
        result["risk_level"]
        == "CRITICAL"
    ]

    if len(critical) > 0:

        print(
            critical[
                [
                    "frame",
                    "track_id",
                    "target_track_id",
                    "distance_km",
                    "relative_bearing_deg",
                    "heading_difference_deg",
                    "encounter_type",
                    "risk_level",
                    "recommended_action"
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No critical situations."
        )


    print("\n")
    print("SAMPLE")
    print("-" * 40)

    print(
        result[
            [
                "frame",
                "track_id",
                "target_track_id",
                "distance_km",
                "relative_bearing_deg",
                "encounter_type",
                "risk_level",
                "recommended_action"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


print("\n")
print("Saved to:")
print(OUTPUT_FILE)

print("\nDone.")
