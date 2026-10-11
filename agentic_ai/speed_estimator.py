
import math
import pandas as pd


INPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "georeferenced_detections.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/agentic_ai/"
    "outputs/speed_estimated_detections.csv"
)


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two geographic coordinates."""

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )


def estimate_speed(group):

    group = group.sort_values("frame").copy()

    speeds = [None]

    previous = None

    for _, row in group.iterrows():

        if previous is None:
            previous = row
            continue

        distance_km = haversine_km(
            previous["latitude"],
            previous["longitude"],
            row["latitude"],
            row["longitude"]
        )

        frame_difference = (
            row["frame"] - previous["frame"]
        )

        if frame_difference <= 0:
            speeds.append(None)
            previous = row
            continue

        # Assumption:
        # dataset represents consecutive frames.
        #
        # If frame rate is 1 frame/sec:
        # time = frame_difference seconds.
        #
        # Convert km/sec -> knots.

        time_seconds = frame_difference

        speed_km_per_sec = (
            distance_km / time_seconds
        )

        speed_km_per_hour = (
            speed_km_per_sec * 3600
        )

        speed_knots = (
            speed_km_per_hour / 1.852
        )

        speeds.append(speed_knots)

        previous = row

    group["estimated_speed_knots"] = speeds

    return group


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Input records:", len(df))

print("\nInput columns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# Check required columns
# ---------------------------------------------------------

required_columns = [
    "track_id",
    "frame",
    "latitude",
    "longitude"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ---------------------------------------------------------
# Estimate speed for every track
# ---------------------------------------------------------

df = (
    df.groupby(
        "track_id",
        group_keys=False
    )
    .apply(estimate_speed)
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Fill first observation of each track
# ---------------------------------------------------------

df["estimated_speed_knots"] = (
    df["estimated_speed_knots"]
    .fillna(0)
)


# ---------------------------------------------------------
# Remove unrealistic speed spikes
#
# This is only a safety filter for tracking noise.
# ---------------------------------------------------------

df.loc[
    df["estimated_speed_knots"] > 50,
    "estimated_speed_knots"
] = float("nan")


# Interpolate within each track

df["estimated_speed_knots"] = (
    df.groupby("track_id")[
        "estimated_speed_knots"
    ]
    .transform(
        lambda x: x.replace(
            0,
            float("nan")
        ).interpolate(
            limit_direction="both"
        )
    )
)

df["estimated_speed_knots"] = (
    df["estimated_speed_knots"]
    .fillna(0)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("\n================================")
print("SPEED ESTIMATION COMPLETED")
print("================================")

print(
    "Output records:",
    len(df)
)

print(
    "Unique tracks:",
    df["track_id"].nunique()
)

print("\nSpeed statistics (knots):")

print(
    df[
        "estimated_speed_knots"
    ].describe()
)

print("\nSample:")

print(
    df[
        [
            "frame",
            "track_id",
            "latitude",
            "longitude",
            "heading_deg",
            "estimated_speed_knots"
        ]
    ].head(20).to_string(
        index=False
    )
)

print(
    "\nSaved to:",
    OUTPUT_FILE
)
