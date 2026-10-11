
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ============================================================
# Haversine distance
# ============================================================

def haversine_matrix(
    det_lat,
    det_lon,
    ais_lat,
    ais_lon
):
    R = 6371.0

    det_lat = np.radians(np.asarray(det_lat))
    det_lon = np.radians(np.asarray(det_lon))

    ais_lat = np.radians(np.asarray(ais_lat))
    ais_lon = np.radians(np.asarray(ais_lon))

    lat1 = det_lat[:, None]
    lon1 = det_lon[:, None]

    lat2 = ais_lat[None, :]
    lon2 = ais_lon[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    return (
        2
        * R
        * np.arcsin(np.sqrt(a))
    )


# ============================================================
# Multi-vessel AIS Matcher
# ============================================================

class MultiVesselAISMatcher:

    def __init__(
        self,
        max_distance_km=5.0,
        reliable_distance_km=1.0,
        moderate_distance_km=2.0
    ):

        # Maximum distance allowed for an association
        self.max_distance_km = max_distance_km

        # Distance categories
        self.reliable_distance_km = reliable_distance_km
        self.moderate_distance_km = moderate_distance_km


    # ========================================================
    # Distance category
    # ========================================================

    def distance_category(self, distance):

        if distance <= self.reliable_distance_km:

            return "reliable"

        elif distance <= self.moderate_distance_km:

            return "moderate"

        elif distance <= self.max_distance_km:

            return "weak"

        else:

            return "unmatched"


    # ========================================================
    # Match one frame
    # ========================================================

    def match_frame(
        self,
        detections,
        ais
    ):

        if detections.empty or ais.empty:

            return []

        distances = haversine_matrix(
            detections["latitude"].to_numpy(),
            detections["longitude"].to_numpy(),
            ais["LAT"].to_numpy(),
            ais["LON"].to_numpy()
        )

        # ----------------------------------------------------
        # Only allow matches within max distance
        # ----------------------------------------------------

        cost = distances.copy()

        cost[
            cost > self.max_distance_km
        ] = 1e6

        # ----------------------------------------------------
        # One-to-one assignment
        # ----------------------------------------------------

        detection_indices, ais_indices = (
            linear_sum_assignment(cost)
        )

        matches = []

        for det_idx, ais_idx in zip(
            detection_indices,
            ais_indices
        ):

            distance = distances[
                det_idx,
                ais_idx
            ]

            if distance > self.max_distance_km:

                continue

            matches.append(
                (
                    det_idx,
                    ais_idx,
                    distance
                )
            )

        return matches


    # ========================================================
    # Main matching
    # ========================================================

    def match(
        self,
        detections_file,
        ais_file,
        output_file
    ):

        print("Loading detections...")

        detections = pd.read_csv(
            detections_file
        )

        print(
            "Total detection records:",
            len(detections)
        )

        print("Loading AIS data...")

        ais = pd.read_csv(
            ais_file
        )

        print(
            "Total AIS records:",
            len(ais)
        )

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required_detection = [
            "frame",
            "latitude",
            "longitude"
        ]

        required_ais = [
            "MMSI",
            "LAT",
            "LON"
        ]

        for column in required_detection:

            if column not in detections.columns:

                raise ValueError(
                    f"Missing detection column: {column}"
                )

        for column in required_ais:

            if column not in ais.columns:

                raise ValueError(
                    f"Missing AIS column: {column}"
                )

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        detections["latitude"] = pd.to_numeric(
            detections["latitude"],
            errors="coerce"
        )

        detections["longitude"] = pd.to_numeric(
            detections["longitude"],
            errors="coerce"
        )

        ais["LAT"] = pd.to_numeric(
            ais["LAT"],
            errors="coerce"
        )

        ais["LON"] = pd.to_numeric(
            ais["LON"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Remove invalid coordinates
        # ----------------------------------------------------

        detections = detections.dropna(
            subset=[
                "latitude",
                "longitude"
            ]
        ).copy()

        ais = ais.dropna(
            subset=[
                "LAT",
                "LON"
            ]
        ).copy()

        # ----------------------------------------------------
        # Remove duplicate AIS positions
        # ----------------------------------------------------

        ais = ais.drop_duplicates(
            subset=[
                "MMSI",
                "LAT",
                "LON"
            ]
        ).copy()

        print(
            "Valid AIS records:",
            len(ais)
        )

        print(
            "Unique AIS vessels:",
            ais["MMSI"].nunique()
        )

        # ----------------------------------------------------
        # Result storage
        # ----------------------------------------------------

        results = []

        frames = sorted(
            detections["frame"].unique()
        )

        print(
            "\nProcessing",
            len(frames),
            "frames..."
        )

        # ====================================================
        # FRAME-BY-FRAME MATCHING
        # ====================================================

        for counter, frame in enumerate(
            frames,
            start=1
        ):

            frame_detections = detections[
                detections["frame"] == frame
            ].copy()

            if frame_detections.empty:

                continue

            # ------------------------------------------------
            # Geographic candidate region
            # ------------------------------------------------

            lat_min = (
                frame_detections["latitude"].min()
                - 0.10
            )

            lat_max = (
                frame_detections["latitude"].max()
                + 0.10
            )

            lon_min = (
                frame_detections["longitude"].min()
                - 0.10
            )

            lon_max = (
                frame_detections["longitude"].max()
                + 0.10
            )

            frame_ais = ais[
                (ais["LAT"] >= lat_min)
                &
                (ais["LAT"] <= lat_max)
                &
                (ais["LON"] >= lon_min)
                &
                (ais["LON"] <= lon_max)
            ].copy()

            # ------------------------------------------------
            # No AIS candidates
            # ------------------------------------------------

            if frame_ais.empty:

                for _, detection in (
                    frame_detections.iterrows()
                ):

                    result = detection.to_dict()

                    result["MMSI"] = None
                    result["VesselName"] = None
                    result["VesselType"] = None

                    result["AIS_LAT"] = np.nan
                    result["AIS_LON"] = np.nan

                    result["SOG"] = np.nan
                    result["COG"] = np.nan

                    result["distance_to_ais_km"] = np.nan

                    result["match_quality"] = "unmatched"

                    result["ais_match"] = False

                    results.append(result)

                continue

            # ------------------------------------------------
            # One-to-one assignment
            # ------------------------------------------------

            matches = self.match_frame(
                frame_detections,
                frame_ais
            )

            matched_detection_indices = set()

            # ------------------------------------------------
            # Save matches
            # ------------------------------------------------

            for (
                det_idx,
                ais_idx,
                distance
            ) in matches:

                detection = (
                    frame_detections.iloc[
                        det_idx
                    ]
                )

                vessel = (
                    frame_ais.iloc[
                        ais_idx
                    ]
                )

                category = (
                    self.distance_category(
                        distance
                    )
                )

                result = detection.to_dict()

                result["MMSI"] = vessel.get(
                    "MMSI",
                    None
                )

                result["VesselName"] = vessel.get(
                    "VesselName",
                    None
                )

                result["VesselType"] = vessel.get(
                    "VesselType",
                    None
                )

                result["AIS_LAT"] = vessel["LAT"]
                result["AIS_LON"] = vessel["LON"]

                result["SOG"] = vessel.get(
                    "SOG",
                    None
                )

                result["COG"] = vessel.get(
                    "COG",
                    None
                )

                result["distance_to_ais_km"] = distance

                result["match_quality"] = category

                result["ais_match"] = True

                results.append(result)

                matched_detection_indices.add(
                    det_idx
                )

            # ------------------------------------------------
            # Save unmatched detections
            # ------------------------------------------------

            for det_idx in range(
                len(frame_detections)
            ):

                if det_idx in matched_detection_indices:

                    continue

                detection = (
                    frame_detections.iloc[
                        det_idx
                    ]
                )

                result = detection.to_dict()

                result["MMSI"] = None
                result["VesselName"] = None
                result["VesselType"] = None

                result["AIS_LAT"] = np.nan
                result["AIS_LON"] = np.nan

                result["SOG"] = np.nan
                result["COG"] = np.nan

                result["distance_to_ais_km"] = np.nan

                result["match_quality"] = "unmatched"

                result["ais_match"] = False

                results.append(result)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                counter == 1
                or counter % 20 == 0
                or counter == len(frames)
            ):

                print(
                    f"Processed frame "
                    f"{counter}/{len(frames)}"
                )

        # ====================================================
        # OUTPUT
        # ====================================================

        matched = pd.DataFrame(
            results
        )

        sort_columns = [
            c
            for c in [
                "frame",
                "track_id"
            ]
            if c in matched.columns
        ]

        if sort_columns:

            matched = matched.sort_values(
                sort_columns
            )

        os.makedirs(
            os.path.dirname(output_file),
            exist_ok=True
        )

        matched.to_csv(
            output_file,
            index=False
        )

        # ====================================================
        # STATISTICS
        # ====================================================

        total = len(matched)

        matched_count = int(
            matched["ais_match"].sum()
        )

        unmatched_count = (
            total - matched_count
        )

        print("\n========================================")
        print("MULTI-VESSEL AIS MATCHING")
        print("========================================")

        print(
            "Total records:",
            total
        )

        print(
            "Matched:",
            matched_count
        )

        print(
            "Unmatched:",
            unmatched_count
        )

        if total > 0:

            print(
                "Match rate:",
                f"{matched_count / total * 100:.2f}%"
            )

        # ----------------------------------------------------
        # Quality statistics
        # ----------------------------------------------------

        print("\nMatch quality:")

        print(
            matched["match_quality"]
            .value_counts()
        )

        print("\nDistance statistics (km):")

        if matched_count > 0:

            print(
                matched.loc[
                    matched["ais_match"],
                    "distance_to_ais_km"
                ].describe()
            )

        # ----------------------------------------------------
        # MMSI statistics
        # ----------------------------------------------------

        print("\nUnique MMSI:")

        print(
            matched.loc[
                matched["ais_match"],
                "MMSI"
            ].nunique()
        )

        print("\nMMSI distribution:")

        print(
            matched.loc[
                matched["ais_match"],
                "MMSI"
            ].value_counts()
        )

        print(
            "\nOutput saved to:",
            output_file
        )

        return matched


# ============================================================
# CONFIGURATION
# ============================================================

GEOREFERENCED_FILE = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "georeferenced_detections.csv"
)

AIS_FILE = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "clean_ais.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "matched_vessels.csv"
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    matcher = MultiVesselAISMatcher(

        # Hard maximum:
        # anything farther than 5 km is NOT matched
        max_distance_km=5.0,

        # <= 1 km = reliable
        reliable_distance_km=1.0,

        # 1-2 km = moderate
        moderate_distance_km=2.0
    )

    result = matcher.match(
        GEOREFERENCED_FILE,
        AIS_FILE,
        OUTPUT_FILE
    )

    print("\nFirst 10 records:")

    columns = [
        "frame",
        "track_id",
        "latitude",
        "longitude",
        "MMSI",
        "AIS_LAT",
        "AIS_LON",
        "distance_to_ais_km",
        "match_quality",
        "ais_match"
    ]

    columns = [
        c
        for c in columns
        if c in result.columns
    ]

    print(
        result[columns].head(10)
    )
