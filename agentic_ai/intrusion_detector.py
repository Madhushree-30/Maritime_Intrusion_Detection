
import os
import pandas as pd


class IntrusionDetector:

    def __init__(
        self,
        zone_file,
        reliable_only=False
    ):

        self.zone_file = zone_file
        self.reliable_only = reliable_only

        self.zones = pd.read_csv(zone_file)

        required = [
            "zone_name",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max"
        ]

        for column in required:

            if column not in self.zones.columns:

                raise ValueError(
                    f"Missing zone column: {column}"
                )

        print("Intrusion zones loaded successfully")
        print("Number of zones:", len(self.zones))


    def check_position(self, latitude, longitude):

        for _, zone in self.zones.iterrows():

            inside = (
                latitude >= zone["lat_min"]
                and
                latitude <= zone["lat_max"]
                and
                longitude >= zone["lon_min"]
                and
                longitude <= zone["lon_max"]
            )

            if inside:

                return (
                    "INTRUSION",
                    zone["zone_name"]
                )

        return "NORMAL", None


    def classify_record(self, row):

        latitude = row["latitude"]
        longitude = row["longitude"]

        # --------------------------------------------------
        # No AIS association
        # --------------------------------------------------

        if not bool(row["ais_match"]):

            return (
                "UNKNOWN_AIS",
                None
            )

        # --------------------------------------------------
        # Optional high-confidence mode
        # --------------------------------------------------

        if self.reliable_only:

            if row["match_quality"] != "reliable":

                return (
                    "NORMAL_LOW_CONFIDENCE",
                    None
                )

        # --------------------------------------------------
        # Check surveillance zones
        # --------------------------------------------------

        return self.check_position(
            latitude,
            longitude
        )


    def detect(
        self,
        input_file,
        output_file
    ):

        print("\nLoading matched vessel data...")

        df = pd.read_csv(input_file)

        print(
            "Total input records:",
            len(df)
        )

        required = [
            "frame",
            "track_id",
            "latitude",
            "longitude",
            "ais_match",
            "match_quality"
        ]

        for column in required:

            if column not in df.columns:

                raise ValueError(
                    f"Missing required column: {column}"
                )

        # --------------------------------------------------
        # Numeric conversion
        # --------------------------------------------------

        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )

        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        statuses = []
        zones = []

        for _, row in df.iterrows():

            status, zone = self.classify_record(
                row
            )

            statuses.append(status)
            zones.append(zone)

        df["intrusion_status"] = statuses
        df["intrusion_zone"] = zones

        # --------------------------------------------------
        # Alert level
        # --------------------------------------------------

        def alert_level(status):

            if status == "INTRUSION":
                return "HIGH"

            if status == "UNKNOWN_AIS":
                return "MEDIUM"

            if status == "NORMAL_LOW_CONFIDENCE":
                return "LOW"

            return "NONE"

        df["alert_level"] = (
            df["intrusion_status"]
            .apply(alert_level)
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        os.makedirs(
            os.path.dirname(output_file),
            exist_ok=True
        )

        df.to_csv(
            output_file,
            index=False
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        print("\n========================================")
        print("INTRUSION DETECTION RESULTS")
        print("========================================")

        print(
            "Total records:",
            len(df)
        )

        print("\nStatus distribution:")

        print(
            df["intrusion_status"]
            .value_counts()
        )

        print("\nAlert distribution:")

        print(
            df["alert_level"]
            .value_counts()
        )

        print(
            "\nIntrusion records:",
            (
                df["intrusion_status"]
                == "INTRUSION"
            ).sum()
        )

        print(
            "Unknown AIS records:",
            (
                df["intrusion_status"]
                == "UNKNOWN_AIS"
            ).sum()
        )

        print("\nIntrusions by zone:")

        intrusion_df = df[
            df["intrusion_status"]
            == "INTRUSION"
        ]

        if len(intrusion_df) > 0:

            print(
                intrusion_df[
                    "intrusion_zone"
                ].value_counts()
            )

        else:

            print("No intrusions detected")

        print(
            "\nOutput saved to:",
            output_file
        )

        return df


# ============================================================
# CONFIGURATION
# ============================================================

MATCHED_FILE = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "matched_vessels.csv"
)

ZONE_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/intrusion_zones.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/intrusion_results.csv"
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    detector = IntrusionDetector(
        zone_file=ZONE_FILE,

        # Use all AIS associations for the first
        # pipeline test.
        reliable_only=False
    )

    result = detector.detect(
        input_file=MATCHED_FILE,
        output_file=OUTPUT_FILE
    )

    print("\nSample results:")

    columns = [
        "frame",
        "track_id",
        "MMSI",
        "latitude",
        "longitude",
        "match_quality",
        "intrusion_status",
        "intrusion_zone",
        "alert_level"
    ]

    columns = [
        c
        for c in columns
        if c in result.columns
    ]

    print(
        result[columns].head(15)
    )
