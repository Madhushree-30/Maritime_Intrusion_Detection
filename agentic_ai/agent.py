
import os
import pandas as pd


class MaritimeAgent:
    """
    Agentic AI controller for maritime surveillance.

    Pipeline:
        OBSERVE
            ↓
        ASSESS
            ↓
        REASON
            ↓
        CLASSIFY
            ↓
        DECIDE

    IMPORTANT:
        The frame column is preserved from intrusion_results.csv
        so that Agent Decisions can later be correctly joined
        with Encounter Classification using:

            frame + track_id
    """

    def __init__(self, intrusion_zone_file):

        self.zones = pd.read_csv(
            intrusion_zone_file
        )

        print("Maritime Agent initialized")
        print(
            f"Surveillance zones: {len(self.zones)}"
        )

    # =====================================================
    # OBSERVE
    # =====================================================

    def observe(self, row):

        return {
            "frame": row.get(
                "frame",
                None
            ),

            "track_id": row.get(
                "track_id",
                None
            ),

            "MMSI": row.get(
                "MMSI",
                None
            ),

            "latitude": row.get(
                "latitude",
                None
            ),

            "longitude": row.get(
                "longitude",
                None
            ),

            "speed": row.get(
                "Speed",
                row.get(
                    "SOG",
                    None
                )
            ),

            "heading": row.get(
                "heading_deg",
                row.get(
                    "Heading",
                    None
                )
            ),

            "ais_match": row.get(
                "ais_match",
                False
            ),

            "match_quality": row.get(
                "match_quality",
                "unmatched"
            ),

            "distance_to_ais_km": row.get(
                "distance_to_ais_km",
                None
            ),

            "intrusion_status": row.get(
                "intrusion_status",
                "NORMAL"
            ),

            "intrusion_zone": row.get(
                "intrusion_zone",
                None
            )
        }

    # =====================================================
    # ASSESS
    # =====================================================

    def assess(self, observation):

        assessment = {}

        # -------------------------------------------------
        # AIS STATE
        # -------------------------------------------------

        if bool(observation["ais_match"]):

            assessment["ais_state"] = (
                "IDENTIFIED"
            )

        else:

            assessment["ais_state"] = (
                "UNKNOWN"
            )

        # -------------------------------------------------
        # IDENTITY CONFIDENCE
        # -------------------------------------------------

        quality = str(
            observation["match_quality"]
        ).lower()

        if quality == "reliable":

            assessment[
                "identity_confidence"
            ] = "HIGH"

        elif quality == "moderate":

            assessment[
                "identity_confidence"
            ] = "MEDIUM"

        else:

            assessment[
                "identity_confidence"
            ] = "LOW"

        # -------------------------------------------------
        # ZONE STATE
        # -------------------------------------------------

        status = observation[
            "intrusion_status"
        ]

        if status == "INTRUSION":

            assessment[
                "zone_state"
            ] = "INSIDE"

        else:

            assessment[
                "zone_state"
            ] = "OUTSIDE"

        return assessment

    # =====================================================
    # REASON
    # =====================================================

    def reason(
        self,
        observation,
        assessment
    ):

        intrusion_status = observation[
            "intrusion_status"
        ]

        ais_state = assessment[
            "ais_state"
        ]

        # -------------------------------------------------
        # UNKNOWN AIS
        # -------------------------------------------------

        if ais_state == "UNKNOWN":

            return {
                "intrusion": False,
                "intrusion_type":
                    "AIS_UNIDENTIFIED",
                "severity":
                    "MEDIUM",
                "reason":
                    "Vessel detected without "
                    "a reliable AIS association."
            }

        # -------------------------------------------------
        # CONFIRMED INTRUSION
        # -------------------------------------------------

        if intrusion_status == "INTRUSION":

            return {
                "intrusion": True,
                "intrusion_type":
                    "UNAUTHORIZED_ENTRY",
                "severity":
                    "HIGH",
                "reason":
                    "Identified vessel has entered "
                    "the surveillance zone."
            }

        # -------------------------------------------------
        # NORMAL
        # -------------------------------------------------

        return {
            "intrusion": False,
            "intrusion_type":
                "NORMAL",
            "severity":
                "NONE",
            "reason":
                "No intrusion condition detected."
        }

    # =====================================================
    # DECIDE
    # =====================================================

    def decide(self, reasoning):

        intrusion_type = reasoning[
            "intrusion_type"
        ]

        severity = reasoning[
            "severity"
        ]

        # -------------------------------------------------
        # UNAUTHORIZED ENTRY
        # -------------------------------------------------

        if intrusion_type == (
            "UNAUTHORIZED_ENTRY"
        ):

            return {
                "decision":
                    "ALERT",

                "action":
                    "ALERT_AND_MONITOR"
            }

        # -------------------------------------------------
        # UNKNOWN AIS
        # -------------------------------------------------

        if intrusion_type == (
            "AIS_UNIDENTIFIED"
        ):

            return {
                "decision":
                    "MONITOR",

                "action":
                    "CONTINUE_OBSERVATION"
            }

        # -------------------------------------------------
        # NORMAL
        # -------------------------------------------------

        return {
            "decision":
                "NORMAL",

            "action":
                "NO_ACTION"
        }

    # =====================================================
    # PROCESS
    # =====================================================

    def process(
        self,
        input_file,
        output_file
    ):

        print(
            "\nLoading surveillance observations..."
        )

        df = pd.read_csv(
            input_file
        )

        print(
            "Input records:",
            len(df)
        )

        # =================================================
        # REQUIRED COLUMNS
        # =================================================

        required = [
            "frame",
            "track_id",
            "latitude",
            "longitude",
            "ais_match",
            "match_quality",
            "intrusion_status"
        ]

        missing = [
            c
            for c in required
            if c not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing required columns: "
                + str(missing)
            )

        # =================================================
        # MAKE SURE FRAME IS NUMERIC
        # =================================================

        df["frame"] = pd.to_numeric(
            df["frame"],
            errors="coerce"
        )

        # =================================================
        # PROCESS EACH OBSERVATION
        # =================================================

        results = []

        for _, row in df.iterrows():

            observation = self.observe(
                row
            )

            assessment = self.assess(
                observation
            )

            reasoning = self.reason(
                observation,
                assessment
            )

            decision = self.decide(
                reasoning
            )

            result = {

                # -----------------------------------------
                # KEY IDENTIFIERS
                # -----------------------------------------

                "frame":
                    observation["frame"],

                "track_id":
                    observation["track_id"],

                "MMSI":
                    observation["MMSI"],

                # -----------------------------------------
                # POSITION
                # -----------------------------------------

                "latitude":
                    observation["latitude"],

                "longitude":
                    observation["longitude"],

                # -----------------------------------------
                # MOTION
                # -----------------------------------------

                "speed":
                    observation["speed"],

                "heading":
                    observation["heading"],

                # -----------------------------------------
                # AIS
                # -----------------------------------------

                "ais_match":
                    observation["ais_match"],

                "match_quality":
                    observation["match_quality"],

                "distance_to_ais_km":
                    observation[
                        "distance_to_ais_km"
                    ],

                # -----------------------------------------
                # INTRUSION
                # -----------------------------------------

                "intrusion_status":
                    observation[
                        "intrusion_status"
                    ],

                "intrusion_zone":
                    observation[
                        "intrusion_zone"
                    ],

                # -----------------------------------------
                # ASSESSMENT
                # -----------------------------------------

                "ais_state":
                    assessment[
                        "ais_state"
                    ],

                "identity_confidence":
                    assessment[
                        "identity_confidence"
                    ],

                "zone_state":
                    assessment[
                        "zone_state"
                    ],

                # -----------------------------------------
                # REASONING
                # -----------------------------------------

                "intrusion":
                    reasoning[
                        "intrusion"
                    ],

                "intrusion_type":
                    reasoning[
                        "intrusion_type"
                    ],

                "severity":
                    reasoning[
                        "severity"
                    ],

                "reason":
                    reasoning[
                        "reason"
                    ],

                # -----------------------------------------
                # DECISION
                # -----------------------------------------

                "decision":
                    decision[
                        "decision"
                    ],

                "action":
                    decision[
                        "action"
                    ]
            }

            results.append(
                result
            )

        result_df = pd.DataFrame(
            results
        )

        # =================================================
        # SAFETY CHECK
        # =================================================

        if len(result_df) != len(df):

            raise RuntimeError(
                "Agent output row count changed. "
                f"Input={len(df)}, "
                f"Output={len(result_df)}"
            )

        # =================================================
        # FRAME VALIDATION
        # =================================================

        frame_count = (
            result_df["frame"]
            .notna()
            .sum()
        )

        print(
            "\nFrame preservation:"
        )

        print(
            "Rows with frame:",
            frame_count
        )

        print(
            "Rows without frame:",
            len(result_df) - frame_count
        )

        print(
            "Unique frames:",
            result_df["frame"]
            .nunique()
        )

        print(
            "Unique tracks:",
            result_df["track_id"]
            .nunique()
        )

        # =================================================
        # DUPLICATE CHECK
        # =================================================

        valid_keys = result_df[
            ["frame", "track_id"]
        ].dropna()

        duplicate_count = (
            valid_keys.duplicated(
                subset=[
                    "frame",
                    "track_id"
                ],
                keep=False
            ).sum()
        )

        print(
            "\nDuplicate (frame, track_id) rows:",
            duplicate_count
        )

        # =================================================
        # SAVE
        # =================================================

        os.makedirs(
            os.path.dirname(
                output_file
            ),
            exist_ok=True
        )

        result_df.to_csv(
            output_file,
            index=False
        )

        # =================================================
        # RESULTS
        # =================================================

        print(
            "\n======================================"
        )

        print(
            "AGENTIC AI RESULTS"
        )

        print(
            "======================================"
        )

        print(
            "Total observations:",
            len(result_df)
        )

        print(
            "\nIntrusion distribution:"
        )

        print(
            result_df[
                "intrusion"
            ].value_counts()
        )

        print(
            "\nIntrusion types:"
        )

        print(
            result_df[
                "intrusion_type"
            ].value_counts()
        )

        print(
            "\nSeverity:"
        )

        print(
            result_df[
                "severity"
            ].value_counts()
        )

        print(
            "\nDecisions:"
        )

        print(
            result_df[
                "decision"
            ].value_counts()
        )

        print(
            "\nOutput saved to:"
        )

        print(
            output_file
        )

        return result_df


# =========================================================
# CONFIGURATION
# =========================================================

ZONE_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/intrusion_zones.csv"
)

INPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/intrusion_results.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/agent_decisions.csv"
)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    agent = MaritimeAgent(
        ZONE_FILE
    )

    results = agent.process(
        INPUT_FILE,
        OUTPUT_FILE
    )

    print(
        "\nSample agent decisions:"
    )

    columns = [
        "frame",
        "track_id",
        "MMSI",
        "intrusion_type",
        "severity",
        "decision",
        "action"
    ]

    columns = [
        c
        for c in columns
        if c in results.columns
    ]

    print(
        results[
            columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )
