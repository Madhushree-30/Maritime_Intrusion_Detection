
import os
import pandas as pd


class NavigationDecision:

    def __init__(self):
        print("Navigation Decision module initialized")

    # ==============================================================
    # NAVIGATION DECISION LOGIC
    # ==============================================================

    def decide(self, row):

        intrusion_type = str(
            row.get("intrusion_type", "NORMAL")
        ).upper()

        severity = str(
            row.get("severity", "NONE")
        ).upper()

        encounter_type = str(
            row.get("encounter_type", "SAFE")
        ).upper()

        risk_level = str(
            row.get("risk_level", "LOW")
        ).upper()

        recommended_action = str(
            row.get("recommended_action", "MAINTAIN_COURSE")
        ).upper()

        risk_score = row.get(
            "risk_score", 0
        )

        try:
            risk_score = float(risk_score)
        except:
            risk_score = 0

        # ==========================================================
        # 1. CRITICAL ENCOUNTER
        # ==========================================================

        if risk_level == "CRITICAL":

            if recommended_action == "STOP":

                return {
                    "navigation_action":
                        "ALERT_AND_STOP",

                    "navigation_priority":
                        "CRITICAL",

                    "navigation_reason":
                        "Critical collision-risk encounter "
                        "detected. Immediate stop required."
                }

            if recommended_action == "TURN_LEFT":

                return {
                    "navigation_action":
                        "ALERT_AND_TURN_LEFT",

                    "navigation_priority":
                        "CRITICAL",

                    "navigation_reason":
                        "Critical collision-risk encounter "
                        "detected. Immediate left turn required."
                }

            if recommended_action == "TURN_RIGHT":

                return {
                    "navigation_action":
                        "ALERT_AND_TURN_RIGHT",

                    "navigation_priority":
                        "CRITICAL",

                    "navigation_reason":
                        "Critical collision-risk encounter "
                        "detected. Immediate right turn required."
                }

            if recommended_action == "SLOW_DOWN":

                return {
                    "navigation_action":
                        "ALERT_AND_SLOW_DOWN",

                    "navigation_priority":
                        "CRITICAL",

                    "navigation_reason":
                        "Critical collision-risk encounter "
                        "detected. Immediate speed reduction required."
                }

            return {
                "navigation_action":
                    "ALERT_AND_MONITOR",

                "navigation_priority":
                    "CRITICAL",

                "navigation_reason":
                    "Critical collision-risk encounter "
                    "detected."
            }

        # ==========================================================
        # 2. UNAUTHORIZED INTRUSION + HIGH RISK
        # ==========================================================

        if (
            intrusion_type == "UNAUTHORIZED_ENTRY"
            and risk_level == "HIGH"
        ):

            if recommended_action == "TURN_LEFT":

                return {
                    "navigation_action":
                        "ALERT_AND_TURN_LEFT",

                    "navigation_priority":
                        "HIGH",

                    "navigation_reason":
                        "Unauthorized vessel entry combined "
                        "with high-risk encounter. Turn left."
                }

            if recommended_action == "TURN_RIGHT":

                return {
                    "navigation_action":
                        "ALERT_AND_TURN_RIGHT",

                    "navigation_priority":
                        "HIGH",

                    "navigation_reason":
                        "Unauthorized vessel entry combined "
                        "with high-risk encounter. Turn right."
                }

            if recommended_action == "STOP":

                return {
                    "navigation_action":
                        "ALERT_AND_STOP",

                    "navigation_priority":
                        "HIGH",

                    "navigation_reason":
                        "Unauthorized vessel entry with "
                        "high-risk encounter. Stop vessel."
                }

            if recommended_action == "SLOW_DOWN":

                return {
                    "navigation_action":
                        "ALERT_AND_SLOW_DOWN",

                    "navigation_priority":
                        "HIGH",

                    "navigation_reason":
                        "Unauthorized vessel entry with "
                        "high-risk encounter. Slow down."
                }

            return {
                "navigation_action":
                    "ALERT_AND_MONITOR",

                "navigation_priority":
                    "HIGH",

                "navigation_reason":
                    "Unauthorized vessel has entered "
                    "the surveillance zone."
            }

        # ==========================================================
        # 3. HIGH-RISK ENCOUNTER
        # ==========================================================

        if risk_level == "HIGH":

            if recommended_action == "TURN_LEFT":

                return {
                    "navigation_action":
                        "TURN_LEFT",

                    "navigation_priority":
                        "HIGH",

                    "navigation_reason":
                        "High-risk vessel encounter detected."
                }

            if recommended_action == "TURN_RIGHT":

                return {
                    "navigation_action":
                        "TURN_RIGHT",

                    "navigation_priority":
                        "HIGH",

                    "navigation_reason":
                        "High-risk vessel encounter detected."
                }

            if recommended_action == "STOP":

                return {
                    "navigation_action":
                        "STOP",

                    "navigation_priority":
                        "HIGH",

                    "navigation_reason":
                        "High-risk encounter requires "
                        "immediate stopping."
                }

            if recommended_action == "SLOW_DOWN":

                return {
                    "navigation_action":
                        "SLOW_DOWN",

                    "navigation_priority":
                        "HIGH",

                    "navigation_reason":
                        "High-risk vessel encounter detected."
                }

            return {
                "navigation_action":
                    "ALERT_AND_MONITOR",

                "navigation_priority":
                    "HIGH",

                "navigation_reason":
                    "High-risk vessel encounter detected."
            }

        # ==========================================================
        # 4. MODERATE-RISK ENCOUNTER
        # ==========================================================

        if risk_level == "MODERATE":

            if recommended_action == "TURN_LEFT":

                return {
                    "navigation_action":
                        "TURN_LEFT",

                    "navigation_priority":
                        "MEDIUM",

                    "navigation_reason":
                        "Moderate vessel encounter risk "
                        "detected. Turn left."
                }

            if recommended_action == "TURN_RIGHT":

                return {
                    "navigation_action":
                        "TURN_RIGHT",

                    "navigation_priority":
                        "MEDIUM",

                    "navigation_reason":
                        "Moderate vessel encounter risk "
                        "detected. Turn right."
                }

            if recommended_action == "SLOW_DOWN":

                return {
                    "navigation_action":
                        "SLOW_DOWN",

                    "navigation_priority":
                        "MEDIUM",

                    "navigation_reason":
                        "Moderate vessel encounter risk "
                        "detected."
                }

            return {
                "navigation_action":
                    "CONTINUE_OBSERVATION",

                "navigation_priority":
                    "MEDIUM",

                "navigation_reason":
                    "Moderate encounter risk requires "
                    "continued observation."
            }

        # ==========================================================
        # 5. AIS UNIDENTIFIED
        # ==========================================================

        if intrusion_type == "AIS_UNIDENTIFIED":

            return {
                "navigation_action":
                    "CONTINUE_OBSERVATION",

                "navigation_priority":
                    "MEDIUM",

                "navigation_reason":
                    "Vessel detected without a reliable "
                    "AIS association."
            }

        # ==========================================================
        # 6. LOW / SAFE ENCOUNTER
        # ==========================================================

        if (
            risk_level == "LOW"
            or encounter_type == "SAFE"
        ):

            return {
                "navigation_action":
                    "CONTINUE",

                "navigation_priority":
                    "NONE",

                "navigation_reason":
                    "No significant navigation threat detected."
            }

        # ==========================================================
        # 7. FALLBACK
        # ==========================================================

        return {
            "navigation_action":
                "MONITOR",

            "navigation_priority":
                "LOW",

            "navigation_reason":
                "Situation requires continued observation."
        }

    # ==============================================================
    # PROCESS
    # ==============================================================

    def process(
        self,
        input_file,
        output_file
    ):

        print("\nLoading merged Agent + Encounter data...")

        df = pd.read_csv(input_file)

        print("Total records:", len(df))

        required_columns = [
            "frame",
            "track_id",
            "intrusion_type",
            "severity",
            "decision",
            "encounter_type",
            "risk_level",
            "risk_score",
            "recommended_action"
        ]

        missing = [
            c for c in required_columns
            if c not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing required columns: "
                + str(missing)
            )

        results = []

        for _, row in df.iterrows():

            result = self.decide(row)

            results.append(result)

        navigation_df = pd.DataFrame(
            results
        )

        output = pd.concat(
            [
                df.reset_index(drop=True),
                navigation_df.reset_index(drop=True)
            ],
            axis=1
        )

        os.makedirs(
            os.path.dirname(output_file),
            exist_ok=True
        )

        output.to_csv(
            output_file,
            index=False
        )

        # ==========================================================
        # RESULTS
        # ==========================================================

        print("\n" + "=" * 70)
        print("NAVIGATION DECISION COMPLETE")
        print("=" * 70)

        print("\nOutput:")
        print(output_file)

        print("\nOutput shape:")
        print(output.shape)

        print("\nNavigation action distribution:")

        print(
            output[
                "navigation_action"
            ].value_counts()
        )

        print("\nNavigation priority distribution:")

        print(
            output[
                "navigation_priority"
            ].value_counts()
        )

        print("\nFirst 20 final decisions:")

        sample_columns = [
            "frame",
            "track_id",
            "intrusion_type",
            "severity",
            "encounter_type",
            "risk_level",
            "risk_score",
            "recommended_action",
            "navigation_action",
            "navigation_priority",
            "navigation_reason"
        ]

        print(
            output[
                sample_columns
            ]
            .head(20)
            .to_string(index=False)
        )

        return output


# ==============================================================
# CONFIGURATION
# ==============================================================

INPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/agent_encounter_merged.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/navigation_decisions.csv"
)


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":

    navigator = NavigationDecision()

    result = navigator.process(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE
    )
