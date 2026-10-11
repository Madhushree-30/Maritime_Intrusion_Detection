
import os
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/agentic_ai/"
    "outputs/encounter_classification.csv"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/ResearchProject/agentic_ai/"
    "outputs"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "agent_navigation_decisions.csv"
)


# ============================================================
# AGENT
# ============================================================

class AgenticNavigationAgent:

    def __init__(self):

        self.action_priority = {
            "STOP": 5,
            "TURN_LEFT": 4,
            "TURN_RIGHT": 4,
            "SLOW_DOWN": 3,
            "MAINTAIN_COURSE": 1
        }


    # ========================================================
    # THINK
    # ========================================================

    def think(self, row):

        encounter = row["encounter_type"]
        risk = row["risk_level"]

        distance = row["distance_km"]

        relative_bearing = row[
            "relative_bearing_deg"
        ]

        heading_difference = row[
            "heading_difference_deg"
        ]


        # ----------------------------------------------------
        # SAFE
        # ----------------------------------------------------

        if encounter == "SAFE":

            return {
                "decision": "MAINTAIN_COURSE",
                "confidence": 0.95,
                "reasoning":
                    "No nearby vessel requiring an avoidance maneuver."
            }


        # ----------------------------------------------------
        # HEAD-ON
        # ----------------------------------------------------

        if encounter == "HEAD_ON":

            if risk == "CRITICAL":

                return {
                    "decision": "STOP",
                    "confidence": 0.95,
                    "reasoning":
                        "Critical head-on encounter with extremely small separation."
                }

            if risk == "HIGH":

                return {
                    "decision": "TURN_RIGHT",
                    "confidence": 0.90,
                    "reasoning":
                        "High-risk head-on encounter. A right-side avoidance maneuver is selected."
                }

            return {
                "decision": "SLOW_DOWN",
                "confidence": 0.82,
                "reasoning":
                    "Head-on encounter detected but separation is not yet critical."
            }


        # ----------------------------------------------------
        # CROSSING STARBOARD
        # ----------------------------------------------------

        if encounter == "CROSSING_STARBOARD":

            if risk == "CRITICAL":

                return {
                    "decision": "TURN_LEFT",
                    "confidence": 0.94,
                    "reasoning":
                        "Critical crossing encounter detected on the starboard side. Left maneuver selected."
                }

            if risk == "HIGH":

                return {
                    "decision": "TURN_LEFT",
                    "confidence": 0.88,
                    "reasoning":
                        "High-risk starboard crossing. Left maneuver selected to increase separation."
                }

            if risk == "MODERATE":

                return {
                    "decision": "SLOW_DOWN",
                    "confidence": 0.78,
                    "reasoning":
                        "Moderate-risk starboard crossing. Speed reduction is selected before stronger avoidance."
                }


        # ----------------------------------------------------
        # CROSSING PORT
        # ----------------------------------------------------

        if encounter == "CROSSING_PORT":

            if risk == "CRITICAL":

                return {
                    "decision": "TURN_RIGHT",
                    "confidence": 0.94,
                    "reasoning":
                        "Critical crossing encounter detected on the port side. Right maneuver selected."
                }

            if risk == "HIGH":

                return {
                    "decision": "TURN_RIGHT",
                    "confidence": 0.88,
                    "reasoning":
                        "High-risk port crossing. Right maneuver selected to increase separation."
                }

            if risk == "MODERATE":

                return {
                    "decision": "SLOW_DOWN",
                    "confidence": 0.78,
                    "reasoning":
                        "Moderate-risk port crossing. Speed reduction is selected."
                }


        # ----------------------------------------------------
        # OVERTAKING
        # ----------------------------------------------------

        if encounter == "OVERTAKING":

            if risk == "CRITICAL":

                return {
                    "decision": "STOP",
                    "confidence": 0.90,
                    "reasoning":
                        "Critical overtaking geometry with extremely small separation."
                }

            if risk == "HIGH":

                return {
                    "decision": "SLOW_DOWN",
                    "confidence": 0.86,
                    "reasoning":
                        "High-risk overtaking situation. Speed reduction is selected."
                }

            return {
                "decision": "SLOW_DOWN",
                "confidence": 0.76,
                "reasoning":
                    "Overtaking situation detected. Conservative speed reduction selected."
            }


        # ----------------------------------------------------
        # UNCERTAIN
        # ----------------------------------------------------

        if encounter == "UNCERTAIN":

            # Extremely close + uncertain geometry
            if (
                risk == "CRITICAL"
                and distance is not None
                and distance <= 0.005
            ):

                return {
                    "decision": "STOP",
                    "confidence": 0.85,
                    "reasoning":
                        "Encounter geometry is uncertain but vessel separation is extremely small."
                }


            if risk == "CRITICAL":

                return {
                    "decision": "SLOW_DOWN",
                    "confidence": 0.70,
                    "reasoning":
                        "Critical proximity detected but encounter geometry is uncertain. Conservative reduction selected."
                }


            if risk == "HIGH":

                return {
                    "decision": "SLOW_DOWN",
                    "confidence": 0.78,
                    "reasoning":
                        "High-risk situation with uncertain encounter geometry. Conservative response selected."
                }


            if risk == "MODERATE":

                return {
                    "decision": "SLOW_DOWN",
                    "confidence": 0.65,
                    "reasoning":
                        "Moderate risk with uncertain encounter geometry."
                }


            return {
                "decision": "MAINTAIN_COURSE",
                "confidence": 0.55,
                "reasoning":
                    "No sufficiently strong evidence for an avoidance maneuver."
            }


        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        return {
            "decision": "SLOW_DOWN",
            "confidence": 0.50,
            "reasoning":
                "Unrecognized situation. Conservative action selected."
        }


    # ========================================================
    # PROCESS ONE ROW
    # ========================================================

    def decide(self, row):

        result = self.think(row)

        return pd.Series({

            "agent_decision":
                result["decision"],

            "agent_confidence":
                result["confidence"],

            "agent_reasoning":
                result["reasoning"]
        })


# ============================================================
# MAIN
# ============================================================

print("=" * 65)
print("AGENTIC NAVIGATION AGENT")
print("=" * 65)

print("\nLoading encounter classification...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Input records:",
    len(df)
)


# ============================================================
# CREATE AGENT
# ============================================================

agent = AgenticNavigationAgent()


# ============================================================
# AGENT THINKING / DECISION
# ============================================================

agent_results = df.apply(
    agent.decide,
    axis=1
)


result = pd.concat(
    [
        df,
        agent_results
    ],
    axis=1
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 65)
print("AGENT DECISION DISTRIBUTION")
print("=" * 65)

print(
    result[
        "agent_decision"
    ].value_counts()
)


print("\n")
print("=" * 65)
print("AGENT DECISION PERCENTAGES")
print("=" * 65)

print(
    (
        result[
            "agent_decision"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


print("\n")
print("=" * 65)
print("AVERAGE CONFIDENCE")
print("=" * 65)

print(
    round(
        result[
            "agent_confidence"
        ].mean(),
        3
    )
)


print("\n")
print("=" * 65)
print("SAMPLE AGENT DECISIONS")
print("=" * 65)

columns = [
    "frame",
    "track_id",
    "target_track_id",
    "distance_km",
    "relative_bearing_deg",
    "encounter_type",
    "risk_level",
    "agent_decision",
    "agent_confidence",
    "agent_reasoning"
]

print(
    result[
        columns
    ]
    .head(30)
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 65)
print("OUTPUT")
print("=" * 65)

print(
    OUTPUT_FILE
)

print("\nAgent processing completed.")
