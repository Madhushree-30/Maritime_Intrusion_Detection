
import math
import pandas as pd


class MaritimeActionAgent:
    """
    Research prototype for autonomous maritime action selection.

    The agent:
        OBSERVE -> ASSESS -> GENERATE ACTIONS
        -> SCORE ACTIONS -> SELECT ACTION

    It outputs a recommended maneuver for simulation/research.
    It does not directly control a real vessel.
    """

    ACTIONS = [
        "STOP",
        "SLOW_DOWN",
        "TURN_LEFT",
        "TURN_RIGHT",
        "MAINTAIN_COURSE"
    ]

    def __init__(self):

        print("Maritime Action Agent initialized")


    # =====================================================
    # OBSERVE
    # =====================================================

    def observe(self, row):

        return {
            "track_id": row.get("track_id"),
            "mmsi": row.get("MMSI"),

            "latitude": self.number(
                row.get("latitude")
            ),

            "longitude": self.number(
                row.get("longitude")
            ),

            "speed": self.number(
                row.get("Speed", row.get("SOG", 0))
            ),

            "heading": self.number(
                row.get("heading_deg",
                        row.get("Heading", 0))
            ),

            "intrusion": row.get(
                "intrusion",
                False
            ),

            "intrusion_type": row.get(
                "intrusion_type",
                "NORMAL"
            ),

            "severity": row.get(
                "severity",
                "NONE"
            ),

            "ais_state": row.get(
                "ais_state",
                "UNKNOWN"
            ),

            "match_quality": row.get(
                "match_quality",
                "unmatched"
            ),

            "distance_to_ais_km": self.number(
                row.get("distance_to_ais_km")
            )
        }


    # =====================================================
    # UTILITY
    # =====================================================

    @staticmethod
    def number(value):

        try:

            if pd.isna(value):
                return 0.0

            return float(value)

        except:

            return 0.0


    # =====================================================
    # RISK ASSESSMENT
    # =====================================================

    def assess_risk(self, state):

        risk = 0.0
        reasons = []

        if state["intrusion"] is True:

            risk += 50

            reasons.append(
                "vessel is inside surveillance zone"
            )

        if state["severity"] == "CRITICAL":

            risk += 40

            reasons.append(
                "critical intrusion condition"
            )

        elif state["severity"] == "HIGH":

            risk += 25

            reasons.append(
                "high severity intrusion"
            )

        elif state["severity"] == "MEDIUM":

            risk += 10

            reasons.append(
                "medium uncertainty"
            )

        if state["ais_state"] == "UNKNOWN":

            risk += 20

            reasons.append(
                "AIS identity is unknown"
            )

        if state["match_quality"] == "weak":

            risk += 10

            reasons.append(
                "weak AIS association"
            )

        # Cap risk at 100
        risk = min(risk, 100)

        return risk, reasons


    # =====================================================
    # ACTION SCORING
    # =====================================================

    def score_actions(self, state, risk):

        scores = {
            "STOP": 0.0,
            "SLOW_DOWN": 0.0,
            "TURN_LEFT": 0.0,
            "TURN_RIGHT": 0.0,
            "MAINTAIN_COURSE": 0.0
        }

        explanations = {}

        # -------------------------------------------------
        # Normal situation
        # -------------------------------------------------

        if risk < 20:

            scores["MAINTAIN_COURSE"] += 80

            explanations[
                "MAINTAIN_COURSE"
            ] = "Low assessed risk."

            scores["SLOW_DOWN"] += 20

            explanations[
                "SLOW_DOWN"
            ] = "Conservative alternative."

        # -------------------------------------------------
        # Moderate risk
        # -------------------------------------------------

        elif risk < 50:

            scores["SLOW_DOWN"] += 60

            explanations[
                "SLOW_DOWN"
            ] = (
                "Reducing speed provides more "
                "time for observation and response."
            )

            scores["TURN_LEFT"] += 40
            scores["TURN_RIGHT"] += 40

            explanations[
                "TURN_LEFT"
            ] = "Course alteration can reduce exposure."

            explanations[
                "TURN_RIGHT"
            ] = "Course alteration can reduce exposure."

        # -------------------------------------------------
        # High risk
        # -------------------------------------------------

        elif risk < 80:

            scores["SLOW_DOWN"] += 55

            scores["TURN_LEFT"] += 65
            scores["TURN_RIGHT"] += 65

            explanations[
                "SLOW_DOWN"
            ] = "High risk requires conservative motion."

            explanations[
                "TURN_LEFT"
            ] = "Avoid continuing the current course."

            explanations[
                "TURN_RIGHT"
            ] = "Avoid continuing the current course."

            scores["STOP"] += 50

            explanations[
                "STOP"
            ] = "Stopping provides maximum reaction time."

        # -------------------------------------------------
        # Critical risk
        # -------------------------------------------------

        else:

            scores["STOP"] += 100

            explanations[
                "STOP"
            ] = (
                "Critical risk; stopping is the "
                "most conservative research action."
            )

            scores["TURN_LEFT"] += 70
            scores["TURN_RIGHT"] += 70

            explanations[
                "TURN_LEFT"
            ] = "Emergency course alteration candidate."

            explanations[
                "TURN_RIGHT"
            ] = "Emergency course alteration candidate."

        return scores, explanations


    # =====================================================
    # DECISION
    # =====================================================

    def choose_action(
        self,
        state,
        risk,
        scores,
        explanations
    ):

        # Highest score wins
        action = max(
            scores,
            key=scores.get
        )

        max_score = scores[action]

        # Convert score to approximate
        # decision confidence.
        confidence = min(
            max_score / 100.0,
            1.0
        )

        return {
            "recommended_action": action,
            "action_score": round(
                max_score,
                2
            ),
            "action_confidence": round(
                confidence,
                3
            ),
            "reason": explanations[action]
        }


    # =====================================================
    # COMPLETE AGENT CYCLE
    # =====================================================

    def run(self, row):

        # 1. Observe
        state = self.observe(row)

        # 2. Assess
        risk, risk_reasons = (
            self.assess_risk(state)
        )

        # 3. Generate and score actions
        scores, explanations = (
            self.score_actions(
                state,
                risk
            )
        )

        # 4. Select action
        decision = self.choose_action(
            state,
            risk,
            scores,
            explanations
        )

        result = {}

        result.update(state)

        result["risk_score"] = round(
            risk,
            2
        )

        result["risk_reasons"] = (
            "; ".join(risk_reasons)
        )

        result.update(decision)

        return result


    # =====================================================
    # DATASET PROCESSING
    # =====================================================

    def process(
        self,
        input_file,
        output_file
    ):

        print(
            "\nLoading Agentic AI observations..."
        )

        df = pd.read_csv(
            input_file
        )

        print(
            "Input records:",
            len(df)
        )

        results = []

        for _, row in df.iterrows():

            result = self.run(row)

            results.append(result)

        output = pd.DataFrame(
            results
        )

        output.to_csv(
            output_file,
            index=False
        )

        print(
            "\n======================================"
        )

        print(
            "AGENTIC ACTION PLANNING RESULTS"
        )

        print(
            "======================================"
        )

        print(
            "\nRecommended actions:"
        )

        print(
            output[
                "recommended_action"
            ].value_counts()
        )

        print(
            "\nRisk distribution:"
        )

        print(
            output[
                "risk_score"
            ].describe()
        )

        print(
            "\nOutput saved to:"
        )

        print(output_file)

        return output
