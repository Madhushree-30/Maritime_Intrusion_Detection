import os
import pandas as pd


# ======================================================================
# NAVIGATION COMMAND MODULE
# ======================================================================

class NavigationCommand:

    def __init__(self):
        print("Navigation Command module initialized")

    # ==================================================================
    # GENERATE NAVIGATION COMMAND
    # ==================================================================

    def generate_command(self, row):

        navigation_action = row.get(
            "navigation_action",
            "CONTINUE"
        )

        navigation_priority = row.get(
            "navigation_priority",
            "NONE"
        )

        risk_level = row.get(
            "risk_level",
            "LOW"
        )

        risk_score = row.get(
            "risk_score",
            0
        )

        encounter_type = row.get(
            "encounter_type",
            "SAFE"
        )

        # --------------------------------------------------------------
        # CRITICAL STOP
        # --------------------------------------------------------------

        if navigation_action == "ALERT_AND_STOP":

            return {
                "command_alert": True,
                "command_action": "STOP",
                "command_urgency": "CRITICAL",
                "command_status": "EXECUTE",
                "command_reason":
                    "Critical collision-risk encounter detected. "
                    "Stop command generated."
            }

        # --------------------------------------------------------------
        # CRITICAL LEFT TURN
        # --------------------------------------------------------------

        if navigation_action == "ALERT_AND_TURN_LEFT":

            return {
                "command_alert": True,
                "command_action": "TURN_LEFT",
                "command_urgency": "CRITICAL",
                "command_status": "EXECUTE",
                "command_reason":
                    "Critical collision-risk encounter detected. "
                    "Left-turn command generated."
            }

        # --------------------------------------------------------------
        # CRITICAL RIGHT TURN
        # --------------------------------------------------------------

        if navigation_action == "ALERT_AND_TURN_RIGHT":

            return {
                "command_alert": True,
                "command_action": "TURN_RIGHT",
                "command_urgency": "CRITICAL",
                "command_status": "EXECUTE",
                "command_reason":
                    "Critical collision-risk encounter detected. "
                    "Right-turn command generated."
            }

        # --------------------------------------------------------------
        # HIGH-RISK SLOW DOWN
        # --------------------------------------------------------------

        if navigation_action == "ALERT_AND_SLOW_DOWN":

            return {
                "command_alert": True,
                "command_action": "SLOW_DOWN",
                "command_urgency": "HIGH",
                "command_status": "EXECUTE",
                "command_reason":
                    "High-risk encounter detected. "
                    "Slow-down command generated."
            }

        # --------------------------------------------------------------
        # TURN LEFT
        # --------------------------------------------------------------

        if navigation_action == "TURN_LEFT":

            return {
                "command_alert":
                    navigation_priority in
                    ["HIGH", "CRITICAL"],

                "command_action":
                    "TURN_LEFT",

                "command_urgency":
                    navigation_priority,

                "command_status":
                    "EXECUTE",

                "command_reason":
                    "Left-turn navigation command generated."
            }

        # --------------------------------------------------------------
        # TURN RIGHT
        # --------------------------------------------------------------

        if navigation_action == "TURN_RIGHT":

            return {
                "command_alert":
                    navigation_priority in
                    ["HIGH", "CRITICAL"],

                "command_action":
                    "TURN_RIGHT",

                "command_urgency":
                    navigation_priority,

                "command_status":
                    "EXECUTE",

                "command_reason":
                    "Right-turn navigation command generated."
            }

        # --------------------------------------------------------------
        # SLOW DOWN
        # --------------------------------------------------------------

        if navigation_action == "SLOW_DOWN":

            return {
                "command_alert":
                    navigation_priority in
                    ["HIGH", "CRITICAL"],

                "command_action":
                    "SLOW_DOWN",

                "command_urgency":
                    navigation_priority,

                "command_status":
                    "EXECUTE",

                "command_reason":
                    "Encounter risk requires speed reduction."
            }

        # --------------------------------------------------------------
        # CONTINUE OBSERVATION
        # --------------------------------------------------------------

        if navigation_action == "CONTINUE_OBSERVATION":

            return {
                "command_alert":
                    False,

                "command_action":
                    "CONTINUE_OBSERVATION",

                "command_urgency":
                    "MEDIUM",

                "command_status":
                    "MONITOR",

                "command_reason":
                    "Vessel requires continued observation."
            }

        # --------------------------------------------------------------
        # NORMAL CONTINUE
        # --------------------------------------------------------------

        if navigation_action == "CONTINUE":

            return {
                "command_alert":
                    False,

                "command_action":
                    "MAINTAIN_COURSE",

                "command_urgency":
                    "NONE",

                "command_status":
                    "CONTINUE",

                "command_reason":
                    "No significant navigation threat detected."
            }

        # --------------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------------

        return {
            "command_alert":
                True,

            "command_action":
                "MONITOR",

            "command_urgency":
                navigation_priority,

            "command_status":
                "MONITOR",

            "command_reason":
                "Navigation situation requires monitoring."
        }

    # ==================================================================
    # PROCESS NAVIGATION DECISIONS
    # ==================================================================

    def process(
        self,
        input_file,
        output_file
    ):

        print("\n" + "=" * 70)
        print("NAVIGATION COMMAND GENERATION")
        print("=" * 70)

        # --------------------------------------------------------------
        # Load navigation decisions
        # --------------------------------------------------------------

        print("\nLoading Navigation Decisions:")
        print(input_file)

        df = pd.read_csv(input_file)

        print(
            "\nNavigation decision records:",
            len(df)
        )

        # --------------------------------------------------------------
        # Required columns
        # --------------------------------------------------------------

        required_columns = [
            "navigation_action",
            "navigation_priority"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + str(missing_columns)
            )

        # --------------------------------------------------------------
        # Generate commands
        # --------------------------------------------------------------

        print("\nGenerating navigation commands...")

        commands = []

        for _, row in df.iterrows():

            command = self.generate_command(row)

            commands.append(command)

        command_df = pd.DataFrame(
            commands
        )

        # --------------------------------------------------------------
        # Combine original navigation decisions
        # with generated commands
        # --------------------------------------------------------------

        output = pd.concat(
            [
                df.reset_index(drop=True),
                command_df.reset_index(drop=True)
            ],
            axis=1
        )

        # --------------------------------------------------------------
        # Create output directory
        # --------------------------------------------------------------

        output_directory = os.path.dirname(
            output_file
        )

        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

        # --------------------------------------------------------------
        # Save output
        # --------------------------------------------------------------

        output.to_csv(
            output_file,
            index=False
        )

        # ==============================================================
        # RESULTS
        # ==============================================================

        print("\n" + "=" * 70)
        print("NAVIGATION COMMAND COMPLETE")
        print("=" * 70)

        print("\nInput records:")
        print(len(df))

        print("\nOutput shape:")
        print(output.shape)

        print("\nCommand action distribution:")

        print(
            output[
                "command_action"
            ].value_counts()
            .to_string()
        )

        print("\nCommand urgency distribution:")

        print(
            output[
                "command_urgency"
            ].value_counts()
            .to_string()
        )

        print("\nCommand status distribution:")

        print(
            output[
                "command_status"
            ].value_counts()
            .to_string()
        )

        print("\nAlert distribution:")

        print(
            output[
                "command_alert"
            ].value_counts()
            .to_string()
        )

        print("\nOutput saved to:")

        print(output_file)

        return output


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/navigation_decisions.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/navigation_commands.csv"
)


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":

    navigator = NavigationCommand()

    result = navigator.process(
        INPUT_FILE,
        OUTPUT_FILE
    )

    # ==============================================================
    # DISPLAY SAMPLE RESULTS
    # ==============================================================

    print("\n" + "=" * 70)
    print("SAMPLE NAVIGATION COMMANDS")
    print("=" * 70)

    display_columns = [
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
        "command_alert",
        "command_action",
        "command_urgency",
        "command_status",
        "command_reason"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in result.columns
    ]

    print(
        result[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    # ==============================================================
    # CRITICAL COMMANDS
    # ==============================================================

    print("\n" + "=" * 70)
    print("CRITICAL NAVIGATION COMMANDS")
    print("=" * 70)

    critical = result[
        result["command_urgency"] == "CRITICAL"
    ]

    print(
        critical[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("MODULE COMPLETE")
    print("=" * 70)
