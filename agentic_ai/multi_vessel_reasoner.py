
import math
import pandas as pd


class MultiVesselReasoner:

    def __init__(
        self,
        proximity_km=2.0,
        critical_distance_km=0.5
    ):
        self.proximity_km = proximity_km
        self.critical_distance_km = critical_distance_km

        print("Multi-vessel reasoning module initialized")


    # --------------------------------------------------
    # Distance between two geographic positions
    # --------------------------------------------------

    def distance_km(
        self,
        lat1,
        lon1,
        lat2,
        lon2
    ):

        R = 6371.0

        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            +
            math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        return R * 2 * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )


    # --------------------------------------------------
    # Bearing from vessel A to vessel B
    # --------------------------------------------------

    def bearing(
        self,
        lat1,
        lon1,
        lat2,
        lon2
    ):

        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        dlon = math.radians(
            lon2 - lon1
        )

        x = (
            math.sin(dlon)
            * math.cos(lat2)
        )

        y = (
            math.cos(lat1)
            * math.sin(lat2)
            -
            math.sin(lat1)
            * math.cos(lat2)
            * math.cos(dlon)
        )

        bearing = math.degrees(
            math.atan2(x, y)
        )

        return (bearing + 360) % 360


    # --------------------------------------------------
    # Difference between two headings
    # --------------------------------------------------

    def heading_difference(
        self,
        h1,
        h2
    ):

        diff = abs(h1 - h2)

        if diff > 180:
            diff = 360 - diff

        return diff


    # --------------------------------------------------
    # Find vessels near the current vessel
    # --------------------------------------------------

    def find_neighbors(
        self,
        frame_df,
        current_index
    ):

        current = frame_df.loc[
            current_index
        ]

        neighbors = []

        for idx, vessel in frame_df.iterrows():

            if idx == current_index:
                continue

            try:

                distance = self.distance_km(
                    float(current["latitude"]),
                    float(current["longitude"]),
                    float(vessel["latitude"]),
                    float(vessel["longitude"])
                )

            except:

                continue

            if distance <= self.proximity_km:

                bearing = self.bearing(
                    float(current["latitude"]),
                    float(current["longitude"]),
                    float(vessel["latitude"]),
                    float(vessel["longitude"])
                )

                neighbors.append({
                    "index": idx,
                    "track_id": vessel.get(
                        "track_id"
                    ),
                    "distance_km": distance,
                    "bearing_deg": bearing,
                    "heading_deg": float(
                        vessel.get(
                            "heading_deg",
                            0
                        )
                    )
                })

        return neighbors


    # --------------------------------------------------
    # Analyze the maritime situation
    # --------------------------------------------------

    def analyze(
        self,
        frame_df,
        current_index
    ):

        current = frame_df.loc[
            current_index
        ]

        neighbors = self.find_neighbors(
            frame_df,
            current_index
        )

        situation = {
            "neighbor_count": len(neighbors),
            "closest_distance_km": None,
            "closest_bearing_deg": None,
            "closest_track_id": None,
            "risk_score": 0.0,
            "risk_level": "LOW",
            "reason": "No nearby vessel."
        }

        if not neighbors:
            return situation

        closest = min(
            neighbors,
            key=lambda x: x["distance_km"]
        )

        distance = closest[
            "distance_km"
        ]

        situation[
            "closest_distance_km"
        ] = round(
            distance,
            4
        )

        situation[
            "closest_bearing_deg"
        ] = round(
            closest["bearing_deg"],
            2
        )

        situation[
            "closest_track_id"
        ] = closest[
            "track_id"
        ]

        # ------------------------------------------
        # Distance-based risk
        # ------------------------------------------

        if distance <= self.critical_distance_km:

            risk = 100
            level = "CRITICAL"

        elif distance <= 1.0:

            risk = 75
            level = "HIGH"

        elif distance <= 1.5:

            risk = 50
            level = "MEDIUM"

        else:

            risk = 25
            level = "LOW"

        situation[
            "risk_score"
        ] = risk

        situation[
            "risk_level"
        ] = level

        situation[
            "reason"
        ] = (
            f"{len(neighbors)} nearby vessel(s); "
            f"closest vessel is "
            f"{distance:.3f} km away at bearing "
            f"{closest['bearing_deg']:.1f} degrees."
        )

        return situation


    # --------------------------------------------------
    # Recommend action
    # --------------------------------------------------

    def recommend_action(
        self,
        situation
    ):

        risk = situation[
            "risk_score"
        ]

        bearing = situation[
            "closest_bearing_deg"
        ]

        if bearing is None:

            return (
                "MAINTAIN_COURSE",
                "No nearby vessel detected."
            )

        # ------------------------------------------
        # Critical proximity
        # ------------------------------------------

        if risk >= 100:

            return (
                "STOP",
                "Critical proximity to another vessel."
            )

        # ------------------------------------------
        # High risk
        # ------------------------------------------

        if risk >= 75:

            # Vessel is ahead
            if 315 <= bearing or bearing <= 45:

                return (
                    "SLOW_DOWN",
                    "High-risk vessel detected ahead."
                )

            # Vessel on right
            if 45 < bearing < 180:

                return (
                    "TURN_LEFT",
                    "High-risk vessel detected on starboard side."
                )

            # Vessel on left
            if 180 <= bearing < 315:

                return (
                    "TURN_RIGHT",
                    "High-risk vessel detected on port side."
                )

        # ------------------------------------------
        # Medium risk
        # ------------------------------------------

        if risk >= 50:

            return (
                "SLOW_DOWN",
                "Nearby vessel requires increased caution."
            )

        # ------------------------------------------
        # Low risk
        # ------------------------------------------

        return (
            "MAINTAIN_COURSE",
            "Nearby vessel is outside the immediate risk zone."
        )


    # --------------------------------------------------
    # Process complete dataset
    # --------------------------------------------------

    def process(
        self,
        input_file,
        output_file
    ):

        df = pd.read_csv(
            input_file
        )

        required = [
            "frame",
            "track_id",
            "latitude",
            "longitude"
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing columns: "
                + str(missing)
            )

        results = []

        print(
            "Processing frames..."
        )

        for frame, frame_df in df.groupby(
            "frame"
        ):

            frame_df = frame_df.copy()

            for index in frame_df.index:

                situation = self.analyze(
                    frame_df,
                    index
                )

                action, action_reason = (
                    self.recommend_action(
                        situation
                    )
                )

                results.append({
                    "frame": frame,
                    "track_id":
                        frame_df.loc[
                            index,
                            "track_id"
                        ],

                    "neighbor_count":
                        situation[
                            "neighbor_count"
                        ],

                    "closest_distance_km":
                        situation[
                            "closest_distance_km"
                        ],

                    "closest_bearing_deg":
                        situation[
                            "closest_bearing_deg"
                        ],

                    "closest_track_id":
                        situation[
                            "closest_track_id"
                        ],

                    "spatial_risk_score":
                        situation[
                            "risk_score"
                        ],

                    "spatial_risk_level":
                        situation[
                            "risk_level"
                        ],

                    "recommended_action":
                        action,

                    "action_reason":
                        action_reason
                })

        result = pd.DataFrame(
            results
        )

        result.to_csv(
            output_file,
            index=False
        )

        print(
            "\n================================"
        )

        print(
            "MULTI-VESSEL REASONING RESULTS"
        )

        print(
            "================================"
        )

        print(
            "\nTotal decisions:",
            len(result)
        )

        print(
            "\nActions:"
        )

        print(
            result[
                "recommended_action"
            ].value_counts()
        )

        print(
            "\nRisk levels:"
        )

        print(
            result[
                "spatial_risk_level"
            ].value_counts()
        )

        print(
            "\nSaved to:"
        )

        print(output_file)

        return result
