
import math
import pandas as pd


class CPATCPAAnalyzer:

    def __init__(self, neighbor_radius_km=2.0):
        self.neighbor_radius_km = neighbor_radius_km

        print("CPA/TCPA analyzer initialized")


    # =========================================================
    # Geographic distance
    # =========================================================

    def distance_km(self, lat1, lon1, lat2, lon2):

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

        return (
            2
            * R
            * math.atan2(
                math.sqrt(a),
                math.sqrt(1 - a)
            )
        )


    # =========================================================
    # Convert speed + heading to east/north velocity
    #
    # SOG assumed to be knots
    # =========================================================

    def velocity_components(self, speed_knots, heading_deg):

        # knots → km/hour
        speed_kmh = speed_knots * 1.852

        heading = math.radians(
            heading_deg
        )

        # Heading 0 = north
        north = (
            speed_kmh
            * math.cos(heading)
        )

        east = (
            speed_kmh
            * math.sin(heading)
        )

        return east, north


    # =========================================================
    # Approximate local x/y coordinates
    # x = east-west km
    # y = north-south km
    # =========================================================

    def local_coordinates(
        self,
        lat,
        lon,
        ref_lat,
        ref_lon
    ):

        lat_scale = 111.32

        lon_scale = (
            111.32
            * math.cos(
                math.radians(ref_lat)
            )
        )

        x = (
            lon - ref_lon
        ) * lon_scale

        y = (
            lat - ref_lat
        ) * lat_scale

        return x, y


    # =========================================================
    # CPA / TCPA
    #
    # Returns:
    # CPA distance in km
    # TCPA in hours
    # =========================================================

    def calculate_cpa_tcpa(
        self,
        own,
        target
    ):

        ref_lat = own["latitude"]
        ref_lon = own["longitude"]

        x1, y1 = self.local_coordinates(
            own["latitude"],
            own["longitude"],
            ref_lat,
            ref_lon
        )

        x2, y2 = self.local_coordinates(
            target["latitude"],
            target["longitude"],
            ref_lat,
            ref_lon
        )

        rx = x2 - x1
        ry = y2 - y1

        own_east, own_north = (
            self.velocity_components(
                own["speed"],
                own["heading"]
            )
        )

        target_east, target_north = (
            self.velocity_components(
                target["speed"],
                target["heading"]
            )
        )

        rvx = (
            target_east
            - own_east
        )

        rvy = (
            target_north
            - own_north
        )

        relative_speed_sq = (
            rvx ** 2
            +
            rvy ** 2
        )

        relative_position_sq = (
            rx ** 2
            +
            ry ** 2
        )

        # Already stationary relative to each other
        if relative_speed_sq < 1e-8:

            cpa = math.sqrt(
                relative_position_sq
            )

            return cpa, 0.0


        # Time to closest approach
        tcpa = -(
            rx * rvx
            +
            ry * rvy
        ) / relative_speed_sq

        # Negative TCPA means vessels are
        # moving away from each other.
        if tcpa < 0:

            tcpa = 0.0

        cpa_x = (
            rx
            +
            rvx * tcpa
        )

        cpa_y = (
            ry
            +
            rvy * tcpa
        )

        cpa = math.sqrt(
            cpa_x ** 2
            +
            cpa_y ** 2
        )

        return cpa, tcpa


    # =========================================================
    # Risk classification
    # =========================================================

    def classify_risk(
        self,
        cpa_km,
        tcpa_hours
    ):

        tcpa_minutes = (
            tcpa_hours * 60
        )

        # Critical:
        # very close predicted encounter
        # within short time.

        if (
            cpa_km <= 0.05
            and tcpa_minutes <= 5
        ):

            return (
                "CRITICAL",
                100
            )

        # High

        if (
            cpa_km <= 0.10
            and tcpa_minutes <= 10
        ):

            return (
                "HIGH",
                80
            )

        # Moderate

        if (
            cpa_km <= 0.25
            and tcpa_minutes <= 20
        ):

            return (
                "MODERATE",
                55
            )

        # Low

        if (
            cpa_km <= 0.50
            and tcpa_minutes <= 30
        ):

            return (
                "LOW",
                30
            )

        return (
            "SAFE",
            0
        )


    # =========================================================
    # Find nearest vessels in same frame
    # =========================================================

    def analyze_frame(
        self,
        frame_df
    ):

        results = []

        for own_index, own_row in frame_df.iterrows():

            best = None

            for target_index, target_row in frame_df.iterrows():

                if own_index == target_index:
                    continue

                try:

                    own = {
                        "latitude":
                            float(own_row["latitude"]),

                        "longitude":
                            float(own_row["longitude"]),

                        "speed":
                            float(
                                own_row.get(
                                    "speed",
                                    own_row.get(
                                        "SOG",
                                        0
                                    )
                                )
                            ),

                        "heading":
                            float(
                                own_row.get(
                                    "heading_deg",
                                    0
                                )
                            )
                    }

                    target = {
                        "latitude":
                            float(
                                target_row[
                                    "latitude"
                                ]
                            ),

                        "longitude":
                            float(
                                target_row[
                                    "longitude"
                                ]
                            ),

                        "speed":
                            float(
                                target_row.get(
                                    "speed",
                                    target_row.get(
                                        "SOG",
                                        0
                                    )
                                )
                            ),

                        "heading":
                            float(
                                target_row.get(
                                    "heading_deg",
                                    0
                                )
                            )
                    }

                except Exception:
                    continue


                distance = self.distance_km(
                    own["latitude"],
                    own["longitude"],
                    target["latitude"],
                    target["longitude"]
                )

                if (
                    distance
                    > self.neighbor_radius_km
                ):
                    continue


                cpa, tcpa = (
                    self.calculate_cpa_tcpa(
                        own,
                        target
                    )
                )

                risk_level, risk_score = (
                    self.classify_risk(
                        cpa,
                        tcpa
                    )
                )

                candidate = {
                    "target_track_id":
                        target_row.get(
                            "track_id"
                        ),

                    "current_distance_km":
                        distance,

                    "cpa_km":
                        cpa,

                    "tcpa_minutes":
                        tcpa * 60,

                    "risk_level":
                        risk_level,

                    "risk_score":
                        risk_score
                }

                if (
                    best is None
                    or candidate["risk_score"]
                    > best["risk_score"]
                ):

                    best = candidate


            # No nearby vessel

            if best is None:

                results.append({

                    "track_id":
                        own_row.get(
                            "track_id"
                        ),

                    "target_track_id":
                        None,

                    "current_distance_km":
                        None,

                    "cpa_km":
                        None,

                    "tcpa_minutes":
                        None,

                    "risk_level":
                        "SAFE",

                    "risk_score":
                        0

                })

            else:

                result = {
                    "track_id":
                        own_row.get(
                            "track_id"
                        )
                }

                result.update(best)

                results.append(
                    result
                )

        return pd.DataFrame(
            results
        )


    # =========================================================
    # Process dataset
    # =========================================================

    def process(
        self,
        input_file,
        output_file
    ):

        df = pd.read_csv(
            input_file
        )

        print(
            "Input records:",
            len(df)
        )

        all_results = []

        for frame, frame_df in df.groupby(
            "frame"
        ):

            frame_result = self.analyze_frame(
                frame_df
            )

            frame_result["frame"] = frame

            all_results.append(
                frame_result
            )

        result = pd.concat(
            all_results,
            ignore_index=True
        )

        result.to_csv(
            output_file,
            index=False
        )

        print(
            "\n=============================="
        )

        print(
            "CPA/TCPA ANALYSIS"
        )

        print(
            "=============================="
        )

        print(
            "\nRisk levels:"
        )

        print(
            result[
                "risk_level"
            ].value_counts()
        )

        print(
            "\nCPA statistics:"
        )

        print(
            result[
                "cpa_km"
            ].describe()
        )

        print(
            "\nTCPA statistics:"
        )

        print(
            result[
                "tcpa_minutes"
            ].describe()
        )

        print(
            "\nSaved to:"
        )

        print(
            output_file
        )

        return result
