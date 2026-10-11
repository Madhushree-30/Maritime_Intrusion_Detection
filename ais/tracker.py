
import os
import numpy as np
import pandas as pd


class VesselTracker:

    def __init__(
        self,
        max_age=10,
        min_hits=2,
        iou_threshold=0.15,
        max_center_distance=150.0
    ):
        """
        max_age:
            Maximum number of frames a track can disappear
            before being removed.

        min_hits:
            Minimum number of successful detections required
            before a track is considered confirmed.

        iou_threshold:
            Minimum IoU for bounding-box matching.

        max_center_distance:
            Maximum center-to-center distance in pixels when
            IoU is insufficient.
        """

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance

        self.next_track_id = 1
        self.tracks = {}


    # =========================================================
    # IoU
    # =========================================================

    @staticmethod
    def calculate_iou(box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(
            0,
            x2 - x1
        )

        intersection_height = max(
            0,
            y2 - y1
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area1 = max(
            0,
            box1[2] - box1[0]
        ) * max(
            0,
            box1[3] - box1[1]
        )

        area2 = max(
            0,
            box2[2] - box2[0]
        ) * max(
            0,
            box2[3] - box2[1]
        )

        union = (
            area1
            + area2
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union


    # =========================================================
    # Center distance
    # =========================================================

    @staticmethod
    def center_distance(
        cx1,
        cy1,
        cx2,
        cy2
    ):

        return np.sqrt(
            (cx2 - cx1) ** 2
            +
            (cy2 - cy1) ** 2
        )


    # =========================================================
    # Create track
    # =========================================================

    def create_track(
        self,
        detection,
        frame
    ):

        track_id = self.next_track_id

        self.next_track_id += 1

        self.tracks[track_id] = {

            "track_id": track_id,

            "class": detection["class"],

            "source": detection.get(
                "source",
                None
            ),

            "x1": detection["x1"],
            "y1": detection["y1"],
            "x2": detection["x2"],
            "y2": detection["y2"],

            "cx": detection["cx"],
            "cy": detection["cy"],

            "last_frame": frame,

            "age": 0,

            "hits": 1,

            "confirmed": (
                self.min_hits <= 1
            ),

            "previous_cx": detection["cx"],
            "previous_cy": detection["cy"],

            "movement_pixels": 0.0,

            "heading_deg": 0.0
        }

        return track_id


    # =========================================================
    # Update track
    # =========================================================

    def update_track(
        self,
        track_id,
        detection,
        frame
    ):

        track = self.tracks[track_id]

        previous_cx = track["cx"]
        previous_cy = track["cy"]

        current_cx = detection["cx"]
        current_cy = detection["cy"]

        dx = (
            current_cx
            - previous_cx
        )

        dy = (
            current_cy
            - previous_cy
        )

        movement = np.sqrt(
            dx ** 2
            +
            dy ** 2
        )

        # -----------------------------------------------------
        # Image heading
        #
        # atan2 gives direction in image coordinates.
        # Convert to degrees and normalize to 0-360.
        # -----------------------------------------------------

        if movement > 0:

            heading = np.degrees(
                np.arctan2(
                    dy,
                    dx
                )
            )

            heading = (
                heading + 360
            ) % 360

        else:

            heading = track[
                "heading_deg"
            ]

        track.update({

            "class": detection["class"],

            "source": detection.get(
                "source",
                track["source"]
            ),

            "x1": detection["x1"],
            "y1": detection["y1"],
            "x2": detection["x2"],
            "y2": detection["y2"],

            "previous_cx": previous_cx,
            "previous_cy": previous_cy,

            "cx": current_cx,
            "cy": current_cy,

            "last_frame": frame,

            "age": 0,

            "hits": track["hits"] + 1,

            "confirmed": (
                track["hits"] + 1
                >= self.min_hits
            ),

            "movement_pixels": movement,

            "heading_deg": heading
        })


    # =========================================================
    # Find best detection for track
    # =========================================================

    def find_best_match(
        self,
        track,
        detections,
        used_detections
    ):

        best_index = None
        best_score = -1

        track_box = [
            track["x1"],
            track["y1"],
            track["x2"],
            track["y2"]
        ]

        for index, detection in detections.iterrows():

            if index in used_detections:
                continue

            # -------------------------------------------------
            # Class must match
            # -------------------------------------------------

            if (
                str(track["class"])
                != str(detection["class"])
            ):
                continue

            detection_box = [
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"]
            ]

            iou = self.calculate_iou(
                track_box,
                detection_box
            )

            distance = self.center_distance(
                track["cx"],
                track["cy"],
                detection["cx"],
                detection["cy"]
            )

            # -------------------------------------------------
            # Matching rule
            # -------------------------------------------------

            valid = (
                iou >= self.iou_threshold
                or
                distance
                <= self.max_center_distance
            )

            if not valid:
                continue

            # -------------------------------------------------
            # Combined score
            # -------------------------------------------------

            distance_score = max(
                0.0,
                1.0
                -
                (
                    distance
                    /
                    self.max_center_distance
                )
            )

            score = (
                0.7 * iou
                +
                0.3 * distance_score
            )

            if score > best_score:

                best_score = score
                best_index = index

        return best_index, best_score


    # =========================================================
    # Process frame
    # =========================================================

    def process_frame(
        self,
        detections,
        frame
    ):

        used_detections = set()

        # -----------------------------------------------------
        # Existing tracks
        # -----------------------------------------------------

        existing_tracks = list(
            self.tracks.keys()
        )

        # -----------------------------------------------------
        # First match existing tracks
        # -----------------------------------------------------

        for track_id in existing_tracks:

            if track_id not in self.tracks:
                continue

            track = self.tracks[
                track_id
            ]

            best_index, best_score = (
                self.find_best_match(
                    track,
                    detections,
                    used_detections
                )
            )

            if best_index is not None:

                detection = detections.loc[
                    best_index
                ]

                self.update_track(
                    track_id,
                    detection,
                    frame
                )

                used_detections.add(
                    best_index
                )

            else:

                track["age"] += 1

        # -----------------------------------------------------
        # Create new tracks for unmatched detections
        # -----------------------------------------------------

        for index, detection in detections.iterrows():

            if index in used_detections:
                continue

            self.create_track(
                detection,
                frame
            )

        # -----------------------------------------------------
        # Remove old tracks
        # -----------------------------------------------------

        remove_ids = []

        for track_id, track in self.tracks.items():

            if track["age"] > self.max_age:

                remove_ids.append(
                    track_id
                )

        for track_id in remove_ids:

            del self.tracks[
                track_id
            ]

        # -----------------------------------------------------
        # Return tracks updated in this frame
        # -----------------------------------------------------

        frame_results = []

        for track_id, track in self.tracks.items():

            if track["last_frame"] != frame:
                continue

            # -------------------------------------------------
            # Only confirmed tracks
            # -------------------------------------------------

            if not track["confirmed"]:
                continue

            frame_results.append({

                "frame": frame,

                "track_id": track_id,

                "class": track["class"],

                "source": track["source"],

                "x1": track["x1"],
                "y1": track["y1"],
                "x2": track["x2"],
                "y2": track["y2"],

                "cx": track["cx"],
                "cy": track["cy"],

                "movement_pixels":
                    track[
                        "movement_pixels"
                    ],

                "heading_deg":
                    track[
                        "heading_deg"
                    ],

                "track_hits":
                    track["hits"]
            })

        return frame_results


    # =========================================================
    # Main tracking function
    # =========================================================

    def track(
        self,
        detections_file,
        output_file
    ):

        print(
            "Loading detections..."
        )

        detections = pd.read_csv(
            detections_file
        )

        print(
            "Total detections:",
            len(detections)
        )

        required_columns = [
            "frame",
            "class",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy"
        ]

        for column in required_columns:

            if column not in detections.columns:

                raise ValueError(
                    f"Missing required column: "
                    f"{column}"
                )

        detections = detections.sort_values(
            by="frame"
        ).copy()

        results = []

        frames = sorted(
            detections["frame"].unique()
        )

        print(
            "Frames:",
            len(frames)
        )

        # -----------------------------------------------------
        # Process each frame
        # -----------------------------------------------------

        for counter, frame in enumerate(
            frames,
            start=1
        ):

            frame_detections = detections[
                detections["frame"] == frame
            ].copy()

            frame_results = self.process_frame(
                frame_detections,
                frame
            )

            results.extend(
                frame_results
            )

            if (
                counter == 1
                or counter % 20 == 0
                or counter == len(frames)
            ):

                print(
                    f"Processed frame "
                    f"{counter}/{len(frames)}"
                )

        # -----------------------------------------------------
        # Create dataframe
        # -----------------------------------------------------

        tracked = pd.DataFrame(
            results
        )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        os.makedirs(
            os.path.dirname(output_file),
            exist_ok=True
        )

        tracked.to_csv(
            output_file,
            index=False
        )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        print("\n======================================")
        print("TRACKING COMPLETED")
        print("======================================")

        print(
            "Input detections:",
            len(detections)
        )

        print(
            "Tracked detections:",
            len(tracked)
        )

        if len(tracked) > 0:

            print(
                "Unique tracks:",
                tracked["track_id"].nunique()
            )

            print(
                "Unique frames:",
                tracked["frame"].nunique()
            )

            print(
                "Confirmed tracks:",
                tracked["track_id"].nunique()
            )

        print(
            "\nOutput saved to:",
            output_file
        )

        return tracked


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "fused_detections.csv"
)

OUTPUT = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "tracked_fused_objects.csv"
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    tracker = VesselTracker(

        # Allow a vessel to disappear temporarily
        max_age=10,

        # Require at least 2 detections
        min_hits=2,

        # Minimum bounding-box overlap
        iou_threshold=0.15,

        # Maximum movement between frames
        max_center_distance=150.0
    )

    result = tracker.track(
        INPUT,
        OUTPUT
    )

    print("\nFirst 10 records:")

    if len(result) > 0:
        print(
            result.head(10)
        )
    else:
        print(
            "No confirmed tracks found."
        )
