
import os
import pandas as pd


class GeoReferencer:

    def __init__(self, tile_bounds_file):

        self.bounds = pd.read_csv(tile_bounds_file)

        # Make sure frame is integer
        self.bounds["frame"] = self.bounds["frame"].astype(int)

        print("Tile bounds loaded successfully")
        print("Number of frame bounds:", len(self.bounds))


    def pixel_to_geo(self, cx, cy, frame, image_width, image_height):

        # Find geographic bounds for this frame
        tile = self.bounds[
            self.bounds["frame"] == int(frame)
        ]

        if tile.empty:
            return None, None

        tile = tile.iloc[0]

        lat_min = tile["lat_min"]
        lat_max = tile["lat_max"]

        lon_min = tile["lon_min"]
        lon_max = tile["lon_max"]

        # Pixel X -> longitude
        lon = lon_min + (
            cx / image_width
        ) * (lon_max - lon_min)

        # Pixel Y -> latitude
        # Image Y increases downward,
        # geographic latitude increases upward.
        lat = lat_max - (
            cy / image_height
        ) * (lat_max - lat_min)

        return lat, lon


    def process(
        self,
        tracked_file,
        output_file,
        image_width=640,
        image_height=640
    ):

        df = pd.read_csv(tracked_file)

        print("Tracked detections:", len(df))

        latitudes = []
        longitudes = []

        for _, row in df.iterrows():

            lat, lon = self.pixel_to_geo(
                row["cx"],
                row["cy"],
                row["frame"],
                image_width,
                image_height
            )

            latitudes.append(lat)
            longitudes.append(lon)

        df["latitude"] = latitudes
        df["longitude"] = longitudes

        os.makedirs(
            os.path.dirname(output_file),
            exist_ok=True
        )

        df.to_csv(
            output_file,
            index=False
        )

        print("\nGeoreferencing completed successfully")
        print("Output saved to:")
        print(output_file)

        print("\nSample:")
        print(
            df[
                [
                    "frame",
                    "track_id",
                    "class",
                    "latitude",
                    "longitude"
                ]
            ].head()
        )

        return df


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

TILE_BOUNDS = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "tile_bounds_los_angeles.csv"
)

TRACKED_FILE = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "tracked_fused_objects.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/ais/"
    "georeferenced_detections.csv"
)


# -------------------------------------------------------
# Run
# -------------------------------------------------------

if __name__ == "__main__":

    geo = GeoReferencer(
        TILE_BOUNDS
    )

    result = geo.process(
        TRACKED_FILE,
        OUTPUT_FILE
    )

    print("\nFirst 10 rows:")
    print(result.head(10))
