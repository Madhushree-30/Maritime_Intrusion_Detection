
import pandas as pd
import os


class AISParser:
    """
    Loads and provides access to the cleaned AIS dataset.
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

    def load_dataset(self):
        self.df = pd.read_csv(self.csv_path)

        self.df["BaseDateTime"] = pd.to_datetime(
            self.df["BaseDateTime"]
        )

        print("\nDataset Loaded Successfully")
        print(f"Total Records : {len(self.df)}")

        return self.df

    def get_all_vessels(self):
        return self.df

    def get_vessel_by_mmsi(self, mmsi):
        vessel = self.df[self.df["MMSI"] == mmsi]
        return vessel

    def get_latest_position(self, mmsi):
        vessel = self.get_vessel_by_mmsi(mmsi)

        if vessel.empty:
            return None

        latest = vessel.sort_values(
            by="BaseDateTime"
        ).iloc[-1]

        return {
            "MMSI": latest["MMSI"],
            "Latitude": latest["LAT"],
            "Longitude": latest["LON"],
            "Speed": latest["SOG"],
            "Course": latest["COG"],
            "Heading": latest["Heading"],
            "Status": latest["Status"],
            "Timestamp": latest["BaseDateTime"]
        }

    def get_unique_vessels(self):
        return self.df["MMSI"].nunique()

    def extract_los_angeles_region(self, output_file):
        """
        Extract vessels within the Los Angeles Port region.
        """

        la = self.df[
            (self.df["LAT"] >= 33.65) &
            (self.df["LAT"] <= 33.80) &
            (self.df["LON"] >= -118.30) &
            (self.df["LON"] <= -118.15)
        ].copy()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        la.to_csv(output_file, index=False)

        print("\nLos Angeles AIS dataset created")
        print(f"Records : {len(la)}")
        print(f"Saved to : {output_file}")

        return la


if __name__ == "__main__":
    INPUT = "/content/drive/MyDrive/ResearchProject/ais/clean_ais.csv"
    OUTPUT = "/content/drive/MyDrive/ResearchProject/ais/locations/los_angeles.csv"

    parser = AISParser(INPUT)
    parser.load_dataset()
    parser.extract_los_angeles_region(OUTPUT)
