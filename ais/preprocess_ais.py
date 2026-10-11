
# Paste your entire preprocess_ais.py code below this line
import os
import pandas as pd

# ==========================================================
# AIS DATASET PATH (KaggleHub)
# ==========================================================

DATASET_PATH = "/root/.cache/kagglehub/datasets/satyamrajput7913/ais-ship-tracking-vessel-dynamics-and-eta-data/versions/1"

INPUT_FILE = os.path.join(DATASET_PATH, "processed_AIS_dataset.csv")

# ==========================================================
# OUTPUT PATH (Google Drive)
# ==========================================================

OUTPUT_DIR = "/content/drive/MyDrive/ResearchProject/ais"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clean_ais.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class AISPreprocessor:

    def __init__(self, input_file):
        self.input_file = input_file

    def load_dataset(self):

        print("Loading AIS Dataset...\n")

        df = pd.read_csv(self.input_file)

        print(f"Total Records : {len(df)}")
        print(f"Total Columns : {len(df.columns)}")

        return df

    def preprocess(self, df):

        print("\nCleaning Dataset...")

        # Keep only useful columns
        df = df[
            [
                "MMSI",
                "BaseDateTime",
                "LAT",
                "LON",
                "SOG",
                "COG",
                "Heading",
                "VesselName",
                "VesselType",
                "Status"
            ]
        ]

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove missing values
        df = df.dropna()

        # Convert timestamp
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

        # Sort vessel history
        df = df.sort_values(
            by=["MMSI", "BaseDateTime"]
        )

        # Reset index
        df = df.reset_index(drop=True)

        print("Cleaning Completed")

        return df

    def save_dataset(self, df):

        df.to_csv(OUTPUT_FILE, index=False)

        print("\nDataset Saved Successfully")
        print(f"Location : {OUTPUT_FILE}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    processor = AISPreprocessor(INPUT_FILE)

    df = processor.load_dataset()

    clean_df = processor.preprocess(df)

    processor.save_dataset(clean_df)
