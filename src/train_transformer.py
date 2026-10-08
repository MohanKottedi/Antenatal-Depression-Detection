import pandas as pd
from pathlib import Path

csv_path = Path("data/pakdataset.csv")

df = pd.read_csv(csv_path)

df["Labelling"] = (
    df["Labelling"]
    .astype(str)
    .str.strip()
)

print("\nLabel distribution:")
print(df["Labelling"].value_counts())

print("\nScalling statistics by label:")
print(
    df.groupby("Labelling")["Scalling"]
    .describe()
)

print("\nMean Scalling by label:")
print(
    df.groupby("Labelling")["Scalling"]
    .mean()
)