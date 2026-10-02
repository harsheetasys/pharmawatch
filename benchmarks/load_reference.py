"""Lesson 2: download the OMOP reference set (expert-labelled drug-outcome pairs)."""
from pathlib import Path

import pyreadr
import requests

URL = "https://raw.githubusercontent.com/OHDSI/MethodEvaluation/main/data/omopReferenceSet.rda"
DATA_DIR = Path(__file__).parent / "data"
RDA_PATH = DATA_DIR / "omopReferenceSet.rda"
CSV_PATH = DATA_DIR / "omop_reference_set.csv"


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not RDA_PATH.exists():
        r = requests.get(URL, timeout=60)
        r.raise_for_status()
        RDA_PATH.write_bytes(r.content)


def load():
    download()
    df = pyreadr.read_r(str(RDA_PATH))["omopReferenceSet"]
    df = df[["exposureName", "outcomeName", "groundTruth"]].copy()
    df["exposureName"] = df["exposureName"].str.lower()
    df.to_csv(CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    ref = load()
    print(f"{len(ref)} pairs saved to {CSV_PATH}\n")
    print(ref.groupby(["outcomeName", "groundTruth"]).size().unstack(), "\n")
    print("Example positives (drug really causes outcome):")
    print(ref[ref.groundTruth == 1].sample(5, random_state=1).to_string(index=False), "\n")
    print("Example negatives (drug does NOT cause outcome):")
    print(ref[ref.groundTruth == 0].sample(5, random_state=1).to_string(index=False))