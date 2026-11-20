"""
Data Loader Module for F1 Race Predictor
Loads and prepares raw Ergast Formula 1 datasets.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def load_raw_data(raw_dir="data/raw"):
    """
    Loads all relevant raw CSV files from the raw directory,
    replaces '\\N' markers with np.nan, and casts key identifier columns.
    """
    files = {
        "races": "races.csv",
        "results": "results.csv",
        "drivers": "drivers.csv",
        "constructors": "constructors.csv",
        "circuits": "circuits.csv",
        "qualifying": "qualifying.csv",
        "driver_standings": "driver_standings.csv",
        "constructor_standings": "constructor_standings.csv",
        "status": "status.csv",
    }

    data = {}
    for key, filename in files.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required raw dataset file not found: {filepath}")
        
        df = pd.read_csv(filepath, na_values=[r"\N", "null", "NULL", ""])
        data[key] = df

    # Data types and date parsing
    data["races"]["date"] = pd.to_datetime(data["races"]["date"], errors="coerce")
    data["races"]["year"] = pd.to_numeric(data["races"]["year"], errors="coerce").astype(int)
    data["races"]["round"] = pd.to_numeric(data["races"]["round"], errors="coerce").astype(int)
    
    data["results"]["grid"] = pd.to_numeric(data["results"]["grid"], errors="coerce").fillna(20).astype(int)
    data["results"]["positionOrder"] = pd.to_numeric(data["results"]["positionOrder"], errors="coerce").astype(int)
    data["results"]["points"] = pd.to_numeric(data["results"]["points"], errors="coerce").fillna(0.0)

    if "position" in data["qualifying"].columns:
        data["qualifying"]["qualifying_pos"] = pd.to_numeric(data["qualifying"]["position"], errors="coerce")
    
    # Altitude cleaning in circuits
    data["circuits"]["alt"] = pd.to_numeric(data["circuits"]["alt"], errors="coerce").fillna(0)

    return data


def get_base_race_dataframe(data):
    """
    Creates a base merged DataFrame of all driver race entries,
    sorted strictly chronologically by year, round, and raceId.
    """
    results = data["results"].copy()
    races = data["races"][["raceId", "year", "round", "circuitId", "name", "date"]].rename(
        columns={"name": "race_name"}
    )
    drivers = data["drivers"][["driverId", "driverRef", "code", "forename", "surname", "nationality", "dob"]].rename(
        columns={"nationality": "driver_nationality"}
    )
    constructors = data["constructors"][["constructorId", "constructorRef", "name", "nationality"]].rename(
        columns={"name": "constructor_name", "nationality": "constructor_nationality"}
    )
    circuits = data["circuits"][["circuitId", "circuitRef", "name", "location", "country", "lat", "lng", "alt"]].rename(
        columns={"name": "circuit_name"}
    )
    status = data["status"][["statusId", "status"]]

    # Merge core tables
    merged = results.merge(races, on="raceId", how="inner")
    merged = merged.merge(drivers, on="driverId", how="inner")
    merged = merged.merge(constructors, on="constructorId", how="inner")
    merged = merged.merge(circuits, on="circuitId", how="inner")
    merged = merged.merge(status, on="statusId", how="inner")

    # Qualifying merge if available
    if "qualifying" in data and not data["qualifying"].empty:
        quali_dedup = data["qualifying"][["raceId", "driverId", "qualifying_pos"]].drop_duplicates(
            subset=["raceId", "driverId"]
        )
        merged = merged.merge(quali_dedup, on=["raceId", "driverId"], how="left")
    else:
        merged["qualifying_pos"] = np.nan

    # Fill qualifying position from grid if missing
    merged["qualifying_pos"] = merged["qualifying_pos"].fillna(merged["grid"])

    # Full driver display name
    merged["driver_full_name"] = merged["forename"] + " " + merged["surname"]

    # Define binary target: Podium = 1 if positionOrder in [1, 2, 3] else 0
    merged["podium"] = (merged["positionOrder"] <= 3).astype(int)

    # Sort strictly chronologically
    merged = merged.sort_values(by=["date", "year", "round", "raceId", "positionOrder"]).reset_index(drop=True)

    return merged


if __name__ == "__main__":
    print("Testing data loader...")
    raw_data = load_raw_data("data/raw")
    base_df = get_base_race_dataframe(raw_data)
    print(f"Loaded {len(base_df)} driver race entries across {base_df['raceId'].nunique()} races.")
    print(f"Seasons: {base_df['year'].min()} to {base_df['year'].max()}")
    print(f"Target distribution (podium=1): {base_df['podium'].mean():.4f}")
    print(base_df[["year", "race_name", "driver_full_name", "constructor_name", "grid", "positionOrder", "podium"]].head())
