"""
Feature Engineering Module for F1 Race Predictor
Calculates pre-race features strictly using historical data prior to each race.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from collections import defaultdict


def calculate_form_score(positions, max_pos=20):
    """
    Calculates a 0.0 to 10.0 form score from recent finishes with exponential decay.
    """
    if not positions:
        return 5.0  # neutral mid-grid default
    
    # Exponential decay weights for up to 5 recent races
    n = len(positions)
    weights = np.exp(np.linspace(0, 1, n))
    weights /= weights.sum()
    
    # Calculate weighted position
    weighted_pos = np.sum(np.array(positions) * weights)
    
    # Map position 1 -> 10.0, position 20 -> 0.0
    score = 10.0 * (1.0 - (weighted_pos - 1.0) / (max_pos - 1.0))
    return float(np.clip(score, 0.0, 10.0))


def generate_prerace_features(df, driver_standings=None, constructor_standings=None):
    """
    Processes the chronological base DataFrame and computes strictly leak-free
    pre-race features for every driver-race entry.
    """
    # Sort strictly by date, year, round, raceId
    df = df.sort_values(by=["date", "year", "round", "raceId", "grid"]).reset_index(drop=True)

    # State tracking dictionaries
    driver_history = defaultdict(lambda: {
        "finishes": [],
        "starts": 0,
        "podiums": 0,
        "wins": 0,
        "dnfs": [],
        "quali_pos": [],
        "points_season": defaultdict(float),
    })

    constructor_history = defaultdict(lambda: {
        "finishes": [],
        "starts": 0,
        "podiums": 0,
        "wins": 0,
        "dnfs": [],
        "quali_pos": [],
        "points_season": defaultdict(float),
    })

    # Track historical performance (driver_id, circuit_id) -> finishes
    driver_circuit_history = defaultdict(lambda: {"finishes": [], "podiums": 0, "starts": 0})
    constructor_circuit_history = defaultdict(lambda: {"finishes": [], "podiums": 0, "starts": 0})
    circuit_races_count = defaultdict(int)

    # Output feature records
    feature_rows = []

    # Group by raceId to process each race chronologically
    race_groups = df.groupby("raceId", sort=False)

    for race_id, race_df in race_groups:
        year = race_df["year"].iloc[0]
        round_num = race_df["round"].iloc[0]
        circuit_id = race_df["circuitId"].iloc[0]
        altitude = race_df["alt"].iloc[0] if "alt" in race_df.columns else 0.0

        # First pass for current race: compute constructor's mean qualifying position in this race
        team_quali_in_race = race_df.groupby("constructorId")["qualifying_pos"].mean().to_dict()

        race_pre_features = []

        for idx, row in race_df.iterrows():
            d_id = row["driverId"]
            c_id = row["constructorId"]
            grid = float(row["grid"])
            # Handle grid 0 as 20 (started from pit lane)
            if grid <= 0:
                grid = 20.0
            
            quali = float(row["qualifying_pos"]) if not pd.isna(row["qualifying_pos"]) else grid
            if quali <= 0:
                quali = grid

            # Driver historical state prior to this race
            d_hist = driver_history[d_id]
            d_finishes = d_hist["finishes"]
            d_last_3 = d_finishes[-3:] if len(d_finishes) >= 1 else []
            d_last_5 = d_finishes[-5:] if len(d_finishes) >= 1 else []
            
            driver_avg_pos_last_3 = np.mean(d_last_3) if d_last_3 else 14.0
            driver_avg_pos_last_5 = np.mean(d_last_5) if d_last_5 else 14.0
            driver_recent_form_score = calculate_form_score(d_last_5)
            driver_career_starts = d_hist["starts"]
            driver_career_podiums = d_hist["podiums"]
            driver_career_podium_rate = driver_career_podiums / max(1, driver_career_starts)
            driver_career_win_rate = d_hist["wins"] / max(1, driver_career_starts)
            
            d_dnfs_last_5 = d_hist["dnfs"][-5:] if d_hist["dnfs"] else []
            driver_recent_dnf_rate_last_5 = np.mean(d_dnfs_last_5) if d_dnfs_last_5 else 0.15

            d_quali_last_5 = d_hist["quali_pos"][-5:] if d_hist["quali_pos"] else []
            driver_qualifying_avg_last_5 = np.mean(d_quali_last_5) if d_quali_last_5 else quali
            driver_points_prior = d_hist["points_season"][year]

            # Constructor historical state prior to this race
            c_hist = constructor_history[c_id]
            c_finishes = c_hist["finishes"]
            c_last_3 = c_finishes[-6:] if len(c_finishes) >= 1 else []
            c_last_5 = c_finishes[-10:] if len(c_finishes) >= 1 else []
            
            constructor_avg_pos_last_3 = np.mean(c_last_3) if c_last_3 else 12.0
            constructor_avg_pos_last_5 = np.mean(c_last_5) if c_last_5 else 12.0
            constructor_recent_performance_score = calculate_form_score(c_last_5)
            constructor_career_starts = c_hist["starts"]
            constructor_career_podiums = c_hist["podiums"]
            constructor_career_podium_rate = constructor_career_podiums / max(1, constructor_career_starts)
            
            c_dnfs_last_5 = c_hist["dnfs"][-10:] if c_hist["dnfs"] else []
            constructor_recent_dnf_rate_last_5 = np.mean(c_dnfs_last_5) if c_dnfs_last_5 else 0.15

            c_quali_last_5 = c_hist["quali_pos"][-10:] if c_hist["quali_pos"] else []
            constructor_qualifying_avg_last_5 = np.mean(c_quali_last_5) if c_quali_last_5 else quali
            constructor_points_prior = c_hist["points_season"][year]

            # Qualifying diff with team
            team_race_quali_avg = team_quali_in_race.get(c_id, quali)
            qualifying_diff_team = quali - team_race_quali_avg

            # Circuit historical stats prior to this race
            dc_hist = driver_circuit_history[(d_id, circuit_id)]
            driver_circuit_starts = dc_hist["starts"]
            driver_circuit_avg_finish = np.mean(dc_hist["finishes"]) if dc_hist["finishes"] else driver_avg_pos_last_5
            driver_circuit_podium_rate = dc_hist["podiums"] / max(1, driver_circuit_starts)

            cc_hist = constructor_circuit_history[(c_id, circuit_id)]
            constructor_circuit_starts = cc_hist["starts"]
            constructor_circuit_podium_rate = cc_hist["podiums"] / max(1, constructor_circuit_starts)
            circuit_total_historical_races = circuit_races_count[circuit_id]

            # Compile feature dictionary
            feat = {
                "resultId": row["resultId"],
                "raceId": race_id,
                "year": year,
                "round": round_num,
                "date": row["date"],
                "driverId": d_id,
                "constructorId": c_id,
                "circuitId": circuit_id,
                "driver_full_name": row["driver_full_name"],
                "constructor_name": row["constructor_name"],
                "circuit_name": row["circuit_name"],
                "race_name": row["race_name"],
                # --- Engineered Pre-Race Features ---
                "grid_position": grid,
                "qualifying_position": quali,
                "driver_avg_pos_last_3": driver_avg_pos_last_3,
                "driver_avg_pos_last_5": driver_avg_pos_last_5,
                "driver_recent_form_score": round(driver_recent_form_score, 2),
                "driver_career_starts": driver_career_starts,
                "driver_career_podiums": driver_career_podiums,
                "driver_career_podium_rate": round(driver_career_podium_rate, 4),
                "driver_career_win_rate": round(driver_career_win_rate, 4),
                "driver_recent_dnf_rate_last_5": round(driver_recent_dnf_rate_last_5, 4),
                "driver_qualifying_avg_last_5": round(driver_qualifying_avg_last_5, 2),
                "driver_points_prior": driver_points_prior,
                "constructor_avg_pos_last_3": round(constructor_avg_pos_last_3, 2),
                "constructor_avg_pos_last_5": round(constructor_avg_pos_last_5, 2),
                "constructor_recent_performance_score": round(constructor_recent_performance_score, 2),
                "constructor_career_podium_rate": round(constructor_career_podium_rate, 4),
                "constructor_recent_dnf_rate_last_5": round(constructor_recent_dnf_rate_last_5, 4),
                "constructor_qualifying_avg_last_5": round(constructor_qualifying_avg_last_5, 2),
                "constructor_points_prior": constructor_points_prior,
                "qualifying_diff_team": round(qualifying_diff_team, 2),
                "driver_circuit_avg_finish": round(driver_circuit_avg_finish, 2),
                "driver_circuit_podium_rate": round(driver_circuit_podium_rate, 4),
                "driver_circuit_starts": driver_circuit_starts,
                "constructor_circuit_podium_rate": round(constructor_circuit_podium_rate, 4),
                "circuit_altitude": float(altitude),
                "circuit_total_historical_races": circuit_total_historical_races,
                # --- Target Label (Only known after race) ---
                "positionOrder": row["positionOrder"],
                "podium": int(row["podium"]),
            }
            race_pre_features.append(feat)

        # Append all computed features for this race
        feature_rows.extend(race_pre_features)

        # AFTER recording pre-race features, update state trackers with current race results!
        circuit_races_count[circuit_id] += 1

        for idx, row in race_df.iterrows():
            d_id = row["driverId"]
            c_id = row["constructorId"]
            pos = float(row["positionOrder"])
            points = float(row["points"])
            is_podium = 1 if pos <= 3 else 0
            is_win = 1 if pos == 1 else 0
            # DNF check: statusId != 1 (Finished) and status not containing 'Lap'
            status_text = str(row["status"]).lower()
            is_dnf = 0 if ("finished" in status_text or "lap" in status_text) else 1

            # Update Driver History
            d_hist = driver_history[d_id]
            d_hist["finishes"].append(pos)
            d_hist["starts"] += 1
            d_hist["podiums"] += is_podium
            d_hist["wins"] += is_win
            d_hist["dnfs"].append(is_dnf)
            d_hist["quali_pos"].append(float(row["qualifying_pos"]) if not pd.isna(row["qualifying_pos"]) else float(row["grid"]))
            d_hist["points_season"][year] += points

            # Update Constructor History
            c_hist = constructor_history[c_id]
            c_hist["finishes"].append(pos)
            c_hist["starts"] += 1
            c_hist["podiums"] += is_podium
            c_hist["wins"] += is_win
            c_hist["dnfs"].append(is_dnf)
            c_hist["quali_pos"].append(float(row["qualifying_pos"]) if not pd.isna(row["qualifying_pos"]) else float(row["grid"]))
            c_hist["points_season"][year] += points

            # Update Circuit Histories
            dc_hist = driver_circuit_history[(d_id, circuit_id)]
            dc_hist["finishes"].append(pos)
            dc_hist["starts"] += 1
            dc_hist["podiums"] += is_podium

            cc_hist = constructor_circuit_history[(c_id, circuit_id)]
            cc_hist["finishes"].append(pos)
            cc_hist["starts"] += 1
            cc_hist["podiums"] += is_podium

    features_df = pd.DataFrame(feature_rows)
    return features_df, driver_history, constructor_history, driver_circuit_history, constructor_circuit_history


def save_processed_features(features_df, output_path="data/processed/f1_features.csv"):
    """
    Saves the engineered features to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Engineered features successfully saved to {output_path} ({len(features_df)} rows).")


if __name__ == "__main__":
    from src.data_loader import load_raw_data, get_base_race_dataframe

    print("Running feature engineering pipeline...")
    raw_data = load_raw_data("data/raw")
    base_df = get_base_race_dataframe(raw_data)
    features_df, _, _, _, _ = generate_prerace_features(base_df)
    save_processed_features(features_df)
    
    # Inspect modern era sample
    modern_sample = features_df[features_df["year"] >= 2023].head(5)
    print("\nModern Era Features Sample (2023+):")
    print(modern_sample[["year", "race_name", "driver_full_name", "constructor_name", "grid_position", "driver_recent_form_score", "constructor_recent_performance_score", "podium"]])
