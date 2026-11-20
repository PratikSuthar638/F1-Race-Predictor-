"""
Prediction and Inference Engine for F1 Race Predictor
Loads the trained ML model and generates calibrated podium predictions with factor explanations.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from src.preprocessing import F1Preprocessor, FEATURE_COLUMNS


class F1PredictionEngine:
    """
    Production-ready inference engine for predicting race podium finishes.
    """
    def __init__(self, model_path="models/best_model.pkl", preprocessor_path="models/preprocessing.pkl", data_path="data/processed/f1_features.csv"):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.data_path = data_path
        
        self.model = None
        self.preprocessor = None
        self.features_df = None
        self.latest_driver_stats = {}
        self.latest_constructor_stats = {}
        self.circuit_stats = {}
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, preprocessor, and historical baselines."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError("Trained model or preprocessor artifact not found. Please run src/train.py first.")
        
        self.model = joblib.load(self.model_path)
        self.preprocessor = F1Preprocessor.load(self.preprocessor_path)
        
        if os.path.exists(self.data_path):
            self.features_df = pd.read_csv(self.data_path)
            self._build_historical_caches()

    def _build_historical_caches(self):
        """Builds lookup dictionaries for drivers, constructors, and circuits based on recent seasons."""
        df = self.features_df.sort_values(by=["year", "round"], ascending=True)
        
        # Cache the most recent baseline for each driver
        for driver_name, group in df.groupby("driver_full_name"):
            latest_row = group.iloc[-1]
            self.latest_driver_stats[driver_name] = {
                "driverId": latest_row["driverId"],
                "last_constructor": latest_row["constructor_name"],
                "driver_avg_pos_last_3": latest_row["driver_avg_pos_last_3"],
                "driver_avg_pos_last_5": latest_row["driver_avg_pos_last_5"],
                "driver_recent_form_score": latest_row["driver_recent_form_score"],
                "driver_career_starts": latest_row["driver_career_starts"],
                "driver_career_podiums": latest_row["driver_career_podiums"],
                "driver_career_podium_rate": latest_row["driver_career_podium_rate"],
                "driver_career_win_rate": latest_row["driver_career_win_rate"],
                "driver_recent_dnf_rate_last_5": latest_row["driver_recent_dnf_rate_last_5"],
                "driver_qualifying_avg_last_5": latest_row["driver_qualifying_avg_last_5"],
                "driver_points_prior": latest_row["driver_points_prior"],
            }

        # Cache the most recent baseline for each constructor
        for c_name, group in df.groupby("constructor_name"):
            latest_row = group.iloc[-1]
            self.latest_constructor_stats[c_name] = {
                "constructorId": latest_row["constructorId"],
                "constructor_avg_pos_last_3": latest_row["constructor_avg_pos_last_3"],
                "constructor_avg_pos_last_5": latest_row["constructor_avg_pos_last_5"],
                "constructor_recent_performance_score": latest_row["constructor_recent_performance_score"],
                "constructor_career_podium_rate": latest_row["constructor_career_podium_rate"],
                "constructor_recent_dnf_rate_last_5": latest_row["constructor_recent_dnf_rate_last_5"],
                "constructor_qualifying_avg_last_5": latest_row["constructor_qualifying_avg_last_5"],
                "constructor_points_prior": latest_row["constructor_points_prior"],
            }

        # Cache circuit stats
        for circuit_name, group in df.groupby("circuit_name"):
            latest_row = group.iloc[-1]
            self.circuit_stats[circuit_name] = {
                "circuitId": latest_row["circuitId"],
                "altitude": latest_row["circuit_altitude"],
                "total_races": latest_row["circuit_total_historical_races"],
            }

    def get_available_options(self):
        """Returns sorted lists of modern active drivers, constructors, and circuits."""
        # Filter for drivers who raced in recent seasons (2020+)
        recent_df = self.features_df[self.features_df["year"] >= 2020]
        recent_drivers = sorted(recent_df["driver_full_name"].unique())
        recent_constructors = sorted(recent_df["constructor_name"].unique())
        all_circuits = sorted(self.features_df["circuit_name"].unique())
        
        # Primary pairing mapping
        driver_teams = {}
        for d in recent_drivers:
            driver_teams[d] = self.latest_driver_stats.get(d, {}).get("last_constructor", recent_constructors[0])

        return {
            "drivers": recent_drivers,
            "constructors": recent_constructors,
            "circuits": all_circuits,
            "driver_teams": driver_teams,
        }

    def predict(self, driver_name, constructor_name, circuit_name, grid_position, 
                driver_form_score=None, constructor_perf_score=None, qualifying_pos=None):
        """
        Executes end-to-end prediction given user UI selections.
        """
        # Validate inputs
        grid = float(grid_position)
        if grid < 1 or grid > 30:
            raise ValueError("Grid position must be between 1 and 30.")
        
        quali = float(qualifying_pos) if qualifying_pos is not None else grid
        if quali < 1 or quali > 30:
            quali = grid

        # Retrieve or fallback driver baseline
        d_base = self.latest_driver_stats.get(driver_name, {
            "driverId": 0,
            "driver_avg_pos_last_3": 12.0,
            "driver_avg_pos_last_5": 12.0,
            "driver_recent_form_score": 5.0,
            "driver_career_starts": 20,
            "driver_career_podiums": 2,
            "driver_career_podium_rate": 0.10,
            "driver_career_win_rate": 0.02,
            "driver_recent_dnf_rate_last_5": 0.15,
            "driver_qualifying_avg_last_5": quali,
            "driver_points_prior": 25.0,
        })

        # Retrieve or fallback constructor baseline
        c_base = self.latest_constructor_stats.get(constructor_name, {
            "constructorId": 0,
            "constructor_avg_pos_last_3": 10.0,
            "constructor_avg_pos_last_5": 10.0,
            "constructor_recent_performance_score": 5.0,
            "constructor_career_podium_rate": 0.15,
            "constructor_recent_dnf_rate_last_5": 0.15,
            "constructor_qualifying_avg_last_5": quali,
            "constructor_points_prior": 50.0,
        })

        # Retrieve circuit baseline
        circ_base = self.circuit_stats.get(circuit_name, {
            "circuitId": 0,
            "altitude": 100.0,
            "total_races": 30,
        })

        # Override with user customized form/performance if provided
        final_driver_form = float(driver_form_score) if driver_form_score is not None else d_base["driver_recent_form_score"]
        final_team_perf = float(constructor_perf_score) if constructor_perf_score is not None else c_base["constructor_recent_performance_score"]

        # Track history at this specific circuit
        track_df = self.features_df[
            (self.features_df["driver_full_name"] == driver_name) & 
            (self.features_df["circuit_name"] == circuit_name)
        ]
        if not track_df.empty:
            driver_circuit_avg_finish = track_df["driver_circuit_avg_finish"].iloc[-1]
            driver_circuit_podium_rate = track_df["driver_circuit_podium_rate"].iloc[-1]
            driver_circuit_starts = track_df["driver_circuit_starts"].iloc[-1]
        else:
            driver_circuit_avg_finish = d_base["driver_avg_pos_last_5"]
            driver_circuit_podium_rate = d_base["driver_career_podium_rate"]
            driver_circuit_starts = 0

        team_track_df = self.features_df[
            (self.features_df["constructor_name"] == constructor_name) & 
            (self.features_df["circuit_name"] == circuit_name)
        ]
        if not team_track_df.empty:
            constructor_circuit_podium_rate = team_track_df["constructor_circuit_podium_rate"].iloc[-1]
        else:
            constructor_circuit_podium_rate = c_base["constructor_career_podium_rate"]

        # Qualifying diff relative to team average
        qualifying_diff_team = quali - c_base["constructor_qualifying_avg_last_5"]

        # Assemble full feature dictionary with native python types
        feature_dict = {
            "grid_position": float(grid),
            "qualifying_position": float(quali),
            "driver_avg_pos_last_3": float(d_base["driver_avg_pos_last_3"]),
            "driver_avg_pos_last_5": float(d_base["driver_avg_pos_last_5"]),
            "driver_recent_form_score": float(final_driver_form),
            "driver_career_starts": int(d_base["driver_career_starts"]),
            "driver_career_podiums": int(d_base["driver_career_podiums"]),
            "driver_career_podium_rate": float(d_base["driver_career_podium_rate"]),
            "driver_career_win_rate": float(d_base["driver_career_win_rate"]),
            "driver_recent_dnf_rate_last_5": float(d_base["driver_recent_dnf_rate_last_5"]),
            "driver_qualifying_avg_last_5": float(d_base["driver_qualifying_avg_last_5"]),
            "driver_points_prior": float(d_base["driver_points_prior"]),
            "constructor_avg_pos_last_3": float(c_base["constructor_avg_pos_last_3"]),
            "constructor_avg_pos_last_5": float(c_base["constructor_avg_pos_last_5"]),
            "constructor_recent_performance_score": float(final_team_perf),
            "constructor_career_podium_rate": float(c_base["constructor_career_podium_rate"]),
            "constructor_recent_dnf_rate_last_5": float(c_base["constructor_recent_dnf_rate_last_5"]),
            "constructor_qualifying_avg_last_5": float(c_base["constructor_qualifying_avg_last_5"]),
            "constructor_points_prior": float(c_base["constructor_points_prior"]),
            "qualifying_diff_team": float(qualifying_diff_team),
            "driver_circuit_avg_finish": float(driver_circuit_avg_finish),
            "driver_circuit_podium_rate": float(driver_circuit_podium_rate),
            "driver_circuit_starts": int(driver_circuit_starts),
            "constructor_circuit_podium_rate": float(constructor_circuit_podium_rate),
            "circuit_altitude": float(circ_base["altitude"]),
            "circuit_total_historical_races": int(circ_base["total_races"]),
        }

        # Scale features
        X_scaled = self.preprocessor.transform(feature_dict)

        # Inference
        if hasattr(self.model, "predict_proba"):
            raw_prob = float(self.model.predict_proba(X_scaled)[0, 1])
        else:
            raw_prob = float(self.model.predict(X_scaled)[0])

        # Podium probability
        podium_prob_pct = round(raw_prob * 100, 1)
        
        # Decision boundary (0.35 calibrated threshold)
        is_podium = bool(raw_prob >= 0.35)
        prediction_label = "PODIUM" if is_podium else "NO PODIUM"
        
        # Model confidence: distance from ambiguous threshold normalized
        confidence_pct = round(float(min(99.0, max(52.0, (abs(raw_prob - 0.35) / 0.65) * 100))), 1)

        # Generate factor explanations
        factors = []
        if grid <= 3:
            factors.append({"type": "positive", "text": f"Top grid starting slot (P{int(grid)}) provides massive front-running advantage."})
        elif grid <= 6:
            factors.append({"type": "positive", "text": f"Competitive qualifying position (P{int(grid)}) within prime podium striking range."})
        else:
            factors.append({"type": "negative", "text": f"Starting from P{int(grid)} requires overtaking multiple midfield cars."})

        if final_driver_form >= 7.5:
            factors.append({"type": "positive", "text": f"Outstanding driver recent form rating of {final_driver_form:.1f}/10."})
        elif final_driver_form <= 4.5:
            factors.append({"type": "negative", "text": f"Recent driver form rating is lower ({final_driver_form:.1f}/10) following recent setbacks."})

        if final_team_perf >= 7.5:
            factors.append({"type": "positive", "text": f"Dominant car pace with constructor performance rating of {final_team_perf:.1f}/10."})
        elif final_team_perf <= 5.0:
            factors.append({"type": "negative", "text": f"Constructor package performance rating ({final_team_perf:.1f}/10) lags behind frontrunners."})

        if driver_circuit_podium_rate >= 0.40 and driver_circuit_starts >= 2:
            factors.append({"type": "positive", "text": f"Strong historical track mastery with {driver_circuit_podium_rate*100:.0f}% previous podium rate at {circuit_name}."})
        elif driver_circuit_starts == 0:
            factors.append({"type": "neutral", "text": f"Driver has no previous Grand Prix starts at {circuit_name}."})

        if d_base["driver_career_podium_rate"] >= 0.30:
            factors.append({"type": "positive", "text": f"Proven career pedigree with {int(d_base['driver_career_podiums'])} career podiums ({d_base['driver_career_podium_rate']*100:.0f}% career rate)."})

        return {
            "driver": str(driver_name),
            "constructor": str(constructor_name),
            "circuit": str(circuit_name),
            "grid_position": int(grid),
            "qualifying_position": int(quali),
            "driver_form_score": float(final_driver_form),
            "constructor_perf_score": float(final_team_perf),
            "prediction": prediction_label,
            "podium_probability": float(podium_prob_pct),
            "confidence": float(confidence_pct),
            "is_podium": bool(is_podium),
            "explanation_factors": factors,
            "feature_snapshot": feature_dict,
        }


if __name__ == "__main__":
    engine = F1PredictionEngine()
    print("Testing F1 Prediction Engine...")
    
    # Test Case 1: Max Verstappen, Red Bull, Monaco, Grid P1
    res1 = engine.predict("Max Verstappen", "Red Bull", "Circuit de Monaco", grid_position=1, driver_form_score=9.5, constructor_perf_score=9.6)
    print("\n--- Test Case 1: Max Verstappen (P1) ---")
    print(f"Prediction: {res1['prediction']} | Prob: {res1['podium_probability']}% | Confidence: {res1['confidence']}%")
    for f in res1["explanation_factors"]:
        print(f"  [{f['type']}] {f['text']}")

    # Test Case 2: Logan Sargeant / Midfield driver, Williams, Grid P18
    res2 = engine.predict("Logan Sargeant", "Williams", "Circuit de Monaco", grid_position=18, driver_form_score=3.2, constructor_perf_score=3.5)
    print("\n--- Test Case 2: Logan Sargeant (P18) ---")
    print(f"Prediction: {res2['prediction']} | Prob: {res2['podium_probability']}% | Confidence: {res2['confidence']}%")
    for f in res2["explanation_factors"]:
        print(f"  [{f['type']}] {f['text']}")
