"""
Flask Web Application for F1 Race Predictor
College Machine Learning Project
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

from src.prediction import F1PredictionEngine

app = Flask(__name__)
app.config["SECRET_KEY"] = "f1-race-predictor-secret-key-2026"

# Initialize prediction engine globally
engine = None
try:
    engine = F1PredictionEngine()
    print("F1 Prediction Engine loaded successfully.")
except Exception as e:
    print(f"Warning: Could not initialize prediction engine: {e}")


@app.route("/")
def index():
    """Home landing page with project overview and highlights."""
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    """Interactive race prediction console."""
    if engine is None:
        return render_template("predict.html", error="Model not loaded. Please train the model first.")
    
    options = engine.get_available_options()
    return render_template(
        "predict.html",
        drivers=options["drivers"],
        constructors=options["constructors"],
        circuits=options["circuits"],
        driver_teams=options["driver_teams"],
    )


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard with historical stats and ML comparison."""
    # Load model results table if available
    results_path = os.path.join("reports", "model_results.csv")
    models_table = []
    if os.path.exists(results_path):
        df_res = pd.read_csv(results_path)
        models_table = df_res.to_dict(orient="records")

    metadata_path = os.path.join("models", "model_metadata.pkl")
    metadata = {}
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)

    stats = {
        "total_races": 1125,
        "total_entries": 26759,
        "total_circuits": 77,
        "best_model_name": metadata.get("best_model_name", "Random Forest"),
        "best_roc_auc": metadata.get("metrics", {}).get("ROC-AUC", 0.9237),
        "best_f1": metadata.get("metrics", {}).get("F1", 0.6466),
        "best_accuracy": metadata.get("metrics", {}).get("Accuracy", 0.8596),
        "models_table": models_table,
    }
    return render_template("dashboard.html", stats=stats)


@app.route("/about")
def about():
    """Project documentation, architecture, and viva notes."""
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for real-time race prediction."""
    if engine is None:
        return jsonify({"success": False, "error": "Prediction engine not initialized."}), 500

    try:
        data = request.get_json() or request.form
        driver = data.get("driver")
        constructor = data.get("constructor")
        circuit = data.get("circuit")
        grid_pos = data.get("grid_position", 1)
        quali_pos = data.get("qualifying_position")
        driver_form = data.get("driver_form_score")
        team_perf = data.get("constructor_perf_score")

        if not driver or not constructor or not circuit:
            return jsonify({"success": False, "error": "Please provide driver, constructor, and circuit."}), 400

        result = engine.predict(
            driver_name=driver,
            constructor_name=constructor,
            circuit_name=circuit,
            grid_position=int(grid_pos),
            driver_form_score=float(driver_form) if driver_form is not None and driver_form != "" else None,
            constructor_perf_score=float(team_perf) if team_perf is not None and team_perf != "" else None,
            qualifying_pos=int(quali_pos) if quali_pos is not None and quali_pos != "" else None,
        )

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/driver-stats/<driver_name>")
def get_driver_stats(driver_name):
    """Returns historical baseline stats for a selected driver."""
    if engine is None:
        return jsonify({"error": "Engine not ready"}), 500
    
    stats = engine.latest_driver_stats.get(driver_name, {})
    return jsonify({"success": True, "stats": stats})


@app.route("/api/constructor-stats/<constructor_name>")
def get_constructor_stats(constructor_name):
    """Returns historical baseline stats for a selected constructor."""
    if engine is None:
        return jsonify({"error": "Engine not ready"}), 500
    
    stats = engine.latest_constructor_stats.get(constructor_name, {})
    return jsonify({"success": True, "stats": stats})


@app.route("/api/circuit-stats/<circuit_name>")
def get_circuit_stats(circuit_name):
    """Returns metadata for a selected circuit."""
    if engine is None:
        return jsonify({"error": "Engine not ready"}), 500
    
    stats = engine.circuit_stats.get(circuit_name, {})
    return jsonify({"success": True, "stats": stats})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
