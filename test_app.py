"""
Automated Test Suite for F1 Race Predictor Flask Application
Tests all endpoints, prediction engine, invalid inputs, and 5 distinct driver test cases.
"""

import sys
import unittest
from app import app


class F1AppTestCase(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page(self):
        """Test GET / returns 200 and contains hero text."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"FORMULA 1 PODIUM PREDICTOR", response.data)

    def test_predict_page(self):
        """Test GET /predict returns 200 and contains driver selectors."""
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Select Track", response.data)
        self.assertIn(b"Select Driver", response.data)

    def test_dashboard_page(self):
        """Test GET /dashboard returns 200 and contains KPI metrics."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"F1 PREDICTION ANALYTICS DASHBOARD", response.data)
        self.assertIn(b"Random Forest", response.data)

    def test_about_page(self):
        """Test GET /about returns 200 and contains project methodology."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PROJECT ARCHITECTURE", response.data)

    def test_prediction_api_success_cases(self):
        """Test 5 distinct driver/race prediction combinations."""
        test_cases = [
            {
                "name": "Case 1: Max Verstappen P1 at Monaco",
                "payload": {
                    "driver": "Max Verstappen",
                    "constructor": "Red Bull",
                    "circuit": "Circuit de Monaco",
                    "grid_position": 1,
                    "qualifying_position": 1,
                    "driver_form_score": 9.5,
                    "constructor_perf_score": 9.6,
                },
                "expected_podium": True,
            },
            {
                "name": "Case 2: Charles Leclerc P2 at Monza",
                "payload": {
                    "driver": "Charles Leclerc",
                    "constructor": "Ferrari",
                    "circuit": "Autodromo Nazionale di Monza",
                    "grid_position": 2,
                    "qualifying_position": 2,
                    "driver_form_score": 8.5,
                    "constructor_perf_score": 8.8,
                },
                "expected_podium": True,
            },
            {
                "name": "Case 3: Lando Norris P3 at Silverstone",
                "payload": {
                    "driver": "Lando Norris",
                    "constructor": "McLaren",
                    "circuit": "Silverstone Circuit",
                    "grid_position": 3,
                    "qualifying_position": 3,
                    "driver_form_score": 8.8,
                    "constructor_perf_score": 9.0,
                },
                "expected_podium": True,
            },
            {
                "name": "Case 4: Lewis Hamilton P5 at Spa",
                "payload": {
                    "driver": "Lewis Hamilton",
                    "constructor": "Mercedes",
                    "circuit": "Circuit de Spa-Francorchamps",
                    "grid_position": 5,
                    "qualifying_position": 5,
                    "driver_form_score": 7.5,
                    "constructor_perf_score": 7.8,
                },
                "expected_podium": None,  # Can be podium or close
            },
            {
                "name": "Case 5: Logan Sargeant P19 at Monaco",
                "payload": {
                    "driver": "Logan Sargeant",
                    "constructor": "Williams",
                    "circuit": "Circuit de Monaco",
                    "grid_position": 19,
                    "qualifying_position": 19,
                    "driver_form_score": 3.0,
                    "constructor_perf_score": 3.2,
                },
                "expected_podium": False,
            },
        ]

        for tc in test_cases:
            res = self.client.post("/api/predict", json=tc["payload"])
            self.assertEqual(res.status_code, 200, f"Failed on {tc['name']}")
            json_data = res.get_json()
            self.assertTrue(json_data["success"])
            d = json_data["data"]
            self.assertIn("podium_probability", d)
            self.assertIn("confidence", d)
            self.assertIn("explanation_factors", d)
            print(f"\n[PASS] {tc['name']}: Result={d['prediction']} (Prob={d['podium_probability']}%, Conf={d['confidence']}%)")

    def test_prediction_api_invalid_inputs(self):
        """Test API error handling with missing or out-of-range inputs."""
        # Missing driver
        res1 = self.client.post("/api/predict", json={"constructor": "Ferrari", "circuit": "Monaco"})
        self.assertEqual(res1.status_code, 400)
        self.assertFalse(res1.get_json()["success"])

        # Grid out of bounds (> 30)
        res2 = self.client.post("/api/predict", json={
            "driver": "Charles Leclerc",
            "constructor": "Ferrari",
            "circuit": "Circuit de Monaco",
            "grid_position": 45
        })
        self.assertEqual(res2.status_code, 400)
        self.assertFalse(res2.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
