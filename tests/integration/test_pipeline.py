# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from backend.data_validation.dataset_validator import run_validation_checks
from backend.mlops.drift.data_drift_monitor import calculate_feature_drift
from backend.ml.eif_detector import load_or_train_eif

def test_data_validation():
    print("Running data validation test...")
    res = run_validation_checks()
    assert "crop_recommendation" in res
    assert res["crop_recommendation"]["rows"] > 0
    print("Data validation test PASSED.")

def test_drift_calculation():
    print("Running drift calculation test...")
    ref = np.random.normal(10, 1, 100)
    curr = np.random.normal(10, 1, 100)
    res = calculate_feature_drift(ref, curr)
    assert "p_value" in res
    assert "drift_detected" in res
    print("Drift calculation test PASSED.")

def test_eif_anomaly_scores():
    print("Running EIF anomaly scores test...")
    model = load_or_train_eif()
    x = np.random.normal(40, 2, (1, 7))
    score = model.compute_anomaly_score(x)[0]
    assert 0.0 <= score <= 1.0
    print("EIF anomaly scores test PASSED.")

if __name__ == "__main__":
    test_data_validation()
    test_drift_calculation()
    test_eif_anomaly_scores()
    print("All integration tests executed successfully.")
