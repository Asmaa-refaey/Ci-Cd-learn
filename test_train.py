from train import model
import numpy as np

def test_prediction():
    # Prediction for 2 تقريبًا لازم تكون قريبة من 4 (لأن y = x^2 في التدريب)
    pred = model.predict([[2]])[0]
    assert abs(pred - 4) < 2, f"Expected ~4 but got {pred}"

