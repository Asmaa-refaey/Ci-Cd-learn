from train import model
import numpy as np

def test_prediction():
    # هنا بندخل 4 قيم (عشان الموديل متدرب على 4 features)
    sample_input = np.array([[0, 0, 0, 0]])  
    
    pred = model.predict(sample_input)[0]
    
    # نفترض الموديل بيصنف 0 أو 1
    assert pred in [0, 1], f"Expected class 0 or 1 but got {pred}"

