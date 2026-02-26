import os
import pandas as pd
import numpy as np
from model import EndoFusionModel
from data_processor import ClinicalDataProcessor
from image_processor import ImageProcessor

def test_model_validation():
    """
    Test that the model properly validates inputs
    """
    print("\nTesting model validation...")
    model = EndoFusionModel()
    
    # Test missing clinical data
    result = model.predict(None, "path/to/image.jpg")
    print(f"Missing clinical data test: {'PASSED' if 'error' in result else 'FAILED'}")
    if 'error' in result:
        print(f"  Error message: {result['error']}")
    
    # Test missing image path
    clinical_data = {"Age": 35, "BMI": 25}
    result = model.predict(clinical_data, None)
    print(f"Missing image path test: {'PASSED' if 'error' in result else 'FAILED'}")
    if 'error' in result:
        print(f"  Error message: {result['error']}")
    
    # Test non-existent image path
    result = model.predict(clinical_data, "non_existent_image.jpg")
    print(f"Non-existent image path test: {'PASSED' if 'error' in result else 'FAILED'}")
    if 'error' in result:
        print(f"  Error message: {result['error']}")
    
    # Test missing required clinical features
    incomplete_data = {"Age": 35}  # Missing BMI
    result = model.predict(incomplete_data, "path/to/image.jpg")
    print(f"Missing clinical features test: {'PASSED' if 'error' in result else 'FAILED'}")
    if 'error' in result:
        print(f"  Error message: {result['error']}")
    
    print("Model validation tests completed.")

def test_heuristic_prediction():
    """
    Test the heuristic prediction functionality
    """
    print("\nTesting heuristic prediction...")
    model = EndoFusionModel()
    
    # Create test clinical data
    clinical_data = {
        "Age": 45,
        "BMI": 32,
        "Pelvic_Pain": 1,
        "Fatigue": 1,
        "GI_Issues": 1
    }
    
    # Create a test image (simple numpy array)
    test_image = np.ones((224, 224, 3)) * 0.5  # Gray image
    
    # Test heuristic prediction
    result = model._heuristic_prediction(clinical_data, test_image)
    print(f"Heuristic prediction result: {result}")
    print(f"Prediction: {result.get('prediction', 'N/A')}")
    print(f"Probability: {result.get('probability', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    # Test with different values
    clinical_data2 = {
        "Age": 25,
        "BMI": 22,
        "Pelvic_Pain": 0,
        "Fatigue": 0,
        "GI_Issues": 0
    }
    
    result2 = model._heuristic_prediction(clinical_data2, test_image)
    print(f"\nHeuristic prediction with different values:")
    print(f"Prediction: {result2.get('prediction', 'N/A')}")
    print(f"Probability: {result2.get('probability', 'N/A')}")
    
    print("Heuristic prediction tests completed.")

def main():
    print("Running model tests...")
    
    # Run validation tests
    test_model_validation()
    
    # Run heuristic prediction tests
    test_heuristic_prediction()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()