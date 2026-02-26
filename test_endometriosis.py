import os
import sys
import pandas as pd
from model import EndoFusionModel
from data_processor import ClinicalDataProcessor, ImageProcessor

def test_endometriosis_detection():
    """
    Test the Endometriosis detection system with sample data
    """
    print("Testing Endometriosis Detection System")
    print("======================================")
    
    # Initialize model and processors
    model = EndoFusionModel()
    clinical_processor = ClinicalDataProcessor()
    image_processor = ImageProcessor()
    
    # Set paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    clinical_data_path = os.path.join(project_dir, 'endometriosis_clinical_data.csv')
    image_folder_path = os.path.join(project_dir, 'Image')
    
    # Check if paths exist
    if not os.path.exists(clinical_data_path):
        print(f"Error: Clinical data file not found at {clinical_data_path}")
        return False
    
    if not os.path.exists(image_folder_path):
        print(f"Error: Image folder not found at {image_folder_path}")
        return False
    
    # Train the model
    print("\nTraining model...")
    training_result = model.train(clinical_data_path, image_folder_path)
    print(f"Training completed with accuracy: {training_result.get('accuracy', 'N/A')}")
    
    # Test with sample clinical data - match the expected shape (15 features)
    sample_clinical_data = {
        'Age': 32,
        'BMI': 24.5,
        'CRP': 3,
        'TSH': 2,
        'Cycle_Length': 33,
        'Period_Duration': 7,
        'Pelvic_Pain': 1,
        'Fatigue': 1,
        'GI_Issues': 1,
        'Menstrual_Pain': 1,
        'Infertility': 0,
        'Dyspareunia': 1,
        'Bloating': 1,
        'Irregular_Bleeding': 0,
        'Family_History': 0
    }
    
    # Test with infected image
    infected_folder = os.path.join(image_folder_path, 'Infected')
    if os.path.exists(infected_folder) and os.listdir(infected_folder):
        infected_image = os.path.join(infected_folder, os.listdir(infected_folder)[0])
        print(f"\nTesting with infected image: {os.path.basename(infected_image)}")
        infected_result = model.predict(sample_clinical_data, infected_image)
        print(f"Prediction: {infected_result['prediction']}")
        print(f"Probability: {infected_result['probability']:.2f}")
        print(f"Confidence: {infected_result['confidence']}")
        if 'stage' in infected_result['analysis']:
            print(f"Stage: {infected_result['analysis']['stage']}")
    else:
        print("No infected images found for testing")
    
    # Test with non-infected image
    non_infected_folder = os.path.join(image_folder_path, 'Non-infected')
    if os.path.exists(non_infected_folder) and os.listdir(non_infected_folder):
        non_infected_image = os.path.join(non_infected_folder, os.listdir(non_infected_folder)[0])
        print(f"\nTesting with non-infected image: {os.path.basename(non_infected_image)}")
        non_infected_result = model.predict(sample_clinical_data, non_infected_image)
        print(f"Prediction: {non_infected_result['prediction']}")
        print(f"Probability: {non_infected_result['probability']:.2f}")
        print(f"Confidence: {non_infected_result['confidence']}")
    else:
        print("No non-infected images found for testing")
    
    print("\nEndometriosis Detection System Test Completed")
    return True

if __name__ == "__main__":
    test_endometriosis_detection()