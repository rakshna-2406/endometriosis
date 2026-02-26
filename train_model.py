import os
import shutil
import pandas as pd
from model import EndoFusionModel
from data_processor import ClinicalDataProcessor

def main():
    """
    Simple script to train the endometriosis detection model
    """
    print("Endometriosis Detection Model Training")
    print("======================================")
    
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'endometriosis_clinical_data.csv')
    image_folder = os.path.join(current_dir, 'Image')
    upload_dir = os.path.join(current_dir, 'uploads', 'clinical_data')
    
    # Ensure upload directory exists
    os.makedirs(upload_dir, exist_ok=True)
    
    # Copy CSV to uploads directory
    upload_path = os.path.join(upload_dir, 'endometriosis_clinical_data.csv')
    shutil.copy2(csv_path, upload_path)
    print(f"Copied CSV file to {upload_path}")
    
    # Verify image folder structure
    infected_dir = os.path.join(image_folder, 'Infected')
    non_infected_dir = os.path.join(image_folder, 'Non-infected')
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return
    
    if not os.path.exists(infected_dir):
        print(f"Error: Infected folder not found at {infected_dir}")
        return
    
    if not os.path.exists(non_infected_dir):
        print(f"Error: Non-infected folder not found at {non_infected_dir}")
        return
    
    # Count images
    infected_images = [f for f in os.listdir(infected_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    non_infected_images = [f for f in os.listdir(non_infected_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(infected_images)} infected images")
    print(f"Found {len(non_infected_images)} non-infected images")
    
    # Validate minimum image requirements
    min_images_required = 5
    if len(infected_images) < min_images_required or len(non_infected_images) < min_images_required:
        print(f"Error: Insufficient training data. Each category needs at least {min_images_required} images.")
        print(f"Currently have: {len(infected_images)} infected and {len(non_infected_images)} non-infected images.")
        return
    
    # Validate CSV file
    try:
        df = pd.read_csv(upload_path)
        if df.empty:
            print(f"Error: The clinical data CSV file is empty.")
            return
        print(f"CSV file validated: {len(df)} clinical records found.")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return
    
    # Initialize model
    model = EndoFusionModel()
    
    # Train model
    print("\nTraining model...")
    try:
        result = model.train(upload_path, image_folder)
        if "error" in result:
            print(f"Training failed: {result['error']}")
        else:
            print("\nTraining completed successfully!")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Validation Accuracy: {result['val_accuracy']:.4f}")
            print(f"Loss: {result['loss']:.4f}")
            print(f"Validation Loss: {result['val_loss']:.4f}")
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()