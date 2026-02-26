import os
import shutil
import pandas as pd
from model import EndoFusionModel

def test_training_validation():
    """
    Test that the model properly validates training inputs
    """
    print("\nTesting training validation...")
    model = EndoFusionModel()
    
    # Setup test directories
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_training')
    os.makedirs(test_dir, exist_ok=True)
    
    # Test with empty CSV
    empty_csv_path = os.path.join(test_dir, 'empty.csv')
    with open(empty_csv_path, 'w') as f:
        f.write('')
    
    # Test with invalid image folder
    invalid_image_folder = os.path.join(test_dir, 'invalid_images')
    os.makedirs(invalid_image_folder, exist_ok=True)
    
    print("Testing training with empty CSV...")
    result = model.train(empty_csv_path, invalid_image_folder)
    print(f"Empty CSV test: {'PASSED' if 'error' in result else 'FAILED'}")
    if 'error' in result:
        print(f"  Error message: {result['error']}")
    
    # Test with missing Infected/Non-infected folders
    print("\nTesting training with missing image subfolders...")
    result = model.train(empty_csv_path, invalid_image_folder)
    print(f"Missing subfolders test: {'PASSED' if 'error' in result else 'FAILED'}")
    if 'error' in result:
        print(f"  Error message: {result['error']}")
    
    # Create proper structure but with insufficient images
    infected_dir = os.path.join(invalid_image_folder, 'Infected')
    non_infected_dir = os.path.join(invalid_image_folder, 'Non-infected')
    os.makedirs(infected_dir, exist_ok=True)
    os.makedirs(non_infected_dir, exist_ok=True)
    
    # Create a valid CSV with minimal data
    valid_csv_path = os.path.join(test_dir, 'valid.csv')
    df = pd.DataFrame({
        'Age': [30, 40, 50],
        'BMI': [22, 25, 30],
        'Label': [1, 0, 1]
    })
    df.to_csv(valid_csv_path, index=False)
    
    print("\nTesting training with insufficient images...")
    result = model.train(valid_csv_path, invalid_image_folder)
    print(f"Insufficient images test: {'PASSED' if 'error' in result else 'FAILED'}")
    if 'error' in result:
        print(f"  Error message: {result['error']}")
    
    # Clean up test directory
    print("\nCleaning up test files...")
    shutil.rmtree(test_dir)
    
    print("Training validation tests completed.")

def main():
    print("Running training validation tests...")
    test_training_validation()
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()