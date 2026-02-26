import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os

class ClinicalDataProcessor:
    """
    Process clinical data for endometriosis detection
    """
    def __init__(self):
        self.required_features = ['Age', 'BMI']
        self.categorical_features = []
        self.numerical_features = ['Age', 'BMI']
        
    def process_file(self, file_path):
        """
        Process a CSV file containing clinical data
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Clinical data file not found: {file_path}")
                
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                raise ValueError("Clinical data file is empty")
                
            # Read the CSV file
            try:
                data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                raise ValueError("Clinical data file is empty or has no valid data")
            except pd.errors.ParserError:
                raise ValueError("Clinical data file is not a valid CSV format")
                
            # Check if dataframe is empty
            if data.empty:
                raise ValueError("Clinical data contains no records")
                
            # Check if required features are present
            missing_features = [f for f in self.required_features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Process numerical features
            for feature in self.numerical_features:
                if feature in data.columns:
                    data[feature] = pd.to_numeric(data[feature], errors='coerce')
                    
                    # Check if all values are NaN after conversion
                    if data[feature].isna().all():
                        raise ValueError(f"Feature '{feature}' contains no valid numerical values")
            
            # Fill missing values
            data = self.handle_missing_values(data)
            
            # Normalize numerical features
            data = self.normalize_features(data)
            
            return data
        except Exception as e:
            print(f"Error processing clinical data: {e}")
            raise
    
    def handle_missing_values(self, data):
        """
        Handle missing values in the clinical data
        """
        # Fill numerical features with median
        for feature in self.numerical_features:
            if feature in data.columns:
                data[feature] = data[feature].fillna(data[feature].median())
        
        # Fill categorical features with mode
        for feature in self.categorical_features:
            if feature in data.columns:
                data[feature] = data[feature].fillna(data[feature].mode()[0])
        
        return data
    
    def normalize_features(self, data):
        """
        Normalize numerical features
        """
        for feature in self.numerical_features:
            if feature in data.columns:
                mean = data[feature].mean()
                std = data[feature].std()
                if std > 0:
                    data[feature] = (data[feature] - mean) / std
        
        return data
    
    def process_data(self, data):
        """
        Process clinical data DataFrame directly
        """
        # Check if required features are present
        missing_features = [f for f in self.required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Process numerical features
        for feature in self.numerical_features:
            if feature in data.columns:
                data[feature] = pd.to_numeric(data[feature], errors='coerce')
        
        # Fill missing values
        data = self.handle_missing_values(data)
        
        # Normalize numerical features
        data = self.normalize_features(data)
        
        return data
        
    def process_input(self, clinical_data):
        """
        Process clinical data input for prediction
        """
        # Convert to DataFrame if it's a dictionary
        if isinstance(clinical_data, dict):
            clinical_data = pd.DataFrame([clinical_data])
        
        return self.process_data(clinical_data)


class ImageProcessor:
    """
    Process ultrasound images for endometriosis detection
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def process_image(self, image_path):
        """
        Process a single image for model input
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.target_size)
            
            # Normalize pixel values to [0, 1]
            img = img / 255.0
            
            return img
        except Exception as e:
            print(f"Error processing image: {e}")
            raise
    
    def process_directory(self, directory_path, label=None):
        """
        Process all images in a directory
        """
        images = []
        labels = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory_path, filename)
                try:
                    processed_image = self.process_image(image_path)
                    images.append(processed_image)
                    if label is not None:
                        labels.append(label)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return np.array(images), np.array(labels) if label is not None else None
    
    def test_processing(self, image_path):
        """
        Test image processing and return processed image with extracted features
        """
        processed_image = self.process_image(image_path)
        
        # Extract basic features for testing
        features = {
            'mean_intensity': np.mean(processed_image),
            'std_intensity': np.std(processed_image),
            'min_intensity': np.min(processed_image),
            'max_intensity': np.max(processed_image)
        }
        
        return processed_image, features