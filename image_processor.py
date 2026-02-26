import cv2
import numpy as np
import os
from PIL import Image
import logging

class ImageProcessor:
    """
    Process ultrasound images for endometriosis detection
    """
    def __init__(self):
        self.target_size = (224, 224)  # Standard size for many CNN models
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
    def process_image(self, image_path):
        """
        Process an image for model input
        """
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Check file extension
        _, ext = os.path.splitext(image_path.lower())
        if ext not in self.valid_extensions:
            raise ValueError(f"Invalid image format: {ext}. Supported formats: {', '.join(self.valid_extensions)}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}. The file may be corrupted.")
        
        # Basic image validation
        self._validate_image(img, image_path)
        
        # Resize and normalize
        img = cv2.resize(img, self.target_size)
        img = img / 255.0  # Normalize to 0-1 range
        
        return img
    
    def _validate_image(self, img, image_path):
        """
        Validate image quality and characteristics
        """
        # Check dimensions
        height, width, channels = img.shape
        if height < 10 or width < 10:
            raise ValueError(f"Image at {image_path} is too small ({width}x{height}). Minimum size is 10x10.")
        
        # Check if image is not empty or completely uniform
        std_dev = np.std(img)
        if std_dev < 5:
            raise ValueError(f"Image at {image_path} has very low variation (std={std_dev:.2f}). It may be blank or uniform.")
        
        # Check if image is not too dark or too bright
        mean_value = np.mean(img)
        if mean_value < 10:
            raise ValueError(f"Image at {image_path} is too dark (mean={mean_value:.2f}). Please provide a clearer image.")
        if mean_value > 245:
            raise ValueError(f"Image at {image_path} is too bright (mean={mean_value:.2f}). Please provide a clearer image.")
        
    def process_directory(self, directory_path):
        """
        Process all images in a directory
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        processed_images = []
        errors = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(self.valid_extensions):
                try:
                    processed_img = self.process_image(file_path)
                    processed_images.append((processed_img, filename))
                except Exception as e:
                    errors.append((filename, str(e)))
        
        return processed_images, errors