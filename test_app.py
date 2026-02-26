import os
import unittest
import tempfile
from app import app
from model import EndoFusionModel
from image_processor import ImageProcessor
from PIL import Image
import numpy as np

class EndoAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        
        # Create test directory for images
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test image
        self.test_image_path = os.path.join(self.test_dir, 'test_image.jpg')
        test_img = Image.new('RGB', (100, 100), color=(100, 100, 100))
        # Add some variation to the image
        pixels = test_img.load()
        for i in range(test_img.width):
            for j in range(test_img.height):
                if i % 10 == 0 or j % 10 == 0:
                    pixels[i, j] = (200, 200, 200)
                elif i % 20 == 0 or j % 20 == 0:
                    pixels[i, j] = (50, 50, 50)
        test_img.save(self.test_image_path)
        
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
        
        self.app_context.pop()
    
    def test_home_page(self):
        """Test that the home page loads correctly"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Endometriosis Detection', response.data)
    
    def test_analyze_with_missing_data(self):
        """Test analyze endpoint with missing data"""
        response = self.app.post('/analyze', data={})
        # The app returns 200 with an error message instead of 400
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'error', response.data)
    
    def test_analyze_with_invalid_image(self):
        """Test analyze endpoint with invalid image"""
        # Create an invalid image file (text file)
        invalid_image_path = os.path.join(self.test_dir, 'invalid.jpg')
        with open(invalid_image_path, 'w') as f:
            f.write('Not an image')
        
        with open(invalid_image_path, 'rb') as img:
            response = self.app.post(
                '/analyze',
                data={
                    'image': (img, 'invalid.jpg'),
                    'age': '35',
                    'bmi': '25'
                },
                content_type='multipart/form-data'
            )
        
        # Clean up
        os.remove(invalid_image_path)
        
        # The app should handle this gracefully
        self.assertEqual(response.status_code, 200)
        # Check if there's an error message in the response
        self.assertIn(b'error', response.data)
    
    def test_analyze_with_valid_data(self):
        """Test analyze endpoint with valid data"""
        with open(self.test_image_path, 'rb') as img:
            response = self.app.post(
                '/analyze',
                data={
                    'image': (img, 'test_image.jpg'),
                    'age': '35',
                    'bmi': '25',
                    'pain': 'yes',
                    'fatigue': 'yes'
                },
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200)
        # Since the model is not trained, we should get an error message
        self.assertIn(b'error', response.data)
        self.assertIn(b'Model not trained yet', response.data)
    
    def test_model_integration(self):
        """Test that the model is properly integrated with the app"""
        model = EndoFusionModel()
        
        # Test with valid data - model is not trained, so we expect an error
        result = model.predict(self.test_image_path, {'Age': 35, 'BMI': 25, 'Pelvic_Pain': 1, 'Fatigue': 1})
        
        # Check that the result has the expected error structure
        self.assertIn('error', result)
        self.assertIn('Model has not been trained yet', result['error'])
        
        # Test with invalid image path
        result = model.predict('nonexistent.jpg', {'Age': 35, 'BMI': 25})
        self.assertIn('error', result)
    
    def test_image_processor_integration(self):
        """Test that the image processor is properly integrated"""
        processor = ImageProcessor()
        
        # Test with valid image
        try:
            img = processor.process_image(self.test_image_path)
            self.assertIsNotNone(img)
            self.assertEqual(img.shape, (224, 224, 3))
        except Exception as e:
            self.fail(f"Image processor failed with valid image: {str(e)}")
        
        # Test with invalid path
        with self.assertRaises(FileNotFoundError):
            processor.process_image('nonexistent.jpg')

def main():
    unittest.main()

if __name__ == '__main__':
    main()