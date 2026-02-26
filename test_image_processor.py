import os
import numpy as np
import cv2
from PIL import Image
from image_processor import ImageProcessor

def test_image_processor():
    """
    Test the ImageProcessor class functionality
    """
    print("\nTesting ImageProcessor...")
    processor = ImageProcessor()
    
    # Setup test directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test images
    print("\nCreating test images...")
    
    # Valid image with variation
    valid_image_path = os.path.join(test_dir, 'valid.jpg')
    valid_img = Image.new('RGB', (100, 100), color=(100, 100, 100))
    # Add some variation to the image
    pixels = valid_img.load()
    for i in range(valid_img.width):
        for j in range(valid_img.height):
            if i % 10 == 0 or j % 10 == 0:
                pixels[i, j] = (200, 200, 200)
            elif i % 20 == 0 or j % 20 == 0:
                pixels[i, j] = (50, 50, 50)
    valid_img.save(valid_image_path)
    
    # Small image
    small_image_path = os.path.join(test_dir, 'small.jpg')
    small_img = Image.new('RGB', (5, 5), color=(100, 100, 100))
    small_img.save(small_image_path)
    
    # Dark image
    dark_image_path = os.path.join(test_dir, 'dark.jpg')
    dark_img = Image.new('RGB', (100, 100), color=(1, 1, 1))
    dark_img.save(dark_image_path)
    
    # Bright image
    bright_image_path = os.path.join(test_dir, 'bright.jpg')
    bright_img = Image.new('RGB', (100, 100), color=(254, 254, 254))
    bright_img.save(bright_image_path)
    
    # Test non-existent path
    print("\nTesting non-existent image path...")
    non_existent_path = os.path.join(test_dir, 'non_existent.jpg')
    try:
        processor.process_image(non_existent_path)
        print("Non-existent path test: FAILED")
    except FileNotFoundError as e:
        print(f"Non-existent path test: PASSED")
        print(f"  Error message: {str(e)}")
    
    # Test invalid format
    invalid_format_path = os.path.join(test_dir, 'invalid.txt')
    with open(invalid_format_path, 'w') as f:
        f.write('Not an image')
    
    try:
        processor.process_image(invalid_format_path)
        print("Invalid format test: FAILED")
    except ValueError as e:
        print(f"Invalid format test: PASSED")
        print(f"  Error message: {str(e)}")
    
    # Test valid image processing
    print("\nTesting valid image processing...")
    try:
        processed_img = processor.process_image(valid_image_path)
        print(f"Process valid image test: PASSED")
        print(f"  Image shape: {processed_img.shape}")
        print(f"  Normalized values: min={processed_img.min():.2f}, max={processed_img.max():.2f}, mean={processed_img.mean():.2f}")
    except Exception as e:
        print(f"Process valid image test: FAILED")
        print(f"  Error: {str(e)}")
    
    # Test small image
    print("\nTesting small image...")
    try:
        processor.process_image(small_image_path)
        print("Small image test: FAILED")
    except ValueError as e:
        print(f"Small image test: PASSED")
        print(f"  Error message: {str(e)}")
    
    # Test dark image
    print("\nTesting dark image...")
    try:
        processor.process_image(dark_image_path)
        print("Dark image test: FAILED")
    except ValueError as e:
        print(f"Dark image test: PASSED")
        print(f"  Error message: {str(e)}")
    
    # Test bright image
    print("\nTesting bright image...")
    try:
        processor.process_image(bright_image_path)
        print("Bright image test: FAILED")
    except ValueError as e:
        print(f"Bright image test: PASSED")
        print(f"  Error message: {str(e)}")
    
    # Test directory processing
    print("\nTesting directory processing...")
    try:
        processed_images, errors = processor.process_directory(test_dir)
        print(f"Directory processing test: PASSED")
        print(f"  Processed {len(processed_images)} images successfully")
        print(f"  Encountered {len(errors)} errors")
    except Exception as e:
        print(f"Directory processing test: FAILED")
        print(f"  Error: {str(e)}")
    
    # Clean up test files
    print("\nCleaning up test files...")
    for file in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, file))
    os.rmdir(test_dir)
    
    print("ImageProcessor tests completed.")

def main():
    print("Running image processor tests...")
    test_image_processor()
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()