#!/usr/bin/env python3
"""
Health check script for the Endometriosis Detection System
Verifies that all components are working correctly
"""

import os
import sys
import importlib
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_modules = [
        'flask', 'numpy', 'pandas', 'sklearn', 
        'cv2', 'PIL', 'werkzeug'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"   ❌ {module}")
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def check_optional_dependencies():
    """Check optional dependencies"""
    print("\n🔍 Checking optional dependencies...")
    
    try:
        import tensorflow as tf
        print("   ✅ TensorFlow available - Deep learning mode enabled")
        return True
    except ImportError:
        print("   ⚠️  TensorFlow not available - Heuristic mode will be used")
        print("   💡 For deep learning: pip install tensorflow")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    print("\n🔍 Checking project structure...")
    
    required_files = [
        'app.py', 'model.py', 'data_processor.py', 'image_processor.py',
        'requirements.txt', 'templates/index.html'
    ]
    
    required_dirs = [
        'uploads/clinical_data', 'uploads/images', 'static', 'templates'
    ]
    
    missing_items = []
    
    # Check files
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(file_path)
            print(f"   ❌ {file_path}")
        else:
            print(f"   ✅ {file_path}")
    
    # Check directories
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_items.append(dir_path)
            print(f"   ❌ {dir_path}")
        else:
            print(f"   ✅ {dir_path}")
    
    if missing_items:
        print(f"\n⚠️  Missing items: {', '.join(missing_items)}")
        return False
    
    print("✅ Project structure is correct")
    return True

def check_data_files():
    """Check if data files are available"""
    print("\n🔍 Checking data files...")
    
    # Check clinical data
    clinical_data_path = "endometriosis_clinical_data.csv"
    if os.path.exists(clinical_data_path):
        print(f"   ✅ {clinical_data_path}")
        
        # Check file size
        size = os.path.getsize(clinical_data_path)
        if size > 0:
            print(f"      📊 File size: {size} bytes")
        else:
            print("      ⚠️  File is empty")
    else:
        print(f"   ❌ {clinical_data_path}")
    
    # Check image folders
    image_folders = ["Image/Infected", "Image/Non-infected"]
    for folder in image_folders:
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   ✅ {folder} ({len(images)} images)")
        else:
            print(f"   ❌ {folder}")
    
    return True

def check_imports():
    """Check if main modules can be imported"""
    print("\n🔍 Checking module imports...")
    
    modules_to_test = [
        'app', 'model', 'data_processor', 'image_processor'
    ]
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {module_name}.py")
        except Exception as e:
            print(f"   ❌ {module_name}.py - Error: {str(e)}")
            return False
    
    print("✅ All modules import successfully")
    return True

def run_basic_tests():
    """Run basic functionality tests"""
    print("\n🔍 Running basic functionality tests...")
    
    try:
        # Test model creation
        from model import EndoFusionModel
        model = EndoFusionModel()
        print("   ✅ Model creation")
        
        # Test processors
        from data_processor import ClinicalDataProcessor
        from image_processor import ImageProcessor
        
        clinical_processor = ClinicalDataProcessor()
        image_processor = ImageProcessor()
        print("   ✅ Processor creation")
        
        # Test heuristic prediction
        test_clinical_data = {
            'Age': 30,
            'BMI': 25,
            'Pelvic_Pain': 1,
            'Fatigue': 0
        }
        
        # Create a dummy image array
        import numpy as np
        dummy_image = np.random.rand(224, 224, 3)
        
        result = model._heuristic_prediction(test_clinical_data, dummy_image)
        if 'prediction' in result:
            print("   ✅ Heuristic prediction")
        else:
            print("   ❌ Heuristic prediction failed")
            return False
        
    except Exception as e:
        print(f"   ❌ Basic tests failed: {str(e)}")
        return False
    
    print("✅ Basic functionality tests passed")
    return True

def main():
    """Main health check function"""
    print("🏥 Endometriosis Detection System - Health Check")
    print("=" * 55)
    
    checks = [
        check_dependencies,
        check_optional_dependencies,
        check_project_structure,
        check_data_files,
        check_imports,
        run_basic_tests
    ]
    
    all_passed = True
    for check in checks:
        try:
            result = check()
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 Health check passed! System is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open: http://127.0.0.1:5001")
        print("   3. Upload data and start analyzing!")
    else:
        print("⚠️  Some issues found. Please address them before running the system.")
        print("\n💡 Common fixes:")
        print("   • Run: pip install -r requirements.txt")
        print("   • Check file permissions")
        print("   • Ensure all files are present")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)