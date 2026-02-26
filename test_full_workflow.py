#!/usr/bin/env python3
"""
Full workflow test for the Endometriosis Detection System
Tests the complete pipeline from data upload to prediction
"""

import os
import sys
import requests
import json
from pathlib import Path

def test_full_workflow():
    """Test the complete workflow of the application"""
    
    base_url = "http://127.0.0.1:5001"
    
    print("🧪 Testing Endometriosis Detection System Full Workflow")
    print("=" * 60)
    
    # Test 1: Check if server is running
    print("1. Testing server connectivity...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is running and accessible")
        else:
            print(f"   ❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Cannot connect to server: {e}")
        print("   💡 Make sure to run 'python app.py' first")
        return False
    
    # Test 2: Upload clinical data
    print("\n2. Testing clinical data upload...")
    clinical_data_path = "endometriosis_clinical_data.csv"
    
    if not os.path.exists(clinical_data_path):
        print(f"   ❌ Clinical data file not found: {clinical_data_path}")
        return False
    
    try:
        with open(clinical_data_path, 'rb') as f:
            files = {'clinical_data_file': f}
            response = requests.post(f"{base_url}/upload_clinical_data", files=files)
            result = response.json()
            
        if result.get('success'):
            print("   ✅ Clinical data uploaded successfully")
        else:
            print(f"   ❌ Clinical data upload failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"   ❌ Error uploading clinical data: {e}")
        return False
    
    # Test 3: Set image path
    print("\n3. Testing image path configuration...")
    image_folder = os.path.join(os.getcwd(), "Image")
    
    try:
        data = {'image_path': image_folder}
        response = requests.post(f"{base_url}/set_image_path", data=data)
        result = response.json()
        
        if result.get('success'):
            print("   ✅ Image path set successfully")
        else:
            print(f"   ❌ Image path setting failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"   ❌ Error setting image path: {e}")
        return False
    
    # Test 4: Train model (this will use heuristic mode since TensorFlow is optional)
    print("\n4. Testing model training...")
    try:
        response = requests.post(f"{base_url}/train_model")
        result = response.json()
        
        if result.get('success'):
            print("   ✅ Model training completed successfully")
            if 'mode' in result.get('result', {}):
                print(f"   ℹ️  Training mode: {result['result']['mode']}")
        else:
            print(f"   ❌ Model training failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"   ❌ Error during model training: {e}")
        return False
    
    # Test 5: Upload test image for analysis
    print("\n5. Testing image upload for analysis...")
    test_image_path = "Image/Infected/00f87e09-9a60-4a79-bc37-073c296a4bb5.JPG"
    
    if not os.path.exists(test_image_path):
        print(f"   ❌ Test image not found: {test_image_path}")
        return False
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image_file': f}
            response = requests.post(f"{base_url}/upload_image", files=files)
            result = response.json()
            
        if result.get('success'):
            print("   ✅ Test image uploaded successfully")
        else:
            print(f"   ❌ Test image upload failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"   ❌ Error uploading test image: {e}")
        return False
    
    # Test 6: Perform analysis
    print("\n6. Testing endometriosis analysis...")
    try:
        # Sample clinical data for analysis
        analysis_data = {
            'age': '35',
            'weight': '65',
            'height': '165',
            'cycle_length': '28',
            'period_duration': '5',
            'crp': '2.5',
            'tsh': '2.0',
            'pelvic_pain': 'Moderate',
            'fatigue': 'on',
            'gi_issues': 'on'
        }
        
        # Upload the test image again for analysis
        with open(test_image_path, 'rb') as f:
            files = {'ultrasound_image': f}
            response = requests.post(f"{base_url}/analyze", data=analysis_data, files=files)
            result = response.json()
        
        if result.get('success'):
            print("   ✅ Analysis completed successfully")
            analysis_result = result.get('result', {})
            print(f"   📊 Prediction: {analysis_result.get('prediction', 'Unknown')}")
            print(f"   📊 Probability: {analysis_result.get('probability', 0):.2%}")
            if analysis_result.get('stage'):
                print(f"   📊 Stage: {analysis_result.get('stage')}")
        else:
            print(f"   ❌ Analysis failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"   ❌ Error during analysis: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! The Endometriosis Detection System is working correctly.")
    print("\n💡 You can now:")
    print("   • Open http://127.0.0.1:5001 in your browser")
    print("   • Upload your own clinical data and images")
    print("   • Train the model with your data")
    print("   • Perform endometriosis detection analysis")
    
    return True

if __name__ == "__main__":
    success = test_full_workflow()
    sys.exit(0 if success else 1)