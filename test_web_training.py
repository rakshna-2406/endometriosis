#!/usr/bin/env python3
"""
Test the web training endpoint to verify the fix
"""

import requests
import time
import subprocess
import os
import signal
import sys

def start_server():
    """Start the Flask server in background"""
    print("Starting Flask server...")
    
    # Start server
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    process = subprocess.Popen(
        ['venv/bin/python', 'app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    
    # Wait for server to start
    time.sleep(3)
    
    return process

def test_training_endpoint():
    """Test the training endpoint"""
    base_url = "http://127.0.0.1:5001"
    
    try:
        # Test server connectivity
        response = requests.get(base_url, timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not accessible: {response.status_code}")
            return False
        
        print("✅ Server is running")
        
        # Test training endpoint
        print("Testing training endpoint...")
        response = requests.post(f"{base_url}/train_model", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training endpoint responded successfully")
            print(f"Response: {result}")
            
            if result.get('success'):
                print("✅ Training completed successfully!")
                return True
            else:
                print(f"❌ Training failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Training endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Web Training Endpoint")
    print("=" * 40)
    
    # Start server
    server_process = start_server()
    
    try:
        # Test training
        success = test_training_endpoint()
        
        print("\n" + "=" * 40)
        if success:
            print("🎉 Web training test passed!")
        else:
            print("❌ Web training test failed!")
            
    finally:
        # Stop server
        print("Stopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()