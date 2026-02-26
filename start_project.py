#!/usr/bin/env python3
"""
Startup script for the Endometriosis Detection System
This script helps you get the project running quickly
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_virtual_environment():
    """Check if virtual environment exists and activate it"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found")
        print("💡 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    else:
        print("✅ Virtual environment found")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        if os.name == 'nt':  # Windows
            pip_path = "venv\\Scripts\\pip"
        else:  # macOS/Linux
            pip_path = "venv/bin/pip"
        
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "endometriosis_clinical_data.csv",
        "Image/Infected",
        "Image/Non-infected"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Some data files are missing:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("💡 The system will still work with heuristic predictions")
    else:
        print("✅ All data files found")
    
    return True

def start_application():
    """Start the Flask application"""
    print("🚀 Starting Endometriosis Detection System...")
    print("📍 The application will be available at: http://127.0.0.1:5001")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        if os.name == 'nt':  # Windows
            python_path = "venv\\Scripts\\python"
        else:  # macOS/Linux
            python_path = "venv/bin/python"
        
        subprocess.run([python_path, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🏥 Endometriosis Detection System - Startup Script")
    print("=" * 55)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check/create virtual environment
    if not check_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check data files
    if not check_data_files():
        return False
    
    print("\n✅ Setup complete!")
    print("\n" + "=" * 55)
    
    # Start application
    return start_application()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)