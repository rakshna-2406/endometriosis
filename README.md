# Endometriosis Detection System

An AI-powered system for detecting endometriosis using multi-modal early fusion of clinical data and ultrasound images.

##  Features

- **Multi-modal Analysis**: Combines clinical data and ultrasound images for accurate detection
- **Intelligent Fallback**: Uses heuristic predictions when deep learning models aren't available
- **Robust Validation**: Comprehensive error handling and input validation
- **Interactive Web Interface**: User-friendly interface for data input and visualization
- **Flexible Training**: Supports both deep learning and heuristic prediction modes
- **Real-time Detection**: Instant endometriosis analysis with staging information

##  Quick Start

### Option 1: Automated Setup (Recommended)
```bash
python3 start_project.py
```

### Option 2: Manual Setup
1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://127.0.0.1:5001
   ```

## Usage Guide

### Step 1: Data Collection
1. **Upload Clinical Data**: Upload a CSV file with patient clinical data
2. **Set Image Path**: Specify the folder containing training images (Image folder)

### Step 2: Model Training
1. Click "Train Model" to train the detection system
2. The system will automatically use heuristic mode if TensorFlow is not available

### Step 3: Analysis
1. **Enter Clinical Information**: Fill in patient details (age, weight, height, symptoms)
2. **Upload Ultrasound Image**: Upload the patient's ultrasound image
3. **Analyze**: Click "Analyse" to get endometriosis detection results with staging

### Expected Results
- **Prediction**: Infected/Non-infected
- **Probability**: Confidence score (0-100%)
- **Stage**: If infected, shows endometriosis stage (I-IV)
- **Recommendations**: Treatment suggestions based on stage

## Project Structure

- `app.py`: Main Flask application
- `model.py`: Multi-modal fusion model implementation with heuristic fallback
- `image_processor.py`: Image validation and processing module
- `data_processor.py`: Clinical data processing module
- `templates/`: HTML templates
- `static/`: CSS and JavaScript files
- `uploads/`: Directory for uploaded files
- `Image/`: Directory containing training images
- `test_model.py`: Tests for model validation and heuristic prediction
- `test_training.py`: Tests for model training validation
- `test_image_processor.py`: Tests for image processing functionality
- `test_app.py`: Tests for web application endpoints

## Requirements

- Python 3.6+
- Flask
- TensorFlow
- NumPy
- Pandas
- OpenCV
- Pillow
- Scikit-learn

## Testing

### Run All Tests
```bash
# Activate virtual environment first
source venv/bin/activate

# Test individual components
python test_model.py          # Model validation and heuristic prediction
python test_training.py       # Model training validation  
python test_image_processor.py # Image processing functionality
python test_app.py           # Web application endpoints

# Test complete workflow (requires running server)
python test_full_workflow.py
```

### Test Results Expected
- All basic tests should pass
- Heuristic predictions should work without TensorFlow
- Web interface should be accessible
- File uploads should work correctly

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'flask'"**
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**2. "Training failed: Insufficient images"**
- Ensure both `Image/Infected` and `Image/Non-infected` folders contain images
- The system needs at least 5 images in each folder for training

**3. "TensorFlow not available"**
- This is normal! The system will use heuristic predictions
- For deep learning, install TensorFlow: `pip install tensorflow`

**4. "Port 5001 already in use"**
- Change the port in `app.py`: `app.run(host='0.0.0.0', port=5002, debug=True)`

**5. Image processing errors**
- Ensure images are in supported formats: JPG, JPEG, PNG
- Check image file size (max 200MB)
- Verify images are not corrupted

### Performance Tips
- Use smaller images (224x224 recommended) for faster processing
- Ensure sufficient RAM for image processing
- Close other applications if experiencing memory issues
