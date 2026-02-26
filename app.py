from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import numpy as np
import pandas as pd
import json
from werkzeug.utils import secure_filename
from model import EndoFusionModel
from data_processor import ClinicalDataProcessor, ImageProcessor

app = Flask(__name__)
app.secret_key = "endometriosis_detection_secret_key"

# Configure upload folders
CLINICAL_DATA_UPLOAD_FOLDER = 'uploads/clinical_data'
IMAGE_UPLOAD_FOLDER = 'uploads/images'
ALLOWED_CLINICAL_EXTENSIONS = {'csv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB limit

# Set model parameters
MODEL_THRESHOLD = 0.5  # Threshold for endometriosis detection

# Create upload directories if they don't exist
os.makedirs(CLINICAL_DATA_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)

app.config['CLINICAL_DATA_UPLOAD_FOLDER'] = CLINICAL_DATA_UPLOAD_FOLDER
app.config['IMAGE_UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize processors and model
clinical_processor = ClinicalDataProcessor()
image_processor = ImageProcessor()
model = EndoFusionModel()

def allowed_clinical_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CLINICAL_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_clinical_data', methods=['POST'])
def upload_clinical_data():
    if 'clinical_data_file' not in request.files:
        flash('No file part')
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['clinical_data_file']
    if file.filename == '':
        flash('No selected file')
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and allowed_clinical_file(file.filename):
        try:
            # Save as endometriosis_clinical_data.csv for consistency
            filename = 'endometriosis_clinical_data.csv'
            filepath = os.path.join(app.config['CLINICAL_DATA_UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the clinical data
            clinical_data = clinical_processor.process_file(filepath)
            
            # Store the path in session for later use
            session['clinical_data_path'] = filepath
            
            return jsonify({'success': True, 'filename': filename, 'message': 'Clinical data uploaded successfully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file format'})

@app.route('/set_image_path', methods=['POST'])
def set_image_path():
    try:
        image_path = request.form.get('image_path')
        
        if not image_path:
            return jsonify({'success': False, 'error': 'No image path provided'})
        
        # Validate the path exists
        if not os.path.exists(image_path):
            # Try to resolve relative to the project directory
            project_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(project_dir, image_path)
            
            if not os.path.exists(full_path):
                return jsonify({'success': False, 'error': f'Image path not found: {image_path}'})
            
            image_path = full_path
        
        # Store the path in session for later use
        session['image_folder_path'] = image_path
        
        return jsonify({'success': True, 'message': f'Image path set to: {image_path}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image_file' not in request.files:
        flash('No file part')
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['image_file']
    if file.filename == '':
        flash('No selected file')
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and allowed_image_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Store the path in session for later use
            session['analysis_image_path'] = filepath
            
            return jsonify({'success': True, 'filename': filename, 'message': 'Image uploaded successfully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file format'})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Get training data paths from session or form
        clinical_data_path = session.get('clinical_data_path', 
                                        request.form.get('clinical_data_path', 
                                                        os.path.join(app.config['CLINICAL_DATA_UPLOAD_FOLDER'], 'endometriosis_clinical_data.csv')))
        
        # Get image folder path from session or form
        image_folder_path = session.get('image_folder_path',
                                       request.form.get('image_folder_path',
                                                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Image')))
        
        # Check if paths exist
        if not os.path.exists(clinical_data_path):
            return jsonify({'success': False, 'error': f'Clinical data file not found at {clinical_data_path}'})
        
        if not os.path.exists(image_folder_path):
            return jsonify({'success': False, 'error': f'Image folder not found at {image_folder_path}'})
        
        # Train the model
        training_result = model.train(clinical_data_path, image_folder_path)
        
        # Check if there was an error during training
        if 'error' in training_result:
            return jsonify({'success': False, 'error': training_result['error']})
        
        # Only mark model as trained if there was no error
        if 'error' not in training_result:
            session['model_trained'] = True
        else:
            session['model_trained'] = False
        
        return jsonify({'success': True, 'message': 'Model trained successfully', 'result': training_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if model is trained
        if not session.get('model_trained', False) and not model.trained:
            return jsonify({'success': False, 'error': 'Model not trained yet. Please train the model first.'})
        
        # Get clinical data from form and validate required fields
        required_fields = ['age', 'bmi', 'pelvic_pain']
        missing_fields = [field for field in required_fields if not request.form.get(field)]
        
        if missing_fields:
            return jsonify({'success': False, 'error': f'Missing required clinical data: {", ".join(missing_fields)}'})
            
        # Process clinical data with validation
        try:
            clinical_data = {
                'Age': float(request.form.get('age')),
                'BMI': float(request.form.get('bmi', 0)) or float(request.form.get('weight', 0)) / ((float(request.form.get('height', 1)) / 100) ** 2),
                # Add other clinical features as needed
                'CRP': float(request.form.get('crp', 0)),
                'TSH': float(request.form.get('tsh', 0)),
                'Cycle_Length': float(request.form.get('cycle_length', 0)),
                'Period_Duration': float(request.form.get('period_duration', 0)),
                'Pelvic_Pain': 1 if request.form.get('pelvic_pain') == 'Moderate' else 0,
                'Fatigue': 1 if request.form.get('fatigue') == 'on' else 0,
                'GI_Issues': 1 if request.form.get('gi_issues') == 'on' else 0
            }
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid clinical data format: {str(e)}'})
        
        # Check if image file is uploaded
        image_path = None
        if 'ultrasound_image' in request.files and request.files['ultrasound_image'].filename != '':
            file = request.files['ultrasound_image']
            if allowed_image_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_path = filepath
                session['analysis_image_path'] = image_path
            else:
                return jsonify({'success': False, 'error': 'Invalid image format. Please upload JPG, JPEG, or PNG files.'})
        else:
            # Try to get image path from session
            image_path = session.get('analysis_image_path')
        
        if not image_path:
            return jsonify({'success': False, 'error': 'Please upload an ultrasound image first'})
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': f'Image not found at {image_path}'})
        
        # Make prediction with error handling
        try:
            prediction = model.predict(clinical_data, image_path)
        except Exception as pred_error:
            return jsonify({'success': False, 'error': f'Prediction failed: {str(pred_error)}'})
        
        # Check if prediction contains an error
        if 'error' in prediction:
            return jsonify({'success': False, 'error': prediction['error']})
        
        # Add stage information if infected prediction
        result = {
            'prediction': prediction.get('prediction', 'Unknown'),
            'probability': prediction.get('probability', 0),
            'stage': None
        }
        
        if prediction.get('probability', 0) > MODEL_THRESHOLD:
            # Determine endometriosis stage based on probability
            prob = prediction.get('probability', 0)
            # Ensure prediction is set to "Infected" when we assign a stage
            result['prediction'] = "Infected"
            
            if prob > 0.9:
                result['stage'] = 'Stage IV (Severe)'
            elif prob > 0.8:
                result['stage'] = 'Stage III (Moderate to Severe)'
            elif prob > 0.7:
                result['stage'] = 'Stage II (Mild to Moderate)'
            else:
                result['stage'] = 'Stage I (Minimal)'
        
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_image_processing', methods=['POST'])
def test_image_processing():
    if 'image_file' not in request.files:
        flash('No file part')
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['image_file']
    if file.filename == '':
        flash('No selected file')
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and allowed_image_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image for testing
            processed_image, features = image_processor.test_processing(filepath)
            
            return jsonify({'success': True, 'features': features})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)