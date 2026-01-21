from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
IMAGE_SIZE = (64, 64)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models and scaler
try:
    svm_model = joblib.load('models/svm_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    models = {
        'svm': svm_model,
        'random_forest': rf_model,
        'logistic_regression': lr_model,
        'knn': knn_model
    }
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    models = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Preprocess image: resize, convert to grayscale, flatten, and normalize
    """
    try:
        # Read image
        img = cv2.imread(image_path)

        if img is None:
            return None

        # Resize to 64x64
        img_resized = cv2.resize(img, IMAGE_SIZE)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Flatten to 1D array
        img_flattened = img_gray.flatten()

        # Normalize pixel values
        img_normalized = img_flattened / 255.0

        return img_normalized.reshape(1, -1)

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    """
    Handle image classification request
    Accepts: uploaded image file and selected model
    Returns: JSON with classification result
    """
    try:
        # Check if file and model are in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        if 'model' not in request.form:
            return jsonify({'error': 'No model selected'}), 400

        file = request.files['file']
        selected_model = request.form['model']

        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400

        # Validate model selection
        if selected_model not in models:
            return jsonify({'error': f'Invalid model: {selected_model}'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        processed_image = preprocess_image(filepath)

        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400

        # Scale features (SVM, Logistic Regression, KNN need scaling)
        if selected_model in ['svm', 'logistic_regression', 'knn']:
            processed_image = scaler.transform(processed_image)

        # Get the selected model
        model = models[selected_model]

        # Make prediction
        prediction = model.predict(processed_image)[0]

        # Get probability if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(processed_image)[0]
            prediction_proba = {
                'cat': float(proba[0]),
                'dog': float(proba[1])
            }

        # Map prediction to class name
        result_class = 'Dog' if prediction == 1 else 'Cat'

        # Return response
        response = {
            'success': True,
            'classification': result_class,
            'prediction_value': int(prediction),
            'model_used': selected_model,
            'filename': filename,
            'probabilities': prediction_proba
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Error in classification: {e}")
        return jsonify({'error': f'Classification error: {str(e)}'}), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    available_models = [
        {'name': 'SVM', 'id': 'svm'},
        {'name': 'Random Forest', 'id': 'random_forest'},
        {'name': 'Logistic Regression', 'id': 'logistic_regression'},
        {'name': 'KNN (K-Nearest Neighbors)', 'id': 'knn'}
    ]
    return jsonify({'models': available_models}), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = len(models) == 4
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'available_models': list(models.keys())
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
