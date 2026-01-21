# Cat vs Dog Image Classifier - ML Project

A machine learning project that classifies images as cats or dogs using multiple ML models trained on the Kaggle Dogs vs Cats dataset.

## 📋 Project Overview

This project demonstrates:

- **Multiple ML Models**: SVM, Random Forest, Logistic Regression, and KNN
- **Image Preprocessing**: OpenCV for image resizing, grayscale conversion, and normalization
- **Web Interface**: Flask backend with HTML/CSS/JavaScript frontend
- **Model Selection**: Dropdown menu to choose which model to use for classification
- **Kaggle Integration**: Uses Kaggle API to access the dataset without local download

## 🏗️ Project Structure

```
Experiment - 2/
├── training/
│   └── train_models.ipynb          # Google Colab notebook for training
├── models/                          # Directory for saved trained models
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── knn_model.pkl
│   └── scaler.pkl
├── static/
│   ├── uploads/                     # Directory for uploaded images
│   ├── css/
│   │   └── style.css               # Styling for the web interface
│   └── js/
│       └── script.js               # Frontend functionality
├── templates/
│   └── index.html                  # Main HTML page
├── app.py                          # Flask backend application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Step 1: Train Models on Google Colab

1. Open `training/train_models.ipynb` in Google Colab
2. Upload your `kaggle.json` credentials when prompted
3. Run all cells to train the models
4. Download the trained model files (.pkl files) to the `models/` directory

**Note**: You'll need:

- Kaggle account
- Kaggle API credentials (download from https://www.kaggle.com/settings/account)

### Step 2: Setup Local Environment

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 3: Place Model Files

Download the trained models from Google Colab and place them in the `models/` directory:

- `svm_model.pkl`
- `random_forest_model.pkl`
- `logistic_regression_model.pkl`
- `knn_model.pkl`
- `scaler.pkl`

### Step 4: Run Flask Application

```bash
python app.py
```

The web application will be available at: `http://localhost:5000`

## 🎯 Usage

1. **Open the web application** in your browser
2. **Upload an image** by dragging and dropping or clicking the upload area
3. **Select a model** from the dropdown menu
4. **Click "Classify Image"** to get the prediction
5. **View results** including confidence scores
6. **Download result** as a text file (optional)

## 📊 Model Details

### Models Trained

1. **Support Vector Machine (SVM)**
   - Kernel: RBF (Radial Basis Function)
   - Best for: Clear decision boundaries

2. **Random Forest**
   - Estimators: 100
   - Best for: Handling complex patterns and feature interactions

3. **Logistic Regression**
   - Solver: LBFGS
   - Best for: Interpretable probability estimates

4. **K-Nearest Neighbors (KNN)**
   - Neighbors: 5
   - Best for: Instance-based learning

### Data Preprocessing

- **Image Size**: 64×64 pixels
- **Color Space**: Grayscale
- **Feature Extraction**: Flattened pixel values (4,096 features)
- **Normalization**: Pixel values scaled to 0-1 range
- **Scaling**: StandardScaler applied to SVM, Logistic Regression, and KNN

### Training Dataset

- **Source**: Kaggle Dogs vs Cats Dataset
- **Images per class**: 500 (1000 total)
- **Train/Test Split**: 80/20
- **Classes**: Cat (0), Dog (1)

## 🔧 Configuration

### Flask Settings

Located in `app.py`:

- `UPLOAD_FOLDER`: Directory for temporary file uploads
- `ALLOWED_EXTENSIONS`: {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
- `MAX_CONTENT_LENGTH`: 16 MB file size limit
- `DEBUG`: True (change to False for production)

### Image Preprocessing

Located in `app.py`, `preprocess_image()` function:

- `IMAGE_SIZE`: (64, 64) - Change for different image sizes

## 📈 Model Performance

The training notebook displays:

- Accuracy scores (Training and Testing)
- Precision, Recall, and F1-Score
- Confusion matrices for all models
- Probability distributions

## 🌐 Web Interface Features

- **Drag & Drop Upload**: Easy image selection
- **Real-time Preview**: See uploaded image before classification
- **Model Dropdown**: Choose from 4 different models
- **Confidence Bars**: Visual representation of prediction probabilities
- **Error Handling**: Clear error messages for invalid inputs
- **Responsive Design**: Works on desktop and mobile devices
- **Download Results**: Save classification results as text file

## 🔐 Security Considerations

- File upload validation (file type and size)
- Secure filename handling
- Input sanitization
- CORS headers can be added for cross-origin requests

## 📝 API Endpoints

### POST `/classify`

Classify an uploaded image

**Request:**

```
Content-Type: multipart/form-data
- file: image file
- model: model selection (svm, random_forest, logistic_regression, knn)
```

**Response:**

```json
{
  "success": true,
  "classification": "Cat|Dog",
  "prediction_value": 0|1,
  "model_used": "model_name",
  "filename": "uploaded_filename",
  "probabilities": {
    "cat": 0.75,
    "dog": 0.25
  }
}
```

### GET `/models`

Get list of available models

**Response:**

```json
{
  "models": [
    {"name": "SVM", "id": "svm"},
    ...
  ]
}
```

### GET `/health`

Health check endpoint

**Response:**

```json
{
  "status": "healthy|unhealthy",
  "models_loaded": true|false,
  "available_models": [...]
}
```

## 🐛 Troubleshooting

### Models not loading?

- Ensure model files (.pkl) are in the `models/` directory
- Check file names match exactly
- Run `/health` endpoint to check status

### Upload fails?

- Check file is a valid image format
- Ensure file size < 16MB
- Try a different image

### Poor classification results?

- Try different model options
- Ensure image quality and clarity
- Check if image contains clear cat or dog face

## 🚀 Future Enhancements

- [ ] Add deep learning models (CNN)
- [ ] Batch image processing
- [ ] Model performance comparison dashboard
- [ ] User authentication
- [ ] Image history and results database
- [ ] Docker containerization
- [ ] Deployment to Azure/AWS

## 📚 References

- [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/shaunachamberlain/dogs-vs-cats)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created as an ML Project for image classification demonstration.

---

**Last Updated**: January 2024
