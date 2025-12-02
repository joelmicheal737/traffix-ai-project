# Traffix AI Backend

Advanced Python backend with Machine Learning and Deep Learning capabilities for traffic management and prediction.

## 🚀 Features

### 🤖 Machine Learning & Deep Learning
- **LSTM Neural Networks** - Deep learning for traffic prediction
- **Prophet Time Series** - Facebook's robust forecasting model
- **Hybrid Models** - Combining multiple ML approaches
- **Real-time Training** - Dynamic model updates
- **Model Performance Tracking** - Comprehensive evaluation metrics

### 🎥 Computer Vision
- **YOLOv8 Integration** - State-of-the-art object detection
- **Multi-model Support** - YOLOv8n, YOLOv8s, YOLOv8m
- **Video Analysis** - Automated vehicle counting and classification
- **Real-time Processing** - Efficient frame-by-frame analysis

### 📊 Advanced Analytics
- **Time Series Analysis** - Seasonal pattern detection
- **Feature Engineering** - Automated feature creation
- **Data Preprocessing** - Cleaning and validation
- **Cross-validation** - Robust model evaluation

### 🔧 API Endpoints

#### Core Functionality
- `POST /upload` - Upload CSV traffic data with validation
- `GET /traffic-data` - Retrieve traffic data with filtering
- `GET /locations` - Get all monitored locations
- `GET /health` - System health and GPU status

#### Machine Learning
- `POST /train-model` - Train LSTM/Hybrid models
- `POST /predict` - Generate ML/DL predictions
- `GET /model-performance` - Model evaluation metrics
- `GET /analytics/summary` - Comprehensive analytics

#### Computer Vision
- `POST /video-detect` - Advanced video analysis with YOLOv8

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Optional: CUDA-compatible GPU for faster training

### Quick Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies (recommended method)
python install_dependencies.py

# OR install manually
# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

### Troubleshooting Installation

If you encounter issues with OpenCV or scikit-learn:

```bash
# Method 1: Use the dependency installer
python install_dependencies.py

# Method 2: Install problematic packages individually
pip install opencv-python==4.8.1.78
pip install scikit-learn==1.3.2
pip install pandas==2.1.3
pip install numpy==1.25.2

# Method 3: For OpenCV issues on some systems
pip install opencv-python-headless==4.8.1.78

# Method 4: Complete fresh installation
rm -rf venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Alternative Start Methods
```bash
# Using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Using Python module
python -m uvicorn main:app --reload

# Production mode
python run.py
```

## 📋 Dependencies

### Core Framework
- **FastAPI** - Modern, fast web framework
- **Uvicorn** - ASGI server implementation
- **Pydantic** - Data validation using Python type hints

### Machine Learning
- **TensorFlow 2.15** - Deep learning framework
- **Scikit-learn** - Traditional ML algorithms
- **Prophet** - Time series forecasting
- **XGBoost, LightGBM, CatBoost** - Gradient boosting

### Computer Vision
- **OpenCV** - Computer vision library
- **Ultralytics YOLOv8** - Object detection
- **Pillow** - Image processing

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib, Seaborn** - Data visualization

## 🏗️ Architecture

### Model Classes

#### LSTMTrafficPredictor
```python
# Advanced LSTM with multiple layers
- 3 LSTM layers with dropout
- Dense layers for final prediction
- Sequence-to-one prediction
- Feature engineering integration
```

#### HybridTrafficPredictor
```python
# Combines Prophet + LSTM
- Prophet for trend/seasonality
- LSTM for complex patterns
- Weighted ensemble predictions
- Automatic fallback mechanisms
```

#### AdvancedVideoAnalyzer
```python
# Multi-model YOLO analysis
- YOLOv8n/s/m model support
- Vehicle classification
- Confidence scoring
- Performance optimization
```

### Data Processing Pipeline

1. **Data Ingestion** - CSV upload with validation
2. **Preprocessing** - Cleaning, outlier removal
3. **Feature Engineering** - Time-based, lag, rolling features
4. **Model Training** - LSTM/Hybrid model training
5. **Prediction** - Real-time traffic forecasting
6. **Evaluation** - Comprehensive metrics calculation

## 🎯 Usage Examples

### Training an LSTM Model
```python
POST /train-model
{
    "location": "Gandhipuram",
    "model_type": "lstm",
    "epochs": 50,
    "batch_size": 32
}
```

### Generating Predictions
```python
POST /predict
{
    "location": "Gandhipuram",
    "days_ahead": 7,
    "model_type": "hybrid"
}
```

### Video Analysis
```python
POST /video-detect
# Upload video file
# Returns: vehicle counts, types, confidence scores
```

## 📊 Model Performance

### Evaluation Metrics
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Square Error
- **R²** - Coefficient of Determination
- **MAPE** - Mean Absolute Percentage Error
- **Directional Accuracy** - Trend prediction accuracy

### Performance Grades
- **A+** - Excellent (90%+ accuracy)
- **A** - Very Good (80-90%)
- **B+** - Good (70-80%)
- **B** - Satisfactory (60-70%)
- **C+/C** - Needs Improvement (40-60%)
- **D** - Poor (<40%)

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=sqlite:///./traffix.db

# CORS Origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Model Configuration
DEFAULT_LSTM_EPOCHS=50
DEFAULT_BATCH_SIZE=32
DEFAULT_SEQUENCE_LENGTH=24

# Video Analysis
VIDEO_CONFIDENCE_THRESHOLD=0.5
MAX_VIDEO_SIZE=500MB

# Performance
MAX_WORKERS=4
LOG_LEVEL=INFO
```

### Model Storage
- **LSTM Models** - Saved as `.h5` files
- **Scalers** - Saved as `.pkl` files
- **Prophet Models** - In-memory storage
- **Performance Metrics** - SQLite database

## 🚀 Advanced Features

### Real-time Model Updates
- Incremental learning support
- Online model adaptation
- Performance monitoring
- Automatic retraining triggers

### GPU Acceleration
- TensorFlow GPU support
- CUDA optimization
- Memory management
- Fallback to CPU

### Scalability
- Async processing
- Thread pool execution
- Batch prediction support
- Model caching

## 🐛 Troubleshooting

### Common Issues

**TensorFlow GPU Issues**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Set memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

**Memory Issues**
```bash
# Reduce batch size
DEFAULT_BATCH_SIZE=16

# Limit TensorFlow memory
export TF_MEMORY_GROWTH=true
```

**Model Training Failures**
- Ensure sufficient data (minimum 50 records)
- Check data quality and format
- Verify timestamp format
- Monitor system resources

## 📈 Performance Optimization

### Training Optimization
- Use GPU acceleration when available
- Implement early stopping
- Batch processing for large datasets
- Feature selection and dimensionality reduction

### Inference Optimization
- Model caching and reuse
- Batch predictions
- Async processing
- Result caching

### Video Processing Optimization
- Frame skipping for efficiency
- Model selection based on requirements
- Parallel processing
- Memory management

## 🔒 Security

- Input validation and sanitization
- File type restrictions
- Size limits for uploads
- SQL injection prevention
- CORS configuration

## 📝 Logging

- Structured logging with timestamps
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- File and console output
- Performance metrics logging
- Error tracking and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.