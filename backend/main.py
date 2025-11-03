from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import os
from typing import List, Optional
import aiofiles
from pydantic import BaseModel
import cv2
from ultralytics import YOLO
import tempfile
from prophet import Prophet
import warnings
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Traffix AI API", version="2.0.0", description="Advanced Traffic Management with ML/DL")

# Get CORS origins from environment variable
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,https://localhost:5173").split(",")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins + ["*"],  # Allow configured origins plus all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add startup event to populate database with sample data
@app.on_event("startup")
async def startup_event():
    """Initialize database with comprehensive sample data"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if we already have data
        cursor.execute("SELECT COUNT(*) FROM traffic_data")
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("Populating database with comprehensive sample data...")
            
            # Comprehensive Coimbatore locations
            locations_data = [
                # Major Commercial Areas
                ('Gandhipuram', 280, 25, 'high'),
                ('RS Puram', 220, 30, 'medium'),
                ('Cross Cut Road', 240, 28, 'high'),
                ('Town Hall', 260, 26, 'high'),
                ('Race Course', 200, 32, 'medium'),
                
                # IT Corridor & Tech Areas
                ('Peelamedu', 180, 35, 'medium'),
                ('Tidel Park', 320, 22, 'very_high'),
                ('ELCOT IT Park', 190, 34, 'medium'),
                ('Coimbatore IT Park', 160, 38, 'low'),
                
                # Educational & Residential Areas
                ('Saibaba Colony', 150, 40, 'low'),
                ('Vadavalli', 170, 36, 'medium'),
                ('Thudiyalur', 140, 42, 'low'),
                ('PSG College Area', 200, 32, 'medium'),
                
                # Industrial Areas
                ('Singanallur', 210, 30, 'medium'),
                ('Kurichi', 250, 28, 'high'),
                ('Kalapatti', 180, 35, 'medium'),
                ('Neelambur', 160, 38, 'low'),
                
                # Transport Hubs
                ('Ukkadam Bus Stand', 380, 18, 'very_high'),
                ('Railway Station', 350, 20, 'very_high'),
                ('Gandhipuram Bus Stand', 340, 19, 'very_high'),
                
                # Major Roads
                ('Avinashi Road', 230, 29, 'high'),
                ('Pollachi Road', 190, 33, 'medium'),
                ('Mettupalayam Road', 170, 36, 'medium'),
                ('Trichy Road', 180, 35, 'medium'),
                
                # Shopping Areas
                ('Brookefields Mall', 240, 27, 'high'),
                ('Fun Mall', 200, 31, 'medium'),
                ('Prozone Mall', 180, 34, 'medium'),
                
                # Hospitals
                ('KMCH Hospital', 190, 33, 'medium'),
                ('PSG Hospitals', 170, 36, 'medium'),
                ('Coimbatore Medical College', 160, 38, 'low'),
                
                # Suburban Areas
                ('Saravanampatty', 120, 45, 'low'),
                ('Ondipudur', 110, 48, 'low'),
                ('Kuniyamuthur', 130, 43, 'low'),
                ('Vilankurichi', 125, 44, 'low'),
                
                # ADDITIONAL 70 LOCATIONS FOR COMPREHENSIVE COVERAGE
                
                # Extended Commercial & Business Areas
                ('Sitra', 180, 32, 'medium'),
                ('Ramnagar', 160, 35, 'medium'),
                ('Tatabad', 220, 28, 'high'),
                ('Pappanaickenpalayam', 170, 33, 'medium'),
                ('Ramanathapuram', 150, 36, 'medium'),
                ('Selvapuram', 130, 40, 'low'),
                ('Ganapathy', 160, 35, 'medium'),
                ('Koundampalayam', 120, 42, 'low'),
                ('Kuniamuthur', 140, 38, 'medium'),
                ('Podanur Junction', 230, 26, 'high'),
                
                # Extended IT & Tech Areas
                ('Keeranatham', 170, 34, 'medium'),
                ('Chinnavedampatti', 110, 45, 'low'),
                ('Kovaipudur', 150, 37, 'medium'),
                ('Thondamuthur', 120, 42, 'low'),
                ('Narasimhanaickenpalayam', 100, 48, 'low'),
                ('Vellalore', 90, 50, 'low'),
                ('Madampatti', 80, 52, 'low'),
                ('Idikarai', 70, 55, 'low'),
                ('Chettipalayam', 80, 52, 'low'),
                ('Kinathukadavu', 60, 58, 'low'),
                
                # Extended Educational Areas
                ('Anna University Coimbatore', 160, 35, 'medium'),
                ('Amrita University', 140, 38, 'medium'),
                ('Karunya University', 120, 42, 'low'),
                ('PSGR Krishnammal College', 130, 40, 'low'),
                ('SNS College of Technology', 110, 45, 'low'),
                ('Karpagam University', 150, 37, 'medium'),
                ('Sri Krishna College of Engineering', 100, 48, 'low'),
                ('Kumaraguru College of Technology', 170, 34, 'medium'),
                ('Dr. Mahalingam College of Engineering', 90, 50, 'low'),
                ('Hindusthan College of Engineering', 120, 42, 'low'),
                
                # Extended Residential Areas
                ('Hopes College Junction', 180, 32, 'medium'),
                ('Lakshmi Mills Junction', 250, 25, 'high'),
                ('Flower Market Road', 220, 28, 'high'),
                ('Vegetable Market Area', 230, 26, 'high'),
                ('Textile Market Junction', 170, 33, 'medium'),
                ('Oppanakara Street', 160, 35, 'medium'),
                ('Nehru Street Junction', 150, 37, 'medium'),
                ('Big Bazaar Street', 200, 30, 'high'),
                ('Diwan Bahadur Road', 140, 38, 'medium'),
                ('Bharathi Park Road', 120, 42, 'low'),
                
                # Extended Industrial Areas
                ('SIDCO Industrial Estate', 170, 34, 'medium'),
                ('Coimbatore Export Promotion Industrial Park', 150, 37, 'medium'),
                ('Kurichi Industrial Area', 220, 28, 'high'),
                ('Kalapatti Industrial Estate', 160, 35, 'medium'),
                ('Perur Industrial Area', 140, 38, 'medium'),
                ('Neelambur Industrial Park', 130, 40, 'low'),
                ('Malumichampatti Industrial Area', 110, 45, 'low'),
                ('Ondipudur Industrial Estate', 150, 37, 'medium'),
                ('Sulur Industrial Park', 100, 48, 'low'),
                ('Annur Industrial Area', 90, 50, 'low'),
                
                # Extended Transport & Highway Junctions
                ('Singanallur Bus Stand', 260, 24, 'high'),
                ('Peelamedu Bus Stand', 180, 32, 'medium'),
                ('Vadavalli Bus Stop', 150, 37, 'medium'),
                ('Thudiyalur Bus Stop', 140, 38, 'medium'),
                ('Saravanampatti Bus Stop', 130, 40, 'low'),
                ('Kuniyamuthur Bus Stop', 120, 42, 'low'),
                ('Podanur Railway Station', 170, 34, 'medium'),
                ('Irugur Railway Station', 110, 45, 'low'),
                ('Periyanayakkanpalayam Railway Station', 90, 50, 'low'),
                ('Coimbatore North Railway Station', 160, 35, 'medium'),
                
                # Extended Major Road Intersections
                ('Avinashi Road - Trichy Road Junction', 280, 22, 'high'),
                ('Pollachi Road - Sathy Road Junction', 170, 33, 'medium'),
                ('Mettupalayam Road - Avinashi Road Junction', 150, 37, 'medium'),
                ('Salem Road - Trichy Road Junction', 140, 38, 'medium'),
                ('Palakkad Road - Pollachi Road Junction', 130, 40, 'low'),
                ('Sathy Road - Mettupalayam Road Junction', 120, 42, 'low'),
                ('Trichy Road - Salem Road Junction', 160, 35, 'medium'),
                ('Avinashi Road - Salem Road Junction', 150, 37, 'medium'),
                ('Pollachi Road - Palakkad Road Junction', 140, 38, 'medium'),
                ('Mettupalayam Road - Sathy Road Junction', 110, 45, 'low'),
            ]
            
            # Generate data for the last 7 days with hourly intervals
            from datetime import datetime, timedelta
            import random
            
            base_time = datetime.now() - timedelta(days=7)
            
            for location, base_vehicles, base_speed, base_congestion in locations_data:
                for day in range(7):
                    for hour in range(24):
                        timestamp = base_time + timedelta(days=day, hours=hour)
                        
                        # Apply realistic variations based on time
                        is_rush_hour = hour in [7, 8, 9, 17, 18, 19]
                        is_weekend = timestamp.weekday() >= 5
                        
                        vehicle_multiplier = 1.0
                        speed_multiplier = 1.0
                        
                        if is_rush_hour:
                            vehicle_multiplier = 1.4 if not is_weekend else 1.1
                            speed_multiplier = 0.7 if not is_weekend else 0.9
                        elif hour >= 22 or hour <= 6:
                            vehicle_multiplier = 0.4
                            speed_multiplier = 1.3
                        
                        # Add randomness
                        random_factor = 0.8 + random.random() * 0.4
                        
                        vehicle_count = int(base_vehicles * vehicle_multiplier * random_factor)
                        avg_speed = round(base_speed * speed_multiplier * random_factor, 1)
                        
                        # Determine congestion based on vehicles and speed
                        if vehicle_count > 300 or avg_speed < 20:
                            congestion = 'very_high'
                        elif vehicle_count > 220 or avg_speed < 28:
                            congestion = 'high'
                        elif vehicle_count > 150 or avg_speed < 35:
                            congestion = 'medium'
                        else:
                            congestion = 'low'
                        
                        weather = random.choice(['clear', 'cloudy', 'partly_cloudy'])
                        day_name = timestamp.strftime('%A').lower()
                        
                        cursor.execute("""
                            INSERT INTO traffic_data 
                            (timestamp, location, vehicle_count, avg_speed, congestion_level, weather, day_of_week, hour, is_weekend, temperature, humidity, visibility)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            timestamp.isoformat(),
                            location,
                            vehicle_count,
                            avg_speed,
                            congestion,
                            weather,
                            day_name,
                            hour,
                            is_weekend,
                            25.0 + random.uniform(-5, 10),  # Temperature
                            60.0 + random.uniform(-20, 30),  # Humidity
                            10.0 + random.uniform(-3, 5)    # Visibility
                        ))
            
            conn.commit()
            logger.info(f"Populated database with {len(locations_data) * 7 * 24} traffic records")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

# Database setup
DATABASE_PATH = "traffix.db"
MODELS_PATH = "models/"
os.makedirs(MODELS_PATH, exist_ok=True)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

def init_db():
    """Initialize database with enhanced schema"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Traffic data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            location TEXT NOT NULL,
            vehicle_count INTEGER,
            avg_speed REAL,
            congestion_level TEXT,
            weather TEXT,
            day_of_week TEXT,
            hour INTEGER,
            is_weekend BOOLEAN,
            temperature REAL,
            humidity REAL,
            visibility REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Enhanced predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            predicted_timestamp TEXT NOT NULL,
            predicted_congestion REAL,
            confidence_interval_lower REAL,
            confidence_interval_upper REAL,
            model_type TEXT,
            accuracy_score REAL,
            mae REAL,
            rmse REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Model performance tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            location TEXT,
            accuracy REAL,
            mae REAL,
            rmse REAL,
            training_date TEXT,
            data_points INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Video analysis results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            total_vehicles INTEGER,
            vehicle_types TEXT,
            processing_time REAL,
            confidence_score REAL,
            frame_count INTEGER,
            fps REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Pydantic models
class TrafficData(BaseModel):
    timestamp: str
    location: str
    vehicle_count: int
    avg_speed: float
    congestion_level: str
    weather: str
    day_of_week: str
    temperature: Optional[float] = 25.0
    humidity: Optional[float] = 60.0
    visibility: Optional[float] = 10.0

class PredictionRequest(BaseModel):
    location: str
    days_ahead: int = 7
    model_type: str = "prophet"  # prophet, lstm, hybrid

class VideoAnalysisResult(BaseModel):
    total_vehicles: int
    vehicle_types: dict
    processing_time: float
    confidence_score: float
    frame_count: int
    fps: float

class ModelTrainingRequest(BaseModel):
    location: str
    model_type: str
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 32

# ML/DL Model Classes
class LSTMTrafficPredictor:
    """LSTM-based traffic prediction model"""
    
    def __init__(self, sequence_length=24, features=7):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self):
        """Build LSTM model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def prepare_data(self, df):
        """Prepare data for LSTM training"""
        # Feature engineering
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week_num'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
        
        # Convert congestion level to numeric
        congestion_map = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        df['congestion_numeric'] = df['congestion_level'].map(congestion_map)
        
        # Select features
        features = ['vehicle_count', 'avg_speed', 'congestion_numeric', 'hour', 
                   'day_of_week_num', 'is_weekend', 'temperature']
        
        # Handle missing values
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        
        data = df[features].values
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 2])  # congestion_numeric
            
        return np.array(X), np.array(y)
    
    def train(self, df, epochs=50, batch_size=32):
        """Train LSTM model"""
        try:
            X, y = self.prepare_data(df)
            
            if len(X) < 10:
                raise ValueError("Insufficient data for LSTM training")
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build and train model
            self.model = self.build_model()
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Calculate metrics
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'history': history.history,
                'data_points': len(X)
            }
            
        except Exception as e:
            logger.error(f"LSTM training error: {str(e)}")
            raise
    
    def predict(self, df, steps=24):
        """Generate predictions using LSTM"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            # Prepare last sequence
            X, _ = self.prepare_data(df)
            if len(X) == 0:
                raise ValueError("No data available for prediction")
            
            last_sequence = X[-1].reshape(1, self.sequence_length, self.features)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                pred = self.model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                new_row = current_sequence[0, -1].copy()
                new_row[2] = pred  # Update congestion prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1] = new_row
            
            # Inverse transform predictions
            dummy_array = np.zeros((len(predictions), self.features))
            dummy_array[:, 2] = predictions
            predictions_scaled = self.scaler.inverse_transform(dummy_array)[:, 2]
            
            return predictions_scaled.tolist()
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {str(e)}")
            raise

class HybridTrafficPredictor:
    """Hybrid model combining Prophet and LSTM"""
    
    def __init__(self):
        self.prophet_model = None
        self.lstm_model = LSTMTrafficPredictor()
        self.weights = {'prophet': 0.6, 'lstm': 0.4}
    
    def train(self, df, epochs=50):
        """Train both Prophet and LSTM models"""
        results = {}
        
        # Train Prophet
        try:
            prophet_df = self.prepare_prophet_data(df)
            self.prophet_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.8
            )
            self.prophet_model.fit(prophet_df)
            results['prophet'] = {'status': 'success'}
        except Exception as e:
            logger.error(f"Prophet training error: {str(e)}")
            results['prophet'] = {'status': 'failed', 'error': str(e)}
        
        # Train LSTM
        try:
            lstm_results = self.lstm_model.train(df, epochs=epochs)
            results['lstm'] = {'status': 'success', **lstm_results}
        except Exception as e:
            logger.error(f"LSTM training error: {str(e)}")
            results['lstm'] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def prepare_prophet_data(self, df):
        """Prepare data for Prophet"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        congestion_map = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        df['congestion_numeric'] = df['congestion_level'].map(congestion_map)
        
        prophet_df = df[['timestamp', 'congestion_numeric']].rename(
            columns={'timestamp': 'ds', 'congestion_numeric': 'y'}
        )
        return prophet_df
    
    def predict(self, df, steps=24):
        """Generate hybrid predictions"""
        predictions = []
        
        # Prophet predictions
        if self.prophet_model:
            try:
                future = self.prophet_model.make_future_dataframe(periods=steps, freq='H')
                prophet_forecast = self.prophet_model.predict(future)
                prophet_preds = prophet_forecast.tail(steps)['yhat'].values
            except:
                prophet_preds = np.full(steps, 2.0)  # Default medium congestion
        else:
            prophet_preds = np.full(steps, 2.0)
        
        # LSTM predictions
        try:
            lstm_preds = self.lstm_model.predict(df, steps)
        except:
            lstm_preds = np.full(steps, 2.0)
        
        # Combine predictions
        for i in range(steps):
            hybrid_pred = (
                self.weights['prophet'] * prophet_preds[i] +
                self.weights['lstm'] * lstm_preds[i]
            )
            predictions.append(hybrid_pred)
        
        return predictions

# Enhanced YOLO Video Analysis
class AdvancedVideoAnalyzer:
    """Advanced video analysis with multiple YOLO models"""
    
    def __init__(self):
        self.models = {
            'yolov8n': None,
            'yolov8s': None,
            'yolov8m': None
        }
        self.vehicle_classes = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
            1: 'bicycle', 0: 'person'
        }
    
    def load_model(self, model_name='yolov8n'):
        """Load YOLO model"""
        if self.models[model_name] is None:
            try:
                self.models[model_name] = YOLO(f'{model_name}.pt')
                logger.info(f"Loaded {model_name} model")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
                # Fallback to nano model
                self.models['yolov8n'] = YOLO('yolov8n.pt')
                return 'yolov8n'
        return model_name
    
    def analyze_video(self, video_path, model_name='yolov8n', confidence_threshold=0.5):
        """Analyze video with advanced features"""
        model_name = self.load_model(model_name)
        model = self.models[model_name]
        
        cap = cv2.VideoCapture(video_path)
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        vehicle_counts = {vehicle: 0 for vehicle in self.vehicle_classes.values()}
        total_vehicles = 0
        confidence_scores = []
        
        start_time = datetime.now()
        processed_frames = 0
        
        # Process every nth frame for efficiency
        frame_skip = max(1, int(fps // 2))  # Process 2 frames per second
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if current_frame % frame_skip == 0:
                processed_frames += 1
                
                # Run inference
                results = model(frame, conf=confidence_threshold)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            if class_id in self.vehicle_classes:
                                vehicle_type = self.vehicle_classes[class_id]
                                vehicle_counts[vehicle_type] += 1
                                total_vehicles += 1
                                confidence_scores.append(confidence)
        
        cap.release()
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate average confidence
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            'total_vehicles': total_vehicles,
            'vehicle_types': vehicle_counts,
            'processing_time': processing_time,
            'confidence_score': avg_confidence,
            'frame_count': frame_count,
            'fps': fps,
            'processed_frames': processed_frames,
            'model_used': model_name
        }

# Global instances
lstm_models = {}
hybrid_models = {}
video_analyzer = AdvancedVideoAnalyzer()

# API Endpoints

@app.get("/health")
async def health_check():
    """Enhanced health check with system info"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tensorflow_version": tf.__version__,
        "models_available": list(lstm_models.keys()),
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
    }

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Enhanced CSV upload with data validation"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['timestamp', 'location', 'vehicle_count', 'avg_speed', 
                          'congestion_level', 'weather', 'day_of_week']
        
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_columns}")
        
        # Data preprocessing and enhancement
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6])
        
        # Add default values for optional columns
        if 'temperature' not in df.columns:
            df['temperature'] = 25.0
        if 'humidity' not in df.columns:
            df['humidity'] = 60.0
        if 'visibility' not in df.columns:
            df['visibility'] = 10.0
        
        # Insert enhanced data into database
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql('traffic_data', conn, if_exists='append', index=False)
        conn.close()
        
        # Data quality metrics
        quality_metrics = {
            'completeness': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'unique_locations': df['location'].nunique(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'congestion_distribution': df['congestion_level'].value_counts().to_dict()
        }
        
        return {
            "message": "Data uploaded successfully",
            "rows_inserted": len(df),
            "locations": df['location'].unique().tolist(),
            "quality_metrics": quality_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/train-model")
async def train_model(request: ModelTrainingRequest):
    """Train ML/DL models for specific location"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Get training data
        query = """
            SELECT * FROM traffic_data 
            WHERE location = ? 
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=(request.location,))
        conn.close()
        
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data for training (minimum 50 records required)")
        
        # Train model based on type
        if request.model_type == "lstm":
            if request.location not in lstm_models:
                lstm_models[request.location] = LSTMTrafficPredictor()
            
            results = await asyncio.get_event_loop().run_in_executor(
                executor, 
                lstm_models[request.location].train, 
                df, request.epochs, request.batch_size
            )
            
            # Save model
            model_path = f"{MODELS_PATH}lstm_{request.location}.h5"
            lstm_models[request.location].model.save(model_path)
            
            # Save scaler
            scaler_path = f"{MODELS_PATH}scaler_{request.location}.pkl"
            joblib.dump(lstm_models[request.location].scaler, scaler_path)
            
        elif request.model_type == "hybrid":
            if request.location not in hybrid_models:
                hybrid_models[request.location] = HybridTrafficPredictor()
            
            results = await asyncio.get_event_loop().run_in_executor(
                executor,
                hybrid_models[request.location].train,
                df, request.epochs
            )
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        # Store model performance
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (model_name, model_type, location, accuracy, mae, rmse, training_date, data_points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{request.model_type}_{request.location}",
            request.model_type,
            request.location,
            results.get('accuracy', 0.0),
            results.get('mae', 0.0),
            results.get('rmse', 0.0),
            datetime.now().isoformat(),
            results.get('data_points', len(df))
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "message": f"{request.model_type.upper()} model trained successfully",
            "location": request.location,
            "model_type": request.model_type,
            "performance": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/predict")
async def generate_predictions(request: PredictionRequest):
    """Generate advanced predictions using ML/DL models"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Get historical data
        query = """
            SELECT * FROM traffic_data 
            WHERE location = ? 
            ORDER BY timestamp DESC
            LIMIT 1000
        """
        df = pd.read_sql_query(query, conn, params=(request.location,))
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient historical data for predictions")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        predictions_data = []
        
        if request.model_type == "lstm" and request.location in lstm_models:
            # LSTM predictions
            predictions = await asyncio.get_event_loop().run_in_executor(
                executor,
                lstm_models[request.location].predict,
                df, request.days_ahead * 24
            )
            
            # Generate timestamps
            last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
            
            for i, pred in enumerate(predictions):
                future_timestamp = last_timestamp + timedelta(hours=i+1)
                
                predictions_data.append({
                    'location': request.location,
                    'predicted_timestamp': future_timestamp.isoformat(),
                    'predicted_congestion': float(pred),
                    'confidence_interval_lower': float(pred - 0.3),
                    'confidence_interval_upper': float(pred + 0.3),
                    'model_type': 'LSTM'
                })
                
        elif request.model_type == "hybrid" and request.location in hybrid_models:
            # Hybrid predictions
            predictions = await asyncio.get_event_loop().run_in_executor(
                executor,
                hybrid_models[request.location].predict,
                df, request.days_ahead * 24
            )
            
            last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
            
            for i, pred in enumerate(predictions):
                future_timestamp = last_timestamp + timedelta(hours=i+1)
                
                predictions_data.append({
                    'location': request.location,
                    'predicted_timestamp': future_timestamp.isoformat(),
                    'predicted_congestion': float(pred),
                    'confidence_interval_lower': float(pred - 0.4),
                    'confidence_interval_upper': float(pred + 0.4),
                    'model_type': 'Hybrid'
                })
                
        else:
            # Fallback to Prophet
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            congestion_map = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
            df['congestion_numeric'] = df['congestion_level'].map(congestion_map)
            
            prophet_df = df[['timestamp', 'congestion_numeric']].rename(
                columns={'timestamp': 'ds', 'congestion_numeric': 'y'}
            )
            
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.8
            )
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=request.days_ahead * 24, freq='H')
            forecast = model.predict(future)
            
            future_predictions = forecast.tail(request.days_ahead * 24)
            
            for _, row in future_predictions.iterrows():
                predictions_data.append({
                    'location': request.location,
                    'predicted_timestamp': row['ds'].isoformat(),
                    'predicted_congestion': float(row['yhat']),
                    'confidence_interval_lower': float(row['yhat_lower']),
                    'confidence_interval_upper': float(row['yhat_upper']),
                    'model_type': 'Prophet'
                })
        
        # Store predictions
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_sql('predictions', conn, if_exists='append', index=False)
        conn.close()
        
        return {
            "message": "Predictions generated successfully",
            "location": request.location,
            "model_type": request.model_type,
            "predictions": predictions_data[:48],  # Return first 48 hours
            "total_predictions": len(predictions_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@app.post("/video-detect")
async def analyze_video(file: UploadFile = File(...)):
    """Advanced video analysis with multiple YOLO models"""
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_video_path = temp_file.name
        
        # Analyze video
        results = await asyncio.get_event_loop().run_in_executor(
            executor,
            video_analyzer.analyze_video,
            temp_video_path,
            'yolov8n',  # Use nano model for speed
            0.5  # Confidence threshold
        )
        
        # Store results in database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO video_analysis 
            (filename, total_vehicles, vehicle_types, processing_time, confidence_score, frame_count, fps)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            file.filename,
            results['total_vehicles'],
            json.dumps(results['vehicle_types']),
            results['processing_time'],
            results['confidence_score'],
            results['frame_count'],
            results['fps']
        ))
        
        conn.commit()
        conn.close()
        
        # Clean up temporary file
        os.unlink(temp_video_path)
        
        return VideoAnalysisResult(**results)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_video_path' in locals():
            try:
                os.unlink(temp_video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")

@app.get("/traffic-data")
async def get_traffic_data(location: Optional[str] = None, limit: int = 100):
    """Get traffic data with enhanced filtering"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        if location:
            query = "SELECT * FROM traffic_data WHERE location = ? ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(location, limit))
        else:
            query = "SELECT * FROM traffic_data ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(limit,))
        
        conn.close()
        
        # Add computed features
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_name'] = df['timestamp'].dt.day_name()
        
        return {
            "data": df.to_dict('records'),
            "total_records": len(df),
            "locations": df['location'].unique().tolist() if not df.empty else [],
            "date_range": {
                "start": df['timestamp'].min().isoformat() if not df.empty else None,
                "end": df['timestamp'].max().isoformat() if not df.empty else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

@app.get("/model-performance")
async def get_model_performance(location: Optional[str] = None):
    """Get model performance metrics"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        if location:
            query = "SELECT * FROM model_performance WHERE location = ? ORDER BY created_at DESC"
            df = pd.read_sql_query(query, conn, params=(location,))
        else:
            query = "SELECT * FROM model_performance ORDER BY created_at DESC LIMIT 50"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        return {
            "performance_data": df.to_dict('records'),
            "summary": {
                "total_models": len(df),
                "avg_mae": df['mae'].mean() if not df.empty else 0,
                "avg_rmse": df['rmse'].mean() if not df.empty else 0,
                "model_types": df['model_type'].unique().tolist() if not df.empty else []
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model performance: {str(e)}")

@app.get("/locations")
async def get_locations():
    """Get all available locations"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT location FROM traffic_data ORDER BY location")
        locations = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return {"locations": locations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving locations: {str(e)}")

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get comprehensive analytics summary"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Traffic data summary
        traffic_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT location) as unique_locations,
                AVG(vehicle_count) as avg_vehicle_count,
                AVG(avg_speed) as avg_speed,
                MIN(timestamp) as earliest_record,
                MAX(timestamp) as latest_record
            FROM traffic_data
        """
        traffic_summary = pd.read_sql_query(traffic_query, conn).iloc[0].to_dict()
        
        # Congestion distribution
        congestion_query = """
            SELECT congestion_level, COUNT(*) as count
            FROM traffic_data
            GROUP BY congestion_level
        """
        congestion_dist = pd.read_sql_query(congestion_query, conn)
        
        # Model performance summary
        model_query = """
            SELECT model_type, COUNT(*) as count, AVG(mae) as avg_mae, AVG(rmse) as avg_rmse
            FROM model_performance
            GROUP BY model_type
        """
        model_summary = pd.read_sql_query(model_query, conn)
        
        # Video analysis summary
        video_query = """
            SELECT 
                COUNT(*) as total_videos,
                AVG(total_vehicles) as avg_vehicles_per_video,
                AVG(processing_time) as avg_processing_time,
                AVG(confidence_score) as avg_confidence
            FROM video_analysis
        """
        video_summary = pd.read_sql_query(video_query, conn).iloc[0].to_dict()
        
        conn.close()
        
        return {
            "traffic_summary": traffic_summary,
            "congestion_distribution": congestion_dist.to_dict('records'),
            "model_performance": model_summary.to_dict('records'),
            "video_analysis_summary": video_summary,
            "system_info": {
                "tensorflow_version": tf.__version__,
                "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                "active_models": len(lstm_models) + len(hybrid_models)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analytics summary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)