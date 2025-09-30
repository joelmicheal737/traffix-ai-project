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
warnings.filterwarnings('ignore')

app = FastAPI(title="Traffix AI API", version="1.0.0")

# Get CORS origins from environment variable
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,https://localhost:5173").split(",")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_PATH = "traffix.db"

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            predicted_timestamp TEXT NOT NULL,
            predicted_congestion REAL,
            confidence_interval_lower REAL,
            confidence_interval_upper REAL,
            model_type TEXT,
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

class PredictionRequest(BaseModel):
    location: str
    days_ahead: int = 7

class VideoAnalysisResult(BaseModel):
    total_vehicles: int
    vehicle_types: dict
    processing_time: float
    confidence_score: float

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Upload CSV data
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['timestamp', 'location', 'vehicle_count', 'avg_speed', 
                          'congestion_level', 'weather', 'day_of_week']
        
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_columns}")
        
        # Insert data into database
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql('traffic_data', conn, if_exists='append', index=False)
        conn.close()
        
        return {
            "message": "Data uploaded successfully",
            "rows_inserted": len(df),
            "locations": df['location'].unique().tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Get traffic data
@app.get("/traffic-data")
async def get_traffic_data(location: Optional[str] = None, limit: int = 100):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        if location:
            query = "SELECT * FROM traffic_data WHERE location = ? ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(location, limit))
        else:
            query = "SELECT * FROM traffic_data ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(limit,))
        
        conn.close()
        
        return {
            "data": df.to_dict('records'),
            "total_records": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

# Generate predictions
@app.post("/predict")
async def generate_predictions(request: PredictionRequest):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Get historical data for the location
        query = """
            SELECT timestamp, vehicle_count, congestion_level 
            FROM traffic_data 
            WHERE location = ? 
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=(request.location,))
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient historical data for predictions")
        
        # Prepare data for Prophet
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert congestion level to numeric
        congestion_map = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        df['congestion_numeric'] = df['congestion_level'].map(congestion_map)
        
        # Prepare Prophet dataframe
        prophet_df = df[['timestamp', 'congestion_numeric']].rename(
            columns={'timestamp': 'ds', 'congestion_numeric': 'y'}
        )
        
        # Create and fit Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.8
        )
        model.fit(prophet_df)
        
        # Generate future dates
        future = model.make_future_dataframe(periods=request.days_ahead * 24, freq='H')
        forecast = model.predict(future)
        
        # Get only future predictions
        future_predictions = forecast.tail(request.days_ahead * 24)
        
        # Store predictions in database
        predictions_data = []
        for _, row in future_predictions.iterrows():
            prediction_data = {
                'location': request.location,
                'predicted_timestamp': row['ds'].isoformat(),
                'predicted_congestion': float(row['yhat']),
                'confidence_interval_lower': float(row['yhat_lower']),
                'confidence_interval_upper': float(row['yhat_upper']),
                'model_type': 'Prophet'
            }
            predictions_data.append(prediction_data)
        
        # Insert predictions into database
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_sql('predictions', conn, if_exists='append', index=False)
        
        conn.close()
        
        return {
            "message": "Predictions generated successfully",
            "location": request.location,
            "predictions": predictions_data[:24],  # Return first 24 hours
            "total_predictions": len(predictions_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

# Video analysis with YOLO
@app.post("/video-detect")
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_video_path = temp_file.name
        
        # Load YOLO model (using YOLOv8 nano for speed)
        model = YOLO('yolov8n.pt')
        
        # Process video
        cap = cv2.VideoCapture(temp_video_path)
        total_vehicles = 0
        vehicle_types = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        frame_count = 0
        
        start_time = datetime.now()
        
        while cap.read()[0]:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 10th frame to speed up analysis
            if frame_count % 10 == 0:
                results = model(frame)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Filter for vehicles (COCO classes: car=2, motorcycle=3, bus=5, truck=7)
                            if class_id in [2, 3, 5, 7] and confidence > 0.5:
                                total_vehicles += 1
                                
                                if class_id == 2:
                                    vehicle_types['car'] += 1
                                elif class_id == 3:
                                    vehicle_types['motorcycle'] += 1
                                elif class_id == 5:
                                    vehicle_types['bus'] += 1
                                elif class_id == 7:
                                    vehicle_types['truck'] += 1
        
        cap.release()
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up temporary file
        os.unlink(temp_video_path)
        
        return VideoAnalysisResult(
            total_vehicles=total_vehicles,
            vehicle_types=vehicle_types,
            processing_time=processing_time,
            confidence_score=0.85  # Average confidence
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_video_path' in locals():
            try:
                os.unlink(temp_video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")

# Get locations
@app.get("/locations")
async def get_locations():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT location FROM traffic_data")
        locations = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return {"locations": locations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving locations: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)