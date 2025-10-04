import os
from typing import List

class Settings:
    """Application settings and configuration"""
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./traffix.db")
    
    # API Configuration
    API_TITLE: str = "Traffix AI API"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "Advanced Traffic Management with ML/DL"
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:5173,http://localhost:3000,https://localhost:5173"
    ).split(",")
    
    # File Upload
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads/")
    
    # ML Models
    MODELS_PATH: str = os.getenv("MODELS_PATH", "./models/")
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "./models/yolov8n.pt")
    
    # ML Training Parameters
    DEFAULT_LSTM_EPOCHS: int = int(os.getenv("DEFAULT_LSTM_EPOCHS", "50"))
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))
    DEFAULT_SEQUENCE_LENGTH: int = int(os.getenv("DEFAULT_SEQUENCE_LENGTH", "24"))
    
    # Video Analysis
    VIDEO_CONFIDENCE_THRESHOLD: float = float(os.getenv("VIDEO_CONFIDENCE_THRESHOLD", "0.5"))
    MAX_VIDEO_SIZE: int = int(os.getenv("MAX_VIDEO_SIZE", "500")) * 1024 * 1024  # 500MB
    
    # Performance
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # TensorFlow Configuration
    TF_CPP_MIN_LOG_LEVEL: str = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")

settings = Settings()

# Create necessary directories
os.makedirs(settings.MODELS_PATH, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)