#!/usr/bin/env python3
"""
Traffix AI Backend Server
Advanced Traffic Management with ML/DL
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffix_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and configurations"""
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Disable TensorFlow GPU memory growth warnings
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Create necessary directories
    directories = ['models', 'uploads', 'logs', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("Environment setup completed")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'tensorflow', 'pandas', 'numpy',
        'scikit-learn', 'prophet', 'opencv-python', 'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("All dependencies are installed")

def main():
    """Main function to start the server"""
    try:
        logger.info("Starting Traffix AI Backend Server...")
        
        # Setup environment
        setup_environment()
        
        # Check dependencies
        check_dependencies()
        
        # Import main app after environment setup
        from main import app
        
        # Server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "true").lower() == "true"
        workers = int(os.getenv("WORKERS", "1"))
        
        logger.info(f"Server configuration:")
        logger.info(f"  Host: {host}")
        logger.info(f"  Port: {port}")
        logger.info(f"  Reload: {reload}")
        logger.info(f"  Workers: {workers}")
        
        # Print startup message
        print("\n" + "="*60)
        print("🚦 TRAFFIX AI BACKEND SERVER STARTING 🚦")
        print("="*60)
        print(f"📍 Backend API: http://{host}:{port}")
        print(f"📚 API Docs: http://{host}:{port}/docs")
        print(f"🔄 Auto-reload: {reload}")
        print("="*60 + "\n")
        
        # Start server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()