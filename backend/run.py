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
    core_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'sklearn', 'cv2'
    ]
    
    optional_packages = [
        'tensorflow', 'prophet', 'ultralytics'
    ]
    
    missing_packages = []
    
    # Check core packages
    for package in core_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing core packages: {missing_packages}")
        logger.info("Attempting to install missing packages...")
        
        # Try to install missing packages
        package_map = {
            'sklearn': 'scikit-learn==1.3.2',
            'cv2': 'opencv-python==4.8.1.78',
            'pandas': 'pandas==2.1.3',
            'numpy': 'numpy==1.25.2',
            'fastapi': 'fastapi==0.104.1',
            'uvicorn': 'uvicorn[standard]==0.24.0'
        }
        
        for package in missing_packages:
            install_cmd = package_map.get(package, package)
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", install_cmd])
                logger.info(f"Successfully installed {install_cmd}")
            except subprocess.CalledProcessError:
                logger.error(f"Failed to install {install_cmd}")
        
        # Re-check after installation
        remaining_missing = []
        for package in core_packages:
            try:
                if package == 'sklearn':
                    __import__('sklearn')
                elif package == 'cv2':
                    __import__('cv2')
                else:
                    __import__(package.replace('-', '_'))
            except ImportError:
                remaining_missing.append(package)
        
        if remaining_missing:
            logger.error(f"Still missing packages after installation: {remaining_missing}")
            logger.error("Please manually install: pip install -r requirements.txt")
            sys.exit(1)
    
    # Check optional packages
    missing_optional = []
    for package in optional_packages:
        try:
            if package == 'tensorflow':
                __import__('tensorflow')
            elif package == 'prophet':
                __import__('prophet')
            elif package == 'ultralytics':
                __import__('ultralytics')
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)
    
    if missing_optional:
        logger.warning(f"Optional packages not installed: {missing_optional}")
        logger.warning("Some features may be limited. Install with: pip install tensorflow prophet ultralytics")
    
    logger.info("Core dependencies are installed")

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