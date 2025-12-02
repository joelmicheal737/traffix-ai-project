#!/usr/bin/env python3
"""
Dependency installer for Traffix AI Backend
Ensures all required packages are installed before starting the server
"""

import subprocess
import sys
import logging
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_package(package_name):
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name, "--upgrade"
        ])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        logger.info(f"✅ {package_name} is available")
        return True
    except ImportError:
        logger.warning(f"❌ {package_name} is not available")
        return False

def main():
    """Install all required dependencies"""
    logger.info("🚀 Starting Traffix AI dependency installation...")
    
    # Core dependencies that must be installed
    core_dependencies = [
        ("fastapi==0.104.1", "fastapi"),
        ("uvicorn[standard]==0.24.0", "uvicorn"),
        ("pandas==2.1.3", "pandas"),
        ("numpy==1.25.2", "numpy"),
        ("opencv-python==4.8.1.78", "cv2"),
        ("scikit-learn==1.3.2", "sklearn"),
        ("joblib==1.3.2", "joblib"),
        ("python-multipart==0.0.6", "multipart"),
        ("pydantic==2.5.0", "pydantic"),
        ("aiofiles==23.2.1", "aiofiles"),
        ("requests==2.31.0", "requests"),
        ("python-dotenv==1.0.0", "dotenv"),
        ("Pillow==10.0.1", "PIL"),
    ]
    
    # Optional dependencies for enhanced features
    optional_dependencies = [
        ("tensorflow==2.15.0", "tensorflow"),
        ("prophet==1.1.4", "prophet"),
        ("ultralytics==8.0.206", "ultralytics"),
        ("matplotlib==3.7.2", "matplotlib"),
        ("seaborn==0.12.2", "seaborn"),
        ("plotly==5.17.0", "plotly"),
    ]
    
    # Install core dependencies
    logger.info("📦 Installing core dependencies...")
    failed_core = []
    
    for package, import_name in core_dependencies:
        if not check_package(package.split('==')[0], import_name):
            if install_package(package):
                if check_package(package.split('==')[0], import_name):
                    logger.info(f"✅ Successfully installed {package}")
                else:
                    logger.error(f"❌ Installation verification failed for {package}")
                    failed_core.append(package)
            else:
                failed_core.append(package)
    
    # Install optional dependencies
    logger.info("🔧 Installing optional dependencies...")
    failed_optional = []
    
    for package, import_name in optional_dependencies:
        if not check_package(package.split('==')[0], import_name):
            if install_package(package):
                if check_package(package.split('==')[0], import_name):
                    logger.info(f"✅ Successfully installed {package}")
                else:
                    logger.warning(f"⚠️ Installation verification failed for {package}")
                    failed_optional.append(package)
            else:
                failed_optional.append(package)
    
    # Summary
    logger.info("📋 Installation Summary:")
    
    if not failed_core:
        logger.info("✅ All core dependencies installed successfully!")
    else:
        logger.error(f"❌ Failed to install core dependencies: {failed_core}")
        logger.error("🚨 The application may not work properly without these packages")
        return False
    
    if failed_optional:
        logger.warning(f"⚠️ Some optional features may be limited due to missing packages: {failed_optional}")
    else:
        logger.info("✅ All optional dependencies installed successfully!")
    
    logger.info("🎉 Dependency installation completed!")
    logger.info("🚀 You can now start the server with: python run.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)