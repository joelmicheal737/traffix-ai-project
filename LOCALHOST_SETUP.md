# 🚀 Traffix AI - Localhost Setup Guide

Complete guide to run Traffix AI on your local machine with both frontend and backend.

## 📋 Prerequisites

### Required Software
- **Node.js 18+** - [Download here](https://nodejs.org/)
- **Python 3.8+** - [Download here](https://python.org/)
- **Git** (optional) - [Download here](https://git-scm.com/)

### Check Your Installation
```bash
# Check Node.js version
node --version

# Check Python version
python --version

# Check npm version
npm --version
```

## 🎯 Quick Start (5 Minutes)

### Option 1: Automatic Setup (Recommended)
```bash
# Install frontend dependencies
npm install

# Install concurrently for running both servers
npm install -g concurrently

# Run both frontend and backend together
npm run full-stack
```

### Option 2: Manual Setup (Step by Step)

#### Step 1: Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```
✅ Frontend running on: **http://localhost:5173**

#### Step 2: Backend Setup (New Terminal)
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start backend server
python run.py
```
✅ Backend running on: **http://localhost:8000**

## 🌐 Access Your Application

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | Main web application |
| **Backend API** | http://localhost:8000 | REST API server |
| **API Documentation** | http://localhost:8000/docs | Interactive API docs |
| **Health Check** | http://localhost:8000/health | Server status |

## 🔧 Development Commands

### Frontend Commands
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run typecheck    # TypeScript type checking
```

### Backend Commands
```bash
python run.py                    # Start server with logging
uvicorn main:app --reload       # Alternative start method
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🧪 Testing the Setup

### 1. Test Frontend
- Open http://localhost:5173
- You should see the Traffix AI homepage
- Navigate through different pages (Dashboard, Upload, AI Prediction, About)

### 2. Test Backend
- Open http://localhost:8000/docs
- You should see the FastAPI documentation
- Try the `/health` endpoint to check server status

### 3. Test Integration
- Go to Dashboard page
- Upload a CSV file on Upload page
- Try AI Prediction functionality

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Frontend Issues

**Port 5173 already in use:**
```bash
# Kill process using port 5173
npx kill-port 5173
# Or use different port
npm run dev -- --port 3000
```

**Node modules issues:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### Backend Issues

**Port 8000 already in use:**
```bash
# Kill process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Or use different port
python run.py --port 8001
```

**Python dependencies issues:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Virtual environment issues:**
```bash
# Remove and recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### TensorFlow/GPU Issues
```bash
# For CPU-only installation
pip install tensorflow-cpu

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📊 Development Workflow

### 1. Daily Development
```bash
# Terminal 1: Frontend
npm run dev

# Terminal 2: Backend
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python run.py
```

### 2. Making Changes
- **Frontend changes**: Auto-reload at http://localhost:5173
- **Backend changes**: Auto-reload at http://localhost:8000
- **Database changes**: Restart backend server

### 3. Testing Features
- **Upload CSV**: Use sample data from `data/datasets/`
- **Video Analysis**: Upload sample videos
- **AI Predictions**: Train models with uploaded data

## 🔒 Environment Variables

### Frontend (.env)
```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_GOOGLE_MAPS_API_KEY=your_key_here  # Optional
```

### Backend (.env)
```bash
DATABASE_URL=sqlite:///./traffix.db
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
API_BASE_URL=http://localhost:8000
```

## 📱 Mobile Testing

Test on mobile devices using your local IP:
```bash
# Find your local IP
# Windows:
ipconfig

# macOS/Linux:
ifconfig

# Access from mobile
http://YOUR_LOCAL_IP:5173  # Frontend
http://YOUR_LOCAL_IP:8000  # Backend
```

## 🚀 Production Build

### Build Frontend
```bash
npm run build
npm run preview  # Test production build
```

### Production Backend
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📈 Performance Tips

### Frontend Optimization
- Use `npm run build` for production
- Enable gzip compression
- Optimize images and assets

### Backend Optimization
- Use GPU for ML training if available
- Implement caching for frequent requests
- Use database indexing for large datasets

## 🆘 Getting Help

### Log Files
- **Frontend**: Check browser console (F12)
- **Backend**: Check `traffix_ai.log` file
- **System**: Check terminal output

### Debug Mode
```bash
# Frontend debug
npm run dev -- --debug

# Backend debug
python run.py --log-level DEBUG
```

### Common Commands Reference
```bash
# Check what's running on ports
netstat -tulpn | grep :5173  # Linux/macOS
netstat -ano | findstr :5173  # Windows

# Restart everything
pkill -f "vite"     # Kill frontend
pkill -f "uvicorn"  # Kill backend
npm run full-stack  # Restart both
```

## ✅ Success Checklist

- [ ] Node.js and Python installed
- [ ] Frontend running on http://localhost:5173
- [ ] Backend running on http://localhost:8000
- [ ] Can access API docs at http://localhost:8000/docs
- [ ] Can navigate through all frontend pages
- [ ] Can upload CSV files
- [ ] Can view traffic dashboard
- [ ] No console errors in browser

Your Traffix AI application is now running locally! 🎉