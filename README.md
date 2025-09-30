# Traffix AI - Coimbatore Edition

A comprehensive traffic management and prediction system for Coimbatore using AI/ML technologies.

## Features

### Frontend (React + Vite + Tailwind)
- **Home**: Landing page with overview
- **Dashboard**: Live traffic data and analytics
- **Upload**: CSV data upload functionality
- **AI Prediction**: Traffic congestion forecasting
- **About**: Information about the system

### Backend (FastAPI + SQLite)
- **Data Upload**: Accept and store CSV traffic data
- **AI Predictions**: Prophet/LSTM-based traffic forecasting
- **Video Analysis**: YOLOv8 vehicle detection and counting
- **RESTful API**: Clean endpoints for all operations

## Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS, React Router, Recharts, Leaflet
- **Backend**: FastAPI, SQLite, Prophet, YOLOv8
- **Maps**: Leaflet with OpenStreetMap
- **Charts**: Recharts for data visualization

## Project Structure

```
traffix-ai/
├── frontend/           # React application
├── backend/           # FastAPI server
├── data/             # Sample datasets and videos
│   ├── datasets/     # CSV traffic data
│   └── videos/       # Demo videos for YOLO
├── .env.example      # Environment variables template
└── README.md         # This file
```

## Local Setup

### Prerequisites
- Node.js 18+
- Python 3.8+
- pip

### Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

## API Endpoints

- `POST /upload` - Upload CSV traffic data
- `POST /predict` - Generate traffic predictions
- `POST /video-detect` - Analyze video for vehicle detection
- `GET /traffic-data` - Retrieve traffic data
- `GET /health` - Health check

## Sample Data

The `data/` directory contains:
- Sample Coimbatore traffic datasets (CSV format)
- Demo videos for vehicle detection
- Traffic pattern examples

## Development

1. Start the backend server: `uvicorn main:app --reload`
2. Start the frontend: `npm run dev`
3. Access the application at `http://localhost:5173`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.