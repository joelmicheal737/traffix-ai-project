# Google Maps Integration Setup Guide

## Getting Started with Google Maps API

### Step 1: Get Google Maps API Key

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/google/maps-apis

2. **Create a New Project** (if needed)
   - Click "Select a project" → "New Project"
   - Enter project name: "Traffix AI Coimbatore"
   - Click "Create"

3. **Enable Required APIs**
   - Maps JavaScript API
   - Places API (optional, for search functionality)

4. **Create API Key**
   - Go to "Credentials" → "Create Credentials" → "API Key"
   - Copy the generated API key

### Step 2: Configure API Key Restrictions (Recommended)

1. **Application Restrictions**
   - Choose "HTTP referrers (web sites)"
   - Add your domains:
     - `http://localhost:5173/*` (for development)
     - `https://yourdomain.com/*` (for production)

2. **API Restrictions**
   - Select "Restrict key"
   - Choose: "Maps JavaScript API"

### Step 3: Add API Key to Environment

1. **Update .env file**
   ```
   VITE_GOOGLE_MAPS_API_KEY=YOUR_ACTUAL_API_KEY_HERE
   ```

2. **Restart Development Server**
   ```bash
   npm run dev
   ```

## Features Included

✅ **Interactive Google Maps**
- Real Coimbatore map with accurate coordinates
- Zoom and pan functionality
- Street view integration

✅ **Google Traffic Layer**
- Real-time traffic data from Google
- Color-coded traffic conditions
- Live traffic updates

✅ **Custom Traffic Markers**
- Color-coded congestion levels
- Vehicle count display
- Interactive info windows
- Real traffic monitoring points

✅ **Coimbatore-Specific Locations**
- Gandhipuram (Commercial Hub)
- RS Puram (Mixed Area)
- Peelamedu (IT Corridor)
- Saibaba Colony (Residential/Educational)
- Singanallur (Industrial/Residential)
- Tidel Park (IT/Business District)
- Race Course (Central Area)
- Ukkadam (Transport Hub)

## Map Controls

- **Zoom Controls**: Mouse wheel or +/- buttons
- **Pan**: Click and drag
- **Traffic Layer**: Automatically enabled (red lines show traffic)
- **Map Type**: Road map optimized for traffic viewing
- **Info Windows**: Click markers for detailed traffic information

## Customization Options

The map component supports:
- Custom traffic data points
- Adjustable height
- Real-time data updates
- Responsive design
- Multiple congestion levels
- Custom marker styling

## Troubleshooting

**Map not loading?**
- Check API key is correctly set in .env
- Verify API key has Maps JavaScript API enabled
- Check browser console for error messages

**Markers not showing?**
- Ensure traffic data has valid lat/lng coordinates
- Check congestion level values are valid

**Performance issues?**
- Consider limiting number of markers
- Implement marker clustering for large datasets
- Use map bounds to load only visible markers