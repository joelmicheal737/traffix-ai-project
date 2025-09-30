import { useEffect, useRef, useState, useCallback } from 'react';
import { Wrapper, Status } from '@googlemaps/react-wrapper';

interface TrafficPoint {
  location: string;
  lat: number;
  lng: number;
  congestionLevel: 'low' | 'medium' | 'high' | 'very_high';
  vehicleCount: number;
}

interface CoimbatoreMapProps {
  trafficData?: TrafficPoint[];
  height?: string;
}

const render = (status: Status) => {
  switch (status) {
    case Status.LOADING:
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-gray-600">Loading Coimbatore map...</p>
          </div>
        </div>
      );
    case Status.FAILURE:
      return (
        <div className="flex items-center justify-center h-full bg-red-50">
          <div className="text-center p-6">
            <p className="text-red-600 mb-2">Failed to load Google Maps</p>
            <p className="text-sm text-red-500">Please check your API key configuration</p>
            <div className="mt-4 text-xs text-gray-600">
              <p>Add to .env file:</p>
              <code className="bg-gray-100 p-1 rounded">VITE_GOOGLE_MAPS_API_KEY=your_key</code>
            </div>
          </div>
        </div>
      );
    default:
      return null;
  }
};

interface MapComponentProps {
  trafficData: TrafficPoint[];
  height: string;
}

const MapComponent = ({ trafficData, height }: MapComponentProps) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const map = useRef<google.maps.Map>();
  const markersRef = useRef<google.maps.Marker[]>([]);
  const infoWindowRef = useRef<google.maps.InfoWindow>();
  const trafficLayerRef = useRef<google.maps.TrafficLayer>();

  const getCongestionColor = (congestionLevel: string) => {
    switch (congestionLevel) {
      case 'low': return '#10B981'; // green
      case 'medium': return '#F59E0B'; // yellow
      case 'high': return '#EF4444'; // red
      case 'very_high': return '#7C2D12'; // dark red
      default: return '#6B7280'; // gray
    }
  };

  const getCongestionLabel = (level: string) => {
    return level.replace('_', ' ').toUpperCase();
  };

  const createMarkerIcon = (congestionLevel: string) => {
    const color = getCongestionColor(congestionLevel);
    return {
      path: google.maps.SymbolPath.CIRCLE,
      fillColor: color,
      fillOpacity: 0.9,
      stroke: '#FFFFFF',
      strokeWeight: 3,
      scale: 10,
    };
  };

  const clearMarkers = () => {
    markersRef.current.forEach(marker => marker.setMap(null));
    markersRef.current = [];
  };

  const createInfoWindowContent = (point: TrafficPoint) => {
    const color = getCongestionColor(point.congestionLevel);
    const timestamp = new Date().toLocaleTimeString();
    
    return `
      <div style="padding: 12px; min-width: 220px; font-family: Arial, sans-serif;">
        <h3 style="margin: 0 0 10px 0; font-size: 16px; font-weight: bold; color: #1F2937; border-bottom: 2px solid ${color}; padding-bottom: 5px;">
          📍 ${point.location}
        </h3>
        <div style="margin-bottom: 8px; display: flex; align-items: center;">
          <span style="font-weight: bold; color: #374151;">🚗 Vehicles:</span>
          <span style="margin-left: 8px; font-size: 18px; color: #1F2937;">${point.vehicleCount}</span>
        </div>
        <div style="margin-bottom: 8px; display: flex; align-items: center;">
          <span style="font-weight: bold; color: #374151;">🚦 Status:</span>
          <span style="margin-left: 8px; color: ${color}; font-weight: bold; font-size: 14px;">
            ${getCongestionLabel(point.congestionLevel)}
          </span>
        </div>
        <div style="font-size: 11px; color: #6B7280; margin-top: 10px; padding-top: 8px; border-top: 1px solid #E5E7EB;">
          🕒 Last updated: ${timestamp}
        </div>
      </div>
    `;
  };

  const initializeMap = useCallback(() => {
    if (!mapRef.current) return;

    // Coimbatore center coordinates
    const coimbatoreCenter = { lat: 11.0168, lng: 76.9558 };

    map.current = new google.maps.Map(mapRef.current, {
      center: coimbatoreCenter,
      zoom: 12,
      mapTypeId: google.maps.MapTypeId.ROADMAP,
      styles: [
        {
          featureType: 'poi.business',
          elementType: 'labels',
          stylers: [{ visibility: 'off' }]
        },
        {
          featureType: 'transit.station',
          elementType: 'labels',
          stylers: [{ visibility: 'off' }]
        }
      ],
      mapTypeControl: true,
      streetViewControl: true,
      fullscreenControl: true,
      zoomControl: true,
    });

    // Initialize info window
    infoWindowRef.current = new google.maps.InfoWindow({
      maxWidth: 300,
    });

    // Add traffic layer
    trafficLayerRef.current = new google.maps.TrafficLayer();
    trafficLayerRef.current.setMap(map.current);

  }, []);

  const updateMarkers = useCallback(() => {
    if (!map.current) return;

    // Clear existing markers
    clearMarkers();

    // Add new markers
    trafficData.forEach(point => {
      const marker = new google.maps.Marker({
        position: { lat: point.lat, lng: point.lng },
        map: map.current,
        title: `${point.location} - ${getCongestionLabel(point.congestionLevel)}`,
        icon: createMarkerIcon(point.congestionLevel),
        animation: google.maps.Animation.DROP,
      });

      // Add click listener for info window
      marker.addListener('click', () => {
        if (infoWindowRef.current) {
          infoWindowRef.current.setContent(createInfoWindowContent(point));
          infoWindowRef.current.open(map.current, marker);
        }
      });

      // Add hover effect
      marker.addListener('mouseover', () => {
        marker.setIcon({
          ...createMarkerIcon(point.congestionLevel),
          scale: 12,
        });
      });

      marker.addListener('mouseout', () => {
        marker.setIcon(createMarkerIcon(point.congestionLevel));
      });

      markersRef.current.push(marker);
    });
  }, [trafficData]);

  useEffect(() => {
    initializeMap();
  }, [initializeMap]);

  useEffect(() => {
    updateMarkers();
  }, [updateMarkers]);

  return (
    <div style={{ height, width: '100%' }} className="rounded-lg shadow-md relative overflow-hidden">
      <div ref={mapRef} style={{ height: '100%', width: '100%' }} />
      
      {/* Enhanced Legend */}
      <div className="absolute bottom-4 right-4 z-10 bg-white p-4 rounded-lg shadow-lg border">
        <h4 className="font-bold text-sm text-gray-900 mb-3">🚦 Traffic Status</h4>
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 rounded-full bg-green-500"></div>
            <span className="text-xs font-medium">Low Congestion</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
            <span className="text-xs font-medium">Medium Traffic</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 rounded-full bg-red-500"></div>
            <span className="text-xs font-medium">Heavy Traffic</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 rounded-full bg-red-800"></div>
            <span className="text-xs font-medium">Severe Congestion</span>
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="text-xs text-gray-600 space-y-1">
            <div>🔴 Google Live Traffic</div>
            <div>📍 {trafficData.length} Monitor Points</div>
          </div>
        </div>
      </div>

      {/* Info Panel */}
      <div className="absolute top-4 left-4 z-10 bg-white p-4 rounded-lg shadow-lg border">
        <h3 className="font-bold text-sm text-gray-900 mb-1">🏙️ Coimbatore Traffic</h3>
        <p className="text-xs text-gray-600">Live monitoring • Updated every 5 min</p>
        <div className="text-xs text-blue-600 mt-1">
          📊 {trafficData.length} active locations
        </div>
      </div>
    </div>
  );
};

const CoimbatoreMap = ({ trafficData = [], height = '500px' }: CoimbatoreMapProps) => {
  // Comprehensive Coimbatore locations with accurate coordinates
  const defaultLocations: TrafficPoint[] = [
    // Major Commercial Areas
    { location: 'Gandhipuram', lat: 11.0168, lng: 76.9558, congestionLevel: 'high', vehicleCount: 245 },
    { location: 'RS Puram', lat: 11.0041, lng: 76.9597, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Cross Cut Road', lat: 11.0089, lng: 76.9614, congestionLevel: 'high', vehicleCount: 198 },
    
    // IT Corridor & Tech Areas
    { location: 'Peelamedu', lat: 11.0296, lng: 77.0266, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Tidel Park', lat: 11.0240, lng: 77.0020, congestionLevel: 'high', vehicleCount: 203 },
    { location: 'ELCOT IT Park', lat: 11.0891, lng: 77.0378, congestionLevel: 'medium', vehicleCount: 167 },
    
    // Educational & Residential Areas
    { location: 'Saibaba Colony', lat: 11.0214, lng: 76.9214, congestionLevel: 'low', vehicleCount: 134 },
    { location: 'Vadavalli', lat: 11.0708, lng: 76.9044, congestionLevel: 'medium', vehicleCount: 145 },
    { location: 'Thudiyalur', lat: 11.0708, lng: 76.9378, congestionLevel: 'low', vehicleCount: 112 },
    
    // Industrial Areas
    { location: 'Singanallur', lat: 11.0510, lng: 77.0330, congestionLevel: 'medium', vehicleCount: 178 },
    { location: 'Kurichi', lat: 11.0167, lng: 77.0500, congestionLevel: 'high', vehicleCount: 234 },
    { location: 'Kalapatti', lat: 11.0167, lng: 76.8833, congestionLevel: 'medium', vehicleCount: 156 },
    
    // Transport Hubs
    { location: 'Ukkadam', lat: 10.9995, lng: 76.9569, congestionLevel: 'very_high', vehicleCount: 289 },
    { location: 'Town Hall', lat: 11.0015, lng: 76.9628, congestionLevel: 'high', vehicleCount: 221 },
    { location: 'Railway Station', lat: 11.0018, lng: 76.9628, congestionLevel: 'high', vehicleCount: 267 },
    
    // Major Junctions & Roads
    { location: 'Race Course', lat: 11.0019, lng: 76.9611, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Hopes College', lat: 11.0167, lng: 76.9500, congestionLevel: 'medium', vehicleCount: 143 },
    { location: 'Lakshmi Mills', lat: 11.0000, lng: 76.9667, congestionLevel: 'high', vehicleCount: 198 },
    
    // Outer Areas
    { location: 'Pollachi Road', lat: 10.9833, lng: 76.9167, congestionLevel: 'medium', vehicleCount: 134 },
    { location: 'Mettupalayam Road', lat: 11.0500, lng: 76.9167, congestionLevel: 'low', vehicleCount: 98 },
    { location: 'Avinashi Road', lat: 11.0333, lng: 77.0167, congestionLevel: 'medium', vehicleCount: 176 },
    
    // Shopping & Entertainment
    { location: 'Brookefields Mall', lat: 11.0167, lng: 77.0167, congestionLevel: 'high', vehicleCount: 212 },
    { location: 'Fun Mall', lat: 11.0089, lng: 76.9725, congestionLevel: 'medium', vehicleCount: 154 },
    { location: 'Prozone Mall', lat: 11.0708, lng: 76.9378, congestionLevel: 'medium', vehicleCount: 167 },
    
    // Hospitals & Medical
    { location: 'KMCH', lat: 11.0167, lng: 76.9833, congestionLevel: 'medium', vehicleCount: 145 },
    { location: 'PSG Hospitals', lat: 11.0214, lng: 76.9214, congestionLevel: 'medium', vehicleCount: 132 },
    
    // Suburban Areas
    { location: 'Saravanampatty', lat: 11.0708, lng: 77.0044, congestionLevel: 'low', vehicleCount: 89 },
    { location: 'Ondipudur', lat: 11.0167, lng: 77.0833, congestionLevel: 'low', vehicleCount: 76 },
    { location: 'Kuniyamuthur', lat: 10.9833, lng: 76.9500, congestionLevel: 'medium', vehicleCount: 123 },
  ];

  // Use provided traffic data or comprehensive default locations
  const locations = trafficData.length > 0 ? trafficData : defaultLocations;

  // Check if Google Maps API key is available
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
  
  if (!apiKey) {
    return (
      <div style={{ height, width: '100%' }} className="rounded-lg shadow-md bg-gradient-to-br from-red-50 to-orange-100 border-2 border-red-200 flex items-center justify-center">
        <div className="text-center p-8 max-w-md">
          <div className="text-6xl mb-4">🔑</div>
          <h3 className="text-xl font-bold text-red-800 mb-3">Google Maps API Key Required</h3>
          <p className="text-red-700 mb-4">
            Please add a valid Google Maps API key to display the interactive Coimbatore traffic map.
          </p>
          <div className="bg-white p-4 rounded-lg shadow-sm mb-4">
            <p className="text-sm text-gray-600 mb-2">Add to your .env file:</p>
            <code className="text-xs bg-gray-100 p-2 rounded block font-mono">
              VITE_GOOGLE_MAPS_API_KEY=your_api_key_here
            </code>
          </div>
          <div className="text-sm text-red-600">
            <p className="mb-2 font-semibold">📋 Quick Setup Steps:</p>
            <ol className="text-left space-y-1">
              <li>1. Go to Google Cloud Console</li>
              <li>2. Enable "Maps JavaScript API"</li>
              <li>3. Create API key credentials</li>
              <li>4. Add key to .env file</li>
              <li>5. Restart the development server</li>
            </ol>
          </div>
          <a 
            href="https://console.cloud.google.com/google/maps-apis" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-block mt-4 bg-red-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-red-700 transition-colors"
          >
            🚀 Get API Key →
          </a>
        </div>
      </div>
    );
  }

  return (
    <Wrapper apiKey={apiKey} render={render} libraries={['places', 'geometry']}>
      <MapComponent trafficData={locations} height={height} />
    </Wrapper>
  );
};

export default CoimbatoreMap;