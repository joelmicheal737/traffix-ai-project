import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default markers in Leaflet with Vite
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

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
  showTrafficMarkers?: boolean;
}

const CoimbatoreLeafletMap = ({ 
  trafficData = [], 
  height = '500px', 
  showTrafficMarkers = true 
}: CoimbatoreMapProps) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersRef = useRef<L.Marker[]>([]);

  // Comprehensive Coimbatore locations
  const coimbatoreLocations: TrafficPoint[] = [
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

  const createCustomIcon = (congestionLevel: string) => {
    const color = getCongestionColor(congestionLevel);
    return L.divIcon({
      className: 'custom-marker',
      html: `
        <div style="
          background-color: ${color};
          width: 20px;
          height: 20px;
          border-radius: 50%;
          border: 3px solid white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
          display: flex;
          align-items: center;
          justify-content: center;
        ">
          <div style="
            width: 8px;
            height: 8px;
            background-color: white;
            border-radius: 50%;
          "></div>
        </div>
      `,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });
  };

  const clearMarkers = () => {
    markersRef.current.forEach(marker => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.removeLayer(marker);
      }
    });
    markersRef.current = [];
  };

  const addMarkers = (locations: TrafficPoint[]) => {
    if (!mapInstanceRef.current) return;

    clearMarkers();

    locations.forEach(point => {
      if (!mapInstanceRef.current) return;

      const marker = L.marker([point.lat, point.lng], {
        icon: createCustomIcon(point.congestionLevel)
      });

      const popupContent = `
        <div style="padding: 8px; min-width: 200px; font-family: Arial, sans-serif;">
          <h3 style="margin: 0 0 8px 0; font-size: 16px; font-weight: bold; color: #1F2937; border-bottom: 2px solid ${getCongestionColor(point.congestionLevel)}; padding-bottom: 4px;">
            📍 ${point.location}
          </h3>
          <div style="margin-bottom: 6px; display: flex; align-items: center;">
            <span style="font-weight: bold; color: #374151;">🚗 Vehicles:</span>
            <span style="margin-left: 8px; font-size: 16px; color: #1F2937;">${point.vehicleCount}</span>
          </div>
          <div style="margin-bottom: 6px; display: flex; align-items: center;">
            <span style="font-weight: bold; color: #374151;">🚦 Status:</span>
            <span style="margin-left: 8px; color: ${getCongestionColor(point.congestionLevel)}; font-weight: bold; font-size: 14px;">
              ${getCongestionLabel(point.congestionLevel)}
            </span>
          </div>
          <div style="font-size: 11px; color: #6B7280; margin-top: 8px; padding-top: 6px; border-top: 1px solid #E5E7EB;">
            🕒 Last updated: ${new Date().toLocaleTimeString()}
          </div>
        </div>
      `;

      marker.bindPopup(popupContent);
      marker.addTo(mapInstanceRef.current);
      markersRef.current.push(marker);
    });
  };

  useEffect(() => {
    if (!mapRef.current) return;

    // Initialize map
    const map = L.map(mapRef.current).setView([11.0168, 76.9558], 12);

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    // Add markers if traffic data is provided or show default locations
    if (showTrafficMarkers) {
      const locationsToShow = trafficData.length > 0 ? trafficData : coimbatoreLocations;
      addMarkers(locationsToShow);
    }

    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update markers when traffic data changes
  useEffect(() => {
    if (showTrafficMarkers && mapInstanceRef.current) {
      const locationsToShow = trafficData.length > 0 ? trafficData : coimbatoreLocations;
      addMarkers(locationsToShow);
    }
  }, [trafficData, showTrafficMarkers]);

  return (
    <div className="relative">
      <div 
        ref={mapRef} 
        style={{ height, width: '100%' }} 
        className="rounded-lg shadow-md border border-gray-200"
      />
      
      {/* Legend */}
      {showTrafficMarkers && (
        <div className="absolute bottom-4 right-4 z-[1000] bg-white p-4 rounded-lg shadow-lg border">
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
              <div>🗺️ OpenStreetMap Data</div>
              <div>📍 {trafficData.length > 0 ? trafficData.length : coimbatoreLocations.length} Monitor Points</div>
            </div>
          </div>
        </div>
      )}

      {/* Info Panel */}
      <div className="absolute top-4 left-4 z-[1000] bg-white p-4 rounded-lg shadow-lg border">
        <h3 className="font-bold text-sm text-gray-900 mb-1">🏙️ Coimbatore Traffic</h3>
        <p className="text-xs text-gray-600">Free OpenStreetMap • No API costs</p>
        <div className="text-xs text-blue-600 mt-1">
          📊 {trafficData.length > 0 ? trafficData.length : coimbatoreLocations.length} active locations
        </div>
      </div>
    </div>
  );
};

export default CoimbatoreLeafletMap;