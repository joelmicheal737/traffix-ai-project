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
    // MAJOR COMMERCIAL & BUSINESS DISTRICTS
    { location: 'Gandhipuram', lat: 11.0168, lng: 76.9558, congestionLevel: 'very_high', vehicleCount: 345 },
    { location: 'RS Puram', lat: 11.0041, lng: 76.9597, congestionLevel: 'high', vehicleCount: 289 },
    { location: 'Cross Cut Road', lat: 11.0089, lng: 76.9614, congestionLevel: 'high', vehicleCount: 298 },
    { location: 'Town Hall', lat: 11.0015, lng: 76.9628, congestionLevel: 'high', vehicleCount: 321 },
    { location: 'Race Course', lat: 11.0019, lng: 76.9611, congestionLevel: 'high', vehicleCount: 267 },
    { location: 'Big Bazaar Street', lat: 11.0025, lng: 76.9635, congestionLevel: 'high', vehicleCount: 234 },
    { location: 'Oppanakara Street', lat: 11.0012, lng: 76.9642, congestionLevel: 'medium', vehicleCount: 198 },
    { location: 'Nehru Street', lat: 11.0035, lng: 76.9625, congestionLevel: 'medium', vehicleCount: 187 },
    
    // IT CORRIDOR & TECH PARKS
    { location: 'Peelamedu', lat: 11.0296, lng: 77.0266, congestionLevel: 'high', vehicleCount: 256 },
    { location: 'Tidel Park', lat: 11.0240, lng: 77.0020, congestionLevel: 'very_high', vehicleCount: 403 },
    { location: 'ELCOT IT Park', lat: 11.0891, lng: 77.0378, congestionLevel: 'high', vehicleCount: 267 },
    { location: 'Coimbatore IT Park', lat: 11.0350, lng: 77.0180, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Software Technology Park', lat: 11.0280, lng: 77.0150, congestionLevel: 'medium', vehicleCount: 176 },
    { location: 'Cyber Park', lat: 11.0320, lng: 77.0200, congestionLevel: 'medium', vehicleCount: 165 },
    
    // EDUCATIONAL INSTITUTIONS AREAS
    { location: 'Saibaba Colony', lat: 11.0214, lng: 76.9214, congestionLevel: 'medium', vehicleCount: 234 },
    { location: 'PSG College Area', lat: 11.0250, lng: 76.9180, congestionLevel: 'high', vehicleCount: 298 },
    { location: 'Bharathiar University', lat: 11.1000, lng: 76.9333, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Tamil Nadu Agricultural University', lat: 11.0167, lng: 76.9333, congestionLevel: 'low', vehicleCount: 134 },
    { location: 'Government Arts College', lat: 11.0083, lng: 76.9583, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Coimbatore Institute of Technology', lat: 11.0167, lng: 76.9000, congestionLevel: 'low', vehicleCount: 123 },
    
    // RESIDENTIAL AREAS
    { location: 'Vadavalli', lat: 11.0708, lng: 76.9044, congestionLevel: 'medium', vehicleCount: 245 },
    { location: 'Thudiyalur', lat: 11.0708, lng: 76.9378, congestionLevel: 'medium', vehicleCount: 212 },
    { location: 'Saravanampatty', lat: 11.0708, lng: 77.0044, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Vilankurichi', lat: 11.0500, lng: 76.9500, congestionLevel: 'low', vehicleCount: 145 },
    { location: 'Kuniyamuthur', lat: 10.9833, lng: 76.9500, congestionLevel: 'medium', vehicleCount: 178 },
    { location: 'Podanur', lat: 10.9833, lng: 76.9833, congestionLevel: 'low', vehicleCount: 134 },
    { location: 'Sulur', lat: 11.0333, lng: 77.1167, congestionLevel: 'low', vehicleCount: 98 },
    { location: 'Madukkarai', lat: 10.9000, lng: 76.9500, congestionLevel: 'low', vehicleCount: 87 },
    
    // INDUSTRIAL AREAS
    { location: 'Singanallur', lat: 11.0510, lng: 77.0330, congestionLevel: 'high', vehicleCount: 278 },
    { location: 'Kurichi', lat: 11.0167, lng: 77.0500, congestionLevel: 'high', vehicleCount: 334 },
    { location: 'Kalapatti', lat: 11.0167, lng: 76.8833, congestionLevel: 'medium', vehicleCount: 256 },
    { location: 'Perur', lat: 11.0333, lng: 76.8833, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Neelambur', lat: 11.0500, lng: 77.0667, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Ondipudur', lat: 11.0167, lng: 77.0833, congestionLevel: 'medium', vehicleCount: 176 },
    { location: 'Malumichampatti', lat: 11.0833, lng: 76.9167, congestionLevel: 'low', vehicleCount: 123 },
    
    // TRANSPORT HUBS & TERMINALS
    { location: 'Ukkadam Bus Stand', lat: 10.9995, lng: 76.9569, congestionLevel: 'very_high', vehicleCount: 489 },
    { location: 'Railway Station', lat: 11.0018, lng: 76.9628, congestionLevel: 'very_high', vehicleCount: 467 },
    { location: 'Gandhipuram Bus Stand', lat: 11.0175, lng: 76.9550, congestionLevel: 'very_high', vehicleCount: 445 },
    { location: 'Singanallur Bus Stand', lat: 11.0520, lng: 77.0340, congestionLevel: 'high', vehicleCount: 298 },
    { location: 'Peelamedu Bus Stand', lat: 11.0300, lng: 77.0270, congestionLevel: 'medium', vehicleCount: 234 },
    { location: 'Coimbatore Airport', lat: 11.0297, lng: 77.0436, congestionLevel: 'medium', vehicleCount: 189 },
    
    // MAJOR ROADS & HIGHWAYS
    { location: 'Avinashi Road', lat: 11.0333, lng: 77.0167, congestionLevel: 'high', vehicleCount: 276 },
    { location: 'Pollachi Road', lat: 10.9833, lng: 76.9167, congestionLevel: 'medium', vehicleCount: 234 },
    { location: 'Mettupalayam Road', lat: 11.0500, lng: 76.9167, congestionLevel: 'medium', vehicleCount: 198 },
    { location: 'Palakkad Road', lat: 10.9667, lng: 76.9333, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Trichy Road', lat: 11.0167, lng: 77.0833, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Salem Road', lat: 11.0833, lng: 77.0167, congestionLevel: 'low', vehicleCount: 145 },
    { location: 'Sathy Road', lat: 11.0667, lng: 76.9000, congestionLevel: 'low', vehicleCount: 134 },
    
    // SHOPPING MALLS & ENTERTAINMENT
    { location: 'Brookefields Mall', lat: 11.0167, lng: 77.0167, congestionLevel: 'high', vehicleCount: 312 },
    { location: 'Fun Mall', lat: 11.0089, lng: 76.9725, congestionLevel: 'high', vehicleCount: 254 },
    { location: 'Prozone Mall', lat: 11.0708, lng: 76.9378, congestionLevel: 'medium', vehicleCount: 267 },
    { location: 'DB City Mall', lat: 11.0200, lng: 76.9600, congestionLevel: 'medium', vehicleCount: 198 },
    { location: 'VR Mall', lat: 11.0150, lng: 76.9580, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Lulu Mall', lat: 11.0180, lng: 76.9620, congestionLevel: 'high', vehicleCount: 278 },
    
    // HOSPITALS & MEDICAL CENTERS
    { location: 'KMCH Hospital', lat: 11.0167, lng: 76.9833, congestionLevel: 'medium', vehicleCount: 245 },
    { location: 'PSG Hospitals', lat: 11.0214, lng: 76.9214, congestionLevel: 'medium', vehicleCount: 232 },
    { location: 'Coimbatore Medical College', lat: 11.0083, lng: 76.9667, congestionLevel: 'medium', vehicleCount: 198 },
    { location: 'GEM Hospital', lat: 11.0250, lng: 76.9800, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Sri Ramakrishna Hospital', lat: 11.0200, lng: 76.9750, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Ganga Hospital', lat: 11.0300, lng: 76.9900, congestionLevel: 'medium', vehicleCount: 176 },
    
    // TEMPLES & RELIGIOUS PLACES
    { location: 'Marudamalai Temple', lat: 11.0500, lng: 76.8500, congestionLevel: 'medium', vehicleCount: 234 },
    { location: 'Perur Patteeswarar Temple', lat: 11.0333, lng: 76.8833, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Eachanari Vinayagar Temple', lat: 11.0833, lng: 76.9500, congestionLevel: 'low', vehicleCount: 145 },
    { location: 'Dhyanalinga Temple', lat: 11.0167, lng: 76.7333, congestionLevel: 'low', vehicleCount: 123 },
    { location: 'Arulmigu Subramaniaswamy Temple', lat: 11.0000, lng: 76.9500, congestionLevel: 'low', vehicleCount: 134 },
    
    // MARKETS & COMMERCIAL AREAS
    { location: 'Flower Market', lat: 11.0020, lng: 76.9640, congestionLevel: 'high', vehicleCount: 298 },
    { location: 'Vegetable Market', lat: 11.0010, lng: 76.9650, congestionLevel: 'high', vehicleCount: 267 },
    { location: 'Textile Market', lat: 11.0030, lng: 76.9630, congestionLevel: 'medium', vehicleCount: 234 },
    { location: 'Lakshmi Mills Junction', lat: 11.0000, lng: 76.9667, congestionLevel: 'high', vehicleCount: 298 },
    { location: 'Hopes College Junction', lat: 11.0167, lng: 76.9500, congestionLevel: 'medium', vehicleCount: 243 },
    
    // OUTER SUBURBAN AREAS
    { location: 'Annur', lat: 11.2333, lng: 77.1000, congestionLevel: 'low', vehicleCount: 87 },
    { location: 'Mettupalayam', lat: 11.3000, lng: 76.9333, congestionLevel: 'low', vehicleCount: 98 },
    { location: 'Pollachi', lat: 10.6667, lng: 77.0000, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Udumalaipettai', lat: 10.5833, lng: 77.2500, congestionLevel: 'low', vehicleCount: 76 },
    { location: 'Palladam', lat: 11.1500, lng: 77.2833, congestionLevel: 'low', vehicleCount: 89 },
    { location: 'Tirupur Road', lat: 11.1000, lng: 77.3500, congestionLevel: 'medium', vehicleCount: 134 },
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