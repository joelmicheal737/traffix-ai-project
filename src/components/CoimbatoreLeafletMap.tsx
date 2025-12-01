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

  // Comprehensive Coimbatore locations with accurate coordinates
  const coimbatoreLocations: TrafficPoint[] = [
    // MAJOR COMMERCIAL & BUSINESS DISTRICTS
    { location: 'Gandhipuram', lat: 11.0168, lng: 76.9558, congestionLevel: 'very_high', vehicleCount: 345 },
    { location: 'RS Puram', lat: 11.0041, lng: 76.9597, congestionLevel: 'high', vehicleCount: 289 },
    { location: 'Cross Cut Road', lat: 11.0089, lng: 76.9614, congestionLevel: 'high', vehicleCount: 298 },
    { location: 'Town Hall', lat: 11.0015, lng: 76.9628, congestionLevel: 'high', vehicleCount: 321 },
    { location: 'Race Course', lat: 11.0019, lng: 76.9611, congestionLevel: 'high', vehicleCount: 267 },
    
    // IT CORRIDOR & TECH PARKS
    { location: 'Peelamedu', lat: 11.0296, lng: 77.0266, congestionLevel: 'high', vehicleCount: 256 },
    { location: 'Tidel Park', lat: 11.0240, lng: 77.0020, congestionLevel: 'very_high', vehicleCount: 403 },
    { location: 'ELCOT IT Park', lat: 11.0891, lng: 77.0378, congestionLevel: 'high', vehicleCount: 267 },
    { location: 'Coimbatore IT Park', lat: 11.0350, lng: 77.0180, congestionLevel: 'medium', vehicleCount: 189 },
    
    // EDUCATIONAL INSTITUTIONS AREAS
    { location: 'Saibaba Colony', lat: 11.0214, lng: 76.9214, congestionLevel: 'medium', vehicleCount: 234 },
    { location: 'PSG College Area', lat: 11.0250, lng: 76.9180, congestionLevel: 'high', vehicleCount: 298 },
    { location: 'Bharathiar University', lat: 11.1000, lng: 76.9333, congestionLevel: 'medium', vehicleCount: 167 },
    
    // RESIDENTIAL AREAS
    { location: 'Vadavalli', lat: 11.0708, lng: 76.9044, congestionLevel: 'medium', vehicleCount: 245 },
    { location: 'Thudiyalur', lat: 11.0708, lng: 76.9378, congestionLevel: 'medium', vehicleCount: 212 },
    { location: 'Saravanampatty', lat: 11.0708, lng: 77.0044, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Vilankurichi', lat: 11.0500, lng: 76.9500, congestionLevel: 'low', vehicleCount: 145 },
    { location: 'Kuniyamuthur', lat: 10.9833, lng: 76.9500, congestionLevel: 'medium', vehicleCount: 178 },
    
    // INDUSTRIAL AREAS
    { location: 'Singanallur', lat: 11.0510, lng: 77.0330, congestionLevel: 'high', vehicleCount: 278 },
    { location: 'Kurichi', lat: 11.0167, lng: 77.0500, congestionLevel: 'high', vehicleCount: 334 },
    { location: 'Kalapatti', lat: 11.0167, lng: 76.8833, congestionLevel: 'medium', vehicleCount: 256 },
    { location: 'Neelambur', lat: 11.0500, lng: 77.0667, congestionLevel: 'medium', vehicleCount: 167 },
    
    // TRANSPORT HUBS & TERMINALS
    { location: 'Ukkadam Bus Stand', lat: 10.9995, lng: 76.9569, congestionLevel: 'very_high', vehicleCount: 489 },
    { location: 'Railway Station', lat: 11.0018, lng: 76.9628, congestionLevel: 'very_high', vehicleCount: 467 },
    { location: 'Gandhipuram Bus Stand', lat: 11.0175, lng: 76.9550, congestionLevel: 'very_high', vehicleCount: 445 },
    { location: 'Coimbatore Airport', lat: 11.0297, lng: 77.0436, congestionLevel: 'medium', vehicleCount: 189 },
    
    // MAJOR ROADS & HIGHWAYS
    { location: 'Avinashi Road', lat: 11.0333, lng: 77.0167, congestionLevel: 'high', vehicleCount: 276 },
    { location: 'Pollachi Road', lat: 10.9833, lng: 76.9167, congestionLevel: 'medium', vehicleCount: 234 },
    { location: 'Mettupalayam Road', lat: 11.0500, lng: 76.9167, congestionLevel: 'medium', vehicleCount: 198 },
    { location: 'Trichy Road', lat: 11.0167, lng: 77.0833, congestionLevel: 'medium', vehicleCount: 189 },
    
    // SHOPPING MALLS & ENTERTAINMENT
    { location: 'Brookefields Mall', lat: 11.0167, lng: 77.0167, congestionLevel: 'high', vehicleCount: 312 },
    { location: 'Fun Mall', lat: 11.0089, lng: 76.9725, congestionLevel: 'high', vehicleCount: 254 },
    { location: 'Prozone Mall', lat: 11.0708, lng: 76.9378, congestionLevel: 'medium', vehicleCount: 267 },
    
    // HOSPITALS & MEDICAL CENTERS
    { location: 'KMCH Hospital', lat: 11.0167, lng: 76.9833, congestionLevel: 'medium', vehicleCount: 245 },
    { location: 'PSG Hospitals', lat: 11.0214, lng: 76.9214, congestionLevel: 'medium', vehicleCount: 232 },
    { location: 'Coimbatore Medical College', lat: 11.0083, lng: 76.9667, congestionLevel: 'medium', vehicleCount: 198 },
    
    // ADDITIONAL 70 LOCATIONS FOR COMPREHENSIVE COVERAGE
    
    // Extended Commercial & Business Areas
    { location: 'Sitra', lat: 11.0100, lng: 76.9700, congestionLevel: 'medium', vehicleCount: 198 },
    { location: 'Ramnagar', lat: 11.0200, lng: 76.9800, congestionLevel: 'medium', vehicleCount: 176 },
    { location: 'Tatabad', lat: 11.0300, lng: 76.9900, congestionLevel: 'high', vehicleCount: 234 },
    { location: 'Pappanaickenpalayam', lat: 11.0400, lng: 77.0100, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Ramanathapuram', lat: 11.0150, lng: 76.9750, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Selvapuram', lat: 11.0250, lng: 76.9850, congestionLevel: 'low', vehicleCount: 145 },
    { location: 'Ganapathy', lat: 11.0350, lng: 76.9950, congestionLevel: 'medium', vehicleCount: 178 },
    { location: 'Koundampalayam', lat: 11.0450, lng: 77.0050, congestionLevel: 'low', vehicleCount: 134 },
    { location: 'Kuniamuthur', lat: 10.9900, lng: 76.9600, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Podanur Junction', lat: 10.9850, lng: 76.9850, congestionLevel: 'high', vehicleCount: 245 },
    
    // Extended IT & Tech Areas
    { location: 'Keeranatham', lat: 11.0600, lng: 77.0400, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Chinnavedampatti', lat: 11.0700, lng: 77.0500, congestionLevel: 'low', vehicleCount: 123 },
    { location: 'Kovaipudur', lat: 11.0800, lng: 77.0600, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Thondamuthur', lat: 11.0900, lng: 77.0700, congestionLevel: 'low', vehicleCount: 134 },
    { location: 'Narasimhanaickenpalayam', lat: 11.1000, lng: 77.0800, congestionLevel: 'low', vehicleCount: 112 },
    { location: 'Vellalore', lat: 11.1100, lng: 77.0900, congestionLevel: 'low', vehicleCount: 98 },
    { location: 'Madampatti', lat: 11.1200, lng: 77.1000, congestionLevel: 'low', vehicleCount: 87 },
    { location: 'Idikarai', lat: 11.1300, lng: 77.1100, congestionLevel: 'low', vehicleCount: 76 },
    { location: 'Chettipalayam', lat: 11.1400, lng: 77.1200, congestionLevel: 'low', vehicleCount: 89 },
    { location: 'Kinathukadavu', lat: 10.8000, lng: 76.9000, congestionLevel: 'low', vehicleCount: 65 },
    
    // Extended Educational Areas
    { location: 'Anna University Coimbatore', lat: 11.0180, lng: 76.9300, congestionLevel: 'medium', vehicleCount: 178 },
    { location: 'Amrita University', lat: 10.9000, lng: 76.9000, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Karunya University', lat: 10.9333, lng: 76.7333, congestionLevel: 'low', vehicleCount: 134 },
    { location: 'PSGR Krishnammal College', lat: 11.0200, lng: 76.9250, congestionLevel: 'medium', vehicleCount: 145 },
    { location: 'SNS College of Technology', lat: 11.0500, lng: 77.1000, congestionLevel: 'low', vehicleCount: 123 },
    { location: 'Karpagam University', lat: 11.0167, lng: 76.9000, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Sri Krishna College of Engineering', lat: 11.1000, lng: 77.0500, congestionLevel: 'low', vehicleCount: 112 },
    { location: 'Kumaraguru College of Technology', lat: 11.0700, lng: 76.9200, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Dr. Mahalingam College of Engineering', lat: 10.9500, lng: 77.4833, congestionLevel: 'low', vehicleCount: 98 },
    { location: 'Hindusthan College of Engineering', lat: 11.0333, lng: 76.9167, congestionLevel: 'low', vehicleCount: 134 },
    
    // Extended Residential Areas
    { location: 'Hopes College Junction', lat: 11.0180, lng: 76.9520, congestionLevel: 'medium', vehicleCount: 198 },
    { location: 'Lakshmi Mills Junction', lat: 11.0020, lng: 76.9680, congestionLevel: 'high', vehicleCount: 267 },
    { location: 'Flower Market Road', lat: 11.0030, lng: 76.9650, congestionLevel: 'high', vehicleCount: 234 },
    { location: 'Vegetable Market Area', lat: 11.0015, lng: 76.9660, congestionLevel: 'high', vehicleCount: 245 },
    { location: 'Textile Market Junction', lat: 11.0035, lng: 76.9640, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Oppanakara Street', lat: 11.0025, lng: 76.9655, congestionLevel: 'medium', vehicleCount: 176 },
    { location: 'Nehru Street Junction', lat: 11.0040, lng: 76.9635, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Big Bazaar Street', lat: 11.0030, lng: 76.9645, congestionLevel: 'high', vehicleCount: 212 },
    { location: 'Diwan Bahadur Road', lat: 11.0050, lng: 76.9625, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Bharathi Park Road', lat: 11.0060, lng: 76.9615, congestionLevel: 'low', vehicleCount: 134 },
    
    // Extended Industrial Areas
    { location: 'SIDCO Industrial Estate', lat: 11.0600, lng: 77.0200, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Coimbatore Export Promotion Industrial Park', lat: 11.0700, lng: 77.0300, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Kurichi Industrial Area', lat: 11.0200, lng: 77.0550, congestionLevel: 'high', vehicleCount: 234 },
    { location: 'Kalapatti Industrial Estate', lat: 11.0200, lng: 76.8900, congestionLevel: 'medium', vehicleCount: 178 },
    { location: 'Perur Industrial Area', lat: 11.0400, lng: 76.8900, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Neelambur Industrial Park', lat: 11.0550, lng: 77.0700, congestionLevel: 'medium', vehicleCount: 145 },
    { location: 'Malumichampatti Industrial Area', lat: 11.0900, lng: 76.9200, congestionLevel: 'low', vehicleCount: 123 },
    { location: 'Ondipudur Industrial Estate', lat: 11.0200, lng: 77.0900, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Sulur Industrial Park', lat: 11.0400, lng: 77.1200, congestionLevel: 'low', vehicleCount: 112 },
    { location: 'Annur Industrial Area', lat: 11.2400, lng: 77.1100, congestionLevel: 'low', vehicleCount: 98 },
    
    // Extended Transport & Highway Junctions
    { location: 'Singanallur Bus Stand', lat: 11.0530, lng: 77.0350, congestionLevel: 'high', vehicleCount: 278 },
    { location: 'Peelamedu Bus Stand', lat: 11.0310, lng: 77.0280, congestionLevel: 'medium', vehicleCount: 198 },
    { location: 'Vadavalli Bus Stop', lat: 11.0720, lng: 76.9060, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Thudiyalur Bus Stop', lat: 11.0720, lng: 76.9390, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Saravanampatti Bus Stop', lat: 11.0720, lng: 77.0060, congestionLevel: 'medium', vehicleCount: 145 },
    { location: 'Kuniyamuthur Bus Stop', lat: 10.9850, lng: 76.9520, congestionLevel: 'medium', vehicleCount: 134 },
    { location: 'Podanur Railway Station', lat: 10.9850, lng: 76.9870, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Irugur Railway Station', lat: 11.0167, lng: 76.8833, congestionLevel: 'low', vehicleCount: 123 },
    { location: 'Periyanayakkanpalayam Railway Station', lat: 11.1500, lng: 76.9500, congestionLevel: 'low', vehicleCount: 98 },
    { location: 'Coimbatore North Railway Station', lat: 11.0300, lng: 76.9700, congestionLevel: 'medium', vehicleCount: 178 },
    
    // Extended Major Road Intersections
    { location: 'Avinashi Road - Trichy Road Junction', lat: 11.0400, lng: 77.0200, congestionLevel: 'high', vehicleCount: 298 },
    { location: 'Pollachi Road - Sathy Road Junction', lat: 10.9900, lng: 76.9200, congestionLevel: 'medium', vehicleCount: 189 },
    { location: 'Mettupalayam Road - Avinashi Road Junction', lat: 11.0600, lng: 76.9300, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Salem Road - Trichy Road Junction', lat: 11.0900, lng: 77.0300, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Palakkad Road - Pollachi Road Junction', lat: 10.9700, lng: 76.9400, congestionLevel: 'medium', vehicleCount: 145 },
    { location: 'Sathy Road - Mettupalayam Road Junction', lat: 11.0800, lng: 76.9100, congestionLevel: 'low', vehicleCount: 134 },
    { location: 'Trichy Road - Salem Road Junction', lat: 11.0800, lng: 77.0400, congestionLevel: 'medium', vehicleCount: 178 },
    { location: 'Avinashi Road - Salem Road Junction', lat: 11.0500, lng: 77.0350, congestionLevel: 'medium', vehicleCount: 167 },
    { location: 'Pollachi Road - Palakkad Road Junction', lat: 10.9800, lng: 76.9300, congestionLevel: 'medium', vehicleCount: 156 },
    { location: 'Mettupalayam Road - Sathy Road Junction', lat: 11.0700, lng: 76.9200, congestionLevel: 'low', vehicleCount: 123 },
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
      className: 'custom-div-icon',
      html: `
        <div style="
          background-color: ${color};
          width: 20px;
          height: 20px;
          border-radius: 50%;
          border: 3px solid white;
          box-shadow: 0 2px 6px rgba(0,0,0,0.6);
          position: relative;
        "></div>
      `,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
      popupAnchor: [0, -10],
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
          <h3 style="margin: 0 0 8px 0; font-size: 14px; font-weight: bold; color: #1F2937; border-bottom: 2px solid ${getCongestionColor(point.congestionLevel)}; padding-bottom: 4px;">
            📍 ${point.location}
          </h3>
          <div style="margin-bottom: 6px; display: flex; align-items: center;">
            <span style="font-weight: bold; color: #374151; font-size: 12px;">🚗 Vehicles:</span>
            <span style="margin-left: 8px; font-size: 14px; color: #1F2937;">${point.vehicleCount}</span>
          </div>
          <div style="margin-bottom: 6px; display: flex; align-items: center;">
            <span style="font-weight: bold; color: #374151; font-size: 12px;">🚦 Status:</span>
            <span style="margin-left: 8px; color: ${getCongestionColor(point.congestionLevel)}; font-weight: bold; font-size: 12px;">
              ${getCongestionLabel(point.congestionLevel)}
            </span>
          </div>
          <div style="font-size: 10px; color: #6B7280; margin-top: 8px; padding-top: 6px; border-top: 1px solid #E5E7EB;">
            🕒 Updated: ${new Date().toLocaleTimeString()}
          </div>
        </div>
      `;

      marker.bindPopup(popupContent, {
        maxWidth: 250,
        className: 'custom-popup'
      });
      
      marker.addTo(mapInstanceRef.current);
      markersRef.current.push(marker);
    });
  };

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    // Initialize map with proper options
    const map = L.map(mapRef.current, {
      center: [11.0168, 76.9558], // Coimbatore center
      zoom: 12,
      zoomControl: true,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      boxZoom: true,
      keyboard: true,
      dragging: true,
      touchZoom: true,
      tap: true,
      maxZoom: 18,
      minZoom: 8
    });

    // Add OpenStreetMap tiles with proper attribution
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
      tileSize: 512,
      zoomOffset: -1,
      detectRetina: true,
      crossOrigin: true,
    }).addTo(map);

    mapInstanceRef.current = map;

    // Wait for map to be ready before adding markers
    map.whenReady(() => {
      if (showTrafficMarkers) {
        const locationsToShow = trafficData.length > 0 ? trafficData : coimbatoreLocations;
        setTimeout(() => addMarkers(locationsToShow), 100);
      }
    });

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
        className="rounded-lg shadow-md border border-gray-200 z-0"
      />
      
      {/* Custom CSS for markers */}
      <style jsx>{`
        .custom-div-icon {
          background: transparent !important;
          border: none !important;
        }
        .custom-popup .leaflet-popup-content-wrapper {
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .custom-popup .leaflet-popup-tip {
          background: white;
        }
      `}</style>
    </div>
  );
};

export default CoimbatoreLeafletMap;