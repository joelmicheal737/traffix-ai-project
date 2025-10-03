import { useEffect, useRef } from 'react';
import { Wrapper, Status } from '@googlemaps/react-wrapper';

const render = (status: Status) => {
  switch (status) {
    case Status.LOADING:
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-gray-600">Loading map...</p>
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

interface SimpleMapProps {
  height?: string;
  center?: { lat: number; lng: number };
  zoom?: number;
}

const MapComponent = ({ height, center, zoom }: SimpleMapProps) => {
  const mapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mapRef.current) return;

    const map = new google.maps.Map(mapRef.current, {
      center: center || { lat: 11.0168, lng: 76.9558 }, // Coimbatore center
      zoom: zoom || 12,
      mapTypeId: google.maps.MapTypeId.ROADMAP,
      mapTypeControl: true,
      streetViewControl: true,
      fullscreenControl: true,
      zoomControl: true,
    });

  }, [center, zoom]);

  return <div ref={mapRef} style={{ height: height || '500px', width: '100%' }} />;
};

const SimpleMap = ({ height = '500px', center, zoom }: SimpleMapProps) => {
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
  
  if (!apiKey) {
    return (
      <div style={{ height, width: '100%' }} className="rounded-lg shadow-md bg-gray-100 border flex items-center justify-center">
        <div className="text-center p-8">
          <div className="text-4xl mb-4">🗺️</div>
          <h3 className="text-lg font-bold text-gray-800 mb-2">Google Maps API Key Required</h3>
          <p className="text-gray-600 mb-4">
            Add your Google Maps API key to display the map.
          </p>
          <div className="bg-white p-3 rounded border">
            <code className="text-sm">VITE_GOOGLE_MAPS_API_KEY=your_key</code>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg shadow-md overflow-hidden">
      <Wrapper apiKey={apiKey} render={render}>
        <MapComponent height={height} center={center} zoom={zoom} />
      </Wrapper>
    </div>
  );
};

export default SimpleMap;