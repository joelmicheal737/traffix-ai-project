import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { MapPin, TrendingUp, Clock, AlertTriangle } from 'lucide-react';
import SimpleMap from '../components/SimpleMap';

interface TrafficData {
  timestamp: string;
  location: string;
  vehicle_count: number;
  avg_speed: number;
  congestion_level: string;
  weather: string;
  day_of_week: string;
}

const Dashboard = () => {
  const [trafficData, setTrafficData] = useState<TrafficData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedLocation, setSelectedLocation] = useState<string>('all');

  useEffect(() => {
    fetchTrafficData();
  }, []);

  const fetchTrafficData = async () => {
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
      
      // Check if backend is available
      const healthResponse = await fetch(`${apiBaseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(5000), // 5 second timeout
      });
      
      if (healthResponse.ok) {
        const response = await fetch(`${apiBaseUrl}/traffic-data?limit=200`);
        if (response.ok) {
          const data = await response.json();
          setTrafficData(data.data);
          return;
        }
      }
      
      throw new Error('Backend server not available');
    } catch (error) {
      console.warn('Backend not available, using sample data:', error);
      // Use sample data if API is not available
      setSampleData();
    } finally {
      setLoading(false);
    }
  };

  const setSampleData = () => {
    const sampleData: TrafficData[] = [
      { timestamp: '2024-01-01 08:00:00', location: 'Gandhipuram', vehicle_count: 245, avg_speed: 25.5, congestion_level: 'high', weather: 'clear', day_of_week: 'monday' },
      { timestamp: '2024-01-01 08:00:00', location: 'RS Puram', vehicle_count: 189, avg_speed: 32.1, congestion_level: 'medium', weather: 'clear', day_of_week: 'monday' },
      { timestamp: '2024-01-01 08:00:00', location: 'Peelamedu', vehicle_count: 156, avg_speed: 38.2, congestion_level: 'low', weather: 'clear', day_of_week: 'monday' },
      { timestamp: '2024-01-01 08:00:00', location: 'Saibaba Colony', vehicle_count: 134, avg_speed: 41.2, congestion_level: 'low', weather: 'clear', day_of_week: 'monday' },
    ];
    setTrafficData(sampleData);
  };

  const filteredData = selectedLocation === 'all' 
    ? trafficData 
    : trafficData.filter(item => item.location === selectedLocation);

  const locations = [...new Set(trafficData.map(item => item.location))];

  // Prepare chart data
  const vehicleCountData = locations.map(location => {
    const locationData = trafficData.filter(item => item.location === location);
    const avgCount = locationData.reduce((sum, item) => sum + item.vehicle_count, 0) / locationData.length;
    return { location, vehicle_count: Math.round(avgCount) };
  });

  const speedData = locations.map(location => {
    const locationData = trafficData.filter(item => item.location === location);
    const avgSpeed = locationData.reduce((sum, item) => sum + item.avg_speed, 0) / locationData.length;
    return { location, avg_speed: Math.round(avgSpeed * 10) / 10 };
  });

  const congestionData = [
    { name: 'Low', value: trafficData.filter(item => item.congestion_level === 'low').length, color: '#10B981' },
    { name: 'Medium', value: trafficData.filter(item => item.congestion_level === 'medium').length, color: '#F59E0B' },
    { name: 'High', value: trafficData.filter(item => item.congestion_level === 'high').length, color: '#EF4444' },
    { name: 'Very High', value: trafficData.filter(item => item.congestion_level === 'very_high').length, color: '#7C2D12' },
  ];

  // Map data
  const mapData = locations.map(location => {
    const locationData = trafficData.filter(item => item.location === location);
    const avgCount = locationData.reduce((sum, item) => sum + item.vehicle_count, 0) / locationData.length;
    const mostCommonCongestion = locationData.length > 0 ? locationData[0].congestion_level : 'low';
    
    // Comprehensive Coimbatore coordinates for all major locations
    const coordinates: { [key: string]: { lat: number; lng: number } } = {
      // Major Commercial Areas
      'Gandhipuram': { lat: 11.0168, lng: 76.9558 },
      'RS Puram': { lat: 11.0041, lng: 76.9597 },
      'Cross Cut Road': { lat: 11.0089, lng: 76.9614 },
      
      // IT Corridor & Tech Areas
      'Peelamedu': { lat: 11.0296, lng: 77.0266 },
      'Tidel Park': { lat: 11.0240, lng: 77.0020 },
      'ELCOT IT Park': { lat: 11.0891, lng: 77.0378 },
      
      // Educational & Residential Areas
      'Saibaba Colony': { lat: 11.0214, lng: 76.9214 },
      'Vadavalli': { lat: 11.0708, lng: 76.9044 },
      'Thudiyalur': { lat: 11.0708, lng: 76.9378 },
      
      // Industrial Areas
      'Singanallur': { lat: 11.0510, lng: 77.0330 },
      'Kurichi': { lat: 11.0167, lng: 77.0500 },
      'Kalapatti': { lat: 11.0167, lng: 76.8833 },
      
      // Transport Hubs
      'Ukkadam': { lat: 10.9995, lng: 76.9569 },
      'Town Hall': { lat: 11.0015, lng: 76.9628 },
      'Railway Station': { lat: 11.0018, lng: 76.9628 },
      
      // Major Junctions & Roads
      'Race Course': { lat: 11.0019, lng: 76.9611 },
      'Hopes College': { lat: 11.0167, lng: 76.9500 },
      'Lakshmi Mills': { lat: 11.0000, lng: 76.9667 },
      
      // Outer Areas
      'Pollachi Road': { lat: 10.9833, lng: 76.9167 },
      'Mettupalayam Road': { lat: 11.0500, lng: 76.9167 },
      'Avinashi Road': { lat: 11.0333, lng: 77.0167 },
      
      // Shopping & Entertainment
      'Brookefields Mall': { lat: 11.0167, lng: 77.0167 },
      'Fun Mall': { lat: 11.0089, lng: 76.9725 },
      'Prozone Mall': { lat: 11.0708, lng: 76.9378 },
      
      // Hospitals & Medical
      'KMCH': { lat: 11.0167, lng: 76.9833 },
      'PSG Hospitals': { lat: 11.0214, lng: 76.9214 },
      
      // Suburban Areas
      'Saravanampatty': { lat: 11.0708, lng: 77.0044 },
      'Ondipudur': { lat: 11.0167, lng: 77.0833 },
      'Kuniyamuthur': { lat: 10.9833, lng: 76.9500 },
    };

    return {
      location,
      lat: coordinates[location]?.lat || 11.0168,
      lng: coordinates[location]?.lng || 76.9558,
      congestionLevel: mostCommonCongestion as 'low' | 'medium' | 'high' | 'very_high',
      vehicleCount: Math.round(avgCount),
    };
  });

  const currentTime = new Date().toLocaleString();
  const totalVehicles = trafficData.reduce((sum, item) => sum + item.vehicle_count, 0);
  const avgSpeed = trafficData.length > 0 
    ? Math.round((trafficData.reduce((sum, item) => sum + item.avg_speed, 0) / trafficData.length) * 10) / 10
    : 0;

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading traffic data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Traffic Dashboard</h1>
          <p className="text-gray-600">Real-time traffic monitoring for Coimbatore</p>
          <div className="flex items-center mt-2 text-sm text-gray-500">
            <Clock className="w-4 h-4 mr-1" />
            Last updated: {currentTime}
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center">
              <MapPin className="w-8 h-8 text-blue-600 mr-3" />
              <div>
                <p className="text-sm text-gray-600">Active Locations</p>
                <p className="text-2xl font-bold text-gray-900">{locations.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center">
              <TrendingUp className="w-8 h-8 text-green-600 mr-3" />
              <div>
                <p className="text-sm text-gray-600">Total Vehicles</p>
                <p className="text-2xl font-bold text-gray-900">{totalVehicles.toLocaleString()}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center">
              <Clock className="w-8 h-8 text-orange-600 mr-3" />
              <div>
                <p className="text-sm text-gray-600">Avg Speed</p>
                <p className="text-2xl font-bold text-gray-900">{avgSpeed} km/h</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center">
              <AlertTriangle className="w-8 h-8 text-red-600 mr-3" />
              <div>
                <p className="text-sm text-gray-600">High Congestion</p>
                <p className="text-2xl font-bold text-gray-900">
                  {trafficData.filter(item => item.congestion_level === 'high' || item.congestion_level === 'very_high').length}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Map and Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="lg:col-span-3 bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Coimbatore Map</h2>
            <SimpleMap height="500px" />
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Congestion Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={congestionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {congestionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Vehicle Count by Location</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={vehicleCountData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="location" angle={-45} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="vehicle_count" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Average Speed by Location</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={speedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="location" angle={-45} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="avg_speed" stroke="#10B981" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;