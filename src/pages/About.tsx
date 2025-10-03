import { Brain, MapPin, BarChart3, Upload, Users, Shield, Zap, Globe } from 'lucide-react';

const About = () => {
  const features = [
    {
      icon: <Brain className="w-8 h-8 text-blue-600" />,
      title: 'Advanced AI/ML Models',
      description: 'Utilizes Prophet for time series forecasting and YOLOv8 for real-time vehicle detection and counting.',
    },
    {
      icon: <MapPin className="w-8 h-8 text-green-600" />,
      title: 'Interactive Maps',
      description: 'Real-time traffic visualization on interactive maps with congestion heatmaps and location-based insights.',
    },
    {
      icon: <BarChart3 className="w-8 h-8 text-purple-600" />,
      title: 'Comprehensive Analytics',
      description: 'Detailed charts and visualizations showing traffic patterns, trends, and predictive analytics.',
    },
    {
      icon: <Upload className="w-8 h-8 text-orange-600" />,
      title: 'Data Integration',
      description: 'Easy CSV upload for traffic data and video analysis for vehicle detection and counting.',
    },
  ];

  const techStack = [
    { category: 'Frontend', technologies: ['React 18', 'TypeScript', 'Tailwind CSS', 'Vite', 'Recharts', 'Leaflet'] },
    { category: 'Backend', technologies: ['FastAPI', 'Python', 'SQLite', 'Prophet', 'YOLOv8', 'OpenCV'] },
    { category: 'AI/ML', technologies: ['Facebook Prophet', 'Ultralytics YOLOv8', 'Pandas', 'NumPy', 'Scikit-learn'] },
    { category: 'Infrastructure', technologies: ['RESTful APIs', 'File Upload', 'Real-time Processing', 'Database Storage'] },
  ];

  const benefits = [
    {
      icon: <Zap className="w-6 h-6 text-yellow-600" />,
      title: 'Real-time Insights',
      description: 'Get instant traffic updates and congestion alerts for better route planning.',
    },
    {
      icon: <Shield className="w-6 h-6 text-green-600" />,
      title: 'Reliable Predictions',
      description: 'AI-powered forecasting with confidence intervals for accurate traffic predictions.',
    },
    {
      icon: <Users className="w-6 h-6 text-blue-600" />,
      title: 'User-Friendly Interface',
      description: 'Intuitive dashboard and controls designed for both technical and non-technical users.',
    },
    {
      icon: <Globe className="w-6 h-6 text-purple-600" />,
      title: 'Scalable Solution',
      description: 'Designed to handle multiple locations and large datasets with efficient processing.',
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="flex justify-center mb-8 text-5xl">
            🚦
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">About Traffix AI</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            A comprehensive traffic management and prediction system designed specifically for Coimbatore, 
            leveraging advanced AI and machine learning technologies to optimize urban mobility.
          </p>
        </div>

        {/* Mission Statement */}
        <div className="bg-white p-8 rounded-lg shadow-md mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4 text-center">Our Mission</h2>
          <p className="text-lg text-gray-700 text-center max-w-4xl mx-auto leading-relaxed">
            To revolutionize traffic management in Coimbatore by providing intelligent, data-driven insights 
            that help reduce congestion, improve commute times, and create a more efficient transportation 
            ecosystem for the city's residents and visitors.
          </p>
        </div>

        {/* Key Features */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Key Features</h2>
          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    {feature.icon}
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Technology Stack</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {techStack.map((stack, index) => (
              <div key={index} className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">
                  {stack.category}
                </h3>
                <ul className="space-y-2">
                  {stack.technologies.map((tech, techIndex) => (
                    <li key={techIndex} className="text-sm text-gray-600 text-center">
                      {tech}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Benefits */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Why Choose Traffix AI?</h2>
          <div className="grid md:grid-cols-2 gap-8">
            {benefits.map((benefit, index) => (
              <div key={index} className="flex items-start space-x-4">
                <div className="flex-shrink-0 p-2 bg-gray-100 rounded-lg">
                  {benefit.icon}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {benefit.title}
                  </h3>
                  <p className="text-gray-600">
                    {benefit.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Coimbatore Focus */}
        <div className="bg-blue-50 p-8 rounded-lg shadow-md mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4 text-center">Coimbatore Edition</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Key Locations Covered</h3>
              <ul className="space-y-2 text-gray-700">
                <li>• Gandhipuram - Major commercial hub</li>
                <li>• RS Puram - Residential and commercial area</li>
                <li>• Peelamedu - IT corridor and residential</li>
                <li>• Saibaba Colony - Educational institutions area</li>
                <li>• Singanallur - Industrial and residential</li>
                <li>• Tidel Park - IT and business district</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Local Insights</h3>
              <p className="text-gray-700 leading-relaxed">
                Our system is specifically calibrated for Coimbatore's unique traffic patterns, 
                considering local factors such as peak hours, weather conditions, festival seasons, 
                and the city's growing IT sector impact on traffic flow.
              </p>
            </div>
          </div>
        </div>

        {/* How It Works */}
        <div className="bg-white p-8 rounded-lg shadow-md mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Upload className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">1. Data Collection</h3>
              <p className="text-gray-600">
                Upload traffic data via CSV files or analyze video footage using our AI-powered detection system.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">2. AI Processing</h3>
              <p className="text-gray-600">
                Our machine learning models analyze patterns and generate accurate predictions for traffic congestion.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <BarChart3 className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">3. Insights & Visualization</h3>
              <p className="text-gray-600">
                View real-time dashboards, interactive maps, and detailed analytics to make informed decisions.
              </p>
            </div>
          </div>
        </div>

        {/* Contact/Future Plans */}
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Future Enhancements</h2>
          <p className="text-gray-600 max-w-3xl mx-auto mb-8">
            We're continuously working to improve Traffix AI with features like real-time API integrations, 
            mobile applications, advanced route optimization, and integration with smart city infrastructure.
          </p>
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-lg">
            <h3 className="text-xl font-semibold mb-2">Ready to Get Started?</h3>
            <p className="mb-4">Experience the power of AI-driven traffic management for Coimbatore.</p>
            <div className="space-x-4">
              <a 
                href="/dashboard" 
                className="bg-white text-blue-600 px-6 py-2 rounded-lg font-medium hover:bg-gray-100 transition-colors"
              >
                View Dashboard
              </a>
              <a 
                href="/upload" 
                className="border-2 border-white text-white px-6 py-2 rounded-lg font-medium hover:bg-white hover:text-blue-600 transition-colors"
              >
                Upload Data
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;