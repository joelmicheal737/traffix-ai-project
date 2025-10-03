import { Link } from 'react-router-dom';
import { BarChart3, MapPin, Upload, Brain, Users, Clock } from 'lucide-react';

const Home = () => {
  const features = [
    {
      icon: <MapPin className="w-8 h-8 text-blue-600" />,
      title: 'Real-time Traffic Monitoring',
      description: 'Monitor live traffic conditions across key locations in Coimbatore with interactive maps and real-time data visualization.',
    },
    {
      icon: <Brain className="w-8 h-8 text-green-600" />,
      title: 'AI-Powered Predictions',
      description: 'Advanced machine learning models using Prophet and LSTM to forecast traffic congestion patterns and optimize route planning.',
    },
    {
      icon: <Upload className="w-8 h-8 text-purple-600" />,
      title: 'Data Upload & Analysis',
      description: 'Upload CSV traffic data and video files for comprehensive analysis using YOLOv8 vehicle detection and counting.',
    },
    {
      icon: <BarChart3 className="w-8 h-8 text-orange-600" />,
      title: 'Advanced Analytics',
      description: 'Detailed charts and visualizations showing traffic patterns, congestion trends, and historical data analysis.',
    },
  ];

  const stats = [
    { label: 'Locations Monitored', value: '25+', icon: <MapPin className="w-6 h-6" /> },
    { label: 'Daily Predictions', value: '1000+', icon: <Brain className="w-6 h-6" /> },
    { label: 'Active Users', value: '500+', icon: <Users className="w-6 h-6" /> },
    { label: 'Uptime', value: '99.9%', icon: <Clock className="w-6 h-6" /> },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-900 via-blue-800 to-indigo-900 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex justify-center mb-8 text-6xl">
              🚦
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              Traffix AI
              <span className="block text-3xl md:text-4xl text-blue-300 mt-2">
                Coimbatore Edition
              </span>
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100 max-w-3xl mx-auto">
              Intelligent traffic management and prediction system powered by AI/ML technologies 
              for smarter urban mobility in Coimbatore.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/dashboard"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors shadow-lg"
              >
                View Dashboard
              </Link>
              <Link
                to="/prediction"
                className="bg-transparent border-2 border-white hover:bg-white hover:text-blue-900 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors"
              >
                Try AI Prediction
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="flex justify-center mb-4 text-blue-600">
                  {stat.icon}
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Powerful Features for Smart Traffic Management
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Leverage cutting-edge AI and machine learning technologies to optimize traffic flow 
              and reduce congestion in Coimbatore.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12">
            {features.map((feature, index) => (
              <div key={index} className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    {feature.icon}
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-3">
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
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-blue-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-bold mb-6">
            Ready to Optimize Coimbatore's Traffic?
          </h2>
          <p className="text-xl mb-8 text-blue-100 max-w-2xl mx-auto">
            Join the smart city revolution with AI-powered traffic management. 
            Upload your data and get instant insights.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/upload"
              className="bg-white text-blue-900 hover:bg-gray-100 px-8 py-4 rounded-lg font-semibold text-lg transition-colors"
            >
              Upload Data
            </Link>
            <Link
              to="/about"
              className="border-2 border-white hover:bg-white hover:text-blue-900 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors"
            >
              Learn More
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;