import { Link } from 'react-router-dom'
import { LayoutDashboard, MessageSquare, Leaf, Bug, Sprout, Droplets, TrendingUp, Settings } from 'lucide-react'

const Home = () => {
  const features = [
    {
      icon: LayoutDashboard,
      title: 'Real-time Dashboard',
      description: 'Monitor your farm with live IoT sensor data and 3D visualizations',
      path: '/dashboard',
      color: 'bg-blue-100 text-blue-600',
    },
    {
      icon: MessageSquare,
      title: 'AI Assistant',
      description: 'Chat with our multilingual AI bot for instant farming advice',
      path: '/chatbot',
      color: 'bg-purple-100 text-purple-600',
    },
    {
      icon: Sprout,
      title: 'Crop Recommendations',
      description: 'Get ML-powered crop suggestions based on your soil and climate',
      path: '/crops',
      color: 'bg-green-100 text-green-600',
    },
    {
      icon: Leaf,
      title: 'Disease Detection',
      description: 'Identify plant diseases using computer vision and get treatments',
      path: '/disease',
      color: 'bg-red-100 text-red-600',
    },
    {
      icon: Bug,
      title: 'Weed Management',
      description: 'Detect and manage weeds with AI-powered image analysis',
      path: '/weeds',
      color: 'bg-orange-100 text-orange-600',
    },
    {
      icon: Droplets,
      title: 'Smart Irrigation',
      description: 'Optimize water usage with ET0-based recommendations',
      path: '/irrigation',
      color: 'bg-cyan-100 text-cyan-600',
    },
    {
      icon: TrendingUp,
      title: 'Yield Prediction',
      description: 'Forecast harvest yields using machine learning models',
      path: '/crops',
      color: 'bg-yellow-100 text-yellow-600',
    },
    {
      icon: Settings,
      title: 'Admin Dashboard',
      description: 'Real-time system monitoring and management console',
      path: '/admin',
      color: 'bg-gray-100 text-gray-600',
    },
  ]

  return (
    <div className="space-y-12 animate-fade-in">
      {/* Hero Section */}
      <section className="text-center py-12 px-4">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl font-bold text-green-800 mb-4">
            🌾 Welcome to AgriSense
          </h1>
          <p className="text-xl text-gray-600 mb-2">
            Smart Agriculture IoT Platform with 18+ ML Models
          </p>
          <p className="text-lg text-gray-500 mb-8">
            Revolutionizing farming with AI, IoT sensors, and real-time analytics
          </p>
          
          <div className="flex flex-wrap gap-4 justify-center">
            <Link
              to="/dashboard"
              className="bg-green-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors shadow-lg"
            >
              Open Dashboard
            </Link>
            <Link
              to="/chatbot"
              className="bg-white text-green-600 border-2 border-green-600 px-8 py-3 rounded-lg font-semibold hover:bg-green-50 transition-colors"
            >
              Talk to AI Assistant
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto">
        <div className="bg-white p-6 rounded-lg shadow text-center">
          <div className="text-3xl font-bold text-green-600">18+</div>
          <div className="text-gray-600 text-sm">ML Models</div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow text-center">
          <div className="text-3xl font-bold text-blue-600">Real-time</div>
          <div className="text-gray-600 text-sm">IoT Monitoring</div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow text-center">
          <div className="text-3xl font-bold text-purple-600">5</div>
          <div className="text-gray-600 text-sm">Languages</div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow text-center">
          <div className="text-3xl font-bold text-orange-600">24/7</div>
          <div className="text-gray-600 text-sm">Support</div>
        </div>
      </section>

      {/* Features Grid */}
      <section>
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-8">
          Platform Features
        </h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <Link
                key={index}
                to={feature.path}
                className="bg-white p-6 rounded-lg shadow hover:shadow-lg transition-all transform hover:-translate-y-1 group"
              >
                <div className={`w-12 h-12 ${feature.color} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 text-sm">
                  {feature.description}
                </p>
              </Link>
            )
          })}
        </div>
      </section>

      {/* Technology Stack */}
      <section className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-8">
          Technology Stack
        </h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <h3 className="text-xl font-semibold text-green-600 mb-3">Backend</h3>
            <ul className="space-y-2 text-gray-700">
              <li>• Python 3.12 + FastAPI</li>
              <li>• TensorFlow & PyTorch</li>
              <li>• scikit-learn</li>
              <li>• SQLite + MongoDB</li>
              <li>• Redis + Celery</li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-xl font-semibold text-blue-600 mb-3">Frontend</h3>
            <ul className="space-y-2 text-gray-700">
              <li>• React 18 + TypeScript</li>
              <li>• Vite + TailwindCSS</li>
              <li>• Three.js for 3D</li>
              <li>• React Query</li>
              <li>• i18next (5 languages)</li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-xl font-semibold text-purple-600 mb-3">IoT</h3>
            <ul className="space-y-2 text-gray-700">
              <li>• ESP32 + Arduino Nano</li>
              <li>• DHT22, DS18B20 sensors</li>
              <li>• MQTT protocol</li>
              <li>• WebSocket streaming</li>
              <li>• Real-time data sync</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="text-center py-8 text-gray-600">
        <p className="mb-2">
          Built with ❤️ for modern agriculture
        </p>
        <p className="text-sm">
          Version 2.0.0 | Last Updated: January 27, 2026
        </p>
      </footer>
    </div>
  )
}

export default Home
