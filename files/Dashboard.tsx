import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Thermometer, Droplets, Activity, Zap } from 'lucide-react'
import apiService from '../services/api'

const Dashboard = () => {
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Fetch sensor data
  const { data: sensorData, refetch } = useQuery({
    queryKey: ['sensorLive'],
    queryFn: apiService.getSensorLive,
    refetchInterval: autoRefresh ? 3000 : false,
  })

  // Fetch device status
  const { data: deviceStatus } = useQuery({
    queryKey: ['deviceStatus'],
    queryFn: apiService.getDeviceStatus,
    refetchInterval: 10000,
  })

  const sensors = sensorData?.data || {}

  const sensorCards = [
    {
      label: 'Temperature',
      value: sensors.temperature || 0,
      unit: '°C',
      icon: Thermometer,
      color: 'text-red-600',
      bgColor: 'bg-red-100',
      status: sensors.temperature > 35 ? 'High' : sensors.temperature < 15 ? 'Low' : 'Normal',
    },
    {
      label: 'Humidity',
      value: sensors.humidity || 0,
      unit: '%',
      icon: Droplets,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
      status: sensors.humidity > 80 ? 'High' : sensors.humidity < 40 ? 'Low' : 'Optimal',
    },
    {
      label: 'Soil Moisture',
      value: sensors.soil_moisture || 0,
      unit: '%',
      icon: Droplets,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      status: sensors.soil_moisture > 70 ? 'Wet' : sensors.soil_moisture < 30 ? 'Dry' : 'Good',
    },
    {
      label: 'pH Level',
      value: sensors.ph || 0,
      unit: 'pH',
      icon: Activity,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
      status: sensors.ph > 7.5 ? 'Alkaline' : sensors.ph < 5.5 ? 'Acidic' : 'Neutral',
    },
    {
      label: 'Nitrogen',
      value: sensors.nitrogen || 0,
      unit: 'mg/kg',
      icon: Zap,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-100',
      status: 'Normal',
    },
    {
      label: 'Phosphorus',
      value: sensors.phosphorus || 0,
      unit: 'mg/kg',
      icon: Zap,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100',
      status: 'Normal',
    },
    {
      label: 'Potassium',
      value: sensors.potassium || 0,
      unit: 'mg/kg',
      icon: Zap,
      color: 'text-pink-600',
      bgColor: 'bg-pink-100',
      status: 'Normal',
    },
    {
      label: 'Light',
      value: sensors.light_intensity || 0,
      unit: 'lux',
      icon: Activity,
      color: 'text-amber-600',
      bgColor: 'bg-amber-100',
      status: sensors.light_intensity > 500 ? 'Bright' : 'Low',
    },
  ]

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Dashboard</h1>
          <p className="text-gray-600">Real-time sensor monitoring</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              autoRefresh
                ? 'bg-green-600 text-white'
                : 'bg-gray-200 text-gray-700'
            }`}
          >
            {autoRefresh ? '⚡ Live' : '⏸️ Paused'}
          </button>
          
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Status Banner */}
      <div className="bg-white rounded-lg shadow p-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
          <div>
            <div className="font-semibold text-gray-800">System Operational</div>
            <div className="text-sm text-gray-600">
              {deviceStatus?.total_devices || 0} devices connected
            </div>
          </div>
        </div>
        
        <div className="text-sm text-gray-600">
          Data source: {sensors.source || 'unknown'} • 
          Last update: {new Date().toLocaleTimeString()}
        </div>
      </div>

      {/* Sensor Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {sensorCards.map((sensor, index) => {
          const Icon = sensor.icon
          return (
            <div
              key={index}
              className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`w-12 h-12 ${sensor.bgColor} rounded-lg flex items-center justify-center`}>
                  <Icon className={`w-6 h-6 ${sensor.color}`} />
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  sensor.status === 'Normal' || sensor.status === 'Good' || sensor.status === 'Optimal' || sensor.status === 'Neutral'
                    ? 'bg-green-100 text-green-700'
                    : 'bg-yellow-100 text-yellow-700'
                }`}>
                  {sensor.status}
                </span>
              </div>
              
              <div className="text-gray-600 text-sm mb-1">{sensor.label}</div>
              <div className="text-3xl font-bold text-gray-800">
                {typeof sensor.value === 'number' ? sensor.value.toFixed(1) : sensor.value}
                <span className="text-lg text-gray-500 ml-1">{sensor.unit}</span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg text-blue-700 font-medium transition-colors">
            📊 View History
          </button>
          <button className="p-4 bg-green-50 hover:bg-green-100 rounded-lg text-green-700 font-medium transition-colors">
            🌱 Get Recommendations
          </button>
          <button className="p-4 bg-purple-50 hover:bg-purple-100 rounded-lg text-purple-700 font-medium transition-colors">
            💬 Ask AI
          </button>
          <button className="p-4 bg-orange-50 hover:bg-orange-100 rounded-lg text-orange-700 font-medium transition-colors">
            ⚙️ Configure Sensors
          </button>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
