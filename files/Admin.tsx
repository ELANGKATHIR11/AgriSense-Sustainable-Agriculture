import { useState, useEffect } from 'react'
import { Activity, Database, Zap, AlertTriangle } from 'lucide-react'
import apiService from '../services/api'

const Admin = () => {
  const [metrics, setMetrics] = useState<any>(null)
  const [summary, setSummary] = useState<any>(null)
  const [activities, setActivities] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const fetchData = async () => {
    try {
      const [metricsData, summaryData, activitiesData] = await Promise.all([
        apiService.getDashboardSummary(),
        apiService.getAdminSummary(),
        apiService.getActivities(),
      ])
      setMetrics(metricsData.metrics)
      setSummary(summaryData.summary)
      setActivities(activitiesData.activities || [])
    } catch (error) {
      console.error('Error fetching admin data:', error)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 3000)
    return () => clearInterval(interval)
  }, [])

  const handleAction = async (action: string) => {
    setIsLoading(true)
    try {
      await apiService.performAdminAction({ action, parameters: {} })
      await fetchData()
    } catch (error) {
      console.error('Error performing action:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = async () => {
    if (confirm('Are you sure you want to reset all data? This cannot be undone.')) {
      setIsLoading(true)
      try {
        await apiService.resetSystem()
        await fetchData()
      } catch (error) {
        console.error('Error resetting system:', error)
      } finally {
        setIsLoading(false)
      }
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Admin Dashboard</h1>
          <p className="text-gray-600">Real-time system monitoring & control</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
          <span className="text-sm text-gray-600">Live (3s refresh)</span>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <button
            onClick={() => handleAction('reload_models')}
            disabled={isLoading}
            className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg text-blue-700 font-medium transition-colors disabled:opacity-50"
          >
            🔄 Reload Models
          </button>
          <button
            onClick={() => handleAction('reload_dataset')}
            disabled={isLoading}
            className="p-4 bg-green-50 hover:bg-green-100 rounded-lg text-green-700 font-medium transition-colors disabled:opacity-50"
          >
            📊 Reload Dataset
          </button>
          <button
            onClick={() => handleAction('refresh_data')}
            disabled={isLoading}
            className="p-4 bg-purple-50 hover:bg-purple-100 rounded-lg text-purple-700 font-medium transition-colors disabled:opacity-50"
          >
            ⚡ Refresh Data
          </button>
          <button
            onClick={() => handleAction('sync_weather')}
            disabled={isLoading}
            className="p-4 bg-yellow-50 hover:bg-yellow-100 rounded-lg text-yellow-700 font-medium transition-colors disabled:opacity-50"
          >
            🌤️ Sync Weather
          </button>
          <button
            onClick={handleReset}
            disabled={isLoading}
            className="p-4 bg-red-50 hover:bg-red-100 rounded-lg text-red-700 font-medium transition-colors disabled:opacity-50"
          >
            🗑️ Erase All
          </button>
        </div>
      </div>

      {/* Live Metrics */}
      <div className="grid md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-8 h-8 text-green-600" />
            <span className="text-xs text-gray-500">Live</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">{metrics?.soil_moisture || 0}%</div>
          <div className="text-sm text-gray-600">Soil Moisture</div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-2">
            <Zap className="w-8 h-8 text-red-600" />
            <span className="text-xs text-gray-500">Live</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">{metrics?.temperature || 0}°C</div>
          <div className="text-sm text-gray-600">Temperature</div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-2">
            <Database className="w-8 h-8 text-blue-600" />
            <span className="text-xs text-gray-500">Live</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">{metrics?.humidity || 0}%</div>
          <div className="text-sm text-gray-600">Humidity</div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-8 h-8 text-purple-600" />
            <span className="text-xs text-gray-500">Live</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">{metrics?.ph_level || 0}</div>
          <div className="text-sm text-gray-600">pH Level</div>
        </div>
      </div>

      {/* Activity Log */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Recent Activity</h2>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {activities.length > 0 ? (
            activities.map((activity: any) => (
              <div key={activity.id} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  activity.type === 'success' ? 'bg-green-500' :
                  activity.type === 'warning' ? 'bg-yellow-500' :
                  activity.type === 'error' ? 'bg-red-500' : 'bg-blue-500'
                }`} />
                <div className="flex-1">
                  <p className="text-gray-800">{activity.message}</p>
                  <p className="text-xs text-gray-500">{new Date(activity.timestamp).toLocaleString()}</p>
                </div>
              </div>
            ))
          ) : (
            <p className="text-gray-500 text-center py-8">No recent activity</p>
          )}
        </div>
      </div>
    </div>
  )
}

export default Admin
