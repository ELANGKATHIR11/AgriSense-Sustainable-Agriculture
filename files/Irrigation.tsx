import { useState } from 'react'
import { Droplets } from 'lucide-react'
import apiService from '../services/api'

const Irrigation = () => {
  const [formData, setFormData] = useState({
    crop: 'Rice',
    growth_stage: 'mid',
    temp_min: 20,
    temp_max: 30,
    soil_type: 'loam',
  })
  const [results, setResults] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    
    try {
      const response = await apiService.optimizeWater(formData)
      setResults(response)
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Smart Irrigation</h1>
        <p className="text-gray-600">Calculate optimal water requirements using ET0 method</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Crop</label>
              <select
                value={formData.crop}
                onChange={(e) => setFormData({ ...formData, crop: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              >
                <option>Rice</option>
                <option>Wheat</option>
                <option>Maize</option>
                <option>Cotton</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Growth Stage</label>
              <select
                value={formData.growth_stage}
                onChange={(e) => setFormData({ ...formData, growth_stage: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              >
                <option value="initial">Initial</option>
                <option value="mid">Mid-Season</option>
                <option value="late">Late-Season</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Min Temperature (°C)</label>
              <input
                type="number"
                value={formData.temp_min}
                onChange={(e) => setFormData({ ...formData, temp_min: Number(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Max Temperature (°C)</label>
              <input
                type="number"
                value={formData.temp_max}
                onChange={(e) => setFormData({ ...formData, temp_max: Number(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:bg-gray-400"
          >
            {isLoading ? 'Calculating...' : 'Calculate Water Needs'}
          </button>
        </form>
      </div>

      {results && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Water Requirements</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-4 bg-blue-50 rounded-lg">
              <Droplets className="w-8 h-8 text-blue-600 mb-2" />
              <h3 className="font-semibold text-gray-800">Daily Requirement</h3>
              <p className="text-3xl font-bold text-blue-600">
                {results.water_requirement?.irrigation_mm_day} mm/day
              </p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg">
              <h3 className="font-semibold text-gray-800">Total Volume</h3>
              <p className="text-3xl font-bold text-green-600">
                {results.water_requirement?.irrigation_liters_ha?.toLocaleString()} L/ha
              </p>
            </div>
          </div>
          <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-2">Recommendation</h3>
            <p className="text-gray-700">{results.water_requirement?.recommendation}</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default Irrigation
