import { useState } from 'react'
import { Upload, Leaf } from 'lucide-react'
import apiService from '../services/api'

const DiseaseManagement = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string>('')
  const [results, setResults] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile || !preview) return

    setIsLoading(true)
    try {
      const response = await apiService.detectDisease({
        image_data: preview.split(',')[1],
        crop_type: 'unknown',
      })
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
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Disease Detection</h1>
        <p className="text-gray-600">Upload a plant image for AI-powered disease identification</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          {preview ? (
            <div className="space-y-4">
              <img src={preview} alt="Preview" className="max-h-96 mx-auto rounded-lg" />
              <button
                onClick={() => {
                  setPreview('')
                  setSelectedFile(null)
                  setResults(null)
                }}
                className="text-red-600 hover:text-red-700 font-medium"
              >
                Remove Image
              </button>
            </div>
          ) : (
            <>
              <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-4">Click to upload or drag and drop</p>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="inline-block bg-green-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-green-700 transition-colors"
              >
                Select Image
              </label>
            </>
          )}
        </div>

        {preview && (
          <button
            onClick={handleAnalyze}
            disabled={isLoading}
            className="w-full mt-4 bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors disabled:bg-gray-400"
          >
            {isLoading ? 'Analyzing...' : 'Analyze Image'}
          </button>
        )}
      </div>

      {results && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Detection Results</h2>
          <div className="space-y-4">
            {results.detections?.map((detection: any, index: number) => (
              <div key={index} className="p-4 bg-red-50 rounded-lg">
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-red-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <Leaf className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-800">{detection.disease}</h3>
                    <p className="text-sm text-gray-600 mb-2">
                      Confidence: {(detection.confidence * 100).toFixed(1)}% | Severity: {detection.severity}
                    </p>
                    <div className="mt-3 space-y-2">
                      <div>
                        <h4 className="font-semibold text-gray-700">Treatment:</h4>
                        <p className="text-gray-600">{detection.treatment}</p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-700">Prevention:</h4>
                        <p className="text-gray-600">{detection.prevention}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default DiseaseManagement
