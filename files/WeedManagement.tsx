import { useState } from 'react'
import { Bug } from 'lucide-react'

const WeedManagement = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Weed Management</h1>
        <p className="text-gray-600">AI-powered weed detection and control recommendations</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center py-12">
          <Bug className="w-24 h-24 text-orange-600 mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-gray-800 mb-2">Feature Coming Soon</h2>
          <p className="text-gray-600">Upload field images for automated weed identification</p>
        </div>
      </div>
    </div>
  )
}

export default WeedManagement
