import React, { useState, useEffect } from 'react';
import { fetchCropLibrary } from '../services/api';
import { CropProfile } from '../types';
import { Search, Sprout, Info, Droplets, Sun, BookOpen } from 'lucide-react';

const CropLibrary: React.FC = () => {
  const [crops, setCrops] = useState<CropProfile[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(true);
      fetchCropLibrary(searchQuery).then(data => { setCrops(data); setLoading(false); }).catch(() => setLoading(false));
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-center bg-white p-6 rounded-2xl shadow-sm border border-agri-100">
        <div className="mb-4 md:mb-0">
          <h1 className="text-2xl font-bold text-agri-900 flex items-center"><BookOpen className="w-6 h-6 mr-2 text-agri-600" />Crop Library</h1>
          <p className="text-gray-500 text-sm mt-1">Encyclopedia of Indian crops with cultivation details.</p>
        </div>
        <div className="relative w-full md:w-96">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><Search className="h-5 w-5 text-gray-400" /></div>
          <input type="text" className="block w-full pl-10 pr-3 py-3 border border-gray-200 rounded-full bg-gray-50 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-agri-400 sm:text-sm" placeholder="Search by name (e.g., Rice, Mango)..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
        </div>
      </div>
      {loading ? (
        <div className="flex justify-center items-center h-64"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-agri-600"></div></div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {crops.length > 0 ? crops.map((crop) => (
            <div key={crop.id} className="bg-white rounded-xl shadow-sm border border-agri-100 overflow-hidden hover:shadow-md transition-shadow flex flex-col">
              <div className="bg-agri-50 p-4 border-b border-agri-100 flex items-center justify-between">
                <div className="flex items-center">
                  <div className="p-2 bg-white rounded-full shadow-sm mr-3"><Sprout className="w-5 h-5 text-agri-600" /></div>
                  <div>
                    <h3 className="font-bold text-gray-800 text-lg leading-tight">{crop.name}</h3>
                    <p className="text-xs text-gray-500 italic">{crop.scientificName}</p>
                  </div>
                </div>
                <span className="text-xs font-semibold px-2 py-1 bg-white text-agri-700 rounded-full border border-agri-100">{crop.type || 'General'}</span>
              </div>
              <div className="p-5 flex-1 flex flex-col space-y-3">
                <p className="text-sm text-gray-600 line-clamp-3 leading-relaxed flex-1">
                  Duration: {crop.duration}. Requires {crop.waterReq} water.
                </p>
                <div className="grid grid-cols-2 gap-2 mt-4 pt-4 border-t border-gray-50">
                  <div className="flex items-center text-xs text-gray-500"><Sun className="w-3 h-3 mr-1 text-orange-400" /><span>{crop.season}</span></div>
                  <div className="flex items-center text-xs text-gray-500"><Droplets className="w-3 h-3 mr-1 text-blue-400" /><span className="truncate" title={crop.phRange}>{crop.phRange}</span></div>
                </div>
              </div>
              <div className="bg-gray-50 px-5 py-3 border-t border-gray-100">
                <button className="w-full text-center text-sm font-medium text-agri-600 hover:text-agri-800 flex items-center justify-center"><Info className="w-4 h-4 mr-1" />View Cultivation Guide</button>
              </div>
            </div>
          )) : (
            <div className="col-span-full text-center py-12 text-gray-500">
              <Sprout className="w-12 h-12 mx-auto mb-3 opacity-20" />
              <p className="text-lg">No crops found matching &quot;{searchQuery}&quot;</p>
              <button onClick={() => setSearchQuery('')} className="mt-2 text-agri-600 hover:underline">Clear Search</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CropLibrary;
