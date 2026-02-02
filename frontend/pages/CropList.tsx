import React, { useEffect, useState } from 'react';
import {
  Droplets,
  Thermometer,
  Beaker,
  Clock,
  Calendar,
  BookOpen,
  Search,
  Filter,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { fetchCropLibrary } from '../services/api';
import { CropProfile } from '../types';

const CropList: React.FC = () => {
  const [crops, setCrops] = useState<CropProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(9);
  const [filterSeason, setFilterSeason] = useState('All');

  useEffect(() => {
    loadCrops();
  }, []);

  const loadCrops = async () => {
    try {
      setLoading(true);
      const data = await fetchCropLibrary();
      setCrops(data);
    } catch (error) {
      console.error('Failed to load crops:', error);
    } finally {
      setLoading(false);
    }
  };

  // Filter crops
  const filteredCrops = crops.filter(crop => {
    const matchesSearch = crop.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      crop.scientificName?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSeason = filterSeason === 'All' || crop.season?.includes(filterSeason);
    return matchesSearch && matchesSeason;
  });

  // Pagination
  const totalPages = Math.ceil(filteredCrops.length / itemsPerPage);
  const paginatedCrops = filteredCrops.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold text-agri-900 flex items-center gap-2">
            <BookOpen className="text-agri-600" />
            Crop Library
          </h2>
          <p className="text-stone-500 mt-1">
            Comprehensive database of {crops.length > 0 ? crops.length : '96+'} agricultural crops from our AI models.
          </p>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="bg-white p-4 rounded-xl border border-stone-200 shadow-sm flex flex-col md:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400 h-5 w-5" />
          <input
            type="text"
            placeholder="Search crops by name..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1); // Reset to first page on search
            }}
            className="w-full pl-10 pr-4 py-2 border border-stone-200 rounded-lg focus:ring-2 focus:ring-agri-500 focus:border-agri-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="text-stone-400 h-5 w-5" />
          <select
            value={filterSeason}
            onChange={(e) => setFilterSeason(e.target.value)}
            className="border border-stone-200 rounded-lg py-2 px-3 focus:ring-2 focus:ring-agri-500 focus:border-agri-500 bg-white"
            aria-label="Filter by season"
          >
            <option value="All">All Seasons</option>
            <option value="Kharif">Kharif</option>
            <option value="Rabi">Rabi</option>
            <option value="Zaid">Zaid</option>
            <option value="General">General</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-4 border-agri-200 border-t-agri-600 rounded-full animate-spin"></div>
        </div>
      ) : filteredCrops.length === 0 ? (
        <div className="text-center py-12 text-stone-400">
          <BookOpen className="h-16 w-16 mx-auto mb-4 opacity-50" />
          <p>No crops found matching your criteria.</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {paginatedCrops.map((crop) => (
              <div key={crop.id} className="bg-white rounded-2xl border border-stone-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden group hover:border-agri-300">
                <div className="p-6">
                  <div className="flex justify-between items-start mb-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-agri-50 to-agri-100 rounded-xl flex items-center justify-center text-agri-700 font-bold text-xl shadow-inner">
                      {crop.name.charAt(0)}
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${crop.season?.includes('Kharif') ? 'bg-orange-50 text-orange-700' :
                        crop.season?.includes('Rabi') ? 'bg-sky-50 text-sky-700' :
                          'bg-green-50 text-green-700'
                      }`}>
                      {crop.season}
                    </span>
                  </div>

                  <h3 className="text-xl font-bold text-stone-800 mb-0.5">{crop.name}</h3>
                  <p className="text-stone-400 text-sm italic mb-6">{crop.scientificName || 'Scientific name N/A'}</p>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2 text-stone-600">
                        <Thermometer size={16} className="text-orange-400" />
                        <span>Temperature</span>
                      </div>
                      <span className="font-medium text-stone-800">{crop.tempRange}</span>
                    </div>

                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2 text-stone-600">
                        <Droplets size={16} className="text-blue-400" />
                        <span>Water / Humidity</span>
                      </div>
                      <div className="text-right">
                        <span className="font-medium text-stone-800 block">{crop.humidityRange}</span>
                        <span className="text-xs text-stone-400">{crop.waterReq} Req.</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2 text-stone-600">
                        <Beaker size={16} className="text-purple-400" />
                        <span>Soil pH</span>
                      </div>
                      <span className="font-medium text-stone-800">{crop.phRange}</span>
                    </div>
                  </div>
                </div>
                <div className="bg-stone-50 px-6 py-3 border-t border-stone-100 flex justify-between items-center">
                  <span className="text-xs font-semibold text-stone-500 uppercase tracking-widest">{crop.type}</span>
                  <button className="text-agri-600 text-sm font-medium hover:text-agri-700">View Details →</button>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex justify-center items-center gap-4 mt-8">
              <button
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="p-2 rounded-lg border border-stone-200 disabled:opacity-50 hover:bg-stone-50 transition-colors"
                aria-label="Previous page"
              >
                <ChevronLeft className="h-5 w-5 text-stone-600" />
              </button>
              <span className="text-stone-600 font-medium">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="p-2 rounded-lg border border-stone-200 disabled:opacity-50 hover:bg-stone-50 transition-colors"
                aria-label="Next page"
              >
                <ChevronRight className="h-5 w-5 text-stone-600" />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default CropList;