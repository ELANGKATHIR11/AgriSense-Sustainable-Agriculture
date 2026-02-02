import React, { useState } from 'react';
import { ALL_CROPS } from '../constants/crops';
import { Sprout, CheckCircle2, TrendingUp, Droplets, Calendar, FileText } from 'lucide-react';
import { toast } from 'react-toastify';
import { MLService } from '../services/api';
import { RecommendationResult } from '../types';

// Reusable Input Field Component
const InputField = ({ label, name, value, onChange, unit, min, max, step = "1" }: any) => (
  <div className="space-y-1">
    <label className="text-xs font-semibold text-stone-500 dark:text-gray-400 uppercase tracking-wide">{label}</label>
    <div className="relative">
      <input
        type="number"
        name={name}
        value={value === 0 ? 0 : value || ''}
        onChange={onChange}
        min={min}
        max={max}
        step={step}
        className="block w-full rounded-lg border-stone-200 bg-stone-50 p-2.5 text-sm text-stone-900 focus:border-agri-500 focus:ring-agri-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:placeholder-gray-400"
        placeholder="0"
      />
      {unit && <span className="absolute right-3 top-2.5 text-stone-400 dark:text-gray-500 text-xs font-medium">{unit}</span>}
    </div>
  </div>
);

const Crops: React.FC = () => {
  const [activeTab, setActiveTab] = useState('recommendation');
  const [loading, setLoading] = useState(false);

  // Crop Recommendation State
  const [cropRecData, setCropRecData] = useState({
    nitrogen: 50,
    phosphorus: 50,
    potassium: 50,
    temperature: 26,
    humidity: 80,
    ph: 7,
    rainfall: 200
  });
  const [cropRecResult, setCropRecResult] = useState<RecommendationResult | null>(null);

  // Yield Prediction State
  const [yieldData, setYieldData] = useState({
    crop: 'Rice',
    area: 1,
    N: 60,
    P: 40,
    K: 40,
    temperature: 25,
    humidity: 60,
    rainfall: 150,
    water_requirement: 6,
    growth_duration: 120
  });
  const [yieldResult, setYieldResult] = useState<any>(null);

  // Water Requirement State
  const [waterData, setWaterData] = useState({
    temperature: 30,
    humidity: 70,
    rainfall: 100,
    growth_duration: 120,
    crop: 'Rice', // Optional context
    soilType: 'Clay', // Optional context
    season: 'Summer' // Optional context
  });
  const [waterResult, setWaterResult] = useState<any>(null);

  // Season Classification State
  const [seasonData, setSeasonData] = useState({
    temperature: 28,
    rainfall: 200,
    humidity: 75,
    growth_duration: 120
  });
  const [seasonResult, setSeasonResult] = useState<any>(null);

  // Crop Type Classification State
  const [cropTypeData, setCropTypeData] = useState({
    nitrogen: 80,
    phosphorus: 40,
    potassium: 40,
    temperature: 25,
    humidity: 70,
    ph: 6.5,
    rainfall: 150
  });
  const [cropTypeResult, setCropTypeResult] = useState<any>(null);

  // Handlers
  const handleCropRecommendation = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await MLService.recommendCrop(cropRecData);
      setCropRecData(prev => ({ ...prev })); // Force refresh if needed
      setCropRecResult(res);
      toast.success(`Recommended crop: ${res.crop}`);
    } catch (error: any) {
      console.error('Crop recommendation error:', error);
      toast.error(error.message || 'Failed to get recommendation');
    } finally {
      setLoading(false);
    }
  };

  const handleYieldPrediction = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      console.log('Yield Payload:', yieldData);
      const res = await MLService.predictYield(yieldData);
      console.log('Yield API Response:', res);
      setYieldResult(res);
      toast.success('Yield prediction completed');
    } catch (error: any) {
      console.error('Yield prediction error:', error);
      toast.error(error.message || 'Failed to predict yield');
    } finally {
      setLoading(false);
    }
  };

  const handleWaterRequirement = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await MLService.predictWaterRequirement(waterData);
      setWaterResult(res);
      toast.success('Water requirement calculated');
    } catch (error: any) {
      console.error('Water requirement error:', error);
      toast.error(error.message || 'Failed to calculate water requirement');
    } finally {
      setLoading(false);
    }
  };

  const handleSeasonClassification = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await MLService.classifySeason(seasonData);
      setSeasonResult(res);
      toast.success('Season classification completed');
    } catch (error: any) {
      console.error('Season classification error:', error);
      toast.error(error.message || 'Failed to classify season');
    } finally {
      setLoading(false);
    }
  };

  const handleCropType = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await MLService.recommendCrop(cropTypeData); // Using same endpoint
      setCropTypeResult(res);
      toast.success('Crop type identified');
    } catch (error: any) {
      console.error('Crop type error:', error);
      toast.error(error.message || 'Failed to identify crop type');
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'recommendation', label: 'Crop Recommendation', icon: Sprout },
    { id: 'yield', label: 'Yield Prediction', icon: TrendingUp },
    { id: 'water', label: 'Water Requirement', icon: Droplets },
    { id: 'season', label: 'Season Classification', icon: Calendar },
    { id: 'croptype', label: 'Crop Type', icon: FileText },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold text-agri-900 dark:text-white">Crop Guide - ML Models</h2>
        <p className="text-stone-500 dark:text-gray-400 mt-1">Access all AI-powered agricultural analysis tools</p>
      </div>

      {/* Tabs */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl border border-stone-200 dark:border-gray-700 overflow-hidden transition-colors">
        <div className="flex overflow-x-auto border-b border-stone-200 dark:border-gray-700">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors whitespace-nowrap ${activeTab === tab.id
                  ? 'bg-agri-50 dark:bg-agri-900/20 text-agri-700 dark:text-agri-400 border-b-2 border-agri-600'
                  : 'text-stone-600 dark:text-gray-400 hover:bg-stone-50 dark:hover:bg-gray-700'
                  }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="p-8">
          {/* Crop Recommendation Tab */}
          {activeTab === 'recommendation' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                <h3 className="text-xl font-bold text-agri-900 dark:text-white mb-4">Soil & Environment Parameters</h3>
                <form onSubmit={handleCropRecommendation} className="space-y-6">
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    <InputField label="Nitrogen (N)" name="nitrogen" value={cropRecData.nitrogen} onChange={(e: any) => setCropRecData({ ...cropRecData, nitrogen: parseFloat(e.target.value) })} unit="mg/kg" min="0" max="200" />
                    <InputField label="Phosphorus (P)" name="phosphorus" value={cropRecData.phosphorus} onChange={(e: any) => setCropRecData({ ...cropRecData, phosphorus: parseFloat(e.target.value) })} unit="mg/kg" min="0" max="200" />
                    <InputField label="Potassium (K)" name="potassium" value={cropRecData.potassium} onChange={(e: any) => setCropRecData({ ...cropRecData, potassium: parseFloat(e.target.value) })} unit="mg/kg" min="0" max="200" />
                    <InputField label="Temperature" name="temperature" value={cropRecData.temperature} onChange={(e: any) => setCropRecData({ ...cropRecData, temperature: parseFloat(e.target.value) })} unit="°C" min="-10" max="50" />
                    <InputField label="Humidity" name="humidity" value={cropRecData.humidity} onChange={(e: any) => setCropRecData({ ...cropRecData, humidity: parseFloat(e.target.value) })} unit="%" min="0" max="100" />
                    <InputField label="pH Level" name="ph" value={cropRecData.ph} onChange={(e: any) => setCropRecData({ ...cropRecData, ph: parseFloat(e.target.value) })} unit="pH" min="0" max="14" step="0.1" />
                    <InputField label="Rainfall" name="rainfall" value={cropRecData.rainfall} onChange={(e: any) => setCropRecData({ ...cropRecData, rainfall: parseFloat(e.target.value) })} unit="mm" min="0" max="500" />
                  </div>
                  <button type="submit" disabled={loading} className="w-full bg-agri-600 hover:bg-agri-700 text-white px-8 py-3 rounded-xl font-medium transition-colors shadow-lg shadow-agri-600/20 disabled:opacity-70 flex items-center justify-center gap-2">
                    {loading ? <span className="animate-spin text-xl">⟳</span> : <Sprout size={20} />}
                    Analyze & Recommend
                  </button>
                </form>
              </div>

              <div className="lg:col-span-1">
                {cropRecResult ? (
                  <div className="bg-gradient-to-br from-emerald-100 to-emerald-300 text-emerald-900 p-8 rounded-3xl shadow-xl sticky top-8">
                      <div className="flex items-start justify-between mb-8">
                        <div className="bg-white/70 p-3 rounded-2xl backdrop-blur-sm">
                          <CheckCircle2 size={32} className="text-emerald-800" />
                        </div>
                        <span className="bg-white/90 px-3 py-1 rounded-full text-xs font-medium backdrop-blur-sm border border-emerald-200 text-emerald-900">
                          {Math.round(cropRecResult.confidence * 100)}% Confidence
                        </span>
                      </div>
                      <h3 className="text-emerald-700 font-medium mb-1">Recommended Crop</h3>
                      <h2 className="text-4xl font-bold mb-6 tracking-tight text-emerald-900">{cropRecResult.crop}</h2>
                      <p className="text-emerald-800 text-sm leading-relaxed opacity-95 border-t border-emerald-200 pt-6">{cropRecResult.details}</p>
                    </div>
                ) : (
                  <div className="bg-stone-100 border-2 border-dashed border-stone-200 rounded-3xl p-8 h-full flex flex-col items-center justify-center text-center text-stone-400">
                    <Sprout size={48} className="mb-4 opacity-50" />
                    <p className="font-medium">Results will appear here</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Yield Prediction Tab */}
          {activeTab === 'yield' && (
            <div className="max-w-4xl">
              <h3 className="text-xl font-bold text-agri-900 dark:text-white mb-4">Crop Yield Prediction</h3>
              <form onSubmit={handleYieldPrediction} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {/* Crop Selection */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-stone-500 dark:text-gray-400 uppercase tracking-wide">Crop</label>
                    <select
                      aria-label="Select Crop"
                      title="Select Crop"
                      value={yieldData.crop}
                      onChange={(e) => setYieldData({ ...yieldData, crop: e.target.value })}
                      className="block w-full rounded-lg border-stone-200 bg-stone-50 p-2.5 text-sm text-stone-900 focus:border-agri-500 focus:ring-agri-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    >
                      <option value="">Select Crop</option>
                      {ALL_CROPS.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  
                  <InputField label="Area" name="area" value={yieldData.area} onChange={(e: any) => setYieldData({ ...yieldData, area: parseFloat(e.target.value) })} unit="Hectare" min="0.1" max="1000" />
                  <InputField label="Nitrogen (N)" name="N" value={yieldData.N} onChange={(e: any) => setYieldData({ ...yieldData, N: parseFloat(e.target.value) })} unit="kg/ha" min="0" max="200" />
                  <InputField label="Phosphorus (P)" name="P" value={yieldData.P} onChange={(e: any) => setYieldData({ ...yieldData, P: parseFloat(e.target.value) })} unit="kg/ha" min="0" max="200" />
                  <InputField label="Potassium (K)" name="K" value={yieldData.K} onChange={(e: any) => setYieldData({ ...yieldData, K: parseFloat(e.target.value) })} unit="kg/ha" min="0" max="200" />
                  <InputField label="Temperature" name="temperature" value={yieldData.temperature} onChange={(e: any) => setYieldData({ ...yieldData, temperature: parseFloat(e.target.value) })} unit="°C" min="-10" max="50" />
                  <InputField label="Humidity" name="humidity" value={yieldData.humidity} onChange={(e: any) => setYieldData({ ...yieldData, humidity: parseFloat(e.target.value) })} unit="%" min="0" max="100" />
                  <InputField label="Rainfall" name="rainfall" value={yieldData.rainfall} onChange={(e: any) => setYieldData({ ...yieldData, rainfall: parseFloat(e.target.value) })} unit="mm" min="0" max="500" />
                  <InputField label="Water Req" name="water_requirement" value={yieldData.water_requirement} onChange={(e: any) => setYieldData({ ...yieldData, water_requirement: parseFloat(e.target.value) })} unit="mm/day" min="0" max="20" step="0.1" />
                  <InputField label="Growth Duration" name="growth_duration" value={yieldData.growth_duration} onChange={(e: any) => setYieldData({ ...yieldData, growth_duration: parseFloat(e.target.value) })} unit="days" min="30" max="365" />
                </div>
                <button type="submit" disabled={loading} className="w-full bg-agri-600 hover:bg-agri-700 text-white px-8 py-3 rounded-xl font-medium">
                  {loading ? 'Predicting...' : 'Predict Yield'}
                </button>
                {yieldResult && (
                  <div className="bg-gradient-to-br from-emerald-600 to-teal-800 text-white p-8 rounded-3xl shadow-xl mt-6">
                    <div className="flex items-start justify-between mb-6">
                      <div className="bg-white/20 p-3 rounded-2xl backdrop-blur-sm">
                        <TrendingUp size={32} className="text-white" />
                      </div>
                      {yieldResult.confidence && (
                        <span className="bg-white/10 px-3 py-1 rounded-full text-xs font-medium backdrop-blur-sm border border-white/20">
                          {(yieldResult.confidence * 100).toFixed(1)}% Confidence
                        </span>
                      )}
                    </div>
                    <h3 className="text-emerald-100 font-medium mb-1">Predicted Yield</h3>
                    <h2 className="text-4xl font-bold mb-2 tracking-tight">
                      {yieldResult.predicted_yield || yieldResult.prediction || yieldResult.yield || 'N/A'}
                    </h2>
                    <p className="text-emerald-50 text-sm opacity-90">
                      {yieldResult.unit || 'tons per hectare'}
                    </p>
                    {/* Debug Info */}
                    <div className="mt-4 p-2 bg-black/50 rounded text-xs overflow-auto font-mono text-white/70">
                        DEBUG: {JSON.stringify(yieldResult, null, 2)}
                    </div>
                    {yieldResult.yield_range && (
                      <div className="border-t border-white/10 pt-4 mt-4">
                        <p className="text-emerald-50 text-sm leading-relaxed opacity-90">
                          Expected range: {yieldResult.yield_range.minimum} - {yieldResult.yield_range.maximum} {yieldResult.unit || 'tons/ha'}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </form>
            </div>
          )}

          {/* Water Requirement Tab */}
          {activeTab === 'water' && (
            <div className="max-w-4xl">
              <h3 className="text-xl font-bold text-agri-900 dark:text-white mb-4">Water Requirement Calculation</h3>
              <form onSubmit={handleWaterRequirement} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <InputField label="Temperature" name="temperature" value={waterData.temperature} onChange={(e: any) => setWaterData({ ...waterData, temperature: parseFloat(e.target.value) })} unit="°C" min="-10" max="50" />
                  <InputField label="Humidity" name="humidity" value={waterData.humidity} onChange={(e: any) => setWaterData({ ...waterData, humidity: parseFloat(e.target.value) })} unit="%" min="0" max="100" />
                  <InputField label="Rainfall" name="rainfall" value={waterData.rainfall} onChange={(e: any) => setWaterData({ ...waterData, rainfall: parseFloat(e.target.value) })} unit="mm" min="0" max="500" />
                  <InputField label="Growth Duration" name="growth_duration" value={waterData.growth_duration} onChange={(e: any) => setWaterData({ ...waterData, growth_duration: parseFloat(e.target.value) })} unit="days" min="30" max="365" />
                  <InputField label="Area" name="area" value={(waterData as any).area || 1} onChange={(e: any) => setWaterData({ ...waterData, area: parseFloat(e.target.value) } as any)} unit="hectares" min="0.1" max="1000" step="0.1" />
                  
                  {/* Optional Context Fields - kept for UI but less critical for model unless enhanced */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-stone-500 dark:text-gray-400 uppercase tracking-wide">Crop Type</label>
                    <select aria-label="Select Crop Type" title="Select Crop Type" value={waterData.crop} onChange={(e) => setWaterData({ ...waterData, crop: e.target.value })} className="block w-full rounded-lg border-stone-200 bg-stone-50 p-2.5 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white">
                      <option value="">Select Crop</option>
                      {ALL_CROPS.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                </div>
                <button type="submit" disabled={loading} className="w-full bg-agri-600 hover:bg-agri-700 text-white px-8 py-3 rounded-xl font-medium">
                  {loading ? 'Calculating...' : 'Calculate Water Needs'}
                </button>
                {waterResult && (
                  <div className="bg-gradient-to-br from-blue-600 to-cyan-800 text-white p-8 rounded-3xl shadow-xl mt-6">
                    <div className="flex items-start justify-between mb-6">
                      <div className="bg-white/20 p-3 rounded-2xl backdrop-blur-sm">
                        <Droplets size={32} className="text-white" />
                      </div>
                      {waterResult.confidence && (
                        <span className="bg-white/10 px-3 py-1 rounded-full text-xs font-medium backdrop-blur-sm border border-white/20">
                          {(waterResult.confidence * 100).toFixed(1)}% Confidence
                        </span>
                      )}
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div>
                        <h3 className="text-blue-100 font-medium mb-1">Daily Requirement</h3>
                        <h2 className="text-4xl font-bold mb-2 tracking-tight">
                          {waterResult.predicted_water_requirement || 'N/A'}
                        </h2>
                        <p className="text-blue-50 text-sm opacity-90">
                          {waterResult.unit || 'mm/day'}
                        </p>
                      </div>

                      {waterResult.total_liters_per_day && (
                        <div>
                          <h3 className="text-blue-100 font-medium mb-1">Total Water Volume</h3>
                          <h2 className="text-3xl font-bold mb-2 tracking-tight">
                            {(waterResult.total_liters_per_day / 1000).toLocaleString()}
                          </h2>
                          <p className="text-blue-50 text-sm opacity-90">
                            m³/day for {waterResult.input_data?.area || 1} ha
                          </p>
                        </div>
                      )}
                    </div>

                    {waterResult.details && (
                      <div className="border-t border-white/10 pt-4 mt-6">
                        <p className="text-blue-50 text-sm leading-relaxed opacity-90">
                          {waterResult.details}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </form>
            </div>
          )}

          {/* Season Classification Tab */}
          {activeTab === 'season' && (
            <div className="max-w-4xl">
              <h3 className="text-xl font-bold text-agri-900 dark:text-white mb-4">Growing Season Classification</h3>
              <form onSubmit={handleSeasonClassification} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <InputField label="Temperature" name="temperature" value={seasonData.temperature} onChange={(e: any) => setSeasonData({ ...seasonData, temperature: parseFloat(e.target.value) })} unit="°C" min="-10" max="50" />
                  <InputField label="Rainfall" name="rainfall" value={seasonData.rainfall} onChange={(e: any) => setSeasonData({ ...seasonData, rainfall: parseFloat(e.target.value) })} unit="mm" min="0" max="500" />
                  <InputField label="Humidity" name="humidity" value={seasonData.humidity} onChange={(e: any) => setSeasonData({ ...seasonData, humidity: parseFloat(e.target.value) })} unit="%" min="0" max="100" />
                  <InputField label="Growth Duration" name="growth_duration" value={seasonData.growth_duration} onChange={(e: any) => setSeasonData({ ...seasonData, growth_duration: parseFloat(e.target.value) })} unit="days" min="30" max="365" />
                </div>
                <button type="submit" disabled={loading} className="w-full bg-agri-600 hover:bg-agri-700 text-white px-8 py-3 rounded-xl font-medium">
                  {loading ? 'Classifying...' : 'Classify Season'}
                </button>
                {seasonResult && (
                  <div className="bg-gradient-to-br from-amber-600 to-orange-800 text-white p-8 rounded-3xl shadow-xl mt-6">
                    <div className="flex items-start justify-between mb-6">
                      <div className="bg-white/20 p-3 rounded-2xl backdrop-blur-sm">
                        <Calendar size={32} className="text-white" />
                      </div>
                      {seasonResult.confidence && (
                        <span className="bg-white/10 px-3 py-1 rounded-full text-xs font-medium backdrop-blur-sm border border-white/20">
                          {(seasonResult.confidence * 100).toFixed(1)}% Confidence
                        </span>
                      )}
                    </div>
                    <h3 className="text-amber-100 font-medium mb-1">Best Growing Season</h3>
                    <h2 className="text-4xl font-bold mb-4 tracking-tight">
                      {seasonResult.predicted_season || seasonResult.prediction || seasonResult.season || 'N/A'}
                    </h2>
                    <p className="text-amber-50 text-sm leading-relaxed opacity-90 border-t border-white/10 pt-4">
                      Optimal growing season identified based on temperature, rainfall, and humidity patterns.
                    </p>
                  </div>
                )}
              </form>
            </div>
          )}

          {/* Crop Type Tab */}
          {activeTab === 'croptype' && (
            <div className="max-w-4xl">
              <h3 className="text-xl font-bold text-agri-900 dark:text-white mb-4">Crop Type Identification</h3>
              <form onSubmit={handleCropType} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <InputField label="Nitrogen (N)" name="nitrogen" value={cropTypeData.nitrogen} onChange={(e: any) => setCropTypeData({ ...cropTypeData, nitrogen: parseFloat(e.target.value) })} unit="mg/kg" min="0" max="200" />
                  <InputField label="Phosphorus (P)" name="phosphorus" value={cropTypeData.phosphorus} onChange={(e: any) => setCropTypeData({ ...cropTypeData, phosphorus: parseFloat(e.target.value) })} unit="mg/kg" min="0" max="200" />
                  <InputField label="Potassium (K)" name="potassium" value={cropTypeData.potassium} onChange={(e: any) => setCropTypeData({ ...cropTypeData, potassium: parseFloat(e.target.value) })} unit="mg/kg" min="0" max="200" />
                  <InputField label="Temperature" name="temperature" value={cropTypeData.temperature} onChange={(e: any) => setCropTypeData({ ...cropTypeData, temperature: parseFloat(e.target.value) })} unit="°C" min="0" max="50" />
                  <InputField label="Humidity" name="humidity" value={cropTypeData.humidity} onChange={(e: any) => setCropTypeData({ ...cropTypeData, humidity: parseFloat(e.target.value) })} unit="%" min="0" max="100" />
                  <InputField label="pH" name="ph" value={cropTypeData.ph} onChange={(e: any) => setCropTypeData({ ...cropTypeData, ph: parseFloat(e.target.value) })} unit="pH" min="0" max="14" step="0.1" />
                  <InputField label="Rainfall" name="rainfall" value={cropTypeData.rainfall} onChange={(e: any) => setCropTypeData({ ...cropTypeData, rainfall: parseFloat(e.target.value) })} unit="mm" min="0" max="500" />
                </div>
                <button type="submit" disabled={loading} className="w-full bg-agri-600 hover:bg-agri-700 text-white px-8 py-3 rounded-xl font-medium">
                  {loading ? 'Identifying...' : 'Identify Crop Type'}
                </button>
                {cropTypeResult && (
                  <div className="bg-gradient-to-br from-purple-600 to-indigo-800 text-white p-8 rounded-3xl shadow-xl mt-6">
                    <div className="flex items-start justify-between mb-6">
                      <div className="bg-white/20 p-3 rounded-2xl backdrop-blur-sm">
                        <FileText size={32} className="text-white" />
                      </div>
                      {cropTypeResult.confidence && (
                        <span className="bg-white/10 px-3 py-1 rounded-full text-xs font-medium backdrop-blur-sm border border-white/20">
                          {(cropTypeResult.confidence * 100).toFixed(1)}% Confidence
                        </span>
                      )}
                    </div>
                    <h3 className="text-purple-100 font-medium mb-1">Identified Crop Type</h3>
                    <h2 className="text-4xl font-bold mb-4 tracking-tight">
                      {cropTypeResult.crop || 'Unknown'}
                    </h2>
                    <p className="text-purple-50 text-sm leading-relaxed opacity-90 border-t border-white/10 pt-4">
                      Crop identified based on soil nutrient profile and environmental conditions analysis.
                    </p>
                  </div>
                )}
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Crops;