import React, { useEffect, useState } from 'react';
import { fetchLiveSensors, fetchSystemMetrics, MLService } from '../services/api';
import { SensorData, SystemMetrics, AnalysisResult } from '../types';
import { Thermometer, Droplets, Sun, Wind, Cpu, Wifi, Brain, CheckCircle2, TrendingUp, Shovel, CloudRain } from 'lucide-react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const Dashboard: React.FC = () => {
  const [sensorData, setSensorData] = useState<SensorData | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState<'soil' | 'climate' | 'ops'>('soil');
  
  // Input parameters for AI Analysis
  const [params, setParams] = useState({
    N: 140,
    P: 45,
    K: 45,
    ph: 6.5,
    rainfall: 150,
    area: 2.5
  });

  useEffect(() => {
    const getData = async () => {
      const data = await fetchLiveSensors();
      const sysMetrics = await fetchSystemMetrics();
      setSensorData(data);
      setMetrics(sysMetrics);
      setHistory(prev => {
        const newEntry = {
          time: new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          temp: data.temperature,
          moisture: data.soilMoisture,
          humidity: data.humidity
        };
        const newHistory = [...prev, newEntry];
        if (newHistory.length > 20) newHistory.shift();
        return newHistory;
      });
    };
    getData();
    const interval = setInterval(getData, 10000); // 10s is enough for dashboard
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async () => {
    if (!sensorData) return;
    setIsAnalyzing(true);
    try {
      const result = await MLService.analyze({
        ...params,
        temperature: sensorData.temperature,
        humidity: sensorData.humidity,
      });
      setAnalysisResult(result);
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (!sensorData) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-agri-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-br from-agri-600 to-agri-800 rounded-2xl p-8 text-white shadow-lg relative overflow-hidden">
        {/* Dark overlay to improve text contrast over the gradient */}
        <div className="absolute inset-0 bg-black/30 pointer-events-none"></div>
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-white opacity-10 rounded-full blur-3xl"></div>
        <div className="relative z-10">
          <h1 className="text-4xl font-bold mb-4 tracking-tight drop-shadow-md">Welcome to AgriSense</h1>
          <p className="text-slate-200 text-lg max-w-2xl mb-8 leading-relaxed">
            Smart Farming with IoT Sensors, Machine Learning, and Real-time Analytics.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <FeaturePill icon={Wifi} title="Real-time IoT" desc="Live soil & air monitoring" />
            <FeaturePill icon={Brain} title="ML Disease Detection" desc="AI-powered plant health" />
            <FeaturePill icon={Cpu} title="Smart Automation" desc="Automated irrigation" />
          </div>
        </div>
      </div>

      <div className="flex flex-col md:flex-row md:items-center justify-between px-2">
        <div>
          <h2 className="text-2xl font-bold text-agri-900">Live Farm Data</h2>
          <p className="text-gray-500">Real-time sensor network feed</p>
        </div>
        <div className="mt-4 md:mt-0 flex items-center space-x-2 bg-white px-4 py-2 rounded-full shadow-sm border border-agri-100">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
          <span className="text-sm font-medium text-gray-600">IoT Gateway Connected</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <SensorCard title="Air Temperature" value={`${sensorData.temperature.toFixed(1)}°C`} icon={Thermometer} color="text-orange-500" bg="bg-orange-50" trend="+0.2%" />
        <SensorCard title="Humidity" value={`${sensorData.humidity.toFixed(1)}%`} icon={Wind} color="text-blue-500" bg="bg-blue-50" trend="-1.5%" />
        <SensorCard title="Soil Moisture" value={`${sensorData.soilMoisture.toFixed(1)}%`} icon={Droplets} color="text-cyan-600" bg="bg-cyan-50" trend="Optimal" />
        <SensorCard title="Light Intensity" value={`${sensorData.lightIntensity.toFixed(0)} Lux`} icon={Sun} color="text-yellow-500" bg="bg-yellow-50" trend="High" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ML Analysis Input Panel */}
        <div className="bg-white p-0 rounded-xl border border-agri-100 shadow-sm overflow-hidden flex flex-col">
          <div className="bg-gray-50 flex border-b border-agri-100">
            <TabButton active={activeTab === 'soil'} onClick={() => setActiveTab('soil')} label="Soil" icon={Shovel} />
            <TabButton active={activeTab === 'climate'} onClick={() => setActiveTab('climate')} label="Climate" icon={Wind} />
            <TabButton active={activeTab === 'ops'} onClick={() => setActiveTab('ops')} label="Ops" icon={Cpu} />
          </div>
          
          <div className="p-6 space-y-4 flex-grow">
            {activeTab === 'soil' && (
              <div className="animate-in fade-in slide-in-from-left-2 duration-300 space-y-4">
                <ParameterSlider label="Nitrogen (N)" value={params.N} min={0} max={300} onChange={v => setParams({...params, N: v})} unit="mg/kg" />
                <ParameterSlider label="Phosphorus (P)" value={params.P} min={0} max={150} onChange={v => setParams({...params, P: v})} unit="mg/kg" />
                <ParameterSlider label="Potassium (K)" value={params.K} min={0} max={150} onChange={v => setParams({...params, K: v})} unit="mg/kg" />
                <ParameterSlider label="Soil pH" value={params.ph} min={0} max={14} step={0.1} onChange={v => setParams({...params, ph: v})} unit="pH" />
              </div>
            )}
            
            {activeTab === 'climate' && (
              <div className="animate-in fade-in slide-in-from-right-2 duration-300 space-y-4">
                <ParameterSlider label="Rainfall" value={params.rainfall} min={0} max={1000} onChange={v => setParams({...params, rainfall: v})} unit="mm" />
                <p className="text-[10px] text-gray-400 italic">Climate defaults are sync'd with live sensors unless overridden manually.</p>
              </div>
            )}

            {activeTab === 'ops' && (
              <div className="animate-in fade-in slide-in-from-bottom-2 duration-300 space-y-4">
                <ParameterSlider label="Farm Area" value={params.area} min={0.1} max={100} step={0.1} onChange={v => setParams({...params, area: v})} unit="Acre" />
                <div className="p-3 bg-agri-50 rounded-lg border border-agri-100">
                  <p className="text-xs text-agri-700">Area scaling is used by the Stage 4 Yield Predictor to estimate total tonnage.</p>
                </div>
              </div>
            )}
          </div>

          <div className="p-6 pt-0 mt-auto">
            <button 
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className={`w-full py-3 rounded-xl font-bold text-white transition-all shadow-md flex items-center justify-center space-x-2 
                ${isAnalyzing ? 'bg-agri-400 cursor-not-allowed' : 'bg-agri-600 hover:bg-agri-700 active:scale-95'}`}
            >
              {isAnalyzing ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              ) : (
                <>
                  <Brain className="w-5 h-5" />
                  <span>Execute AI Studio</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Prediction Results Display */}
        <div className="lg:col-span-2 bg-gradient-to-br from-white to-agri-50 p-6 rounded-xl border border-agri-100 shadow-sm relative overflow-hidden">
          {!analysisResult ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-400 space-y-4 py-12">
              <Brain className="w-16 h-16 opacity-20" />
              <p className="text-center italic">Adjust your soil parameters and click <br/> "Analyze & Recommend" to see AI predictions</p>
            </div>
          ) : (
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center space-x-2 text-agri-900">
                  <TrendingUp className="w-5 h-5 text-agri-600" />
                  <h3 className="text-lg font-semibold">Agri-Intelligence Report</h3>
                </div>
                <div className="text-xs font-bold text-agri-600 bg-white px-3 py-1 rounded-full border border-agri-100">
                  LOCKED DATASET V2.0
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-grow">
                <ResultItem 
                  icon={CheckCircle2} 
                  label="Recommended Crop" 
                  value={analysisResult.recommended_crop} 
                  subtext={`${analysisResult.crop_group} category`}
                  color="text-green-600"
                />
                <ResultItem 
                  icon={TrendingUp} 
                  label="Expected Yield" 
                  value={`${analysisResult.expected_yield.toFixed(2)} Tons`} 
                  subtext="Based on area scaling"
                  color="text-agri-600"
                />
                <ResultItem 
                  icon={CloudRain} 
                  label="Water Requirement" 
                  value={`${analysisResult.water_requirement.toFixed(1)} m³`} 
                  subtext="Projected for full cycle"
                  color="text-blue-600"
                />
                <ResultItem 
                  icon={Sun} 
                  label="Optimum Season" 
                  value={analysisResult.season} 
                  subtext="Climate compatibility"
                  color="text-orange-600"
                />
              </div>

              <div className="mt-6 p-4 bg-white/50 backdrop-blur-sm rounded-lg border border-agri-100 flex items-start space-x-3">
                <Brain className="w-5 h-5 text-agri-600 mt-0.5" />
                <p className="text-xs text-gray-600 leading-relaxed">
                  <strong>AI Insights:</strong> The {analysisResult.recommended_crop} recommendation utilizes your current soil pH of {params.ph} and Nitrogen levels of {params.N} mg/kg. The hierarchical model predicts high yield potential in the {analysisResult.season} season.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const FeaturePill = ({ icon: Icon, title, desc }: any) => (
  <div className="flex items-start bg-white/10 backdrop-blur-md rounded-lg p-4 border border-white/20">
    <Icon className="w-6 h-6 text-white mr-3 mt-1" />
    <div>
      <h3 className="font-bold text-white text-sm">{title}</h3>
      <p className="text-slate-200 text-xs mt-1">{desc}</p>
    </div>
  </div>
);

const SensorCard = ({ title, value, icon: Icon, color, bg, trend }: any) => (
  <div className="bg-white p-5 rounded-xl border border-agri-100 shadow-sm transition hover:shadow-md">
    <div className="flex items-center justify-between mb-4">
      <div className={`p-3 rounded-lg ${bg}`}><Icon className={`w-6 h-6 ${color}`} /></div>
      <span className={`text-xs font-medium px-2 py-1 rounded-full ${trend === 'Optimal' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}`}>{trend}</span>
    </div>
    <h3 className="text-gray-500 text-sm font-medium">{title}</h3>
    <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
  </div>
);

const NutrientBar = ({ label, value, max, color }: any) => (
  <div>
    <div className="flex justify-between text-sm mb-1">
      <span className="text-gray-600 font-medium">{label}</span>
      <span className="text-gray-900 font-bold">{value.toFixed(0)} <span className="text-xs font-normal text-gray-500">mg/kg</span></span>
    </div>
    <div className="w-full bg-gray-100 rounded-full h-2.5">
      <div 
        className={`h-2.5 rounded-full ${color}`} 
        // eslint-disable-next-line react/forbid-dom-props
        style={{ width: `${Math.min((value / max) * 100, 100)}%` }}
      />
    </div>
  </div>
);

const ParameterSlider = ({ label, value, min, max, step = 1, unit, onChange }: any) => (
  <div className="space-y-1">
    <div className="flex justify-between items-center mb-1">
      <label className="text-gray-500 font-medium text-xs">{label}</label>
      <div className="flex items-center space-x-2">
        <input 
          type="number" 
          value={value} 
          min={min} 
          max={max} 
          step={step}
          onChange={e => {
            const val = parseFloat(e.target.value);
            if (!isNaN(val)) onChange(val);
          }}
          className="w-16 h-7 text-xs font-bold text-agri-700 bg-gray-50 border border-agri-100 rounded text-center focus:ring-1 focus:ring-agri-500 outline-none"
          title={`${label} input`}
          aria-label={`${label} numeric input`}
        />
        <span className="text-[10px] font-normal text-gray-400 w-8">{unit}</span>
      </div>
    </div>
    <input 
      type="range" min={min} max={max} step={step} value={value} 
      onChange={e => onChange(parseFloat(e.target.value))}
      className="w-full h-1.5 bg-gray-100 rounded-lg appearance-none cursor-pointer accent-agri-600"
      aria-label={label}
      title={label}
    />
  </div>
);

const TabButton = ({ active, onClick, label, icon: Icon }: any) => (
  <button 
    onClick={onClick}
    className={`flex-1 flex items-center justify-center space-x-2 py-3 text-sm font-semibold transition-all border-b-2
      ${active ? 'text-agri-600 border-agri-600 bg-white' : 'text-gray-400 border-transparent hover:text-gray-600 bg-gray-50/50'}`}
  >
    <Icon className={`w-4 h-4 ${active ? 'text-agri-600' : 'text-gray-400'}`} />
    <span>{label}</span>
  </button>
);

const ResultItem = ({ icon: Icon, label, value, subtext, color }: any) => (
  <div className="bg-white p-4 rounded-xl border border-agri-100 shadow-sm flex items-start space-x-3">
    <div className={`p-2 rounded-lg bg-gray-50 ${color}`}><Icon className="w-5 h-5" /></div>
    <div>
      <p className="text-xs text-gray-500 font-medium">{label}</p>
      <p className="text-lg font-bold text-gray-900">{value}</p>
      <p className="text-[10px] text-gray-400 uppercase tracking-wider">{subtext}</p>
    </div>
  </div>
);

export default Dashboard;
