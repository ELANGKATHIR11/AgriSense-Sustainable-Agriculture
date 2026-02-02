import React, { useState } from 'react';
import { recommendCrop, MLService } from '../services/api';
import { CropInput } from '../types';
import { Sprout, CloudRain, Thermometer, Droplets, FlaskConical, TrendingUp, Calendar, MapPin, Ruler, Cpu } from 'lucide-react';

const CropManager: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'crop' | 'water' | 'yield' | 'season'>('crop');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  // Separate states for each model with required validation fields
  const [cropInputs, setCropInputs] = useState<CropInput>({
    nitrogen: 90, phosphorus: 42, potassium: 43, temperature: 20.8, humidity: 82.0, ph: 6.5, rainfall: 202.9
  });
  
  const [waterInputs, setWaterInputs] = useState({
    temperature: 25, humidity: 70, rainfall: 150, area: 2.5, growth_duration: 120
  });

  const [yieldInputs, setYieldInputs] = useState({
    crop: 'Rice', area: 2.5, season: 'Kharif', state: 'Tamil Nadu',
    N: 90, P: 42, K: 43, temperature: 25, humidity: 70, rainfall: 150,
    water_requirement: 5.5, growth_duration: 120
  });

  const [seasonInputs, setSeasonInputs] = useState({
    temperature: 25, humidity: 70, rainfall: 150, growth_duration: 120
  });

  const handleRunModel = async () => {
    setLoading(true);
    setResult(null);
    try {
      let data: any;
      switch (activeTab) {
        case 'crop':
          data = await recommendCrop(cropInputs);
          setResult({ type: 'crop', value: data.crop, unit: '', details: 'Best matched crop for these conditions.' });
          break;
        case 'water':
          data = await MLService.predictWaterRequirement(waterInputs);
          setResult({ 
            type: 'water', 
            value: (data.predicted_water_requirement || data.water_requirement || 0).toFixed(2), 
            unit: 'm³', 
            details: data.details || 'Projected water requirement for the growth cycle.' 
          });
          break;
        case 'yield':
          data = await MLService.predictYield(yieldInputs);
          setResult({ 
            type: 'yield', 
            value: (data.predicted_yield || data.yield || 0).toFixed(2), 
            unit: 'Tons', 
            details: data.details || `Estimated production for ${yieldInputs.crop} in ${yieldInputs.state}.` 
          });
          break;
        case 'season':
          data = await MLService.classifySeason(seasonInputs);
          setResult({ type: 'season', value: data.season, unit: '', details: 'Most suitable climate category for these parameters.' });
          break;
      }
    } catch (err) {
      console.error(err);
      alert("Error executing ML model. Please check the backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-agri-900">Agri-Intelligence Studio</h1>
          <p className="text-gray-600 mt-1">Specialized ML models for precision agricultural planning.</p>
        </div>
        <div className="mt-4 md:mt-0 flex bg-white p-1 rounded-xl border border-agri-100 shadow-sm">
          <TabNav active={activeTab === 'crop'} onClick={() => setActiveTab('crop')} label="Crop" icon={Sprout} />
          <TabNav active={activeTab === 'water'} onClick={() => setActiveTab('water')} label="Water" icon={Droplets} />
          <TabNav active={activeTab === 'yield'} onClick={() => setActiveTab('yield')} label="Yield" icon={TrendingUp} />
          <TabNav active={activeTab === 'season'} onClick={() => setActiveTab('season')} label="Season" icon={Calendar} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-sm border border-agri-100 overflow-hidden flex flex-col min-h-[500px]">
          <div className="p-6 border-b border-agri-100 bg-gray-50/50 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {activeTab === 'crop' && <Sprout className="w-5 h-5 text-agri-600" />}
              {activeTab === 'water' && <Droplets className="w-5 h-5 text-blue-600" />}
              {activeTab === 'yield' && <TrendingUp className="w-5 h-5 text-emerald-600" />}
              {activeTab === 'season' && <Calendar className="w-5 h-5 text-orange-600" />}
              <h2 className="font-bold text-agri-900 uppercase tracking-wider text-sm">
                {activeTab === 'crop' && 'Crop Recommendation Engine'}
                {activeTab === 'water' && 'Water Requirement Predictor'}
                {activeTab === 'yield' && 'Yield Estimation Model'}
                {activeTab === 'season' && 'Climate Classification Tool'}
              </h2>
            </div>
          </div>

          <div className="p-8 flex-grow space-y-8">
            {activeTab === 'crop' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-left-4 duration-500">
                <div className="md:col-span-2"><h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Nutrient Profile</h3></div>
                <InputGroup label="Nitrogen (N)" value={cropInputs.nitrogen} onChange={v => setCropInputs({...cropInputs, nitrogen: v})} icon="N" />
                <InputGroup label="Phosphorus (P)" value={cropInputs.phosphorus} onChange={v => setCropInputs({...cropInputs, phosphorus: v})} icon="P" />
                <InputGroup label="Potassium (K)" value={cropInputs.potassium} onChange={v => setCropInputs({...cropInputs, potassium: v})} icon="K" />
                <InputWithIcon label="Soil pH" value={cropInputs.ph} onChange={v => setCropInputs({...cropInputs, ph: v})} Icon={FlaskConical} step="0.1" />
                <div className="md:col-span-2 mt-4"><h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Environmental Sensors</h3></div>
                <InputWithIcon label="Temperature (°C)" value={cropInputs.temperature} onChange={v => setCropInputs({...cropInputs, temperature: v})} Icon={Thermometer} />
                <InputWithIcon label="Humidity (%)" value={cropInputs.humidity} onChange={v => setCropInputs({...cropInputs, humidity: v})} Icon={Droplets} />
                <InputWithIcon label="Rainfall (mm)" value={cropInputs.rainfall} onChange={v => setCropInputs({...cropInputs, rainfall: v})} Icon={CloudRain} />
              </div>
            )}

            {activeTab === 'water' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-right-4 duration-500">
                <InputWithIcon label="Temperature (°C)" value={waterInputs.temperature} onChange={v => setWaterInputs({...waterInputs, temperature: v})} Icon={Thermometer} />
                <InputWithIcon label="Humidity (%)" value={waterInputs.humidity} onChange={v => setWaterInputs({...waterInputs, humidity: v})} Icon={Droplets} />
                <InputWithIcon label="Rainfall (mm)" value={waterInputs.rainfall} onChange={v => setWaterInputs({...waterInputs, rainfall: v})} Icon={CloudRain} />
                <InputWithIcon label="Total Area (Acre)" value={waterInputs.area} onChange={v => setWaterInputs({...waterInputs, area: v})} Icon={Ruler} step="0.1" />
                <InputWithIcon label="Growth Duration (Days)" value={waterInputs.growth_duration} onChange={v => setWaterInputs({...waterInputs, growth_duration: v})} Icon={Calendar} />
              </div>
            )}

            {activeTab === 'yield' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="p-4 bg-white rounded-xl border border-agri-100 shadow-sm md:col-span-2">
                   <label className="block text-xs font-bold text-gray-400 uppercase mb-2">Crop to Analyze</label>
                   <select 
                      value={yieldInputs.crop} 
                      onChange={e => setYieldInputs({...yieldInputs, crop: e.target.value})}
                      className="w-full bg-agri-50 border-none rounded-lg p-3 font-bold text-agri-900 focus:ring-2 focus:ring-agri-500"
                      title="Select Crop"
                   >
                     {['Rice', 'Maize', 'Wheat', 'Sugarcane', 'Cotton', 'Jute'].map(c => <option key={c} value={c}>{c}</option>)}
                   </select>
                </div>
                <InputGroup label="Nitrogen (N)" value={yieldInputs.N} onChange={v => setYieldInputs({...yieldInputs, N: v})} icon="N" />
                <InputGroup label="Phosphorus (P)" value={yieldInputs.P} onChange={v => setYieldInputs({...yieldInputs, P: v})} icon="P" />
                <InputGroup label="Potassium (K)" value={yieldInputs.K} onChange={v => setYieldInputs({...yieldInputs, K: v})} icon="K" />
                <InputWithIcon label="Farm Area (Acre)" value={yieldInputs.area} onChange={v => setYieldInputs({...yieldInputs, area: v})} Icon={Ruler} step="0.1" />
                <InputWithIcon label="Temperature (°C)" value={yieldInputs.temperature} onChange={v => setYieldInputs({...yieldInputs, temperature: v})} Icon={Thermometer} />
                <InputWithIcon label="Humidity (%)" value={yieldInputs.humidity} onChange={v => setYieldInputs({...yieldInputs, humidity: v})} Icon={Droplets} />
                <InputWithIcon label="Growth Duration (Days)" value={yieldInputs.growth_duration} onChange={v => setYieldInputs({...yieldInputs, growth_duration: v})} Icon={Calendar} />
                <InputWithIcon label="Water (mm/day)" value={yieldInputs.water_requirement} onChange={v => setYieldInputs({...yieldInputs, water_requirement: v})} Icon={CloudRain} step="0.1" />
              </div>
            )}

            {activeTab === 'season' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in zoom-in duration-500">
                <InputWithIcon label="Avg Temperature (°C)" value={seasonInputs.temperature} onChange={v => setSeasonInputs({...seasonInputs, temperature: v})} Icon={Thermometer} />
                <InputWithIcon label="Avg Humidity (%)" value={seasonInputs.humidity} onChange={v => setSeasonInputs({...seasonInputs, humidity: v})} Icon={Droplets} />
                <InputWithIcon label="Annual Rainfall (mm)" value={seasonInputs.rainfall} onChange={v => setSeasonInputs({...seasonInputs, rainfall: v})} Icon={CloudRain} />
                <InputWithIcon label="Growth Duration (Days)" value={seasonInputs.growth_duration} onChange={v => setSeasonInputs({...seasonInputs, growth_duration: v})} Icon={Calendar} />
              </div>
            )}
          </div>

          <div className="p-6 bg-gray-50 border-t border-agri-100">
            <button 
              onClick={handleRunModel} 
              disabled={loading} 
              className="w-full py-4 bg-agri-600 text-white rounded-xl font-bold shadow-lg hover:bg-agri-700 transition-all hover:scale-[1.01] active:scale-95 flex justify-center items-center space-x-2"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <>
                  <Cpu className="w-5 h-5" />
                  <span>Execute Model Analysis</span>
                </>
              )}
            </button>
          </div>
        </div>

        <div className="lg:col-span-1">
          <div className={`bg-white rounded-2xl shadow-sm border border-agri-100 h-full p-8 flex flex-col items-center justify-center text-center relative overflow-hidden transition-all duration-500 ${result ? 'bg-gradient-to-br from-agri-50 to-white ring-2 ring-agri-200' : ''}`}>
             <div className="absolute top-0 right-0 p-3">
               <div className="text-[10px] font-black text-agri-200 uppercase tracking-widest rotate-90 origin-right translate-y-8">AgriSense Engine</div>
             </div>
            {result ? (
              <div className="animate-in zoom-in fade-in duration-500 flex flex-col items-center w-full">
                <div className={`w-24 h-24 rounded-full flex items-center justify-center mb-8 shadow-inner ${result.type === 'crop' ? 'bg-agri-100' : result.type === 'water' ? 'bg-blue-100' : result.type === 'yield' ? 'bg-emerald-100' : 'bg-orange-100'}`}>
                  {result.type === 'crop' && <Sprout className="w-12 h-12 text-agri-600" />}
                  {result.type === 'water' && <Droplets className="w-12 h-12 text-blue-600" />}
                  {result.type === 'yield' && <TrendingUp className="w-12 h-12 text-emerald-600" />}
                  {result.type === 'season' && <Calendar className="w-12 h-12 text-orange-600" />}
                </div>
                <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-1">Model Output</h3>
                <div className="text-4xl font-black text-agri-950 flex items-baseline">
                  {result.value}
                  {result.unit && <span className="text-sm font-bold ml-2 text-agri-600">{result.unit}</span>}
                </div>
                <div className="mt-8 p-4 bg-white border border-agri-100 rounded-xl shadow-sm w-full">
                  <p className="text-sm text-gray-700 leading-relaxed font-medium">"{result.details}"</p>
                </div>
                <button 
                  onClick={() => setResult(null)} 
                  className="mt-8 text-xs font-bold text-agri-500 hover:text-agri-700 uppercase tracking-tighter"
                >
                  Clear Result
                </button>
              </div>
            ) : (
              <div className="text-gray-300">
                <Cpu className="w-20 h-20 mx-auto mb-6 opacity-20 animate-pulse" />
                <h4 className="font-bold text-gray-400">Decision Pending</h4>
                <p className="text-sm mt-2 max-w-[200px] mx-auto">Configure model parameters and execute to receive AI output.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const TabNav = ({ active, onClick, label, icon: Icon }: any) => (
  <button 
    onClick={onClick}
    className={`px-6 py-2.5 rounded-lg text-sm font-bold flex items-center space-x-2 transition-all duration-200 ${active ? 'bg-agri-600 text-white shadow-md' : 'text-gray-500 hover:text-agri-600 hover:bg-agri-50'}`}
  >
    <Icon className="w-4 h-4" />
    <span>{label}</span>
  </button>
);

const InputGroup = ({ label, value, onChange, icon }: any) => (
  <div className="bg-white p-4 rounded-xl border border-agri-100 shadow-sm transition-all focus-within:ring-2 focus-within:ring-agri-500 focus-within:border-transparent group">
    <label className="block text-[10px] font-black text-gray-400 uppercase mb-2 group-focus-within:text-agri-600 transition-colors">{label}</label>
    <div className="flex items-center">
      <span className="font-black text-agri-600 mr-3 text-2xl" aria-hidden="true">{icon}</span>
      <input 
          type="number" 
          value={value} 
          onChange={e => onChange(parseFloat(e.target.value))} 
          aria-label={label} 
          title={label}
          className="w-full bg-transparent border-none p-0 text-agri-950 font-black text-xl focus:outline-none focus:ring-0" 
      />
    </div>
  </div>
);

const InputWithIcon = ({ label, value, onChange, Icon, step = "1" }: any) => (
  <div className="bg-white p-4 rounded-xl border border-agri-100 shadow-sm transition-all focus-within:ring-2 focus-within:ring-agri-500 focus-within:border-transparent group">
    <label className="block text-[10px] font-black text-gray-400 uppercase mb-2 group-focus-within:text-agri-600 transition-colors">{label}</label>
    <div className="flex items-center">
      <Icon className="w-6 h-6 text-agri-500 mr-3" aria-hidden="true" />
      <input 
          type="number" 
          value={value} 
          step={step} 
          onChange={e => onChange(parseFloat(e.target.value))} 
          aria-label={label} 
          title={label}
          className="w-full bg-transparent border-none p-0 text-agri-950 font-black text-xl focus:outline-none focus:ring-0" 
      />
    </div>
  </div>
);

export default CropManager;
