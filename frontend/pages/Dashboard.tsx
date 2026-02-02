import React, { useEffect, useState } from 'react';
import { 
  Thermometer, 
  Droplets, 
  Wind, 
  Sun, 
  Beaker,
  Sprout
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { SensorCard } from '../components/SensorCard';
import { fetchLiveSensors } from '../services/api';
import { SensorData } from '../types';

const ChartData = [
  { time: '06:00', temp: 22, hum: 65, soil: 48 },
  { time: '09:00', temp: 24, hum: 60, soil: 46 },
  { time: '12:00', temp: 28, hum: 55, soil: 42 },
  { time: '15:00', temp: 27, hum: 58, soil: 44 },
  { time: '18:00', temp: 25, hum: 62, soil: 47 },
  { time: '21:00', temp: 23, hum: 70, soil: 50 },
];

const Dashboard: React.FC = () => {
  const [data, setData] = useState<SensorData | null>(null);

  useEffect(() => {
    // Initial fetch
    fetchLiveSensors().then(setData);
    
    // Poll every 3 seconds as per requirements
    const interval = setInterval(() => {
      fetchLiveSensors().then(setData);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-agri-900">Farm Overview</h2>
          <p className="text-stone-500">Real-time sensor monitoring from Field A</p>
        </div>
        <div className="text-right hidden sm:block">
          <p className="text-sm text-stone-500">Last Update</p>
          <p className="text-sm font-mono text-agri-700">{data ? new Date(data.timestamp).toLocaleTimeString() : 'Connecting...'}</p>
        </div>
      </div>

      {/* Sensor Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <SensorCard 
          title="Air Temperature" 
          value={data?.temperature.toFixed(1) || '--'} 
          unit="°C" 
          icon={Thermometer}
          trend="up"
          trendValue="+1.2%"
          colorClass="bg-orange-50 text-orange-600"
        />
        <SensorCard 
          title="Soil Moisture" 
          value={data?.soilMoisture.toFixed(1) || '--'} 
          unit="%" 
          icon={Droplets}
          trend="down"
          trendValue="-0.5%"
          colorClass="bg-blue-50 text-blue-600"
        />
        <SensorCard 
          title="Humidity" 
          value={data?.humidity.toFixed(1) || '--'} 
          unit="%" 
          icon={Wind}
          colorClass="bg-sky-50 text-sky-600"
        />
        <SensorCard 
          title="Light Intensity" 
          value={data?.lightIntensity.toFixed(0) || '--'} 
          unit="lux" 
          icon={Sun}
          colorClass="bg-yellow-50 text-yellow-600"
        />
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Chart */}
        <div className="lg:col-span-2 bg-white p-6 rounded-2xl border border-stone-100 shadow-sm">
          <h3 className="text-lg font-semibold text-stone-800 mb-6">Microclimate History</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={ChartData}>
                <defs>
                  <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorHum" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorSoil" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f5f5f4" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#78716c'}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#78716c'}} />
                <Tooltip 
                  contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} 
                />
                <Area type="monotone" dataKey="temp" stroke="#f97316" fillOpacity={1} fill="url(#colorTemp)" strokeWidth={2} name="Temp (°C)" />
                <Area type="monotone" dataKey="hum" stroke="#0ea5e9" fillOpacity={1} fill="url(#colorHum)" strokeWidth={2} name="Humidity (%)" />
                <Area type="monotone" dataKey="soil" stroke="#10b981" fillOpacity={1} fill="url(#colorSoil)" strokeWidth={2} name="Soil Moisture (%)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Soil Analysis */}
        <div className="bg-white p-6 rounded-2xl border border-stone-100 shadow-sm">
          <div className="flex items-center gap-2 mb-6">
            <Beaker className="text-agri-600" size={20} />
            <h3 className="text-lg font-semibold text-stone-800">Soil Nutrients</h3>
          </div>
          
          <div className="space-y-5">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-stone-600">Nitrogen (N)</span>
                <span className="font-semibold text-stone-800">{data?.nitrogen} mg/kg</span>
              </div>
              <div className="h-2 bg-stone-100 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-500 rounded-full w-[75%]"></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-stone-600">Phosphorus (P)</span>
                <span className="font-semibold text-stone-800">{data?.phosphorus} mg/kg</span>
              </div>
              <div className="h-2 bg-stone-100 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-400 rounded-full w-[45%]"></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-stone-600">Potassium (K)</span>
                <span className="font-semibold text-stone-800">{data?.potassium} mg/kg</span>
              </div>
              <div className="h-2 bg-stone-100 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-600 rounded-full w-[60%]"></div>
              </div>
            </div>

            <div className="pt-4 mt-4 border-t border-stone-100 flex items-center justify-between">
               <span className="text-stone-500 text-sm">pH Level</span>
               <div className="flex items-center gap-2">
                 <span className="text-2xl font-bold text-stone-800">{data?.phLevel.toFixed(1)}</span>
                 <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-800 text-xs font-medium">Optimal</span>
               </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;