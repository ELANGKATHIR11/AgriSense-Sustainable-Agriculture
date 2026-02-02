import React, { useState, useEffect } from 'react';
import { fetchIrrigationStatus, toggleIrrigationPump } from '../services/api';
import { IrrigationStatus } from '../types';
import { Droplets, Power, Activity, Clock, Settings, AlertTriangle } from 'lucide-react';

const Irrigation: React.FC = () => {
  const [status, setStatus] = useState<IrrigationStatus | null>(null);
  const [loading, setLoading] = useState(false);

  const refreshStatus = async () => {
    const data = await fetchIrrigationStatus();
    setStatus(data);
  };

  useEffect(() => {
    refreshStatus();
    const interval = setInterval(refreshStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleToggle = async () => {
    if (!status) return;
    setLoading(true);
    try {
      const newState = await toggleIrrigationPump(!status.pump_active);
      setStatus(newState);
    } catch (e) { console.error(e); } finally { setLoading(false); }
  };

  if (!status) return <div className="p-8 text-center text-gray-500">Connecting to Irrigation Controller...</div>;

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-agri-900">Smart Irrigation Control</h1>
          <p className="text-gray-600">Monitor water levels and control distribution pumps.</p>
        </div>
        <div className="mt-4 md:mt-0 flex items-center bg-white px-4 py-2 rounded-lg border border-agri-100 shadow-sm">
          <Activity className={`w-5 h-5 mr-2 ${status.pump_active ? 'text-green-500 animate-pulse' : 'text-gray-400'}`} />
          <span className="font-medium text-sm text-gray-700">System Status: <span className={status.pump_active ? 'text-green-600' : 'text-gray-600'}>{status.pump_active ? 'Active' : 'Standby'}</span></span>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 bg-white rounded-2xl shadow-sm border border-agri-100 p-8 flex flex-col items-center justify-center relative overflow-hidden">
          <h2 className="absolute top-6 left-6 font-semibold text-gray-700 flex items-center"><Droplets className="w-5 h-5 mr-2 text-blue-500" />Water Tank Level</h2>
          <div className="mt-8 relative w-40 h-80 border-4 border-gray-200 rounded-3xl bg-gray-50 overflow-hidden shadow-inner">
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-blue-600 to-blue-400 transition-all duration-1000" style={{ height: `${status.water_level}%` }}></div>
            <div className="absolute inset-0 flex flex-col justify-between py-4 px-2 pointer-events-none">
              {[100, 75, 50, 25, 0].map(mark => (
                <div key={mark} className="flex items-center w-full">
                  <div className="w-3 h-0.5 bg-gray-400"></div>
                  <span className="ml-2 text-[10px] text-gray-400 font-mono">{mark}%</span>
                </div>
              ))}
            </div>
          </div>
          <div className="mt-6 text-center">
            <span className="text-4xl font-bold text-gray-900">{status.water_level.toFixed(1)}%</span>
            <p className="text-sm text-gray-500">Current Capacity</p>
          </div>
        </div>
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-2xl shadow-sm border border-agri-100 p-8">
            <h2 className="font-semibold text-gray-800 mb-6 flex items-center"><Settings className="w-5 h-5 mr-2 text-agri-600" />Pump Controls</h2>
            <div className="flex flex-col md:flex-row items-center justify-between bg-agri-50 p-6 rounded-xl border border-agri-100">
              <div className="flex items-center space-x-4 mb-4 md:mb-0">
                <div className={`p-4 rounded-full ${status.pump_active ? 'bg-green-100' : 'bg-gray-200'}`}>
                  <Power className={`w-8 h-8 ${status.pump_active ? 'text-green-600' : 'text-gray-500'}`} />
                </div>
                <div>
                  <h3 className="font-bold text-lg text-gray-900">Main Irrigation Pump</h3>
                  <p className="text-sm text-gray-500">Zone 1 - Maize Field</p>
                </div>
              </div>
              <button onClick={handleToggle} disabled={loading} className={`px-8 py-3 rounded-full font-bold shadow-lg transition-all ${status.pump_active ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-green-500 hover:bg-green-600 text-white'}`}>
                {loading ? 'Processing...' : status.pump_active ? 'STOP PUMP' : 'START PUMP'}
              </button>
            </div>
            {status.water_level < 20 && (
              <div className="mt-4 p-4 bg-red-50 text-red-700 border border-red-100 rounded-lg flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2" />
                Warning: Tank water level is critically low.
              </div>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-agri-100">
              <div className="flex items-center justify-between mb-4"><h3 className="text-gray-500 font-medium text-sm">Flow Rate</h3><Activity className="w-5 h-5 text-blue-500" /></div>
              <p className="text-3xl font-bold text-gray-900">{status.flow_rate.toFixed(1)} <span className="text-base font-normal text-gray-500">L/min</span></p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-sm border border-agri-100">
              <div className="flex items-center justify-between mb-4"><h3 className="text-gray-500 font-medium text-sm">Last Active</h3><Clock className="w-5 h-5 text-orange-500" /></div>
              <p className="text-lg font-bold text-gray-900">{new Date(status.last_active).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Irrigation;
