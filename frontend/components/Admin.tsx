import React, { useEffect, useState } from 'react';
import { fetchActivityLogs, fetchSystemMetrics, triggerAdminAction } from '../services/api';
import { ActivityLog, SystemMetrics } from '../types';
import { Server, Database, RefreshCw, AlertOctagon, Activity, Code, Globe, Shield } from 'lucide-react';

const Admin: React.FC = () => {
  const [logs, setLogs] = useState<ActivityLog[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [newLogs, newMetrics] = await Promise.all([fetchActivityLogs(), fetchSystemMetrics()]);
        setLogs(newLogs);
        setMetrics(newMetrics);
      } catch (e) { console.error("Polling failed", e); }
    };
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleAction = async (actionName: string) => {
    if (confirm(`Trigger: ${actionName}?`)) {
      await triggerAdminAction(actionName);
      const newLogs = await fetchActivityLogs();
      setLogs(newLogs);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">System Administration</h1>
        <div className="flex items-center space-x-2 text-sm text-green-600 bg-green-50 px-3 py-1 rounded-full border border-green-100">
          <Globe className="w-4 h-4" /><span className="font-medium">Backend Connected</span>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center"><Server className="w-5 h-5 mr-2 text-agri-600" />System Health</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricBox label="CPU Usage" value={`${metrics?.cpuUsage ?? '-'}%`} />
            <MetricBox label="Memory" value={`${metrics?.memoryUsage ?? '-'}%`} />
            <MetricBox label="Uptime" value={metrics?.uptime ?? '-'} />
            <MetricBox label="Active Nodes" value={metrics?.activeConnections ?? '-'} />
          </div>
          <div className="mt-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">ML Model Status</h3>
            <div className="flex items-center space-x-2">
              <span className={`w-3 h-3 rounded-full ${metrics?.modelStatus === 'loaded' ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></span>
              <span className="font-semibold text-gray-700">{metrics?.modelStatus || 'Unknown'}</span>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center"><Shield className="w-5 h-5 mr-2 text-agri-600" />Quick Actions</h2>
          <div className="space-y-3">
            <ActionButton label="Restart ML Service" icon={RefreshCw} onClick={() => handleAction('Reload')} />
            <ActionButton label="Retrain Models" icon={Database} onClick={() => handleAction('Dataset')} />
            <ActionButton label="Emergency Stop" icon={AlertOctagon} color="text-red-600 hover:bg-red-50" onClick={() => handleAction('Emergency Stop')} />
          </div>
        </div>
      </div>
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-4 border-b border-gray-100 bg-gray-50 flex items-center"><Activity className="w-5 h-5 mr-2 text-gray-500" /><h2 className="font-semibold text-gray-700">Live Activity Logs</h2></div>
        <div className="max-h-64 overflow-y-auto divide-y divide-gray-100">
          {logs.map((log) => (
            <div key={log.id} className="p-4 flex items-center justify-between hover:bg-gray-50">
              <div><p className="text-sm font-medium text-gray-900">{log.action}</p><p className="text-xs text-gray-500">{new Date(log.timestamp).toLocaleString()}</p></div>
              <span className={`px-2 py-1 text-xs rounded-full font-medium ${log.status === 'success' ? 'bg-green-100 text-green-700' : log.status === 'warning' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>{log.status}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="bg-slate-900 rounded-xl shadow-sm overflow-hidden text-slate-300">
        <div className="p-4 bg-slate-800 border-b border-slate-700 flex items-center"><Code className="w-5 h-5 mr-2 text-blue-400" /><h2 className="font-semibold text-white">Backend API Reference</h2></div>
        <div className="p-6 grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Endpoints</h3>
            <ul className="space-y-4">
              <ApiEndpoint method="GET" path="/api/admin/summary" desc="CPU, Memory, Uptime." />
              <ApiEndpoint method="GET" path="/api/admin/activities" desc="Activity logs." />
              <ApiEndpoint method="POST" path="/api/admin/action" desc="Trigger admin tasks." />
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">ML Pipeline</h3>
            <div className="bg-slate-950 p-4 rounded-lg font-mono text-xs border border-slate-800">
              <p className="text-green-400">// Crop recommendation</p>
              <p>POST /api/crop-recommendation/predict</p>
              <p className="text-green-400 mt-2">// Disease detection</p>
              <p>POST /api/vlm/analyze-plant</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const MetricBox = ({ label, value }: any) => (
  <div className="bg-gray-50 p-3 rounded-lg text-center"><p className="text-xs text-gray-500 mb-1">{label}</p><p className="text-xl font-bold text-gray-800">{value ?? '-'}</p></div>
);

const ActionButton = ({ label, icon: Icon, onClick, color = "text-gray-700 hover:bg-gray-50" }: any) => (
  <button onClick={onClick} className={`w-full flex items-center p-3 rounded-lg border border-gray-200 transition-colors ${color}`}>
    <Icon className="w-5 h-5 mr-3" /><span className="font-medium text-sm">{label}</span>
  </button>
);

const ApiEndpoint = ({ method, path, desc }: any) => (
  <li className="flex items-start">
    <span className={`text-[10px] font-bold px-2 py-1 rounded mr-3 w-14 text-center ${method === 'GET' ? 'bg-blue-900 text-blue-200' : 'bg-green-900 text-green-200'}`}>{method}</span>
    <div><p className="font-mono text-sm text-white">{path}</p><p className="text-xs text-slate-500">{desc}</p></div>
  </li>
);

export default Admin;
