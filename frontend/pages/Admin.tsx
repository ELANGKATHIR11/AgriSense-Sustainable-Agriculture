import React, { useState, useEffect } from 'react';
import {
  RotateCcw,
  Database,
  RefreshCw,
  CloudRain,
  Trash2,
  Cpu,
  Activity,
  Server,
  ShieldCheck
} from 'lucide-react';
import { fetchSystemMetrics, fetchActivityLogs, triggerAdminAction } from '../services/api';
import { ActivityLog, SystemMetrics } from '../types';

const Admin: React.FC = () => {
  const [activities, setActivities] = useState<ActivityLog[]>([]);
  const [resetting, setResetting] = useState(false);
  const [summary, setSummary] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [logs, metrics] = await Promise.all([
          fetchActivityLogs(),
          fetchSystemMetrics()
        ]);
        setActivities(logs);
        setSummary(metrics);
      } catch (err) {
        console.error(err);
      }
    };
    loadData();
    const t = setInterval(loadData, 3000);
    return () => clearInterval(t);
  }, []);

  const handleAction = async (actionName: string) => {
    if (actionName === "Reset") {
      if (!confirm("Are you sure? This erases all data.")) return;
      setResetting(true);
      try {
        await triggerAdminAction("Reset");
        const logs = await fetchActivityLogs();
        setActivities(logs);
      } catch (e) { console.error(e); }
      setResetting(false);
    } else {
      try {
        await triggerAdminAction(actionName);
        const logs = await fetchActivityLogs();
        setActivities(logs);
      } catch (e) {
        console.error("Action failed:", e);
      }
    }
  };

  const metrics = [
    { label: 'CPU Load', val: summary ? `${summary.cpuUsage}%` : '--', icon: Cpu, color: 'text-blue-600 bg-blue-50' },
    { label: 'Memory', val: summary ? `${summary.memoryUsage}%` : '--', icon: Database, color: 'text-purple-600 bg-purple-50' },
    { label: 'Uptime', val: summary?.uptime ?? '--', icon: Activity, color: 'text-green-600 bg-green-50' },
    { label: 'API Status', val: summary?.modelStatus === 'loaded' ? 'Online' : 'Offline', icon: Server, color: 'text-emerald-600 bg-emerald-50' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-stone-900 flex items-center gap-2">
          <ShieldCheck className="text-agri-600" />
          System Administration
        </h2>
        <p className="text-stone-500">Real-time system monitoring and control panel.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {metrics.map((item, i) => (
          <div key={i} className="bg-white p-4 rounded-xl border border-stone-200 shadow-sm flex items-center gap-4">
            <div className={`p-3 rounded-lg ${item.color}`}>
              <item.icon size={20} />
            </div>
            <div>
              <p className="text-xs text-stone-500 uppercase font-bold">{item.label}</p>
              <p className="text-xl font-bold text-stone-800">{item.val}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white rounded-2xl border border-stone-200 shadow-sm overflow-hidden">
          <div className="p-6 border-b border-stone-100">
            <h3 className="font-bold text-stone-800">Quick Actions</h3>
          </div>
          <div className="p-6 grid grid-cols-2 md:grid-cols-3 gap-4">
            <ActionButton icon={RotateCcw} label="Reload Models" onClick={() => handleAction('Reload')} />
            <ActionButton icon={Database} label="Reload Dataset" onClick={() => handleAction('Dataset')} />
            <ActionButton icon={RefreshCw} label="Refresh Data" onClick={() => handleAction('Refresh')} />
            <ActionButton icon={CloudRain} label="Sync Weather" onClick={() => handleAction('Weather')} />
            <ActionButton
              icon={Trash2}
              label="Erase All Data"
              danger
              onClick={() => handleAction('Reset')}
              loading={resetting}
            />
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-stone-200 shadow-sm overflow-hidden flex flex-col h-[400px]">
          <div className="p-6 border-b border-stone-100 bg-stone-50">
            <h3 className="font-bold text-stone-800">Live Activity</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {activities.map(log => (
              <div key={log.id} className="flex gap-3 text-sm p-3 rounded-lg bg-white border border-stone-100 shadow-sm">
                <div className={`w-2 h-2 mt-1.5 rounded-full flex-shrink-0 ${log.status === 'success' ? 'bg-green-500' : log.status === 'warning' ? 'bg-amber-500' : 'bg-red-500'
                  }`} />
                <div>
                  <p className="font-medium text-stone-800">{log.action}</p>
                  <p className="text-stone-500 text-xs">{log.details}</p>
                  <span className="text-[10px] text-stone-400">{new Date(log.timestamp).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const ActionButton = ({ icon: Icon, label, danger, onClick, loading }: any) => (
  <button
    onClick={onClick}
    disabled={loading}
    className={`flex flex-col items-center justify-center p-6 rounded-xl border transition-all ${danger
        ? 'border-red-100 bg-red-50 text-red-600 hover:bg-red-100'
        : 'border-stone-100 bg-stone-50 text-stone-600 hover:bg-agri-50 hover:border-agri-100 hover:text-agri-700'
      }`}
  >
    {loading ? <div className="animate-spin mb-2">⟳</div> : <Icon size={24} className="mb-2" />}
    <span className="font-medium text-sm">{label}</span>
  </button>
);

export default Admin;