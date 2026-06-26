import React, { useState, useEffect } from "react";
import { Cpu, CpuIcon, RefreshCw, AlertTriangle, Terminal, Play, Square, HardDrive } from "lucide-react";

interface AIModel {
  id: string;
  name: string;
  framework: string;
  type: string;
  status: string;
  accuracy: number;
}

export default function LocalAIHub() {
  const [systemStats, setSystemStats] = useState({
    cpu_usage_pct: 0,
    ram_usage_pct: 0,
    vram_used_mb: 0,
    vram_total_mb: 8192,
    inference_latency_ms: 0,
    api_latency_ms: 0,
    status: "Offline"
  });

  const [models, setModels] = useState<AIModel[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchStats = async () => {
    try {
      const res = await fetch("/api/system/health");
      const data = await res.json();
      setSystemStats(data);
    } catch (e) {
      console.error("Failed to load health stats", e);
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch("/api/mlops/models");
      if (res.ok) {
        const data = await res.json();
        setModels(data.models || []);
      }
    } catch (e) {
      console.error("Failed to load model registry", e);
    }
  };

  const fetchLogs = async () => {
    try {
      const res = await fetch("/api/system/logs");
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (e) {
      console.error("Failed to load system logs", e);
    }
  };

  const handleToggleModel = async (modelId: string, currentStatus: string) => {
    setLoading(true);
    try {
      const newStatus = currentStatus === "active" ? "archived" : "active";
      await fetch(`/api/mlops/models/${modelId}/status?status=${newStatus}`, {
        method: "POST"
      });
      await fetchModels();
    } catch (e) {
      console.error("Failed to toggle model", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchModels();
    fetchLogs();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="page-header-strip px-6 py-6 md:py-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 text-white">
        <div>
          <span className="agri-badge bg-emerald-500/20 border-emerald-400 text-emerald-300 mb-2">Local Computing Node</span>
          <h1 className="text-2xl md:text-3xl font-black tracking-tight font-sans font-sans">Local AI Hub</h1>
          <p className="text-xs text-emerald-100/70 mt-1 font-mono">Observe offline-first inference models, VRAM allocation, and model registry lifecycle.</p>
        </div>
        <button
          onClick={() => {
            fetchStats();
            fetchModels();
            fetchLogs();
          }}
          className="btn-secondary text-[#1a5234] border-[#d1fae5] hover:bg-emerald-50 text-xs font-semibold cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh Dashboard
        </button>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold text-emerald-800/60 uppercase tracking-wider font-mono">CPU Usage</span>
            <p className="text-xl font-black text-emerald-950 font-mono mt-1">{systemStats.cpu_usage_pct}%</p>
          </div>
          <Cpu className="w-8 h-8 text-emerald-600/30" />
        </div>

        <div className="metric-card flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold text-emerald-800/60 uppercase tracking-wider font-mono">System Memory</span>
            <p className="text-xl font-black text-emerald-950 font-mono mt-1">{systemStats.ram_usage_pct}%</p>
          </div>
          <CpuIcon className="w-8 h-8 text-emerald-600/30" />
        </div>

        <div className="metric-card flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold text-emerald-800/60 uppercase tracking-wider font-mono">VRAM Allocation</span>
            <p className="text-lg font-black text-emerald-950 font-mono mt-1">
              {systemStats.vram_used_mb.toFixed(0)} / {systemStats.vram_total_mb.toFixed(0)} MB
            </p>
          </div>
          <HardDrive className="w-8 h-8 text-amber-600/30" />
        </div>

        <div className="metric-card flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold text-emerald-800/60 uppercase tracking-wider font-mono">Inference Latency</span>
            <p className="text-xl font-black text-emerald-950 font-mono mt-1">{systemStats.inference_latency_ms} ms</p>
          </div>
          <Cpu className="w-8 h-8 text-emerald-600/30" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Registry List */}
        <div className="lg:col-span-2 space-y-4">
          <div className="agri-card space-y-4">
            <div className="flex items-center justify-between border-b border-emerald-900/5 pb-3">
              <h3 className="text-sm font-bold text-[#0f2e1e] tracking-tight">Active Edge AI Models</h3>
              <span className="agri-badge">7 Models Indexed</span>
            </div>

            <div className="divide-y divide-emerald-900/5">
              {models.length > 0 ? (
                models.map((model) => (
                  <div key={model.id} className="py-3 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${model.status === "active" ? "bg-emerald-500 animate-pulse" : "bg-gray-400"}`} />
                        <h4 className="text-xs font-bold text-[#0f2e1e]">{model.name}</h4>
                        <span className="text-[9px] font-mono text-emerald-800/50">({model.framework})</span>
                      </div>
                      <p className="text-[10px] text-emerald-900/60 font-mono mt-1">Task Scope: {model.type}</p>
                    </div>
                    <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-end">
                      <span className="text-[10px] font-mono text-emerald-800 font-bold bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded">
                        Accuracy: {(model.accuracy * 100).toFixed(1)}%
                      </span>
                      <button
                        onClick={() => handleToggleModel(model.id, model.status)}
                        disabled={loading}
                        className={`text-[9px] font-bold px-3 py-1 rounded transition-colors cursor-pointer ${
                          model.status === "active"
                            ? "bg-red-50 text-red-800 border border-red-200 hover:bg-red-100"
                            : "bg-emerald-950 text-white hover:bg-emerald-900"
                        }`}
                      >
                        {model.status === "active" ? "Deactivate" : "Activate"}
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="py-8 text-center text-xs text-emerald-800/40">
                  <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-emerald-400" />
                  Loading local AI registry metadata...
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Real-time Logs Console */}
        <div className="space-y-4">
          <div className="agri-card bg-zinc-950 border-zinc-800 text-emerald-400 p-5 rounded-xl h-full flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
                <span className="text-xs font-bold font-mono flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-emerald-400" /> System Inference Console
                </span>
                <span className="text-[8px] font-bold font-mono uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5 rounded">
                  Live Feed
                </span>
              </div>
              <div className="font-mono text-[10px] space-y-2 text-emerald-300 overflow-y-auto max-h-[250px] scrollbar-hide">
                {logs.map((log, index) => (
                  <p key={index} className="leading-relaxed border-l-2 border-emerald-500/20 pl-2">
                    {log}
                  </p>
                ))}
              </div>
            </div>
            <div className="mt-6 pt-3 border-t border-zinc-800 flex justify-between items-center text-[9px] text-zinc-500 font-mono">
              <span>AGRISENSE DAEMON PID: 14820</span>
              <span>127.0.0.1:8000</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
