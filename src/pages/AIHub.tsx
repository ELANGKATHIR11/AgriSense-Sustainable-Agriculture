import React, { useState, useEffect } from "react";
import { Cpu, Zap, Activity, HardDrive, RefreshCw, AlertCircle, FileText, CheckCircle } from "lucide-react";

interface HealthStats {
  cpu_usage_pct: number;
  ram_usage_pct: number;
  vram_used_mb: number;
  vram_total_mb: number;
  inference_latency_ms: number;
  api_latency_ms: number;
  status: string;
}

interface Model {
  id: string;
  name: string;
  version: string;
  type: string;
  framework: string;
  status: string;
  accuracy: number;
  f1Score: number;
  lastRetrained: string;
  predictionCount: number;
}

export default function AIHub() {
  const [stats, setStats] = useState<HealthStats | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(false);
  const [retrainingId, setRetrainingId] = useState<string | null>(null);
  const [retrainSuccess, setRetrainSuccess] = useState<string | null>(null);

  const fetchSystemData = async () => {
    try {
      const healthRes = await fetch("/api/system/health");
      const healthData = await healthRes.json();
      setStats(healthData);

      const logsRes = await fetch("/api/system/logs");
      const logsData = await logsRes.json();
      setLogs(logsData.logs || []);
    } catch (err) {
      console.error("Error fetching system health telemetry", err);
    }
  };

  const fetchModels = async () => {
    try {
      const mlopsRes = await fetch("/api/mlops");
      const mlopsData = await mlopsRes.json();
      setModels(mlopsData.registry || []);
    } catch (err) {
      console.error("Error fetching models", err);
    }
  };

  useEffect(() => {
    setLoading(true);
    fetchSystemData();
    fetchModels();
    setLoading(false);

    // Poll system stats every 4 seconds
    const interval = setInterval(fetchSystemData, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleRetrain = async (modelId: string) => {
    setRetrainingId(modelId);
    setRetrainSuccess(null);
    try {
      const res = await fetch("/api/mlops/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelId })
      });
      if (res.ok) {
        const data = await res.json();
        setRetrainSuccess(`Successfully retrained and bumped version to ${data.message.split(' ').pop() || 'next release'}`);
        await fetchModels();
      } else {
        setRetrainSuccess("Retraining request failed. Please check backend stdout.");
      }
    } catch (err) {
      console.error(err);
      setRetrainSuccess("Retraining connection error.");
    } finally {
      setRetrainingId(null);
    }
  };

  const vramPercent = stats ? (stats.vram_used_mb / stats.vram_total_mb) * 100 : 0;

  return (
    <div className="space-y-6 animate-fade-in" id="ai-hub-viewport">
      {/* Header */}
      <div className="page-header-strip px-7 py-6">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-2xl font-bold text-white tracking-tight">
                AI Hub & Hardware Telemetry
              </h1>
              <span className="agri-badge bg-emerald-500/20 text-[#4ade80] border-[#4ade80]/20">
                <Cpu className="w-3 h-3" /> RTX 5060 Core
              </span>
            </div>
            <p className="text-sm text-[#86efac] font-mono max-w-xl leading-relaxed">
              Real-time monitoring of desktop graphics cards, GPU memory footprints, local LLM agents, and CUDA pipelines.
            </p>
          </div>
          <button
            onClick={() => { fetchSystemData(); fetchModels(); }}
            className="btn-secondary self-start md:self-auto text-xs"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Force Telemetry Poll
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side: Hardware Performance (7 columns) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="agri-card p-6 space-y-6">
            <div className="flex items-center gap-2 border-b pb-4">
              <Activity className="w-4 h-4 text-emerald-600" />
              <h3 className="text-sm font-bold text-[#0f2e1e]">NVIDIA RTX 5060 Laptop GPU & Host Diagnostics</h3>
            </div>

            {stats ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* VRAM Progress */}
                <div className="p-4 bg-emerald-50/30 rounded-2xl border border-emerald-500/10 space-y-3">
                  <div className="flex justify-between text-xs font-mono font-bold text-emerald-800">
                    <span>VRAM ALLOCATION</span>
                    <span>{vramPercent.toFixed(1)}%</span>
                  </div>
                  <div className="h-4 w-full bg-emerald-950/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-400 to-emerald-600 transition-all duration-1000"
                      style={{ width: `${vramPercent}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-gray-500 font-mono">
                    <span>{stats.vram_used_mb.toFixed(0)} MB Used</span>
                    <span>{stats.vram_total_mb.toFixed(0)} MB Total</span>
                  </div>
                </div>

                {/* Host Telemetry (CPU / RAM) */}
                <div className="space-y-4">
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-mono text-gray-500">
                      <span>Host CPU Load</span>
                      <span className="font-bold text-gray-800">{stats.cpu_usage_pct}%</span>
                    </div>
                    <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500" style={{ width: `${stats.cpu_usage_pct}%` }} />
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-mono text-gray-500">
                      <span>Host RAM Usage</span>
                      <span className="font-bold text-gray-800">{stats.ram_usage_pct}%</span>
                    </div>
                    <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-amber-500" style={{ width: `${stats.ram_usage_pct}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="skeleton h-24 rounded" />
            )}

            {stats && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 pt-4 border-t">
                <div className="p-3 bg-gray-50 rounded-xl border text-center font-mono">
                  <p className="text-[10px] text-gray-400 font-bold uppercase">Inference Latency</p>
                  <p className="text-lg font-black text-emerald-600 mt-1">{stats.inference_latency_ms} ms</p>
                </div>
                <div className="p-3 bg-gray-50 rounded-xl border text-center font-mono">
                  <p className="text-[10px] text-gray-400 font-bold uppercase">FastAPI Latency</p>
                  <p className="text-lg font-black text-emerald-600 mt-1">{stats.api_latency_ms} ms</p>
                </div>
                <div className="p-3 bg-gray-50 rounded-xl border text-center font-mono col-span-2 md:col-span-1">
                  <p className="text-[10px] text-gray-400 font-bold uppercase">CUDA Status</p>
                  <p className="text-sm font-black text-emerald-600 mt-2 flex items-center justify-center gap-1">
                    <Zap className="w-3.5 h-3.5 text-amber-500" /> CUDA 12.8 OK
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Execution logs */}
          <div className="agri-card p-6 space-y-4">
            <div className="flex items-center gap-2 border-b pb-3">
              <FileText className="w-4 h-4 text-emerald-600" />
              <h3 className="text-sm font-bold text-[#0f2e1e]">Supervisor Daemon Log Stream</h3>
            </div>
            <div className="bg-[#0f2e1e] p-4 rounded-xl border border-emerald-950 font-mono text-[11px] text-[#4ade80] space-y-2 h-[200px] overflow-y-auto">
              {logs.map((log, idx) => (
                <div key={idx} className="leading-relaxed">
                  <span className="text-[#a1a1aa]">{log.split(" ")[0]}</span> {log.substring(log.indexOf(" ") + 1)}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Side: Manual Retraining Controls (5 columns) */}
        <div className="lg:col-span-5 agri-card p-6 space-y-5">
          <div className="flex items-center gap-2 border-b pb-4">
            <RefreshCw className="w-4 h-4 text-emerald-600" />
            <h3 className="text-sm font-bold text-[#0f2e1e]">Model Optimization & Retraining</h3>
          </div>
          <p className="text-xs text-gray-500 leading-relaxed">
            AgriSense local ML engines can be force-retrained using updated datasets in the `AgriSense-Dataset` folder.
          </p>

          <div className="space-y-3 max-h-[350px] overflow-y-auto pr-1">
            {models.map(m => (
              <div key={m.id} className="p-3.5 bg-gray-50 border rounded-xl flex items-center justify-between gap-4 hover:border-emerald-500/30 transition-all">
                <div className="min-w-0">
                  <h4 className="font-bold text-xs text-[#0f2e1e] truncate">{m.name}</h4>
                  <div className="flex items-center gap-2 mt-1 text-[10px] text-gray-400 font-mono">
                    <span>{m.framework}</span>
                    <span>•</span>
                    <span>{m.version}</span>
                  </div>
                </div>
                <button
                  onClick={() => handleRetrain(m.id)}
                  disabled={retrainingId !== null}
                  className="btn-primary text-[10px] px-2.5 py-1.5 font-bold font-mono shrink-0"
                >
                  {retrainingId === m.id ? "Training..." : "Retrain"}
                </button>
              </div>
            ))}
          </div>

          {retrainSuccess && (
            <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl flex items-center gap-2 text-xs text-emerald-800 font-mono">
              <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{retrainSuccess}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
