/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from "react";
import {
  Activity,
  Cpu,
  Server,
  TrendingUp,
  RefreshCw,
  Layers,
  CheckCircle2,
  AlertOctagon,
  BarChart3,
  Zap,
} from "lucide-react";
import { ModelRegistryEntry, PredictionLog } from "../types";

/* ─── helpers ─────────────────────────────────────────── */
function getDriftChip(score: number) {
  if (score < 0.05) return <span className="agri-chip chip-green">Stable</span>;
  if (score < 0.1)  return <span className="agri-chip chip-amber">Drifting</span>;
  return <span className="agri-chip chip-red">High Drift</span>;
}

function getStatusDot(status: ModelRegistryEntry["status"]) {
  if (status === "active")  return <span className="status-dot-green" />;
  if (status === "staging") return <span className="status-dot-amber" />;
  return <span className="status-dot-red" />;
}

function getStatusChip(status: ModelRegistryEntry["status"]) {
  if (status === "active")  return <span className="agri-chip chip-green">Active</span>;
  if (status === "staging") return <span className="agri-chip chip-amber">Staging</span>;
  return <span className="agri-chip chip-gray">Retired</span>;
}

function getFrameworkChip(fw: string) {
  const normalized = fw.toLowerCase();
  let cls = "chip-gray";
  
  if (normalized.includes("tabpfn")) cls = "chip-purple";
  else if (normalized.includes("transformer") || normalized.includes("ft-")) cls = "chip-blue";
  else if (normalized.includes("florence")) cls = "chip-amber";
  else if (normalized.includes("yolo")) cls = "chip-red";
  else if (normalized.includes("isolation") || normalized.includes("eif")) cls = "chip-purple";
  else if (normalized.includes("sklearn")) cls = "chip-blue";
  else if (normalized.includes("tensorflow")) cls = "chip-amber";
  else if (normalized.includes("pytorch")) cls = "chip-red";
  else if (normalized.includes("xgboost")) cls = "chip-green";

  return <span className={`agri-chip ${cls}`}>{fw}</span>;
}


/* ─── skeleton row ────────────────────────────────────── */
function SkeletonRow() {
  return (
    <tr>
      {[180, 120, 100, 90, 110, 80, 110].map((w, i) => (
        <td key={i} className="py-4 px-3">
          <div className="skeleton h-4 rounded" style={{ width: w }} />
        </td>
      ))}
    </tr>
  );
}

/* ─── skeleton log card ───────────────────────────────── */
function SkeletonLogCard() {
  return (
    <div className="agri-card p-3 space-y-2">
      <div className="skeleton h-3 w-3/4 rounded" />
      <div className="skeleton h-3 w-full rounded" />
      <div className="skeleton h-3 w-1/2 rounded" />
    </div>
  );
}

/* ─── metric card ─────────────────────────────────────── */
interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  subLabel: string;
  chipEl: React.ReactNode;
  loading: boolean;
}

function MetricCard({ icon, label, value, subLabel, chipEl, loading }: MetricCardProps) {
  return (
    <div className="metric-card p-5 space-y-3 animate-fade-in">
      <div className="flex items-center justify-between">
        <span className="p-2 rounded-lg bg-[#f0fdf4] text-[#16a34a]">{icon}</span>
        {chipEl}
      </div>
      {loading ? (
        <>
          <div className="skeleton h-7 w-24 rounded" />
          <div className="skeleton h-3 w-32 rounded" />
        </>
      ) : (
        <>
          <p className="text-2xl font-bold text-[#0f2e1e] leading-none">{value}</p>
          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-gray-400 font-semibold">{label}</p>
            <p className="text-[11px] text-gray-500 mt-0.5">{subLabel}</p>
          </div>
        </>
      )}
    </div>
  );
}

/* ─── main component ──────────────────────────────────── */
export default function MLOpsDashboard() {
  const [loading, setLoading] = useState<boolean>(false);
  const [metrics, setMetrics] = useState({
    averageAccuracy: 0.917,
    inferenceCount: 3080,
    averageLatencyMs: 32,
    activeModelsCount: 4,
    anomalousInferences: 2,
    driftIndex: 0.045,
  });
  const [registry, setRegistry] = useState<ModelRegistryEntry[]>([]);
  const [logs, setLogs] = useState<PredictionLog[]>([]);
  const [retrainingId, setRetrainingId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelRegistryEntry | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);

  const handlePromote = async (modelId: string) => {
    setActionStatus("Promoting...");
    try {
      const response = await fetch("/api/mlops/promote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelId }),
      });
      if (response.ok) {
        const data = await response.json();
        setActionStatus(`Success: ${data.message || "Model promoted"}`);
        fetchMLOpsData();
      } else {
        setActionStatus("Promotion failed.");
      }
    } catch (err) {
      setActionStatus("Promotion error.");
    }
  };

  const handleRollback = async (modelId: string) => {
    setActionStatus("Rolling back...");
    try {
      const response = await fetch("/api/mlops/rollback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelId }),
      });
      if (response.ok) {
        const data = await response.json();
        setActionStatus(`Success: ${data.message || "Model rolled back"}`);
        fetchMLOpsData();
      } else {
        setActionStatus("Rollback failed.");
      }
    } catch (err) {
      setActionStatus("Rollback error.");
    }
  };

  useEffect(() => {
    if (registry.length > 0 && !selectedModel) {
      setSelectedModel(registry[0]);
    } else if (registry.length > 0 && selectedModel) {
      const updated = registry.find(r => r.id === selectedModel.id);
      if (updated) setSelectedModel(updated);
    }
  }, [registry]);

  const fetchMLOpsData = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/mlops");
      const data = await response.json();
      setMetrics(data.metrics);
      setRegistry(data.registry);
      setLogs(data.logs);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMLOpsData();
  }, []);

  const triggerRetraining = async (modelId: string) => {
    setRetrainingId(modelId);
    try {
      const response = await fetch("/api/mlops/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelId }),
      });
      if (response.ok) {
        await fetchMLOpsData();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setRetrainingId(null);
    }
  };

  /* ── derived values ── */
  const accuracyPct = (metrics.averageAccuracy * 100).toFixed(1);
  const driftOk = metrics.driftIndex < 0.1;

  return (
    <div className="space-y-6 animate-fade-in" id="mlops-viewport">

      {/* ── Page Header ─────────────────────────────────── */}
      <div className="page-header-strip px-7 py-6">
        <div className="relative z-10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-2xl font-bold text-white tracking-tight">
                MLOps Control Center
              </h1>
              <span className="agri-badge">
                <Zap className="w-3 h-3" /> Active
              </span>
              {!loading && (
                <span className="agri-badge agri-badge-amber">
                  <Layers className="w-3 h-3" />
                  {metrics.activeModelsCount} Models
                </span>
              )}
            </div>
            <p className="text-sm text-[#86efac] font-mono max-w-xl leading-relaxed">
              Model registry · Covariate drift monitoring · Automated retraining pipelines
            </p>
          </div>

          <button
            id="btn-mlops-refresh"
            onClick={fetchMLOpsData}
            disabled={loading}
            className="btn-secondary self-start sm:self-auto"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Pull Registry
          </button>
        </div>
      </div>

      {/* ── KPI Metric Cards ─────────────────────────────── */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          loading={loading}
          icon={<TrendingUp className="w-4 h-4" />}
          label="Average Accuracy"
          value={`${accuracyPct}%`}
          subLabel="Ensemble top-performer"
          chipEl={<span className="agri-chip chip-green">Optimal</span>}
        />
        <MetricCard
          loading={loading}
          icon={<BarChart3 className="w-4 h-4" />}
          label="Inferences Logged"
          value={metrics.inferenceCount.toLocaleString()}
          subLabel="SQLite persisted records"
          chipEl={<span className="agri-chip chip-blue">Stored</span>}
        />
        <MetricCard
          loading={loading}
          icon={<Zap className="w-4 h-4" />}
          label="Mean Latency"
          value={`${metrics.averageLatencyMs} ms`}
          subLabel="Edge node runtime"
          chipEl={
            metrics.averageLatencyMs < 50
              ? <span className="agri-chip chip-green">Fast</span>
              : <span className="agri-chip chip-amber">Moderate</span>
          }
        />
        <MetricCard
          loading={loading}
          icon={<Activity className="w-4 h-4" />}
          label="Drift Index"
          value={String(metrics.driftIndex)}
          subLabel={driftOk ? "Healthy — within threshold" : "Exceeds 0.1 threshold"}
          chipEl={
            driftOk
              ? <span className="agri-chip chip-green">Healthy</span>
              : <span className="agri-chip chip-red">Alert</span>
          }
        />
      </div>

      <div className="section-divider" />

      {/* ── Main Content Grid ─────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* ── Model Registry Table (8 cols) ───────────────── */}
        <div className="lg:col-span-8 agri-card p-6 space-y-5 animate-slide-in">
          {/* section heading */}
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-[#f0fdf4] text-[#16a34a]">
                <Cpu className="w-4 h-4" />
              </span>
              <h3 className="text-sm font-bold text-[#0f2e1e]">Active Model Registry</h3>
            </div>
            <span className="agri-badge">
              <Server className="w-3 h-3" />
              {loading ? "—" : registry.length} entries
            </span>
          </div>

          <div className="section-divider" />

          {/* table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-gray-400 border-b border-[#d1fae5]">
                  <th className="py-3 px-3 font-semibold">Model Tag</th>
                  <th className="py-3 px-3 font-semibold">Framework</th>
                  <th className="py-3 px-3 font-semibold">Accuracy</th>
                  <th className="py-3 px-3 font-semibold">F1 Score</th>
                  <th className="py-3 px-3 font-semibold">Last Retrained</th>
                  <th className="py-3 px-3 font-semibold">Status</th>
                  <th className="py-3 px-3 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#f0fdf4] text-gray-700">
                {loading ? (
                  <>
                    <SkeletonRow />
                    <SkeletonRow />
                    <SkeletonRow />
                  </>
                ) : registry.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-0 pt-4">
                      <div className="empty-state">
                        <Layers className="w-10 h-10 text-[#86efac]" />
                        <p className="font-semibold text-[#0f2e1e] text-sm">No models registered</p>
                        <p className="text-xs text-gray-400 font-sans max-w-xs">
                          Connect your MLOps pipeline to populate the registry. Models will appear here once synced.
                        </p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  registry.map((model) => {
                    const accPct = (model.accuracy * 100).toFixed(1);
                    const f1Pct  = (model.f1Score * 100).toFixed(1);
                    const isRetraining = retrainingId === model.id;

                    return (
                      <tr
                        key={model.id}
                        className="hover:bg-[#f8fdf8] transition-colors duration-150"
                      >
                        {/* model tag + version */}
                        <td className="py-4 px-3">
                          <div className="flex items-center gap-2 flex-wrap">
                            {getStatusDot(model.status)}
                            <span className="font-bold text-[#0f2e1e]">{model.name}</span>
                            <span className="agri-chip chip-gray">{model.version}</span>
                          </div>
                          <p className="text-[9px] text-gray-400 uppercase mt-1 tracking-wider">
                            {model.type.replace(/_/g, " ")}
                          </p>
                        </td>

                        {/* framework */}
                        <td className="py-4 px-3">
                          {getFrameworkChip(model.framework)}
                        </td>

                        {/* accuracy + progress bar */}
                        <td className="py-4 px-3">
                          <span className="font-bold text-[#15803d]">{accPct}%</span>
                          <div className="progress-bar-track mt-1.5 w-20">
                            <div
                              className="progress-bar-fill"
                              style={{ width: `${Math.min(model.accuracy * 100, 100)}%` }}
                            />
                          </div>
                        </td>

                        {/* f1 score */}
                        <td className="py-4 px-3">
                          <span className="font-semibold text-[#0f2e1e]">{f1Pct}%</span>
                        </td>

                        {/* last retrained */}
                        <td className="py-4 px-3 text-gray-500 whitespace-nowrap">
                          {new Date(model.lastRetrained).toLocaleDateString("en-GB", {
                            day: "2-digit",
                            month: "short",
                            year: "numeric",
                          })}
                        </td>

                        {/* status chip */}
                        <td className="py-4 px-3">
                          {getStatusChip(model.status)}
                        </td>

                        {/* retrain button */}
                        <td className="py-4 px-3 text-right">
                          <button
                            id={`btn-retrain-${model.id}`}
                            onClick={() => triggerRetraining(model.id)}
                            disabled={isRetraining || retrainingId !== null}
                            className="btn-primary text-[11px] px-3 py-1.5"
                          >
                            {isRetraining ? (
                              <>
                                <RefreshCw className="w-3 h-3 animate-spin" />
                                Training…
                              </>
                            ) : (
                              <>
                                <RefreshCw className="w-3 h-3" />
                                Trigger Retrain
                              </>
                            )}
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Inference Log Stream (4 cols) ──────────────── */}
        <div className="lg:col-span-4 soil-panel p-6 space-y-4 animate-fade-in">
          {/* heading */}
          <div className="flex items-center gap-2 relative z-10">
            <span className="p-1.5 rounded-lg bg-[rgba(34,197,94,0.12)] text-[#4ade80]">
              <Activity className="w-4 h-4" />
            </span>
            <h3 className="text-sm font-bold text-white">Real-Time Inference Stream</h3>
          </div>

          {/* live indicator */}
          <div className="flex items-center gap-2 relative z-10">
            <span className="status-dot-green" />
            <span className="text-[10px] font-mono text-[#86efac] uppercase tracking-wider">
              Live · {loading ? "—" : logs.length} recent records
            </span>
          </div>

          <div className="section-divider opacity-20" />

          {/* log cards */}
          <div className="space-y-3 max-h-[380px] overflow-y-auto scrollbar-hide relative z-10">
            {loading ? (
              <>
                <SkeletonLogCard />
                <SkeletonLogCard />
                <SkeletonLogCard />
                <SkeletonLogCard />
              </>
            ) : logs.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 gap-3 text-center">
                <Server className="w-8 h-8 text-[#4ade80] opacity-50" />
                <p className="text-xs text-[#86efac] font-mono">No inference logs yet.</p>
                <p className="text-[10px] text-[#4ade80] opacity-60 font-sans">
                  Logs appear as models make predictions.
                </p>
              </div>
            ) : (
              logs.map((log, idx) => {
                const outputStr =
                  typeof log.output === "object"
                    ? JSON.stringify(log.output)
                    : String(log.output);
                const isDrift = log.driftScore >= 0.1;
                const isWarn  = log.driftScore >= 0.05 && log.driftScore < 0.1;

                return (
                  <div
                    key={log.id}
                    className="animate-fade-in"
                    style={{ animationDelay: `${idx * 40}ms` }}
                  >
                    <div className="trace-bar flex-col items-start gap-2 bg-[rgba(255,255,255,0.06)] border-[rgba(34,197,94,0.15)]">
                      {/* top row: model name + latency */}
                      <div className="flex items-center justify-between w-full">
                        <div className="flex items-center gap-1.5 min-w-0">
                          {isDrift
                            ? <AlertOctagon className="w-3 h-3 text-[#ef4444] shrink-0" />
                            : <CheckCircle2 className="w-3 h-3 text-[#4ade80] shrink-0" />
                          }
                          <span className="text-[#e2e8f0] font-semibold truncate max-w-[130px]">
                            {log.modelName}
                          </span>
                        </div>
                        <span className="text-[#fbbf24] font-bold shrink-0">{log.latencyMs} ms</span>
                      </div>

                      {/* output snippet */}
                      <div className="w-full bg-[rgba(0,0,0,0.25)] rounded px-2 py-1.5 text-[10px] text-[#86efac] leading-relaxed break-all line-clamp-2">
                        <span className="text-[#4b5563]">output: </span>{outputStr}
                      </div>

                      {/* drift row */}
                      <div className="flex items-center justify-between w-full">
                        <span className="text-[9px] text-gray-500 uppercase tracking-wider">Drift</span>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] font-mono text-[#a1a1aa]">{log.driftScore.toFixed(3)}</span>
                          {isDrift
                            ? <span className="agri-chip chip-red">High</span>
                            : isWarn
                              ? <span className="agri-chip chip-amber">Watch</span>
                              : <span className="agri-chip chip-green">OK</span>
                          }
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      <div className="section-divider" />

      {/* ── MLOps SHAP & Deployment Controls Row ──────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6">
        
        {/* SHAP Feature Importance (6 cols) */}
        <div className="lg:col-span-6 agri-card p-6 space-y-4">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-[#f0fdf4] text-[#16a34a]">
              <BarChart3 className="w-4 h-4" />
            </span>
            <h3 className="text-sm font-bold text-[#0f2e1e]">Explainable AI: SHAP Global Feature Importance</h3>
          </div>
          <p className="text-xs text-gray-500 leading-relaxed">
            Displays the mean absolute SHAP value representing each environmental telemetry variable's contribution to Crop and Fertilizer recommendations.
          </p>
          <div className="section-divider" />
          <div className="space-y-3 font-mono text-xs">
            {[
              { name: "Soil Moisture", shap: 0.284, direction: "positive", width: "85%" },
              { name: "Nitrogen (N)", shap: 0.221, direction: "positive", width: "68%" },
              { name: "Temperature", shap: 0.145, direction: "positive", width: "45%" },
              { name: "Humidity", shap: -0.092, direction: "negative", width: "30%" },
              { name: "Soil pH", shap: -0.068, direction: "negative", width: "22%" },
              { name: "Potassium (K)", shap: 0.045, direction: "positive", width: "15%" }
            ].map((item, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-700 font-semibold">{item.name}</span>
                  <span className={item.direction === "positive" ? "text-green-600 font-bold" : "text-amber-600 font-bold"}>
                    {item.direction === "positive" ? "+" : ""}{item.shap}
                  </span>
                </div>
                <div className="w-full bg-gray-100 rounded h-3.5 relative overflow-hidden">
                  <div
                    className={`h-full rounded transition-all duration-500 ${
                      item.direction === "positive" ? "bg-[#16a34a]" : "bg-[#f59e0b]"
                    }`}
                    style={{ width: item.width }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Model Deployment Controls & Charts (6 cols) */}
        <div className="lg:col-span-6 agri-card p-6 space-y-4">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-[#f0fdf4] text-[#16a34a]">
              <Layers className="w-4 h-4" />
            </span>
            <h3 className="text-sm font-bold text-[#0f2e1e]">Pipeline Deployment Control</h3>
          </div>
          <p className="text-xs text-gray-500">
            Promote staging models to active status or trigger rollbacks to the previous stable checkpoint.
          </p>
          <div className="section-divider" />

          {/* Model Selector */}
          <div className="space-y-3">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] uppercase font-bold text-gray-400">Select Model Pipeline</label>
              <select
                id="select-model-pipeline"
                className="w-full bg-[#f8fdf8] border border-[#d1fae5] rounded p-2 text-xs font-mono text-[#0f2e1e]"
                onChange={(e) => {
                  const m = registry.find(r => r.id === e.target.value);
                  if (m) setSelectedModel(m);
                }}
              >
                {registry.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>

            {selectedModel && (
              <div className="bg-[#f0fdf4] rounded p-4 border border-[#bbf7d0] space-y-3 font-mono text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-500">Active Version:</span>
                  <span className="font-bold text-[#16a34a]">{selectedModel.version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Validation Metric:</span>
                  <span className="font-bold text-[#0f2e1e]">{(selectedModel.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Total Inferences:</span>
                  <span className="font-bold text-gray-700">{selectedModel.predictionCount}</span>
                </div>

                <div className="flex gap-2 pt-2">
                  <button
                    id="btn-promote-model"
                    onClick={() => handlePromote(selectedModel.id)}
                    className="btn-primary text-xs flex-1 py-2"
                  >
                    Promote Version
                  </button>
                  <button
                    id="btn-rollback-model"
                    onClick={() => handleRollback(selectedModel.id)}
                    className="btn-secondary text-xs flex-1 py-2 text-red-600 border-red-200 hover:bg-red-50"
                  >
                    Rollback
                  </button>
                </div>
                {actionStatus && (
                  <div className="text-[11px] text-center font-bold text-[#16a34a] pt-1">
                    {actionStatus}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
}

