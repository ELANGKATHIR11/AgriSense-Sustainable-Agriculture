/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useCallback } from "react";
import {
  ScanLine, Upload, Microscope, Activity, Sparkles,
  CheckCircle2, AlertCircle, Leaf, Bug, Zap, RotateCcw, ImageIcon
} from "lucide-react";
import { DiseaseDetectionResult } from "../types";

type VisionMode = "disease" | "weed";

const MODE_CONFIG = {
  disease: {
    label: "Pathogen Detection",
    description: "Identify crop diseases, fungal infections, and foliar defects",
    endpoint: "/api/vision/disease",
    color: "text-red-600",
    bgColor: "bg-red-50",
    borderColor: "border-red-200",
    icon: <Microscope className="w-4 h-4" />,
    presets: [
      {
        name: "Tomato Late Blight",
        type: "disease" as VisionMode,
        svg: `<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'><rect width='80' height='80' fill='%231b2e1e' rx='8'/><ellipse cx='40' cy='40' rx='28' ry='22' fill='%23254d30'/><circle cx='32' cy='32' r='8' fill='%23523525' opacity='0.85'/><circle cx='50' cy='48' r='10' fill='%2342372d' opacity='0.8'/><circle cx='38' cy='44' r='5' fill='%235b2e1b' opacity='0.9'/><line x1='40' y1='18' x2='40' y2='62' stroke='%231a4a24' stroke-width='1.5'/><text x='6' y='74' fill='%234ade80' font-size='6' font-family='monospace' font-weight='bold'>LATE BLIGHT</text></svg>`
      },
      {
        name: "Powdery Mildew",
        type: "disease" as VisionMode,
        svg: `<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'><rect width='80' height='80' fill='%231b2e1e' rx='8'/><path d='M12 40 C12 20, 68 20, 68 40 C68 60, 12 60, 12 40Z' fill='%23254d30'/><circle cx='30' cy='38' r='12' fill='%23c2cfc5' opacity='0.6'/><circle cx='52' cy='42' r='14' fill='%23b7c0b9' opacity='0.55'/><circle cx='40' cy='35' r='8' fill='%23d0d8d2' opacity='0.5'/><text x='8' y='74' fill='%234ade80' font-size='6' font-family='monospace' font-weight='bold'>PWDRY MILDEW</text></svg>`
      },
      {
        name: "Healthy Leaf",
        type: "disease" as VisionMode,
        svg: `<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'><rect width='80' height='80' fill='%230f2113' rx='8'/><path d='M40,8 C56,32 58,56 40,72 C22,56 24,32 40,8Z' fill='%231a7f37'/><path d='M40,8 L40,72' stroke='%230f531d' stroke-width='2'/><path d='M40,30 L28,42' stroke='%230f531d' stroke-width='1'/><path d='M40,30 L52,42' stroke='%230f531d' stroke-width='1'/><text x='8' y='74' fill='%234ade80' font-size='6' font-family='monospace' font-weight='bold'>HEALTHY LEAF</text></svg>`
      }
    ]
  },
  weed: {
    label: "Weed Detection",
    description: "Identify invasive weed species and recommend management",
    endpoint: "/api/vision/weed",
    color: "text-orange-600",
    bgColor: "bg-orange-50",
    borderColor: "border-orange-200",
    icon: <Bug className="w-4 h-4" />,
    presets: [
      {
        name: "Broadleaf Pigweed",
        type: "weed" as VisionMode,
        svg: `<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'><rect width='80' height='80' fill='%23192224' rx='8'/><path d='M20,65 L40,15 L60,65 Z' fill='%23422a21'/><path d='M25,55 L55,55' stroke='%236f331d' stroke-width='1.5' opacity='0.7'/><circle cx='40' cy='32' r='6' fill='%236f331d' opacity='0.8'/><text x='8' y='76' fill='%23fb923c' font-size='6' font-family='monospace' font-weight='bold'>BROADLEAF WEED</text></svg>`
      },
      {
        name: "Crabgrass",
        type: "weed" as VisionMode,
        svg: `<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'><rect width='80' height='80' fill='%23192224' rx='8'/><path d='M40,70 C40,70 20,40 25,20' stroke='%23559044' stroke-width='3' fill='none'/><path d='M40,70 C40,70 55,38 50,15' stroke='%23559044' stroke-width='3' fill='none'/><path d='M40,70 C40,70 60,55 70,45' stroke='%23559044' stroke-width='3' fill='none'/><path d='M40,70 C40,70 18,58 10,46' stroke='%23559044' stroke-width='3' fill='none'/><text x='8' y='76' fill='%23fb923c' font-size='6' font-family='monospace' font-weight='bold'>CRABGRASS WEED</text></svg>`
      },
      {
        name: "Field Bindweed",
        type: "weed" as VisionMode,
        svg: `<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'><rect width='80' height='80' fill='%23192224' rx='8'/><path d='M10,70 C30,50 50,30 70,10' stroke='%23559044' stroke-width='2.5' fill='none'/><ellipse cx='28' cy='52' rx='10' ry='7' fill='%23355a28' opacity='0.85'/><ellipse cx='50' cy='30' rx='10' ry='7' fill='%23355a28' opacity='0.85'/><circle cx='28' cy='52' r='4' fill='%23f8fafc' opacity='0.6'/><circle cx='50' cy='30' r='4' fill='%23f8fafc' opacity='0.6'/><text x='4' y='76' fill='%23fb923c' font-size='6' font-family='monospace' font-weight='bold'>FIELD BINDWEED</text></svg>`
      }
    ]
  }
};

export default function DiseaseDetection() {
  const [mode, setMode] = useState<VisionMode>("disease");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageFileName, setImageFileName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiseaseDetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cfg = MODE_CONFIG[mode];

  const handleImageUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setResult(null);
    setImageFileName(file.name);
    const reader = new FileReader();
    reader.onloadend = () => setImagePreview(reader.result as string);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    setError(null);
    setResult(null);
    setImageFileName(file.name);
    const reader = new FileReader();
    reader.onloadend = () => setImagePreview(reader.result as string);
    reader.readAsDataURL(file);
  }, []);

  const triggerPreset = (preset: { name: string; type: VisionMode; svg: string }) => {
    setMode(preset.type);
    setImagePreview(`data:image/svg+xml;utf8,${preset.svg}`);
    setImageFileName(`${preset.name} (demo)`);
    setResult(null);
    setError(null);
  };

  const runInference = async () => {
    if (!imagePreview) {
      setError("Please upload or select a crop image first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(cfg.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageBase64: imagePreview, mode })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      // Normalize: vision endpoints return { success, results, confidence, recommendations }
      const results = data.results || data;
      setResult({
        disease: results.disease || results.weed || results.label || "Unknown",
        confidence: Math.round((results.confidence ?? data.confidence ?? 0.85) * 100),
        severity: results.severity || "medium",
        symptoms: results.symptoms || results.characteristics || [],
        recommendations: data.recommendations || results.recommendations || []
      });
    } catch (err: any) {
      setError(err.message || "Vision inference failed. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImagePreview(null);
    setImageFileName(null);
    setResult(null);
    setError(null);
  };

  const severityConfig = {
    low:    { label: "Low",    chip: "chip-green",  text: "Minimal threat detected" },
    medium: { label: "Medium", chip: "chip-amber",  text: "Monitor and treat soon" },
    high:   { label: "High",   chip: "chip-red",    text: "Immediate action required" }
  };
  const sev = result ? (severityConfig[result.severity as keyof typeof severityConfig] || severityConfig.medium) : null;

  return (
    <div className="space-y-6 animate-fade-in" id="disease-viewport">

      {/* Page Header */}
      <div className="page-header-strip p-6 text-white">
        <div className="relative z-10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="agri-badge"><span>📷</span> Vision Core</span>
              <span className="agri-badge agri-badge-amber"><span>⚡</span> Florence-2 3B</span>
            </div>
            <h1 className="text-2xl font-black tracking-tight">
              Computer Vision <span className="text-amber-400">Diagnostic Core</span>
            </h1>
            <p className="text-emerald-100/80 text-sm max-w-xl">
              Edge-native Florence-2 3B running on Ollama — detects crop pathogens and invasive weeds from foliar imagery in real time.
            </p>
          </div>

          {/* Mode Toggle */}
          <div className="mode-toggle flex-shrink-0">
            <button
              id="tab-mode-disease"
              onClick={() => { setMode("disease"); setResult(null); setError(null); }}
              className={`mode-toggle-btn ${mode === "disease" ? "active" : ""}`}
            >
              <span className="flex items-center gap-1.5">
                <Microscope className="w-3 h-3" /> Pathology
              </span>
            </button>
            <button
              id="tab-mode-weed"
              onClick={() => { setMode("weed"); setResult(null); setError(null); }}
              className={`mode-toggle-btn ${mode === "weed" ? "active" : ""}`}
            >
              <span className="flex items-center gap-1.5">
                <Bug className="w-3 h-3" /> Weed ID
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Mode description strip */}
      <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${cfg.bgColor} ${cfg.borderColor}`}>
        <span className={cfg.color}>{cfg.icon}</span>
        <div>
          <p className={`text-xs font-bold font-mono uppercase tracking-wider ${cfg.color}`}>{cfg.label}</p>
          <p className="text-xs text-gray-600 mt-0.5">{cfg.description}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Panel — Upload + Presets */}
        <div className="lg:col-span-5 space-y-5">
          {/* Upload Zone */}
          <div className="agri-card p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-gray-800">Sample Image Feed</h3>
              {imagePreview && (
                <button onClick={reset} className="btn-secondary px-2.5 py-1.5 text-xs gap-1.5">
                  <RotateCcw className="w-3 h-3" /> Reset
                </button>
              )}
            </div>

            <label
              className="group relative flex flex-col items-center justify-center border-2 border-dashed border-emerald-200 hover:border-emerald-400 rounded-xl p-6 cursor-pointer transition-all bg-[#f8fdf8] hover:bg-emerald-50/50 min-h-[200px]"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              <input
                id="file-pathology-upload"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
              />
              {imagePreview ? (
                <div className="space-y-3 text-center w-full">
                  <img
                    src={imagePreview}
                    alt="Agricultural sample"
                    className="max-h-44 mx-auto rounded-xl object-contain border border-emerald-100 shadow-sm"
                  />
                  {imageFileName && (
                    <p className="text-[10px] font-mono text-gray-400 truncate max-w-full px-2">{imageFileName}</p>
                  )}
                </div>
              ) : (
                <div className="space-y-3 text-center">
                  <div className="w-12 h-12 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center justify-center mx-auto group-hover:bg-emerald-100 transition-colors">
                    <Upload className="w-5 h-5 text-emerald-600" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-semibold text-gray-700">Drop image or click to upload</p>
                    <p className="text-xs text-gray-400">JPEG, PNG, WebP · up to 10 MB</p>
                  </div>
                </div>
              )}
            </label>

            <button
              id="btn-ml-inference-run"
              onClick={runInference}
              disabled={loading || !imagePreview}
              className="btn-primary w-full py-2.5"
            >
              {loading ? (
                <><Activity className="w-4 h-4 animate-spin" /> Analyzing with Florence-2...</>
              ) : (
                <><Sparkles className="w-4 h-4" /> Run Vision Inference</>
              )}
            </button>
          </div>

          {/* Demo Presets */}
          <div className="agri-card p-5 space-y-3">
            <div className="flex items-center gap-2">
              <ImageIcon className="w-3.5 h-3.5 text-gray-400" />
              <span className="text-xs font-bold text-gray-500 uppercase tracking-wider font-mono">Demo Samples</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {cfg.presets.map((preset, idx) => (
                <button
                  key={preset.name}
                  id={`preset-demo-${idx}`}
                  onClick={() => triggerPreset(preset)}
                  className="group relative p-2 rounded-xl border border-gray-100 hover:border-emerald-300 bg-gray-50 hover:bg-emerald-50/40 cursor-pointer transition-all text-center"
                >
                  <img
                    src={`data:image/svg+xml;utf8,${preset.svg}`}
                    alt={preset.name}
                    className="w-full h-16 object-cover rounded-lg mx-auto"
                  />
                  <p className="text-[10px] font-medium text-gray-600 mt-1.5 group-hover:text-emerald-700 leading-tight line-clamp-2">
                    {preset.name}
                  </p>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Panel — Results */}
        <div className="lg:col-span-7 space-y-5">
          {/* Error */}
          {error && (
            <div className="error-alert animate-fade-in">
              <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-bold text-sm">Inference Error</p>
                <p className="text-xs mt-0.5 opacity-80">{error}</p>
              </div>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="agri-card p-10 flex flex-col items-center justify-center min-h-[320px] space-y-4 animate-fade-in">
              <div className="relative">
                <div className="w-14 h-14 rounded-full border-4 border-emerald-100 border-t-emerald-500 animate-spin" />
                <ScanLine className="w-6 h-6 text-emerald-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
              </div>
              <div className="text-center space-y-1">
                <p className="text-xs font-bold font-mono text-emerald-700 uppercase tracking-widest animate-pulse">
                  Florence-2 Neural Pipeline Active
                </p>
                <p className="text-xs text-gray-400">Parsing chloroplast variance signatures...</p>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!loading && !result && !error && (
            <div className="empty-state animate-fade-in">
              <div className="w-16 h-16 rounded-2xl bg-emerald-50 border border-emerald-100 flex items-center justify-center">
                <Leaf className="w-8 h-8 text-emerald-300" />
              </div>
              <div className="space-y-1.5">
                <h4 className="text-sm font-bold text-gray-700">Awaiting Vision Ingestion</h4>
                <p className="text-xs text-gray-400 max-w-xs">
                  Upload a crop leaf image or select a demo sample on the left, then click Run Inference to trigger Florence-2 analysis.
                </p>
              </div>
            </div>
          )}

          {/* Results */}
          {!loading && result && (
            <div className="space-y-4 animate-slide-in">
              {/* Primary Result Card */}
              <div className="result-card-primary">
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
                  <div className="space-y-2">
                    <p className="text-[10px] font-mono font-bold text-gray-400 uppercase tracking-widest">
                      {mode === "disease" ? "Pathology Classification" : "Weed Species ID"}
                    </p>
                    <h2 className="text-2xl font-black text-gray-900">{result.disease}</h2>
                    {result.farmer_explanation && (
                      <p className="text-sm text-gray-600 leading-relaxed max-w-sm">{result.farmer_explanation}</p>
                    )}
                  </div>
                  <div className="flex flex-col sm:items-end gap-2 flex-shrink-0">
                    <div className="flex items-center gap-2">
                      <span className={`agri-chip ${sev?.chip || "chip-amber"}`}>
                        {sev?.label || "Medium"} severity
                      </span>
                    </div>
                    <div className="text-center">
                      <p className="text-3xl font-black text-emerald-600 font-mono leading-none">
                        {result.confidence}%
                      </p>
                      <p className="text-[9px] font-mono text-gray-400 uppercase mt-0.5">Confidence</p>
                    </div>
                    <div className="w-28 progress-bar-track">
                      <div className="progress-bar-fill" style={{ width: `${result.confidence}%` }} />
                    </div>
                  </div>
                </div>
                {sev && (
                  <div className="mt-4 pt-4 border-t border-emerald-100">
                    <div className="flex items-center gap-2">
                      <Zap className="w-3.5 h-3.5 text-emerald-600" />
                      <p className="text-xs font-mono text-emerald-800">{sev.text}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Symptoms + Recommendations Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Symptoms */}
                {result.symptoms && result.symptoms.length > 0 && (
                  <div className="agri-card p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
                      <h4 className="text-xs font-bold font-mono text-gray-500 uppercase tracking-wider">
                        {mode === "disease" ? "Visual Symptoms" : "Weed Characteristics"}
                      </h4>
                    </div>
                    <ul className="space-y-2">
                      {result.symptoms.map((sym: string, i: number) => (
                        <li key={i} className="flex items-start gap-2 text-xs text-gray-700 leading-relaxed">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0 mt-0.5" />
                          {sym}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommendations */}
                {result.recommendations && result.recommendations.length > 0 && (
                  <div className="agri-card p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-amber-500 flex-shrink-0" />
                      <h4 className="text-xs font-bold font-mono text-gray-500 uppercase tracking-wider">
                        Expert Recommendations
                      </h4>
                    </div>
                    <ul className="space-y-2">
                      {result.recommendations.map((rec: string, i: number) => (
                        <li key={i} className="flex items-start gap-2 text-xs text-gray-700 leading-relaxed">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 flex-shrink-0 mt-1.5" />
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Inference Trace */}
              <div className="trace-bar">
                <span>MODEL: Florence-2-3B (riven/florence-2) · MODE: {mode.toUpperCase()}</span>
                <span className="text-emerald-600 font-bold">LOCAL EDGE · OLLAMA</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
