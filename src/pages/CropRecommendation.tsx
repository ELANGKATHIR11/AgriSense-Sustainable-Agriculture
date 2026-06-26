/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import {
  Sprout,
  Sliders,
  Sparkles,
  RefreshCw,
  CheckCircle2,
  HelpCircle,
  Lightbulb,
  BarChart3,
} from "lucide-react";
import { CropRecommendationResult, CropRecommendationInput } from "../types";
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ALL_CROPS } from "../config/crops";

/* ─── Suitability color helpers ────────────────────────────────────────── */
function getSuitabilityColor(pct: number): string {
  if (pct >= 80) return "#22c55e"; // emerald
  if (pct >= 60) return "#f59e0b"; // amber
  return "#ef4444";               // red
}
function getSuitabilityLabel(pct: number): string {
  if (pct >= 80) return "Excellent";
  if (pct >= 60) return "Good";
  if (pct >= 40) return "Moderate";
  return "Low";
}
function getSuitabilityChipClass(pct: number): string {
  if (pct >= 80) return "agri-chip chip-green";
  if (pct >= 60) return "agri-chip chip-amber";
  return "agri-chip chip-red";
}

/* ─── SVG ring progress component ──────────────────────────────────────── */
function SuitabilityRing({ pct }: { pct: number }) {
  const r = 26;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  const color = getSuitabilityColor(pct);
  return (
    <svg width="68" height="68" viewBox="0 0 68 68" className="shrink-0">
      <circle cx="34" cy="34" r={r} fill="none" stroke="#e5e7eb" strokeWidth="6" />
      <circle
        cx="34" cy="34" r={r}
        fill="none"
        stroke={color}
        strokeWidth="6"
        strokeDasharray={`${dash} ${circ - dash}`}
        strokeDashoffset={circ / 4}
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.7s ease" }}
      />
      <text x="34" y="38" textAnchor="middle" fontSize="11" fontWeight="700"
        fill={color} fontFamily="JetBrains Mono, monospace">
        {pct}%
      </text>
    </svg>
  );
}

/* ─── Main Component ────────────────────────────────────────────────────── */
export default function CropRecommendation() {
  const [inputs, setInputs] = useState<CropRecommendationInput>({
    nitrogen: 65,
    phosphorus: 40,
    potassium: 42,
    pH: 6.4,
    temperature: 28,
    humidity: 60,
    rainfall: 110,
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CropRecommendationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const presets = [
    {
      name: "Wet Rice Field",
      icon: "🌾",
      data: { nitrogen: 85, phosphorus: 42, potassium: 38, pH: 6.1, temperature: 27, humidity: 75, rainfall: 180 },
    },
    {
      name: "Dry Pulse/Chickpea",
      icon: "🫘",
      data: { nitrogen: 22, phosphorus: 65, potassium: 32, pH: 7.2, temperature: 21, humidity: 40, rainfall: 45 },
    },
    {
      name: "High-Nutrient Horticulture",
      icon: "🥬",
      data: { nitrogen: 55, phosphorus: 35, potassium: 85, pH: 6.5, temperature: 29, humidity: 55, rainfall: 65 },
    },
  ];

  const handleInputChange = (field: keyof CropRecommendationInput, val: number) => {
    setInputs((prev) => ({ ...prev, [field]: val }));
  };

  const loadPreset = (data: (typeof presets)[0]["data"]) => {
    setInputs(data);
    setResult(null);
    setError(null);
  };

  const calculateRecommendation = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/crop-recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          N: inputs.nitrogen,
          P: inputs.phosphorus,
          K: inputs.potassium,
          temperature: inputs.temperature,
          humidity: inputs.humidity,
          ph: inputs.pH,
          rainfall: inputs.rainfall,
        }),
      });
      if (!response.ok) {
        throw new Error("Agri-Engine server rejected inputs mapping.");
      }
      const data: CropRecommendationResult = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to parse recommendations.");
    } finally {
      setLoading(false);
    }
  };

  /* ── Slider row helper ─────────────────────────────────────────────── */
  const SliderRow = ({
    id,
    label,
    unit,
    field,
    min,
    max,
    step,
    color,
  }: {
    id: string;
    label: string;
    unit: string;
    field: keyof CropRecommendationInput;
    min: number;
    max: number;
    step: number;
    color: string;
  }) => {
    const val = inputs[field] as number;
    const pct = ((val - min) / (max - min)) * 100;
    return (
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono text-[#4a6741] font-medium">{label}</span>
          <span
            className="text-xs font-mono font-bold px-2 py-0.5 rounded-lg"
            style={{ background: `${color}18`, color }}
          >
            {val} {unit}
          </span>
        </div>
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={val}
          onChange={(e) => handleInputChange(field, parseFloat(e.target.value))}
          className="agri-range w-full"
          style={{ accentColor: color }}
        />
      </div>
    );
  };

  return (
    <div className="space-y-0 animate-fade-in" id="crop-viewport">

      {/* ── Page Header Strip ─────────────────────────────────────────── */}
      <div className="page-header-strip p-6">
        <div className="relative z-10 flex flex-col gap-1">
          <div className="flex items-center gap-2.5 flex-wrap">
            <span className="agri-badge">
              <Sprout className="w-3 h-3" />
              Suitability Engine
            </span>
            <span className="text-emerald-300/80 font-mono text-[10px] uppercase tracking-widest hidden sm:block">
              NPK · pH · Climate Vectors
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-white mt-0.5">
            Crop Suitability AI
          </h1>
          <p className="text-emerald-100/75 text-xs sm:text-sm max-w-2xl leading-relaxed">
            TabPFN + LightGBM ensemble classifier evaluates soil chemistry and
            climate vectors to rank optimal crop candidates with suitability indices.
          </p>
        </div>
        <div className="hidden lg:flex items-center gap-3 mt-1">
          <div className="glass-agri-dark px-4 py-2 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-emerald-300" />
            <div className="text-right">
              <div className="text-[10px] text-emerald-200/90 font-mono uppercase">Models Active</div>
              <div className="text-sm font-bold text-white font-mono">2 / 2</div>
            </div>
          </div>
          <div className="glass-agri-dark px-4 py-2 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-amber-300" />
            <div className="text-right">
              <div className="text-[10px] text-emerald-200/90 font-mono uppercase">Features</div>
              <div className="text-sm font-bold text-white font-mono">7 Params</div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Preset Chips ──────────────────────────────────────────────── */}
      <div className="px-4 pt-4 pb-0 sm:px-0">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[10px] font-bold text-[#6b8f5e] uppercase tracking-widest font-mono shrink-0">
            Soil Profiles:
          </span>
          {presets.map((p) => (
            <button
              key={p.name}
              id={`preset-soil-${p.name.replace(/\s+/g, "-").toLowerCase()}`}
              onClick={() => loadPreset(p.data)}
              className="agri-chip chip-green hover:scale-105 transition-transform cursor-pointer"
            >
              <span>{p.icon}</span>
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* ── Two-column Grid ───────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pt-5">

        {/* ── LEFT PANEL: Inputs ─────────────────────────────────── lg:col-5 */}
        <div className="lg:col-span-5">
          <div className="agri-card h-full flex flex-col gap-5">
            {/* Panel heading */}
            <div className="flex items-center gap-2 border-b border-[#ddecd8] pb-3">
              <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center shrink-0">
                <Sliders className="w-4 h-4 text-emerald-700" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-[#0f2e1e]">Soil &amp; Climate Vectors</h3>
                <p className="text-[10px] text-[#7a9e70] font-mono">Adjust NPK, pH, and environmental factors</p>
              </div>
            </div>

            {/* NPK Sliders */}
            <div className="space-y-4">
              <p className="text-[10px] font-mono font-semibold text-[#6b8f5e] uppercase tracking-wider">
                Macronutrient Ratios
              </p>
              <SliderRow
                id="input-crop-nitrogen"
                label="Nitrogen [N]"
                unit="ppm"
                field="nitrogen"
                min={0} max={150} step={1}
                color="#22c55e"
              />
              <SliderRow
                id="input-crop-phosphorus"
                label="Phosphorus [P]"
                unit="ppm"
                field="phosphorus"
                min={0} max={120} step={1}
                color="#f59e0b"
              />
              <SliderRow
                id="input-crop-potassium"
                label="Potassium [K]"
                unit="ppm"
                field="potassium"
                min={0} max={120} step={1}
                color="#6366f1"
              />
            </div>

            <div className="section-divider" />

            {/* Numeric inputs grid */}
            <div className="space-y-3">
              <p className="text-[10px] font-mono font-semibold text-[#6b8f5e] uppercase tracking-wider">
                Environmental Parameters
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label htmlFor="input-crop-ph" className="text-[10px] font-mono text-[#6b8f5e] font-semibold uppercase tracking-wide">
                    Soil pH
                  </label>
                  <input
                    id="input-crop-ph"
                    type="number"
                    min={3.5} max={9.0} step={0.1}
                    value={inputs.pH}
                    onChange={(e) => handleInputChange("pH", parseFloat(e.target.value) || 6.4)}
                    className="agri-input w-full"
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="input-crop-temp" className="text-[10px] font-mono text-[#6b8f5e] font-semibold uppercase tracking-wide">
                    Temperature (°C)
                  </label>
                  <input
                    id="input-crop-temp"
                    type="number"
                    min={5} max={50} step={0.1}
                    value={inputs.temperature}
                    onChange={(e) => handleInputChange("temperature", parseFloat(e.target.value) || 28)}
                    className="agri-input w-full"
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="input-crop-humidity" className="text-[10px] font-mono text-[#6b8f5e] font-semibold uppercase tracking-wide">
                    Humidity (%)
                  </label>
                  <input
                    id="input-crop-humidity"
                    type="number"
                    min={10} max={100} step={1}
                    value={inputs.humidity}
                    onChange={(e) => handleInputChange("humidity", parseInt(e.target.value) || 60)}
                    className="agri-input w-full"
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="input-crop-rainfall" className="text-[10px] font-mono text-[#6b8f5e] font-semibold uppercase tracking-wide">
                    Rainfall (mm)
                  </label>
                  <input
                    id="input-crop-rainfall"
                    type="number"
                    min={10} max={350} step={1}
                    value={inputs.rainfall}
                    onChange={(e) => handleInputChange("rainfall", parseInt(e.target.value) || 110)}
                    className="agri-input w-full"
                  />
                </div>
              </div>
            </div>

            {/* Current input summary badges */}
            <div className="flex flex-wrap gap-1.5 p-3 rounded-xl bg-[#f0f8ee] border border-[#d5e8ce]">
              {[
                { label: "N", val: inputs.nitrogen, unit: "" },
                { label: "P", val: inputs.phosphorus, unit: "" },
                { label: "K", val: inputs.potassium, unit: "" },
                { label: "pH", val: inputs.pH, unit: "" },
                { label: "T", val: inputs.temperature, unit: "°C" },
                { label: "H", val: inputs.humidity, unit: "%" },
                { label: "R", val: inputs.rainfall, unit: "mm" },
              ].map((item) => (
                <span key={item.label} className="text-[10px] font-mono bg-white border border-[#c5ddc0] text-[#2d5a27] px-2 py-0.5 rounded-md font-semibold">
                  {item.label}: {item.val}{item.unit}
                </span>
              ))}
            </div>

            {/* Submit button */}
            <button
              id="btn-run-crop-recommend"
              onClick={calculateRecommendation}
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2 mt-auto"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Evaluating Ensembles…
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Calculate Top Crop Suitability
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── RIGHT PANEL: Results ───────────────────────────────── lg:col-7 */}
        <div className="lg:col-span-7 flex flex-col gap-5">

          {/* Loading state */}
          {loading && (
            <div className="agri-card flex flex-col items-center justify-center gap-5 min-h-[380px] animate-fade-in">
              <div className="relative">
                <div className="w-16 h-16 rounded-full border-4 border-emerald-100 border-t-emerald-500 animate-spin" />
                <Sprout className="w-6 h-6 text-emerald-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
              </div>
              <div className="text-center space-y-1">
                <p className="text-xs font-mono text-emerald-700 font-bold uppercase tracking-widest animate-pulse">
                  Running Tree Ensembles…
                </p>
                <p className="text-[10px] text-[#7a9e70] font-mono">
                  TabPFN · LightGBM · Weighted Voting
                </p>
              </div>
            </div>
          )}

          {/* Error state */}
          {!loading && error && (
            <div className="error-alert animate-fade-in">
              <div className="font-semibold text-sm">Ensemble Error</div>
              <div className="text-xs mt-0.5 opacity-80">{error}</div>
            </div>
          )}

          {/* Empty state */}
          {!loading && !result && !error && (
            <div className="empty-state min-h-[380px] animate-fade-in">
              <HelpCircle className="w-10 h-10 text-[#b5cdb0]" />
              <h4 className="text-sm font-semibold text-[#4a6741]">Awaiting Soil Analysis</h4>
              <p className="text-xs text-[#8aac82] max-w-xs text-center leading-relaxed">
                Configure soil chemistry and environmental vectors on the left, then run
                the ensemble classifier to receive ranked crop recommendations.
              </p>
              <div className="flex gap-2 mt-1 flex-wrap justify-center">
                <span className="agri-chip chip-gray">TabPFN Ready</span>
                <span className="agri-chip chip-gray">LightGBM Ready</span>
                <span className="agri-chip chip-gray">7-Feature Model</span>
              </div>
            </div>
          )}

          {/* Results */}
          {!loading && result && (
            <div className="space-y-4 animate-slide-in">

              {/* Results header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span className="text-xs font-mono font-bold text-[#0f2e1e] uppercase tracking-wider">
                    Top Recommended Crops
                  </span>
                </div>
                <span className="agri-chip chip-green">
                  <span className="status-dot-green" />
                  {result.crops.length} Results
                </span>
              </div>

              {/* Crop ranking cards */}
              <div className="space-y-3">
                {result.crops.map((crop, index) => (
                  <div
                    key={crop.name}
                    className={`result-card-primary animate-slide-in`}
                    style={{ animationDelay: `${index * 80}ms` }}
                  >
                    {/* Rank badge + name */}
                    <div className="flex items-start gap-4">
                      <SuitabilityRing pct={crop.suitability} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-1">
                          <span className="text-[9px] font-mono font-black text-white px-2 py-0.5 rounded-md"
                            style={{ background: getSuitabilityColor(crop.suitability) }}>
                            RANK #{index + 1}
                          </span>
                          <h4 className="text-base font-black text-[#0f2e1e]">{crop.name}</h4>
                          <span className={getSuitabilityChipClass(crop.suitability)}>
                            {getSuitabilityLabel(crop.suitability)}
                          </span>
                        </div>
                        <p className="text-xs text-[#5a7a52] leading-relaxed line-clamp-2">
                          {crop.description}
                        </p>
                        {/* Suitability progress bar */}
                        <div className="mt-2 space-y-0.5">
                          <div className="flex justify-between">
                            <span className="text-[9px] font-mono text-[#8aac82] uppercase tracking-wider">Suitability Index</span>
                            <span className="text-[9px] font-mono font-bold" style={{ color: getSuitabilityColor(crop.suitability) }}>
                              {crop.suitability}%
                            </span>
                          </div>
                          <div className="progress-bar-track">
                            <div
                              className="progress-bar-fill"
                              style={{
                                width: `${crop.suitability}%`,
                                background: getSuitabilityColor(crop.suitability),
                              }}
                            />
                          </div>
                        </div>
                        {crop.optimalConditions && (
                          <p className="text-[10px] text-[#8aac82] font-mono mt-1.5 italic">
                            {crop.optimalConditions}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Soil advisory panel */}
              <div className="soil-panel animate-fade-in">
                <div className="flex items-center gap-2 border-b border-[#c5ddc0]/60 pb-2.5 mb-3">
                  <Lightbulb className="w-4 h-4 text-amber-500" />
                  <h4 className="text-xs font-mono font-bold text-[#0f2e1e] uppercase tracking-wider">
                    Agronomic Advisory
                  </h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <span className="text-[10px] font-mono font-bold text-[#2d5a27] uppercase tracking-wide block mb-0.5">
                      Soil pH Condition
                    </span>
                    <p className="text-xs text-[#4a6741] leading-relaxed">{result.optimalPH}</p>
                  </div>
                  <div className="border-t border-[#c5ddc0]/50 pt-3">
                    <span className="text-[10px] font-mono font-bold text-[#2d5a27] uppercase tracking-wide block mb-0.5">
                      Nutrition Strategy
                    </span>
                    <p className="text-xs text-[#4a6741] leading-relaxed">{result.nutritionStatus}</p>
                  </div>
                </div>
              </div>

              {/* MLOps trace bar */}
              <div className="trace-bar">
                <span>ENSEMBLE: TabPFN + LightGBM (Weighted Vote)</span>
                <span>·</span>
                <span>LATENCY: 14ms</span>
                <span>·</span>
                <span>FEATURES: N · P · K · pH · T · H · R</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
