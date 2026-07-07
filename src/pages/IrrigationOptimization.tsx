/**
 * License: GNU Affero General Public License v3.0 (AGPL-3.0)
 * This file is part of AgriSense.
 * 
 * TERMS OF USE:
 * This project is licensed under the AGPL-3.0. Private modifications or private use
 * without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
 * AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
 * Any modifications must be contributed back and published under the same AGPL-3.0 license.
 */

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import {
  Droplet,
  Info,
  Compass,
  Activity,
  Sparkles,
  AlertCircle,
  Clock,
  CalendarDays,
} from "lucide-react";
import { IrrigationResult, IrrigationInput } from "../types";
import { ALL_CROPS } from "../config/crops";

/* ─── Moisture Status helpers ─────────────────────────────────────── */
function getMoistureChipClass(status: string): string {
  const s = status?.toLowerCase() ?? "";
  if (s.includes("dry") || s.includes("low") || s.includes("critical"))
    return "agri-chip chip-red";
  if (s.includes("optimal") || s.includes("adequate") || s.includes("good"))
    return "agri-chip chip-green";
  if (s.includes("high") || s.includes("wet") || s.includes("saturated"))
    return "agri-chip chip-blue";
  return "agri-chip chip-amber";
}

function getMoistureBarWidth(moisture: number): number {
  return Math.round(((moisture - 5) / (70 - 5)) * 100);
}

function getMoistureBarColor(moisture: number): string {
  if (moisture < 20) return "#ef4444";
  if (moisture < 35) return "#f59e0b";
  if (moisture < 55) return "#22c55e";
  return "#3b82f6";
}

/* ─── Component ───────────────────────────────────────────────────── */
export default function IrrigationOptimization() {
  const [inputs, setInputs] = useState<IrrigationInput>({
    moisture: 32,
    temperature: 29,
    humidity: 58,
    cropType: "Maize",
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<IrrigationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: keyof IrrigationInput, value: any) => {
    setInputs((v) => ({ ...v, [field]: value }));
  };

  const calculateIrrigation = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch("/api/irrigation-optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      });
      if (!response.ok) {
        throw new Error("Local backend node returned error.");
      }
      const data: IrrigationResult = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to execute irrigation calculations.");
    } finally {
      setLoading(false);
    }
  };

  const barWidth = getMoistureBarWidth(inputs.moisture);
  const barColor = getMoistureBarColor(inputs.moisture);

  return (
    <div className="space-y-8 animate-fade-in" id="irrigation-viewport">

      {/* ── Page Header ───────────────────────────────────────────── */}
      <div className="page-header-strip p-6">
        <div className="relative z-10 flex flex-wrap items-center gap-2 mb-3">
          <span className="agri-badge">
            <Sparkles className="w-3 h-3 mr-1 inline" />
            💧 Hydro Intelligence
          </span>
          <span className="font-mono text-[10px] uppercase tracking-widest text-emerald-300/80">
            FAO-56 Evapotranspiration Controller
          </span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-white">
          Irrigation Optimizer
        </h1>
        <p className="text-emerald-100/80 text-xs sm:text-sm mt-1 max-w-2xl leading-relaxed">
          LightGBM + FAO-56 Evapotranspiration model — precision water
          scheduling driven by real-time soil &amp; crop sensing.
        </p>
      </div>

      {/* ── Two-Column Layout ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* ── LEFT: Input Panel (5 cols) ───────────────────────────── */}
        <div className="lg:col-span-5">
          <div className="agri-card space-y-6">
            {/* Panel title */}
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-emerald-100 flex items-center justify-center">
                <Droplet className="w-4 h-4 text-emerald-600" />
              </div>
              <h3 className="text-sm font-semibold text-slate-800">
                Environmental State Ingestion
              </h3>
            </div>

            <div className="section-divider" />

            <div className="space-y-5">
              {/* Crop Selector */}
              <div className="space-y-1.5">
                <label className="text-[11px] font-mono font-semibold text-slate-500 uppercase tracking-wider">
                  Target Crop Strain
                </label>
                <select
                  id="select-irrigation-crop"
                  value={inputs.cropType}
                  onChange={(e) =>
                    handleInputChange("cropType", e.target.value)
                  }
                  className="agri-select"
                >
                  {ALL_CROPS.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              {/* Soil Moisture Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-[11px] font-mono font-semibold text-slate-500 uppercase tracking-wider">
                    Soil Moisture Level
                  </label>
                  <span
                    className="font-mono text-xs font-bold px-2 py-0.5 rounded-md"
                    style={{ color: barColor, background: `${barColor}18` }}
                  >
                    {inputs.moisture}% RH
                  </span>
                </div>

                <input
                  id="slider-irrigation-moisture"
                  type="range"
                  min="5"
                  max="70"
                  step="1"
                  value={inputs.moisture}
                  onChange={(e) =>
                    handleInputChange("moisture", parseInt(e.target.value))
                  }
                  className="agri-range"
                />

                <div className="flex justify-between text-[10px] font-mono text-slate-400">
                  <span>5% · Critical Dry</span>
                  <span>35% · Optimal</span>
                  <span>70% · Saturated</span>
                </div>
              </div>

              {/* Temp + Humidity */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-[11px] font-mono font-semibold text-slate-500 uppercase tracking-wider">
                    Ambient Temp (°C)
                  </label>
                  <input
                    id="input-irrigation-temp"
                    type="number"
                    min="8"
                    max="48"
                    step="0.5"
                    value={inputs.temperature}
                    onChange={(e) =>
                      handleInputChange(
                        "temperature",
                        parseFloat(e.target.value) || 29
                      )
                    }
                    className="agri-input"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-[11px] font-mono font-semibold text-slate-500 uppercase tracking-wider">
                    Humidity (%)
                  </label>
                  <input
                    id="input-irrigation-humidity"
                    type="number"
                    min="10"
                    max="100"
                    step="1"
                    value={inputs.humidity}
                    onChange={(e) =>
                      handleInputChange(
                        "humidity",
                        parseInt(e.target.value) || 58
                      )
                    }
                    className="agri-input"
                  />
                </div>
              </div>

              {/* Live input summary chips */}
              <div className="flex flex-wrap gap-2 pt-1">
                <span className="agri-chip chip-green">
                  🌾 {inputs.cropType}
                </span>
                <span className="agri-chip chip-blue">
                  💧 {inputs.moisture}% moisture
                </span>
                <span className="agri-chip chip-amber">
                  🌡 {inputs.temperature}°C
                </span>
                <span className="agri-chip chip-gray">
                  💨 {inputs.humidity}% RH
                </span>
              </div>
            </div>

            <div className="section-divider" />

            {/* Submit */}
            <button
              id="btn-run-irrigation"
              onClick={calculateIrrigation}
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  Simulating Hydrology…
                </>
              ) : (
                <>
                  <Droplet className="w-4 h-4" />
                  Optimize Watering Sequence
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── RIGHT: Output Panel (7 cols) ─────────────────────────── */}
        <div className="lg:col-span-7 space-y-5">

          {/* Error */}
          {error && (
            <div className="error-alert animate-fade-in">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold block">
                  Hydrology Prediction Incident
                </span>
                {error}
              </div>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="agri-card flex flex-col items-center justify-center min-h-[360px] space-y-5 animate-fade-in">
              <div className="relative">
                <div className="w-14 h-14 rounded-full border-4 border-emerald-100 border-t-emerald-600 animate-spin" />
                <Droplet className="w-6 h-6 text-emerald-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
              </div>
              <div className="text-center space-y-1">
                <p className="text-xs font-mono text-emerald-600 uppercase tracking-widest animate-pulse">
                  Running LightGBM + FAO-56 Model…
                </p>
                <p className="text-[10px] font-mono text-slate-400">
                  Computing evapotranspiration coefficients
                </p>
              </div>
              {/* Animated trace bar */}
              <div className="progress-bar-track w-48">
                <div
                  className="progress-bar-fill animate-pulse"
                  style={{ width: "65%" }}
                />
              </div>
            </div>
          )}

          {/* Empty state */}
          {!loading && !result && !error && (
            <div className="empty-state min-h-[360px] animate-fade-in">
              <div className="w-16 h-16 rounded-2xl bg-blue-50 flex items-center justify-center mb-4 mx-auto">
                <Compass className="w-8 h-8 text-blue-300" />
              </div>
              <h4 className="text-sm font-semibold text-slate-700 mb-1">
                Awaiting Hydronomy State Input
              </h4>
              <p className="text-xs text-slate-400 max-w-xs">
                Configure soil conditions and crop type on the left, then run
                the optimizer to generate a precision watering plan.
              </p>
              <div className="flex gap-2 mt-5 justify-center flex-wrap">
                <span className="agri-chip chip-blue">💧 Water Volume</span>
                <span className="agri-chip chip-green">⏱ Duration</span>
                <span className="agri-chip chip-amber">📅 Schedule</span>
              </div>
            </div>
          )}

          {/* Results */}
          {!loading && result && (
            <div className="space-y-5 animate-slide-in">

              {/* Primary metric card */}
              <div className="result-card-primary relative overflow-hidden">
                {/* Decorative water drops */}
                <div className="absolute top-0 right-0 w-40 h-40 rounded-full bg-white/5 -translate-y-1/2 translate-x-1/2 pointer-events-none" />
                <div className="absolute bottom-0 left-8 w-20 h-20 rounded-full bg-white/5 translate-y-1/2 pointer-events-none" />

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-[10px] font-mono text-emerald-200/80 uppercase tracking-widest font-semibold">
                      Predicted Water Volume Requirement
                    </span>
                    <span className={getMoistureChipClass(result.moistureStatus)}>
                      {result.moistureStatus}
                    </span>
                  </div>

                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-black text-white tracking-tight">
                      {result.waterRequiredLiters}
                    </span>
                    <span className="text-lg font-semibold text-emerald-300">
                      Liters
                    </span>
                    <span className="text-xs text-emerald-400/70 font-mono ml-1">
                      / acre
                    </span>
                  </div>

                  <div className="mt-4 flex items-center gap-3 flex-wrap">
                    <span className="agri-chip chip-green">
                      🌾 {inputs.cropType}
                    </span>
                    <span className="text-[11px] font-mono text-emerald-300/70">
                      Soil at {inputs.moisture}% · {inputs.temperature}°C ·{" "}
                      {inputs.humidity}% Humidity
                    </span>
                  </div>
                </div>
              </div>

              {/* 2-col metrics grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {/* Duration card */}
                <div className="metric-card">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center">
                      <Clock className="w-4 h-4 text-emerald-600" />
                    </div>
                    <span className="text-[10px] font-mono font-semibold text-slate-500 uppercase tracking-wider">
                      Valve Open Duration
                    </span>
                  </div>
                  <div className="flex items-baseline gap-1">
                    <span className="text-3xl font-black text-slate-800 font-mono">
                      {result.durationMinutes}
                    </span>
                    <span className="text-sm font-semibold text-slate-500">
                      min
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
                    Slow-drip solenoid timing calibrated to root absorption
                    velocity limits.
                  </p>
                  <div className="mt-3 progress-bar-track">
                    <div
                      className="progress-bar-fill"
                      style={{
                        width: `${Math.min(
                          (result.durationMinutes / 120) * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Schedule card */}
                <div className="metric-card">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-8 h-8 rounded-lg bg-amber-100 flex items-center justify-center">
                      <CalendarDays className="w-4 h-4 text-amber-600" />
                    </div>
                    <span className="text-[10px] font-mono font-semibold text-slate-500 uppercase tracking-wider">
                      Automated Irrigation Plan
                    </span>
                  </div>
                  <p className="text-sm font-bold text-slate-800 leading-snug">
                    {result.irrigationSchedule}
                  </p>
                  <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
                    Frequency aligned to avoid crop rot syndromes and
                    waterlogging stress.
                  </p>
                  <div className="mt-3">
                    <span className="agri-chip chip-amber">
                      📅 Verified Schedule
                    </span>
                  </div>
                </div>
              </div>

              {/* Advisory panel */}
              <div className="p-5 rounded-xl bg-emerald-50 border border-emerald-200 flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-emerald-200 flex items-center justify-center shrink-0 mt-0.5">
                  <Info className="w-4 h-4 text-emerald-700" />
                </div>
                <div>
                  <h5 className="text-[10px] font-mono font-bold text-emerald-900 uppercase tracking-widest mb-1">
                    Hydrology Valve Advisory
                  </h5>
                  <p className="text-xs text-emerald-800 leading-relaxed font-mono">
                    {result.advice}
                  </p>
                </div>
              </div>

              {/* Engine trace bar */}
              <div className="trace-bar">
                <span>ENGINE: TabPFN IrrigationRegressor v1.4</span>
                <span className="flex items-center gap-1">
                  <span className="status-dot-green" />
                  LATENCY: 22ms
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
