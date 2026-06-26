/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import {
  CloudSun, Sun, CloudRain, CloudLightning, Cloud,
  Droplets, Wind, AlertTriangle, Sparkles, MapPin, RefreshCw, Thermometer
} from "lucide-react";
import { WeatherDay } from "../types";

export default function WeatherIntelligence() {
  const [locationQuery, setLocationQuery] = useState("Zone A – Central Sector");
  const [soilPrediction, setSoilPrediction] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);

  const forecast: WeatherDay[] = [
    { date: "Wednesday", temperature: 28.5, humidity: 59, rainfall: 4.2,  condition: "cloudy",  windSpeed: 12 },
    { date: "Thursday",  temperature: 29.1, humidity: 55, rainfall: 0.0,  condition: "sunny",   windSpeed: 8  },
    { date: "Friday",    temperature: 26.3, humidity: 85, rainfall: 42.5, condition: "stormy",  windSpeed: 24 },
    { date: "Saturday",  temperature: 24.8, humidity: 78, rainfall: 15.0, condition: "rainy",   windSpeed: 16 },
    { date: "Sunday",    temperature: 27.2, humidity: 62, rainfall: 1.2,  condition: "sunny",   windSpeed: 10 },
  ];

  const predict = () => {
    setPredicting(true);
    setTimeout(() => {
      setSoilPrediction(
        "Friday's 42.5 mm storm event will saturate Zone A soil to ~68% moisture — critical water-clogging risk for root vegetables. Recommend closing micro-drip systems 48h before event and clearing drainage channels preemptively."
      );
      setPredicting(false);
    }, 1200);
  };

  const getWeatherIcon = (cond: string) => {
    switch (cond) {
      case "sunny":  return <Sun className="w-7 h-7 text-amber-400" />;
      case "rainy":  return <CloudRain className="w-7 h-7 text-sky-400" />;
      case "cloudy": return <Cloud className="w-7 h-7 text-slate-400" />;
      case "stormy": return <CloudLightning className="w-7 h-7 text-violet-400" />;
      default:       return <CloudSun className="w-7 h-7 text-slate-300" />;
    }
  };

  const conditionColors: Record<string, string> = {
    sunny:  "text-amber-500 bg-amber-50  border-amber-200",
    rainy:  "text-sky-500   bg-sky-50    border-sky-200",
    cloudy: "text-slate-500 bg-slate-50  border-slate-200",
    stormy: "text-violet-600 bg-violet-50 border-violet-200",
  };

  return (
    <div className="space-y-6 animate-fade-in" id="weather-viewport">
      {/* Header */}
      <div className="page-header-strip p-6 text-white">
        <div className="relative z-10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="agri-badge">☀️ Micro-Climate</span>
            </div>
            <h1 className="text-2xl font-black tracking-tight">
              Weather <span className="text-amber-400">Intelligence</span>
            </h1>
            <p className="text-emerald-100/80 text-sm max-w-xl">
              5-day agronomic forecast synced with soil moisture models and pathogen risk indices.
            </p>
          </div>
          <div className="flex items-center gap-2 bg-black/20 border border-emerald-900/40 rounded-xl px-3 py-2 flex-shrink-0">
            <MapPin className="w-3.5 h-3.5 text-amber-400" />
            <input
              id="input-weather-loc"
              type="text"
              value={locationQuery}
              onChange={(e) => setLocationQuery(e.target.value)}
              className="bg-transparent text-xs font-mono text-white placeholder-emerald-400/60 outline-none w-44"
              placeholder="Enter zone or location..."
            />
          </div>
        </div>
      </div>

      {/* 5-Day Forecast */}
      <div className="space-y-3">
        <p className="text-[10px] font-bold font-mono text-gray-400 uppercase tracking-widest">5-Day Agronomic Forecast Window</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
          {forecast.map((day) => {
            const cc = conditionColors[day.condition] || conditionColors.cloudy;
            return (
              <div
                key={day.date}
                className={`agri-card p-4 flex flex-col items-center gap-3 text-center hover:scale-[1.02] transition-transform relative overflow-hidden ${
                  day.condition === "stormy" ? "border-violet-300" : ""
                }`}
              >
                {day.condition === "stormy" && (
                  <div className="absolute top-0 inset-x-0 h-0.5 bg-violet-400" />
                )}
                <div>
                  <p className="text-xs font-bold text-gray-800">{day.date}</p>
                  <span className={`text-[9px] font-mono font-bold uppercase px-2 py-0.5 rounded-full border ${cc}`}>
                    {day.condition}
                  </span>
                </div>
                <div>{getWeatherIcon(day.condition)}</div>
                <div className="space-y-2 w-full">
                  <p className="text-xl font-black text-gray-900">{day.temperature}°C</p>
                  <div className="section-divider" />
                  <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
                    <div className="flex items-center gap-1 text-sky-600">
                      <Droplets className="w-3 h-3" />{day.humidity}%
                    </div>
                    <div className={`text-right font-bold ${day.rainfall > 0 ? "text-sky-700" : "text-gray-400"}`}>
                      {day.rainfall > 0 ? `${day.rainfall}mm` : "Dry"}
                    </div>
                    <div className="flex items-center gap-1 text-gray-400 col-span-2">
                      <Wind className="w-3 h-3" />{day.windSpeed} km/h
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom panels */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Moisture predictor */}
        <div className="lg:col-span-7 agri-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-gray-800">Soil Moisture Impact Predictor</h3>
              <p className="text-xs text-gray-500 mt-0.5">AI-powered forecast of soil saturation from precipitation data</p>
            </div>
          </div>
          <button
            id="btn-run-hydropredict"
            onClick={predict}
            disabled={predicting}
            className="btn-primary"
          >
            {predicting ? (
              <><RefreshCw className="w-4 h-4 animate-spin" /> Simulating Dynamics...</>
            ) : (
              <><Sparkles className="w-4 h-4" /> Predict Moisture Impact</>
            )}
          </button>
          {soilPrediction && (
            <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-100 space-y-2 animate-fade-in">
              <div className="flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
                <p className="text-[10px] font-bold font-mono text-emerald-700 uppercase tracking-wider">AI Prediction Output</p>
              </div>
              <p className="text-sm text-emerald-900 leading-relaxed">{soilPrediction}</p>
            </div>
          )}
        </div>

        {/* Pathogen Risks */}
        <div className="lg:col-span-5 agri-card p-5 space-y-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            <h3 className="text-sm font-bold text-gray-800">Pathogen Risk Forecast</h3>
          </div>
          <div className="space-y-3">
            <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-100 space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs font-bold font-mono text-amber-800 uppercase">Tomato Late Blight</p>
                <span className="agri-chip chip-amber">62% RISK</span>
              </div>
              <p className="text-xs text-amber-800/80">High humidity forecasted Saturday increases sporulation chance by 22%.</p>
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: "62%", background: "linear-gradient(90deg, #f59e0b, #fbbf24)" }} />
              </div>
            </div>
            <div className="p-3.5 rounded-xl bg-red-50 border border-red-200 space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs font-bold font-mono text-red-800 uppercase">Root Anoxia Risk</p>
                <span className="agri-chip chip-red">87% RISK</span>
              </div>
              <p className="text-xs text-red-800/80">Friday storm will saturate subsurface soil profiles — clear drainage channels urgently.</p>
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: "87%", background: "linear-gradient(90deg, #ef4444, #f87171)" }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
