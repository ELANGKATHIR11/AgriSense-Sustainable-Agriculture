/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from "react";
import {
  CloudSun, Sun, CloudRain, CloudLightning, Cloud,
  Droplets, Wind, AlertTriangle, Sparkles, MapPin, RefreshCw, Thermometer
} from "lucide-react";
import { WeatherDay } from "../types";

export default function WeatherIntelligence() {
  const [locationQuery, setLocationQuery] = useState("Detecting GPS...");
  const [soilPrediction, setSoilPrediction] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [gpsLoading, setGpsLoading] = useState(false);
  
  // Risk levels based on weather data
  const [blightRisk, setBlightRisk] = useState(30);
  const [anoxiaRisk, setAnoxiaRisk] = useState(25);

  const [forecast, setForecast] = useState<WeatherDay[]>([
    { date: "Wednesday", temperature: 28.5, humidity: 59, rainfall: 4.2,  condition: "cloudy",  windSpeed: 12 },
    { date: "Thursday",  temperature: 29.1, humidity: 55, rainfall: 0.0,  condition: "sunny",   windSpeed: 8  },
    { date: "Friday",    temperature: 26.3, humidity: 85, rainfall: 42.5, condition: "stormy",  windSpeed: 24 },
    { date: "Saturday",  temperature: 24.8, humidity: 78, rainfall: 15.0, condition: "rainy",   windSpeed: 16 },
    { date: "Sunday",    temperature: 27.2, humidity: 62, rainfall: 1.2,  condition: "sunny",   windSpeed: 10 },
  ]);

  const mapWmoCodeToCondition = (code: number): "sunny" | "cloudy" | "rainy" | "stormy" => {
    if (code === 0) return "sunny";
    if (code >= 1 && code <= 3) return "cloudy";
    if (code >= 45 && code <= 48) return "cloudy";
    if ((code >= 51 && code <= 67) || (code >= 80 && code <= 82)) return "rainy";
    return "stormy"; // 71-77, 85-86, 95-99
  };

  const getDayName = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { weekday: "long" });
  };

  const fetchLiveWeather = () => {
    if (!navigator.geolocation) {
      setLocationQuery("GPS Not Supported");
      return;
    }

    setGpsLoading(true);
    setLocationQuery("Requesting GPS Position...");

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        setLocationQuery(`Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`);

        try {
          const response = await fetch(
            `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&daily=temperature_2m_max,relative_humidity_2m_mean,precipitation_sum,weathercode,windspeed_10m_max&timezone=auto`
          );
          
          if (response.ok) {
            const data = await response.json();
            const daily = data.daily;
            
            const parsedForecast: WeatherDay[] = daily.time.slice(0, 5).map((timeStr: string, idx: number) => {
              return {
                date: getDayName(timeStr),
                temperature: Math.round(daily.temperature_2m_max[idx] * 10) / 10,
                humidity: Math.round(daily.relative_humidity_2m_mean[idx]),
                rainfall: Math.round(daily.precipitation_sum[idx] * 10) / 10,
                condition: mapWmoCodeToCondition(daily.weathercode[idx]),
                windSpeed: Math.round(daily.windspeed_10m_max[idx])
              };
            });
            
            setForecast(parsedForecast);
            generateAgriSuggestions(parsedForecast);
          } else {
            setLocationQuery("Weather Fetch Error");
          }
        } catch (err) {
          console.error("Open-Meteo fetch failed", err);
          setLocationQuery("Network API Error");
        } finally {
          setGpsLoading(false);
        }
      },
      (error) => {
        console.error("Geolocation failed", error);
        setLocationQuery("GPS Access Denied");
        setGpsLoading(false);
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  const generateAgriSuggestions = (days: WeatherDay[]) => {
    // Computes average values and maximums to produce recommendations
    const maxRain = Math.max(...days.map(d => d.rainfall));
    const avgHumid = days.reduce((acc, d) => acc + d.humidity, 0) / days.length;
    const avgTemp = days.reduce((acc, d) => acc + d.temperature, 0) / days.length;

    let rainDay = days.find(d => d.rainfall === maxRain);
    let suggestion = "";
    
    // Dynamic risk adjustment
    const calculatedBlightRisk = Math.min(95, Math.round((avgHumid * 0.8) + (avgTemp * 0.4)));
    const calculatedAnoxiaRisk = Math.min(98, Math.round(maxRain * 2.0));
    setBlightRisk(calculatedBlightRisk);
    setAnoxiaRisk(calculatedAnoxiaRisk);

    if (maxRain > 15 && rainDay) {
      suggestion = `Alert: High rainfall event of ${maxRain}mm detected on ${rainDay.date}. Soil moisture levels are predicted to spike rapidly, creating severe water-clogging and root anoxia risk (calculated at ${calculatedAnoxiaRisk}%). Recommend pausing all drip-irrigation modules 24h prior, clearing main farm outlet drainage channels, and keeping ESP32 edge nodes shielded.`;
    } else if (avgHumid > 75) {
      suggestion = `Warning: Elevated relative humidity average of ${avgHumid.toFixed(0)}% creates high pathogen susceptibility, boosting Tomato Late Blight risk to ${calculatedBlightRisk}%. Ensure plants are adequately spaced to maximize air circulation. Recommend applying defensive biological neem oil or organic copper hydroxide sprays.`;
    } else if (avgTemp > 32) {
      suggestion = `Warning: High thermal stress average of ${avgTemp.toFixed(1)}°C detected. Micro-climate transpiration index indicates severe evaporation. Increase drip irrigation volume by 25% during early morning hours to maintain root moisture buffers.`;
    } else {
      suggestion = `Optimal conditions detected. Consistent temperature (${avgTemp.toFixed(1)}°C) and moderate relative humidity (${avgHumid.toFixed(0)}%) are ideal for vegetative growth. Maintain standard soil moisture settings (32-35%) via localized schedules.`;
    }

    setSoilPrediction(suggestion);
  };

  // Run weather fetch automatically on mount
  useEffect(() => {
    fetchLiveWeather();
  }, []);

  const predict = () => {
    setPredicting(true);
    setTimeout(() => {
      generateAgriSuggestions(forecast);
      setPredicting(false);
    }, 800);
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
          <button
            onClick={fetchLiveWeather}
            disabled={gpsLoading}
            className="flex items-center gap-2 bg-black/20 border border-emerald-900/40 rounded-xl px-3 py-2 flex-shrink-0 text-white cursor-pointer hover:bg-black/30 transition-all text-xs font-mono font-bold"
          >
            <MapPin className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span>{gpsLoading ? "Syncing GPS..." : locationQuery}</span>
            <RefreshCw className={`w-3.5 h-3.5 text-emerald-400 ${gpsLoading ? "animate-spin" : ""}`} />
          </button>
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
            disabled={predicting || gpsLoading}
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
                <span className="agri-chip chip-amber">{blightRisk}% RISK</span>
              </div>
              <p className="text-xs text-amber-800/80">Relative humidity levels forecasted alter spore germination rate.</p>
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: `${blightRisk}%`, background: "linear-gradient(90deg, #f59e0b, #fbbf24)" }} />
              </div>
            </div>
            <div className="p-3.5 rounded-xl bg-red-50 border border-red-200 space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs font-bold font-mono text-red-800 uppercase">Root Anoxia Risk</p>
                <span className="agri-chip chip-red">{anoxiaRisk}% RISK</span>
              </div>
              <p className="text-xs text-red-800/80">Heavy precipitation rates saturate subsurface soil profiles.</p>
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: `${anoxiaRisk}%`, background: "linear-gradient(90deg, #ef4444, #f87171)" }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
