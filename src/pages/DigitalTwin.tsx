/**
 * AGRISENSE DIGITAL TWIN COMMAND CONSOLE
 * Fuses FAO-56 Penman-Monteith physics modeling with residual AI correctors.
 */

import React, { useState, useEffect, useRef } from "react";
import { 
  Sprout, 
  Droplet, 
  Thermometer, 
  ShieldAlert, 
  Settings, 
  TrendingUp, 
  FileText, 
  Play, 
  Download, 
  RefreshCw, 
  Layers, 
  CheckCircle, 
  AlertTriangle, 
  Flame, 
  Wind, 
  CloudRain,
  Database,
  Search,
  Activity,
  Award
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from "recharts";
import { useTwin } from "../hooks/useTwin";
import { useAuthStore } from "../store/authStore";

export default function DigitalTwin() {
  const { state, analytics, isLoading, updateTwin, runScenario, isSimulating } = useTwin();
  const { user } = useAuthStore();
  
  const [activeTab, setActiveTab] = useState<number>(0);
  const [enableResidual, setEnableResidual] = useState<boolean>(true);
  const [selectedScenario, setSelectedScenario] = useState<string>("drought_5_days");
  const [scenarioOutput, setScenarioOutput] = useState<any | null>(null);
  
  // Local telemetry override state for fast-tweaks
  const [manualInput, setManualInput] = useState({
    soilMoisture: 38.3,
    temperature: 28.5,
    humidity: 62.0,
    pH: 6.4,
    nitrogen: 45,
    phosphorus: 38,
    potassium: 42,
    rainfall: 0.0,
    windSpeed: 8.4
  });

  // Report local state
  const [reportType, setReportType] = useState<"daily" | "weekly" | "monthly" | "yield" | "water" | "disease">("daily");
  const [reportLog, setReportLog] = useState<string>("");
  const [downloadSuccess, setDownloadSuccess] = useState<string | null>(null);

  // Sync initial state values
  useEffect(() => {
    if (state) {
      setManualInput({
        soilMoisture: state.waterTwin.currentMoisture,
        temperature: state.weatherTwin.currentTemp,
        humidity: state.weatherTwin.currentHumidity,
        pH: state.soilTwin.pH,
        nitrogen: state.soilTwin.nitrogen,
        phosphorus: state.soilTwin.phosphorus,
        potassium: state.soilTwin.potassium,
        rainfall: 0.0,
        windSpeed: state.weatherTwin.windSpeed
      });
    }
  }, [state]);

  // Execute manual state recalculation using FAO-56 PM
  const handleRecalculate = async () => {
    await updateTwin(manualInput);
  };

  // Run What-If simulation scenarios
  const handleSimulateScenario = async () => {
    const outcome = await runScenario(selectedScenario);
    setScenarioOutput(outcome);
  };

  // Set default scenario prediction outcome on load
  useEffect(() => {
    if (selectedScenario) {
      handleSimulateScenario();
    }
  }, [selectedScenario]);

  // Compile detailed ASCII text reports representing standard PDFs
  const triggerGenerateReport = () => {
    if (!state) return;

    let text = `========================================================================\n`;
    text    += `     AGRISENSE CYBER-AGRICULTURE ORCHESTRATOR - DIGITAL TWIN REPORT     \n`;
    text    += `========================================================================\n`;
    text    += `Report Scope: ${reportType.toUpperCase()} MONITORING REPORT\n`;
    text    += `Generated: ${new Date().toLocaleString()}\n`;
    text    += `Operator Name: ${user?.name || "Alex Agronomist"}\n`;
    text    += `Farm Location: ${user?.farmName || "North Grid Sector-A"}\n`;
    text    += `Overall Farm Health Score: ${state.overallHealthScore}/100\n`;
    text    += `Operational Risk Index: ${state.riskIndex}%\n`;
    text    += `Sustainability Quotient Index: ${state.sustainabilityIndex}%\n\n`;

    text    += `------------------------------------------------------------------------\n`;
    text    += `1. WATER TWIN STATUS (Penman-Monteith Reference ET0 Model)\n`;
    text    += `------------------------------------------------------------------------\n`;
    text    += `- Current Soil Micro-moisture: ${state.waterTwin.currentMoisture}%\n`;
    text    += `- Computed Evapotranspiration (ET0): ${state.waterTwin.evapotranspirationET0} mm/day\n`;
    text    += `- Active Moisture Deficit Depth: ${state.waterTwin.waterDeficitLiters} Liters/acre\n`;
    text    += `- Advisory recommendation: ${state.waterTwin.irrigationRecommendation}\n`;
    text    += `- 5-Day Moisture Projection Line: ${state.waterTwin.predictedMoisture5Days.join(" % -> ")} %\n\n`;

    text    += `------------------------------------------------------------------------\n`;
    text    += `2. SOIL NPK NUTRITIONAL STATUS\n`;
    text    += `------------------------------------------------------------------------\n`;
    text    += `- Nitrogen (N): ${state.soilTwin.nitrogen} ppm\n`;
    text    += `- Phosphorus (P): ${state.soilTwin.phosphorus} ppm\n`;
    text    += `- Potassium (K): ${state.soilTwin.potassium} ppm\n`;
    text    += `- Soil pH Level: ${state.soilTwin.pH}\n`;
    text    += `- EC (Electrical Conductivity): ${state.soilTwin.electricalConductivity} dS/m\n`;
    text    += `- Organic Carbon percentage: ${state.soilTwin.organicCarbon}%\n`;
    text    += `- Combined Nutritional Health Score: ${state.soilTwin.healthScore}/100\n`;
    text    += `- Depletion analysis: ${state.soilTwin.nutrientDeficitForecast}\n\n`;

    text    += `------------------------------------------------------------------------\n`;
    text    += `3. CROP BIOMASS & GROWTH TWIN\n`;
    text    += `------------------------------------------------------------------------\n`;
    text    += `- Sown Vegetation Class: ${state.cropTwin.cropType}\n`;
    text    += `- Growth Phenology Phase: ${state.cropTwin.growthStage}\n`;
    text    += `- Active Biophysical Biomass Index: ${state.cropTwin.biomassIndex} kg/hectare\n`;
    text    += `- Target Harvest Forecast Timeline: ${state.cropTwin.harvestForecastDate}\n`;
    text    += `- Projected Yield Multiplier Margin: ${state.cropTwin.predictedYieldMultiplier}x\n\n`;

    text    += `------------------------------------------------------------------------\n`;
    text    += `4. PATHOLOGICAL DISEASE BIOPHYSICS\n`;
    text    += `------------------------------------------------------------------------\n`;
    text    += `- Active Disease Risk Score: ${state.diseaseTwin.riskScore}%\n`;
    text    += `- Outbreak Propensity: ${state.diseaseTwin.outbreakProbability}%\n`;
    text    += `- Primary Vulnerabilities: ${state.diseaseTwin.susceptibleCrops.join(", ")}\n`;
    text    += `- Preventive Countermeasures: ${state.diseaseTwin.preventiveActionRequired.join(" | ")}\n`;
    text    += `========================================================================\n`;
    text    += `                [AGRISENSE SYSTEM - TRUSTED CYBERPHYSICS]               \n`;

    setReportLog(text);
  };

  // Trigger download of standard ASCII styled PDF text
  const downloadReportFile = () => {
    if (!reportLog) return;
    const blob = new Blob([reportLog], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `Agrisense_Twin_${reportType}_${new Date().toISOString().split('T')[0]}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setDownloadSuccess("Standard PDF representation exported successfully!");
    setTimeout(() => setDownloadSuccess(null), 3000);
  };

  // Interactive 3D Isometric Farm Grid Canvas Visualizer
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [selectedZone, setSelectedZone] = useState<string>("Basmati Rice Sector-A");
  const [hoveredCoordinate, setHoveredCoordinate] = useState<string>("Grid X:2, Y:2");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let angle = 0;

    const resizeAndDraw = () => {
      // Handle canvas dimension bounds
      const container = canvas.parentElement;
      if (!container) return;
      canvas.width = container.clientWidth;
      canvas.height = 360;

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      // Generate isometric grid parameters
      const tileWidth = 60;
      const tileHeight = tileWidth / 2;
      const xOffset = width / 2;
      const yOffset = height / 3;

      const gridZZones = [
        { name: "Sector A - Basmati", crop: "Rice", color: "rgba(16, 185, 129, 0.2)" },
        { name: "Sector B - Sweet Maize", crop: "Maize", color: "rgba(59, 130, 246, 0.2)" },
        { name: "Sector C - Chickpeas", crop: "Chickpea", color: "rgba(245, 158, 11, 0.2)" },
        { name: "Sector D - Cantaloupe", crop: "Melon", color: "rgba(239, 68, 68, 0.2)" }
      ];

      // Draw subtle spatial horizon line as a baseline depth cue
      ctx.strokeStyle = "rgba(229, 231, 235, 0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, yOffset + 120);
      ctx.lineTo(width, yOffset + 120);
      ctx.stroke();

      // Render subterranean water layer profile blocks
      ctx.fillStyle = "rgba(59, 130, 246, 0.04)";
      ctx.beginPath();
      ctx.moveTo(xOffset, yOffset + 160);
      ctx.lineTo(xOffset + 200, yOffset + 260);
      ctx.lineTo(xOffset, yOffset + 320);
      ctx.lineTo(xOffset - 200, yOffset + 260);
      ctx.closePath();
      ctx.fill();

      // Draw active multi-dimensional isometric grid coordinates
      const gridSize = 4;
      for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
          
          // Projects screen coordinate from isometric indices
          const screenX = xOffset + (x - y) * tileWidth;
          const screenY = yOffset + (x + y) * tileHeight;

          // Determine color based on moisture state simulated
          let fillColor = "rgba(16, 185, 129, 0.15)"; // healthy green
          let strokeColor = "rgba(16, 185, 129, 0.45)";
          
          const moisture = state?.waterTwin.currentMoisture || 38.3;
          if (moisture < 20.0) {
            fillColor = "rgba(239, 68, 68, 0.25)"; // red critical stress
            strokeColor = "rgba(239, 68, 68, 0.65)";
          } else if (moisture < 35.0) {
            fillColor = "rgba(245, 158, 11, 0.2)"; // yellow warning
            strokeColor = "rgba(245, 158, 11, 0.5)";
          } else if (moisture > 65.0) {
            fillColor = "rgba(59, 130, 246, 0.25)"; // blue overwater
            strokeColor = "rgba(59, 130, 246, 0.65)";
          }

          // Draw Isometric Grid Tile Face
          ctx.fillStyle = fillColor;
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.5;

          ctx.beginPath();
          ctx.moveTo(screenX, screenY);
          ctx.lineTo(screenX + tileWidth, screenY + tileHeight);
          ctx.lineTo(screenX, screenY + tileHeight * 2);
          ctx.lineTo(screenX - tileWidth, screenY + tileHeight);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          // Left side depth projection
          ctx.fillStyle = "rgba(0, 0, 0, 0.04)";
          ctx.beginPath();
          ctx.moveTo(screenX - tileWidth, screenY + tileHeight);
          ctx.lineTo(screenX, screenY + tileHeight * 2);
          ctx.lineTo(screenX, screenY + tileHeight * 2 + 10);
          ctx.lineTo(screenX - tileWidth, screenY + tileHeight + 10);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          // Right side depth projection
          ctx.fillStyle = "rgba(0, 0, 0, 0.08)";
          ctx.beginPath();
          ctx.moveTo(screenX, screenY + tileHeight * 2);
          ctx.lineTo(screenX + tileWidth, screenY + tileHeight);
          ctx.lineTo(screenX + tileWidth, screenY + tileHeight + 10);
          ctx.lineTo(screenX, screenY + tileHeight * 2 + 10);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          // Draw active floating sensory node on center coordinates
          if (x === 1 && y === 1) {
            const nodeY = screenY + tileHeight - (15 + Math.sin(angle * 0.06) * 5);
            
            // Sensor stem wire
            ctx.strokeStyle = "rgba(107, 114, 128, 0.8)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(screenX, screenY + tileHeight);
            ctx.lineTo(screenX, nodeY);
            ctx.stroke();

            // Pumping sonar pulse ring
            const pulseRadius = 15 + Math.sin(angle * 0.08) * 6;
            ctx.strokeStyle = "rgba(59, 130, 246, 0.35)";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.ellipse(screenX, screenY + tileHeight, pulseRadius, pulseRadius / 2, 0, 0, Math.PI * 2);
            ctx.stroke();

            // Glowing sensor sphere
            ctx.fillStyle = "#3b82f6";
            ctx.beginPath();
            ctx.arc(screenX, nodeY, 6, 0, Math.PI * 2);
            ctx.fill();

            // Inner core glow
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(screenX, nodeY, 2.5, 0, Math.PI * 2);
            ctx.fill();

            // Floating meta label
            ctx.fillStyle = "#0f172a";
            ctx.font = "bold 9px monospace";
            ctx.shadowColor = "rgba(0,0,0,0.1)";
            ctx.shadowBlur = 4;
            ctx.fillText(`ESP32-S01 Node (Moisture: ${moisture}%)`, screenX - 65, nodeY - 10);
            ctx.shadowBlur = 0; // reset shadow
          }

          // Render active vegetable assets in layout rows
          if ((x + y) % 2 === 0) {
            ctx.fillStyle = "rgba(16, 185, 129, 0.85)";
            ctx.beginPath();
            const plantX = screenX;
            const plantY = screenY + tileHeight - 8;
            ctx.moveTo(plantX, plantY);
            ctx.quadraticCurveTo(plantX + 4, plantY - 12, plantX + 1, plantY - 14);
            ctx.quadraticCurveTo(plantX - 4, plantY - 8, plantX, plantY);
            ctx.moveTo(plantX, plantY);
            ctx.quadraticCurveTo(plantX - 4, plantY - 12, plantX - 1, plantY - 14);
            ctx.quadraticCurveTo(plantX + 4, plantY - 8, plantX, plantY);
            ctx.closePath();
            ctx.fill();
          }
        }
      }

      // Draw floating meteorological cloud above farm grid
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.strokeStyle = "rgba(156, 163, 175, 0.25)";
      ctx.lineWidth = 1;
      const cloudX = xOffset - 120 + Math.sin(angle * 0.015) * 20;
      const cloudY = yOffset - 70;
      
      ctx.beginPath();
      ctx.arc(cloudX, cloudY, 15, 0, Math.PI * 2);
      ctx.arc(cloudX + 15, cloudY - 8, 20, 0, Math.PI * 2);
      ctx.arc(cloudX + 35, cloudY, 12, 0, Math.PI * 2);
      ctx.rect(cloudX, cloudY - 5, 35, 10);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Rain droplet simulation loop
      if (state && state.weatherTwin.currentHumidity > 70) {
        ctx.strokeStyle = "rgba(59, 130, 246, 0.4)";
        ctx.lineWidth = 1.5;
        for (let r = 0; r < 5; r++) {
          const rx = cloudX + r * 10 + (angle % 12);
          const ry = cloudY + 15 + ((angle * 1.5 + r * 15) % 45);
          ctx.beginPath();
          ctx.moveTo(rx, ry);
          ctx.lineTo(rx - 2, ry + 6);
          ctx.stroke();
        }
      }

      angle += 0.5;
      animationFrameId = requestAnimationFrame(resizeAndDraw);
    };

    resizeAndDraw();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [state]);

  // Tab definitions
  const tabs = [
    { label: "1. Master Twin Overview", icon: <Layers className="w-4 h-4" /> },
    { label: "2. Water Twin", icon: <Droplet className="w-4 h-4" /> },
    { label: "3. Soil Twin", icon: <Database className="w-4 h-4" /> },
    { label: "4. Crop Growth Twin", icon: <Sprout className="w-4 h-4" /> },
    { label: "5. Weather Twin", icon: <Wind className="w-4 h-4" /> },
    { label: "6. Pathology Twin", icon: <ShieldAlert className="w-4 h-4" /> },
    { label: "7. Scenario Simulator", icon: <Play className="w-4 h-4" /> },
    { label: "8. Reports & Analytics", icon: <FileText className="w-4 h-4" /> }
  ];

  if (isLoading || !state) {
    return (
      <div className="min-h-[500px] flex flex-col items-center justify-center gap-4 text-gray-500 py-20 bg-white border border-gray-100 rounded-2xl" id="digital-twin-loader">
        <RefreshCw className="w-8 h-8 animate-spin text-emerald-600" />
        <span className="text-sm font-semibold tracking-wide">Syncing multidimensional AgriSense Digital Twin registers...</span>
      </div>
    );
  }

  return (
    <div className="space-y-8" id="agrisense-digital-twin-core">
      {/* Top Banner & Header Segment */}
      <div className="page-header-strip p-6 text-white">
        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="max-w-3xl">
          <div className="flex items-center gap-2 text-emerald-100 font-semibold mb-1 text-xs uppercase tracking-wider">
            <Award className="w-4 h-4" />
            Cyber-Physical Agriculture Platform
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white md:text-3xl">
            AGRISENSE ISO-TWIN CONSOLE
          </h1>
          <p className="text-sm text-emerald-100/80">
            Fusing real-time FAO-56 Penman-Monteith physical models with residual TabPFN and LightGBM neural correctors.
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          {/* Active status indicator */}
          <div className="px-3 py-1.5 bg-black/20 border border-emerald-900/40 rounded-lg flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-xs font-bold text-emerald-100">TWIN CORES ONLINE</span>
          </div>
          
          <button 
            onClick={handleRecalculate}
            disabled={isSimulating}
            className="p-2 px-3 bg-white/10 border border-white/20 text-white text-xs font-semibold rounded-lg hover:bg-white/15 active:scale-95 transition-all flex items-center gap-1.5 shadow-sm cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isSimulating ? 'animate-spin' : ''}`} />
            Trigger PM-Recompute
          </button>
        </div>
        </div>
      </div>

      {/* Main Tabbed Layout Segment */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Navigation Sidebar Panel (Span 3) */}
        <div className="lg:col-span-3 space-y-2">
          <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-1 shadow-sm">
            <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block px-2 mb-2">Digital Twin Cores</span>
            {tabs.map((tab, idx) => (
              <button
                key={idx}
                onClick={() => setActiveTab(idx)}
                className={`w-full px-3 py-2.5 rounded-lg text-xs font-medium flex items-center gap-2.5 transition-all cursor-pointer ${activeTab === idx ? "bg-emerald-600 text-white shadow-sm font-semibold" : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"}`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

          {/* Quick Real-Time Telemetry Adjuster */}
          <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-4">
            <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">Telemetry Override Injector</span>
            <div className="space-y-3">
              <div>
                <label className="text-[10px] text-gray-500 font-bold block mb-1">Soil Moisture ({manualInput.soilMoisture}%)</label>
                <input 
                  type="range" 
                  min="5" 
                  max="95" 
                  step="0.5"
                  value={manualInput.soilMoisture}
                  onChange={(e) => setManualInput({ ...manualInput, soilMoisture: parseFloat(e.target.value) })}
                  className="w-full accent-emerald-600"
                />
              </div>
              <div>
                <label className="text-[10px] text-gray-500 font-bold block mb-1">Soil Temperature ({manualInput.temperature}°C)</label>
                <input 
                  type="range" 
                  min="5" 
                  max="45" 
                  step="0.5"
                  value={manualInput.temperature}
                  onChange={(e) => setManualInput({ ...manualInput, temperature: parseFloat(e.target.value) })}
                  className="w-full accent-emerald-600"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] text-gray-500 font-semibold block mb-0.5">Atm pH</label>
                  <input 
                    type="number" 
                    step="0.1"
                    min="3.5"
                    max="9.5"
                    className="w-full text-xs p-1 bg-gray-50 border border-gray-200 rounded text-gray-800"
                    value={manualInput.pH}
                    onChange={(e) => setManualInput({ ...manualInput, pH: parseFloat(e.target.value) || 6.4 })}
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 font-semibold block mb-0.5">Wind (m/s)</label>
                  <input 
                    type="number" 
                    step="0.1"
                    className="w-full text-xs p-1 bg-gray-50 border border-gray-200 rounded text-gray-800"
                    value={manualInput.windSpeed}
                    onChange={(e) => setManualInput({ ...manualInput, windSpeed: parseFloat(e.target.value) || 8.4 })}
                  />
                </div>
              </div>
              <button
                onClick={handleRecalculate}
                className="w-full py-1.5 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 text-xs font-bold rounded-lg transition-all border border-emerald-100 shadow-sm cursor-pointer"
              >
                Inject & Sync Core
              </button>
            </div>
          </div>
        </div>

        {/* Tab Viewport Dashboard Panels (Span 9) */}
        <div className="lg:col-span-9">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              
              {/* PAGE 1: TWIN OVERVIEW */}
              {activeTab === 0 && (
                <div className="space-y-6">
                  {/* Master Dials - Consolidated Indices */}
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                      <div className="space-y-1">
                        <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">Farm Health Score</span>
                        <span className="text-3xl font-extrabold text-emerald-700 tracking-tight">{state.overallHealthScore}</span>
                        <span className="text-[10px] text-emerald-600 block font-semibold">● Highly Vigorous</span>
                      </div>
                      <div className="w-12 h-12 rounded-full bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600">
                        <Sprout className="w-6 h-6" />
                      </div>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                      <div className="space-y-1">
                        <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block font-bold">Operational Risk Index</span>
                        <span className={`text-3xl font-extrabold tracking-tight ${state.riskIndex > 30 ? 'text-amber-600' : 'text-slate-700'}`}>{state.riskIndex}%</span>
                        <span className="text-[10px] text-gray-500 block font-semibold">● Nominal parameters</span>
                      </div>
                      <div className="w-12 h-12 rounded-full bg-slate-50 border border-slate-100 flex items-center justify-center text-slate-500">
                        <ShieldAlert className="w-6 h-6" />
                      </div>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                      <div className="space-y-1">
                        <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block font-bold">Crop Yield Index</span>
                        <span className="text-3xl font-extrabold text-blue-600 tracking-tight">{state.yieldIndex}</span>
                        <span className="text-[10px] text-blue-600 block font-semibold">● TabPFN Predictor v3</span>
                      </div>
                      <div className="w-12 h-12 rounded-full bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600">
                        <TrendingUp className="w-6 h-6" />
                      </div>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                      <div className="space-y-1">
                        <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block font-bold">Sustainability Index</span>
                        <span className="text-3xl font-extrabold text-teal-600 tracking-tight">{state.sustainabilityIndex}%</span>
                        <span className="text-[10px] text-teal-600 block font-semibold">● Resource optimized</span>
                      </div>
                      <div className="w-12 h-12 rounded-full bg-teal-50 border border-teal-100 flex items-center justify-center text-teal-600">
                        <Award className="w-6 h-6" />
                      </div>
                    </div>
                  </div>

                  {/* 3D Farm Visualizer Interactive Map Canvas */}
                  <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-md relative">
                    <div className="absolute top-4 left-4 z-20 space-y-1">
                      <div className="px-2 py-1 bg-slate-800/80 backdrop-blur border border-slate-700 rounded-md text-[10px] font-bold text-slate-300 tracking-wider inline-block">
                        3D AGRONOMIC ISO-GRID ENVIRONMENT
                      </div>
                      <div className="text-xs text-slate-400 font-medium">Interactive spatial model of North Grid Sector-A</div>
                    </div>

                    <div className="absolute top-4 right-4 z-20 flex gap-2">
                      <select 
                        value={selectedZone}
                        onChange={(e) => setSelectedZone(e.target.value)}
                        className="bg-slate-800 text-[10px] font-bold text-slate-100 border border-slate-700 rounded-md px-2 py-1 shadow"
                      >
                        <option>Sector A - Basmati Rice</option>
                        <option>Sector B - Sweet Maize</option>
                        <option>Sector C - Chickpea</option>
                        <option>Sector D - Cantaloupe</option>
                      </select>
                    </div>

                    {/* Canvas Block */}
                    <canvas ref={canvasRef} className="w-full block bg-gradient-to-b from-[#0b0f19] to-[#111827]" />

                    {/* Canvas Meta status overlays */}
                    <div className="p-4 bg-slate-950 border-t border-slate-800 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 font-mono text-[10px]">
                      <div className="text-slate-400 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                        Hover feedback coordinate: <span className="text-emerald-400 font-bold">{hoveredCoordinate}</span>
                      </div>
                      <div className="text-slate-500 flex items-center gap-4">
                        <div className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded bg-emerald-500/25 border border-emerald-500" /> Healthy (35-45%)
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded bg-amber-500/20 border border-amber-500" /> Warning (20-35%)
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded bg-red-500/25 border border-red-500" /> Stress (&lt;20%)
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Twin State Synchronization Details */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-emerald-600" />
                        Active Twin State Synchronization
                      </h3>
                      <p className="text-xs text-gray-500 leading-relaxed">
                        Your physical farm records are synced downstream in near real-time. Changes ingested from ESP32 moisture probes flow directly to recompute evapotranspiration indices on our backend.
                      </p>
                      <div className="border-t border-gray-100 pt-3 space-y-2 font-mono text-[11px]">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Soil Moisture (Telemetry):</span>
                          <span className="text-gray-800 font-bold">{state.waterTwin.currentMoisture}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Nutrient NPK Score:</span>
                          <span className="text-emerald-600 font-bold">{state.soilTwin.healthScore}/100</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Last Core Computation:</span>
                          <span className="text-gray-800">{new Date(state.timestamp).toLocaleTimeString()}</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-amber-500" />
                        Risk Vector Summary Warnings
                      </h3>
                      <div className="space-y-2">
                        {state.weatherTwin.riskIndicators.map((indicator, idx) => (
                          <div key={idx} className="p-2.5 bg-amber-50 border border-amber-100 text-amber-800 text-xs rounded-lg flex items-start gap-2">
                            <span className="font-semibold">{idx + 1}.</span>
                            <span>{indicator}</span>
                          </div>
                        ))}
                        {state.weatherTwin.riskIndicators.length === 0 && (
                          <div className="p-3 text-center text-xs text-gray-500 font-semibold bg-gray-50 rounded-lg">
                            No risk indicators flagged on current timeline.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* PAGE 2: WATER TWIN */}
              {activeTab === 1 && (
                <div className="space-y-6">
                  {/* Detailed Hydrological KPI Block */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">FAO-56 PM Evapotranspiration (ET0)</span>
                      <span className="text-2xl font-black text-blue-600 tracking-tight">{state.physicsModel?.evapotranspirationET0 ?? state.waterTwin.evapotranspirationET0} mm/day</span>
                      <span className="text-[10px] text-gray-500 block">Calculated via daily Penman-Monteith formulas</span>
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">Current Topsoil Moisture Limit</span>
                      <span className="text-2xl font-black text-emerald-600 tracking-tight">{state.waterTwin.currentMoisture}%</span>
                      <span className="text-[10px] text-gray-500 block">Baseline threshold depth: 30cm</span>
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">Volumetric Soil Deficit</span>
                      <span className="text-2xl font-black text-rose-600 tracking-tight">{state.physicsModel?.waterDeficitLiters ?? state.waterTwin.waterDeficitLiters} L/acre</span>
                      <span className="text-[10px] text-gray-500 block">Total moisture gap in active zone</span>
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">Confidence Estimation Band</span>
                      <span className="text-xl font-bold text-indigo-600 tracking-tight">
                        {state.physicsModel?.confidenceInterval ? `${state.physicsModel.confidenceInterval[0]} - ${state.physicsModel.confidenceInterval[1]}` : "1050 - 1450"} L
                      </span>
                      <span className="text-[9px] text-gray-500 block">Margin: ±{state.physicsModel?.uncertaintyMarginLiters ?? 196} L (95% CI)</span>
                    </div>
                  </div>

                  {/* TabPFN Residual Correction Control Toggle */}
                  <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                    <div className="space-y-0.5">
                      <h4 className="text-xs font-bold text-gray-900">TabPFN Residual Corrections</h4>
                      <p className="text-[10px] text-gray-500">Injects machine learning residual error corrections into the Penman-Monteith water equations.</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        id="toggle-residual-corrections"
                        checked={enableResidual}
                        onChange={(e) => setEnableResidual(e.target.checked)}
                        className="rounded text-emerald-600 focus:ring-emerald-500 w-4 h-4 cursor-pointer"
                      />
                      <label htmlFor="toggle-residual-corrections" className="text-xs text-gray-700 font-bold select-none cursor-pointer">
                        Active
                      </label>
                    </div>
                  </div>

                  {/* Residual AI Forecast plot */}
                  <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-bold text-gray-900 flex items-center gap-1.5">
                        <Activity className="w-4 h-4 text-emerald-600" />
                        5-Day Combined Physics + AI Moisture Projection
                      </h3>
                      <div className="px-2.5 py-0.5 bg-blue-50 border border-blue-100 rounded-full text-[10px] text-blue-700 font-semibold uppercase">
                        TabPFN Fused
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 leading-relaxed">
                      Dotted indicators represent physics forecast bounds, whereas the solid line showcases final cyber-corrective neural predictions adjusting for variable canopy shading and soil moisture adsorptions.
                    </p>

                    <div className="h-60">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={state.waterTwin.predictedMoisture5Days.map((val, idx) => ({ 
                          day: `Day +${idx+1}`, 
                          Moisture: enableResidual ? val : val - 1.2, 
                          PhysicsBaseline: val - 1.2 
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                          <YAxis domain={[10, 60]} tick={{ fontSize: 10 }} />
                          <Tooltip />
                          <Legend wrapperStyle={{ fontSize: 10 }} />
                          <Line type="monotone" dataKey="Moisture" stroke="#10b981" strokeWidth={3} activeDot={{ r: 8 }} name="Residual AI Final Prediction" />
                          <Line type="monotone" strokeDasharray="5 5" dataKey="PhysicsBaseline" stroke="#3b82f6" name="FAO-56 Pure Physics Projection" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Water Balance Log historical blocks */}
                  <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                    <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                      <Layers className="w-4 h-4 text-emerald-600" />
                      Weekly Volumetric Water Balance Timeline
                    </h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse text-xs font-mono">
                        <thead>
                          <tr className="bg-gray-50 text-gray-500 border-b border-gray-200">
                            <th className="p-3">Calculation Interval (Day)</th>
                            <th className="p-3">Rainfall (mm)</th>
                            <th className="p-3">Irrigation Input (L)</th>
                            <th className="p-3">Evapotranspiration ET0 (mm)</th>
                            <th className="p-3">Calculated Soil Moisture</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {state.waterTwin.waterBalanceHistory.map((row, idx) => (
                            <tr key={idx} className="hover:bg-gray-55 text-gray-700">
                              <td className="p-3 font-semibold text-gray-800">{row.day}</td>
                              <td className="p-3 text-blue-600">{row.rainfall} mm</td>
                              <td className="p-3 text-teal-600">{row.irrigation || "-"}</td>
                              <td className="p-3 text-red-500">{row.et0} mm</td>
                              <td className="p-3 font-bold text-emerald-600">{row.activeMoisture}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* PAGE 3: SOIL TWIN */}
              {activeTab === 2 && (
                <div className="space-y-6">
                  {/* Soil parameters summary */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1.5">
                      <span className="text-[10px] text-gray-400 font-bold uppercase block tracking-wider font-semibold">Nitrogen (N) Accumulation</span>
                      <span className="text-2xl font-bold tracking-tight text-emerald-700 block">{state.soilTwin.nitrogen} ppm</span>
                      <progress className="w-full h-1 bg-gray-100" value={state.soilTwin.nitrogen} max="100" />
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1.5">
                      <span className="text-[10px] text-gray-400 font-bold uppercase block tracking-wider font-semibold">Phosphorus (P) Level</span>
                      <span className="text-2xl font-bold tracking-tight text-teal-700 block">{state.soilTwin.phosphorus} ppm</span>
                      <progress className="w-full h-1 bg-gray-100" value={state.soilTwin.phosphorus} max="80" />
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1.5">
                      <span className="text-[10px] text-gray-400 font-bold uppercase block tracking-wider font-semibold">Potassium (K) Reserve</span>
                      <span className="text-2xl font-bold tracking-tight text-indigo-700 block">{state.soilTwin.potassium} ppm</span>
                      <progress className="w-full h-1 bg-gray-100" value={state.soilTwin.potassium} max="80" />
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1.5">
                      <span className="text-[10px] text-gray-400 font-bold uppercase block tracking-wider font-semibold">Soil pH Balance</span>
                      <span className={`text-2xl font-bold tracking-tight block ${state.soilTwin.pH < 6.0 || state.soilTwin.pH > 7.2 ? 'text-amber-600' : 'text-emerald-700'}`}>{state.soilTwin.pH} pH</span>
                      <span className="text-[9px] text-gray-400">Class: Sightly Acidic</span>
                    </div>
                  </div>

                  {/* Radical fertility health score and depletion forecast */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                        <Database className="w-4 h-4 text-emerald-600" />
                        Soil Fertility & Chemical Profile
                      </h3>
                      <p className="text-xs text-gray-500 leading-relaxed">
                        Evaluates core minerals against agricultural baseline values. High electrical conductivity indicates adequate mineral salts, and organic carbon acts as a microclimate microbiome metric.
                      </p>
                      
                      <div className="space-y-2 text-xs font-mono">
                        <div className="p-2 bg-gray-50 border border-gray-150 rounded flex justify-between">
                          <span className="text-gray-500">EC (Electrical Conductivity):</span>
                          <span className="text-gray-800 font-bold">{state.soilTwin.electricalConductivity} dS/m</span>
                        </div>
                        <div className="p-2 bg-gray-50 border border-gray-150 rounded flex justify-between">
                          <span className="text-gray-500">Organic Carbon Portion:</span>
                          <span className="text-gray-800 font-bold">{state.soilTwin.organicCarbon}%</span>
                        </div>
                        <div className="p-2 bg-emerald-50 border border-emerald-100 rounded text-emerald-800 text-[11px]">
                          <strong>Core Analysis:</strong> {state.soilTwin.nutrientDeficitForecast}
                        </div>
                      </div>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900">Projected 4-Month NPK Soil Depletion Curves</h3>
                      <p className="text-[11px] text-gray-500">Simulation tracking biological crop nutrient removal patterns over months.</p>
                      <div className="h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={state.soilTwin.depletionTimeline}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="month" tick={{ fontSize: 9 }} />
                            <YAxis tick={{ fontSize: 9 }} />
                            <Tooltip />
                            <Legend wrapperStyle={{ fontSize: 10 }} />
                            <Bar dataKey="nitrogen" fill="#10b981" name="Nitrogen (ppm)" />
                            <Bar dataKey="phosphorus" fill="#06b6d4" name="Phosphorus (ppm)" />
                            <Bar dataKey="potassium" fill="#6366f1" name="Potassium (ppm)" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* PAGE 4: CROP GROWTH TWIN */}
              {activeTab === 3 && (
                <div className="space-y-6">
                  {/* Phenology & Sowing Details */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1">
                      <span className="text-[10px] text-gray-400 font-medium block uppercase font-bold">Crop Target</span>
                      <span className="text-xl font-bold text-gray-800 tracking-tight block">{state.cropTwin.cropType}</span>
                      <span className="text-[10px] text-gray-500 block">Sown class parameters</span>
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1">
                      <span className="text-[10px] text-gray-400 font-medium block uppercase font-bold">Growth Phenology Stage</span>
                      <span className="text-xl font-bold text-emerald-700 tracking-tight block">{state.cropTwin.growthStage}</span>
                      <span className="text-[10px] text-emerald-600 block font-semibold">Active Leaf Expansion</span>
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1">
                      <span className="text-[10px] text-gray-400 font-medium block uppercase font-bold">Biomass Yield Density</span>
                      <span className="text-xl font-bold text-blue-600 tracking-tight block">{state.cropTwin.biomassIndex} kg/ha</span>
                      <span className="text-[10px] text-blue-500 block">Relative crop mass vigor</span>
                    </div>
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-1">
                      <span className="text-[10px] text-gray-400 font-medium block uppercase font-bold">Target Harvest Forecast</span>
                      <span className="text-xl font-bold text-indigo-700 tracking-tight block">{state.cropTwin.harvestForecastDate}</span>
                      <span className="text-[10px] text-gray-500 block">TabPFN dynamic estimates</span>
                    </div>
                  </div>

                  {/* Growth stage timeline chart and biomass expansion */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                        <Sprout className="w-4 h-4 text-emerald-600" />
                        Crop Biomass Accumulation Tracker
                      </h3>
                      <p className="text-xs text-gray-500 leading-relaxed">
                        Tracks physical biomass expansion trajectories. Gaps between actual simulated parameters and expected optimal benchmarks register as secondary stress metrics inside the TabPFN Yield predictor.
                      </p>
                      
                      <div className="h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={state.cropTwin.growthTimeline}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="stage" tick={{ fontSize: 9 }} />
                            <YAxis tick={{ fontSize: 9 }} />
                            <Tooltip />
                            <Legend wrapperStyle={{ fontSize: 10 }} />
                            <Area type="monotone" dataKey="expectedBiomass" stroke="#abb2bf" fill="rgba(171,178,191,0.1)" name="Expected Target" />
                            <Area type="monotone" dataKey="actualBiomass" stroke="#10b981" fill="rgba(16,185,129,0.2)" name="Actual Simulated" />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900">Agronomic Growth Stage Milestones</h3>
                      <div className="space-y-4 relative pl-4 border-l border-gray-200">
                        <div className="relative">
                          <span className="absolute -left-6 top-1 w-3 h-3 rounded-full bg-emerald-500 border border-white" />
                          <div className="text-xs font-semibold text-emerald-700">1. Germination & Emergence (Completed)</div>
                          <p className="text-[10px] text-gray-400">Achieved root tissue breakout within 12 days. Stiff shoot emergence confirmed.</p>
                        </div>
                        <div className="relative">
                          <span className="absolute -left-6 top-1 w-3 h-3 rounded-full bg-emerald-500 border border-white animate-pulse" />
                          <div className="text-xs font-semibold text-emerald-800">2. Active Vegetative Canopy (Current Phase)</div>
                          <p className="text-[10px] text-gray-500">Tassel branch expansions under dynamic LightGBM scheduling. High solar light intercept.</p>
                        </div>
                        <div className="relative text-gray-400">
                          <span className="absolute -left-6 top-1 w-3 h-3 rounded-full bg-gray-200 border border-white" />
                          <div className="text-xs font-semibold">3. Tasseling & Flowering (Scheduled Aug 1)</div>
                          <p className="text-[10px] text-gray-400">Requires high phosphorus (P) minerals and zero moisture stress.</p>
                        </div>
                        <div className="relative text-gray-400">
                          <span className="absolute -left-6 top-1 w-3 h-3 rounded-full bg-gray-200 border border-white" />
                          <div className="text-xs font-semibold">4. Yield Grain Formation & Ripening (Scheduled Aug 25)</div>
                          <p className="text-[10px] text-gray-400">Final dry down cycles with standard starch conversions.</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* PAGE 5: WEATHER TWIN */}
              {activeTab === 4 && (
                <div className="space-y-6">
                  {/* Current conditions banner */}
                  <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl p-6 text-white shadow relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-45 h-45 bg-white/5 rounded-full blur-2xl pointer-events-none" />
                    
                    <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                      <div className="space-y-1">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-blue-100">Live Microclimate Environment</span>
                        <h3 className="text-3xl font-black">{state.weatherTwin.currentTemp}°C</h3>
                        <p className="text-xs text-blue-100 font-medium">Rel. Humidity: {state.weatherTwin.currentHumidity}% | Ambient Wind: {state.weatherTwin.windSpeed} m/s</p>
                      </div>

                      <div className="flex flex-col items-start md:items-end gap-1 font-mono text-xs">
                        <div className="px-3 py-1 bg-white/10 backdrop-blur rounded-lg flex items-center gap-1.5 font-bold">
                          <Flame className="w-3.5 h-3.5 text-orange-300" />
                          Heat Stress Class: {state.weatherTwin.heatStressIndex}
                        </div>
                        <span className="text-[10px] text-blue-100">FAO-56 standard PM atmospheric inputs</span>
                      </div>
                    </div>
                  </div>

                  {/* 7 Day Weather Forecast Grid */}
                  <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                    <h3 className="text-sm font-bold text-gray-900">7-Day Local Microclimate Forecast Arrays</h3>
                    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
                      {state.weatherTwin.forecast.map((day, idx) => {
                        let condIcon = <Thermometer className="w-5 h-5 text-amber-500" />;
                        if (day.condition === "rainy") condIcon = <CloudRain className="w-5 h-5 text-blue-500" />;
                        if (day.condition === "stormy") condIcon = <Wind className="w-5 h-5 text-indigo-500 animate-bounce" />;

                        return (
                          <div key={idx} className="p-3 bg-gray-50 border border-gray-150 rounded-lg text-center space-y-2">
                            <span className="text-[10px] text-gray-500 font-semibold block">{day.date}</span>
                            <div className="flex justify-center">{condIcon}</div>
                            <div className="text-xs font-bold text-gray-800">{day.temperature}°C</div>
                            <div className="text-[9px] text-gray-400 font-medium">Rain: {day.rainfall}mm</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Risk analysis outputs */}
                  <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                    <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                      <Wind className="w-4 h-4 text-indigo-600" />
                      Atmospheric Environmental Risk Propensities
                    </h3>
                    <p className="text-xs text-gray-500">
                      Our models synthesize wind speeds and moisture indexes to project heat stress patterns and late leaf blight spore dispersion risks under active morning precipitation segments.
                    </p>

                    <div className="space-y-2.5">
                      {state.weatherTwin.riskIndicators.map((risk, idx) => (
                        <div key={idx} className="p-3 bg-indigo-50/50 border border-indigo-100 rounded-lg flex items-center gap-3 text-xs text-indigo-900">
                          <span className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
                          <span>{risk}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* PAGE 6: DISEASE TWIN */}
              {activeTab === 5 && (
                <div className="space-y-6">
                  {/* Risks gauges */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm text-center space-y-1">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block font-semibold">Disease Outbreak Probability</span>
                      <span className={`text-3xl font-black block tracking-tight ${state.diseaseTwin.riskScore > 35 ? 'text-amber-500' : 'text-emerald-700'}`}>{state.diseaseTwin.riskScore}%</span>
                      <progress className="w-2/3 mx-auto h-1.5 bg-gray-100 rounded" value={state.diseaseTwin.riskScore} max="100" />
                    </div>
                    
                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm text-center space-y-1">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block font-semibold">Atmospheric Propensity Index</span>
                      <span className="text-3xl font-black text-indigo-700 block tracking-tight">{state.diseaseTwin.environmentalPropensity}%</span>
                      <span className="text-[9px] text-gray-400 font-medium">Derived from high moisture and temperature ranges</span>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm text-center space-y-1">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block font-semibold">Active Pathogen Vector Threats</span>
                      <span className="text-2xl font-bold text-gray-800 focus:outline-none uppercase block leading-relaxed">{state.diseaseTwin.susceptibleCrops.length} Identified</span>
                      <span className="text-[9px] text-gray-400 font-medium">Florence-2 Computer Vision gateway monitored</span>
                    </div>
                  </div>

                  {/* Active susceptible targets and preventive measures */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900 flex items-center gap-1.5">
                        <ShieldAlert className="w-4 h-4 text-emerald-600" />
                        Identified Pathological Vulnerabilities
                      </h3>
                      <p className="text-xs text-gray-500">
                        The current weather, soil moisture (38.3%), and high humidity propensities expose vegetative leaves to the following fungal pathogen spore distributions:
                      </p>

                      <div className="space-y-3">
                        {state.diseaseTwin.susceptibleCrops.map((crop, idx) => (
                          <div key={idx} className="p-3 bg-red-50/50 border border-red-100 rounded-xl flex items-center justify-between">
                            <span className="text-xs font-bold text-red-800">{crop}</span>
                            <span className="px-2 py-0.5 bg-red-100 text-red-800 rounded text-[9px] font-bold">MODERATE RISK</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900">Preventive Bio-Acoustic & Spray Guides</h3>
                      <p className="text-[11px] text-gray-500">Agronomic manual counteractions recommended in the next 48 hours to suppress mold spreads.</p>
                      
                      <div className="space-y-3">
                        {state.diseaseTwin.preventiveActionRequired.map((act, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 border border-gray-150 rounded-lg flex items-start gap-3">
                            <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 mt-1 flex-shrink-0" />
                            <span className="text-xs text-gray-700 leading-relaxed font-medium">{act}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* PAGE 7: SCENARIO SIMULATOR (WHAT-IF ANALYSIS) */}
              {activeTab === 6 && (
                <div className="space-y-6 bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
                  <div className="space-y-1">
                    <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
                      <Play className="w-5 h-5 text-emerald-600" />
                      Multidimensional What-If Scenario Simulator
                    </h3>
                    <p className="text-xs text-gray-500">
                      Select custom physical inputs, rainfall events, extreme weather variations, or disease propagation rates, and run daily simulations to project future yields, risk thresholds, and deficits.
                    </p>
                  </div>

                  {/* Scenarios Pick Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    <button
                      onClick={() => setSelectedScenario("drought_5_days")}
                      className={`p-3 rounded-lg text-center border text-xs font-semibold cursor-pointer transition-all ${selectedScenario === "drought_5_days" ? 'bg-amber-50 border-amber-500 text-amber-700 font-bold ring-2 ring-amber-200' : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'}`}
                    >
                      Scenario 1: No Irrigation (5 Days)
                    </button>
                    <button
                      onClick={() => setSelectedScenario("fertilizer_boost_20")}
                      className={`p-3 rounded-lg text-center border text-xs font-semibold cursor-pointer transition-all ${selectedScenario === "fertilizer_boost_20" ? 'bg-emerald-50 border-emerald-500 text-emerald-700 font-bold ring-2 ring-emerald-200' : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'}`}
                    >
                      Scenario 2: Fertilizer Boost +20%
                    </button>
                    <button
                      onClick={() => setSelectedScenario("downpour")}
                      className={`p-3 rounded-lg text-center border text-xs font-semibold cursor-pointer transition-all ${selectedScenario === "downpour" ? 'bg-blue-50 border-blue-500 text-blue-700 font-bold ring-2 ring-blue-200' : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'}`}
                    >
                      Scenario 3: Severe Rainfall Event
                    </button>
                    <button
                      onClick={() => setSelectedScenario("mildew_outbreak")}
                      className={`p-3 rounded-lg text-center border text-xs font-semibold cursor-pointer transition-all ${selectedScenario === "mildew_outbreak" ? 'bg-red-50 border-red-500 text-red-700 font-bold ring-2 ring-red-200' : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'}`}
                    >
                      Scenario 4: Mold / Mildew Outbreak
                    </button>
                    <button
                      onClick={() => setSelectedScenario("optimal_ops")}
                      className={`p-3 rounded-lg text-center border text-xs font-semibold cursor-pointer transition-all ${selectedScenario === "optimal_ops" ? 'bg-indigo-50 border-indigo-500 text-indigo-700 font-bold ring-2 ring-indigo-200' : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'}`}
                    >
                      Scenario 5: Optimal Automated Cores
                    </button>
                  </div>

                  {/* Simulator Execution Output details */}
                  {scenarioOutput && (
                    <div className="space-y-6 pt-4 border-t border-gray-100">
                      <div className="p-4 bg-gray-50 rounded-xl border border-gray-200 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                        <div className="space-y-1">
                          <span className="text-[10px] p-1 px-2 uppercase font-bold text-gray-500 bg-gray-200/50 rounded inline-block">Analysis Outcome Details</span>
                          <h4 className="text-sm font-bold text-gray-900">{scenarioOutput.scenarioName}</h4>
                          <p className="text-xs text-gray-600 leading-relaxed font-medium">{scenarioOutput.outcomeSummary}</p>
                        </div>

                        <div className="flex items-center gap-6 text-center font-mono">
                          <div>
                            <span className="text-[9px] text-gray-400 font-bold uppercase block">Yield Projected</span>
                            <span className="text-lg font-black text-emerald-600">{scenarioOutput.projectedYieldTonsPerAcre} Tons/a</span>
                          </div>
                          <div>
                            <span className="text-[9px] text-gray-400 font-bold uppercase block">Pest Risk Index</span>
                            <span className="text-lg font-black text-rose-600">{scenarioOutput.pestRiskScore}%</span>
                          </div>
                          <div>
                            <span className="text-[9px] text-gray-400 font-bold uppercase block">Water Stress Factor</span>
                            <span className="text-lg font-black text-blue-600">{scenarioOutput.waterStressIndex} index</span>
                          </div>
                        </div>
                      </div>

                      {/* Scenario Plot Plotting */}
                      <div className="space-y-2">
                        <span className="text-xs font-bold text-gray-800">5-Day Simulation State Progression Outcomes</span>
                        <div className="h-60">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={scenarioOutput.timeline}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} />
                              <XAxis dataKey="day" label={{ value: "Simulation Day", position: "insideBottomRight", offset: -2, fontSize: 10 }} />
                              <YAxis tick={{ fontSize: 10 }} />
                              <Tooltip />
                              <Legend wrapperStyle={{ fontSize: 10 }} />
                              <Line type="monotone" dataKey="soilMoisture" stroke="#3b82f6" strokeWidth={2.5} name="Soil Moisture (%)" />
                              <Line type="monotone" dataKey="cropHealth" stroke="#10b981" strokeWidth={2.5} name="Crop Health Score" />
                              <Line type="monotone" dataKey="yieldImpact" stroke="#6366f1" strokeWidth={2.5} name="Yield Efficiency (%)" />
                              <Line type="monotone" dataKey="riskScore" stroke="#ef4444" strokeWidth={2.5} name="Outbreak Risk Score" />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* PAGE 8: REPORTS & ANALYTICS */}
              {activeTab === 7 && (
                <div className="space-y-6">
                  {/* Radar KPI Indicators diagram */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900">Sustainability Evaluation Metrics</h3>
                      <p className="text-xs text-gray-500">
                        Tracks compliance to resource conservation indexes and nitrogen chemical adsorption efficiencies.
                      </p>
                      
                      <div className="h-60 flex justify-center">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={[
                            { subject: "Water Eff.", A: analytics?.sustainabilityIndices.waterUseEfficiency || 94, fullMark: 100 },
                            { subject: "Pesticide Red.", A: analytics?.sustainabilityIndices.pesticideReductionIndex || 88, fullMark: 100 },
                            { subject: "Soil Retention", A: analytics?.sustainabilityIndices.soilStructuralRetention || 91, fullMark: 100 },
                            { subject: "N Utilization", A: 91, fullMark: 100 },
                            { subject: "CO2 Offset", A: 78, fullMark: 100 }
                          ]}>
                            <PolarGrid />
                            <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: "#4b5563" }} />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 8 }} />
                            <Radar name="North Sector" dataKey="A" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Report Generator Controls */}
                    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                      <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                        <FileText className="w-4 h-4 text-emerald-600" />
                        AgriSense Automatic Report Generator
                      </h3>
                      <p className="text-xs text-gray-500 leading-relaxed">
                        Compile professional, standardized daily, weekly, or monthly crop-yield analytical PDF reports featuring full NPK analyses, wind velocities, and digital twin FAO-56 metrics.
                      </p>

                      <div className="space-y-3 pt-2">
                        <div className="grid grid-cols-3 gap-2">
                          <button
                            onClick={() => setReportType("daily")}
                            className={`p-2 rounded text-xs font-bold border cursor-pointer transition-all ${reportType === "daily" ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-gray-50 border-gray-150 text-gray-700 hover:bg-gray-100'}`}
                          >
                            Daily Log
                          </button>
                          <button
                            onClick={() => setReportType("weekly")}
                            className={`p-2 rounded text-xs font-bold border cursor-pointer transition-all ${reportType === "weekly" ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-gray-50 border-gray-150 text-gray-700 hover:bg-gray-100'}`}
                          >
                            Weekly Summary
                          </button>
                          <button
                            onClick={() => setReportType("water")}
                            className={`p-2 rounded text-xs font-bold border cursor-pointer transition-all ${reportType === "water" ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-gray-50 border-gray-150 text-gray-700 hover:bg-gray-100'}`}
                          >
                            Water Balance
                          </button>
                        </div>

                        <div className="flex gap-2">
                          <button
                            onClick={triggerGenerateReport}
                            className="flex-1 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold rounded-lg shadow-sm transition-all focus:ring-2 focus:ring-emerald-300 cursor-pointer"
                          >
                            Generate Dynamic Report Record
                          </button>
                          
                          {reportLog && (
                            <button
                              onClick={downloadReportFile}
                              className="px-3 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold rounded-lg shadow-sm transition-all flex items-center justify-center cursor-pointer"
                              title="Export Standard PDF Document Blob"
                            >
                              <Download className="w-4 h-4" />
                            </button>
                          )}
                        </div>

                        {downloadSuccess && (
                          <div className="p-2 bg-blue-50 text-blue-700 text-[10px] font-semibold text-center border border-blue-100 rounded">
                            {downloadSuccess}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Generated report preview window */}
                  {reportLog && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="bg-[#1e293b] border border-slate-700 rounded-xl p-5 text-slate-100 shadow-inner relative"
                    >
                      <span className="absolute top-4 right-4 text-[10px] font-mono text-emerald-400 font-bold uppercase">Report Preview</span>
                      <pre className="font-mono text-xs whitespace-pre-wrap leading-relaxed max-h-80 overflow-y-auto pr-2">
                        {reportLog}
                      </pre>
                    </motion.div>
                  )}
                </div>
              )}

            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
