/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, Suspense, lazy } from "react";
import {
  LayoutDashboard, Layers, ScanLine, Sprout, Droplet,
  Radio, CloudSun, TrendingUp, Bot, Cpu, Settings2,
  Menu, X, Leaf, ChevronRight, Wifi, WifiOff, BookOpen, ShoppingBag, Terminal, LogOut, ChevronDown, Globe
} from "lucide-react";
import { useTranslation } from "./hooks/useTranslation";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const DigitalTwin = lazy(() => import("./pages/DigitalTwin"));
const DiseaseDetection = lazy(() => import("./pages/DiseaseDetection"));
const CropRecommendation = lazy(() => import("./pages/CropRecommendation"));
const CropDatabase = lazy(() => import("./pages/CropDatabase"));
const IrrigationOptimization = lazy(() => import("./pages/IrrigationOptimization"));
const SensorMonitoring = lazy(() => import("./pages/SensorMonitoring"));
const WeatherIntelligence = lazy(() => import("./pages/WeatherIntelligence"));
const YieldPrediction = lazy(() => import("./pages/YieldPrediction"));
const AgriGPT = lazy(() => import("./pages/AgriGPT"));
const MLOpsDashboard = lazy(() => import("./pages/MLOpsDashboard"));
const AgriOpsDashboard = lazy(() => import("./pages/AgriOpsDashboard"));
const Settings = lazy(() => import("./pages/Settings"));
const AgentDashboard = lazy(() => import("./pages/AgentDashboard"));
const Marketplace = lazy(() => import("./pages/Marketplace"));
const LocalAIHub = lazy(() => import("./pages/LocalAIHub"));
const MarketIntelligence = lazy(() => import("./pages/MarketIntelligence"));

import Login from "./pages/Login";
import { SensorReading } from "./types";

interface Farm {
  id: number;
  name: string;
  location: string;
  fields: any[];
}

const navSections = [
  {
    title: "Overview",
    items: [
      { id: "dashboard", label: "Dashboard", icon: <LayoutDashboard className="w-4 h-4" /> },
      { id: "twin",      label: "Digital Twin", icon: <Layers className="w-4 h-4" />, badge: "LIVE" },
    ]
  },
  {
    title: "AI Vision",
    items: [
      { id: "disease",   label: "Disease Vision", icon: <ScanLine className="w-4 h-4" /> },
    ]
  },
  {
    title: "Field Intelligence",
    items: [
      { id: "crop",       label: "Crop Suitability",  icon: <Sprout className="w-4 h-4" /> },
      { id: "crops",      label: "Crop Catalog",      icon: <BookOpen className="w-4 h-4" /> },
      { id: "irrigation", label: "Irrigation",        icon: <Droplet className="w-4 h-4" /> },
      { id: "sensors",    label: "IoT Sensors",       icon: <Radio className="w-4 h-4" /> },
      { id: "weather",    label: "Weather Intel",     icon: <CloudSun className="w-4 h-4" /> },
      { id: "yield",      label: "Yield Forecast",    icon: <TrendingUp className="w-4 h-4" /> },
    ]
  },
  {
    title: "Commerce & Infrastructure",
    items: [
      { id: "marketplace", label: "Agri Marketplace", icon: <ShoppingBag className="w-4 h-4" /> },
      { id: "market_intelligence", label: "Market Intelligence", icon: <TrendingUp className="w-4 h-4" />, badge: "LIVE" },
      { id: "aihub",       label: "Local AI Hub",     icon: <Terminal className="w-4 h-4" />, badge: "GPU" },
      { id: "agrigpt",     label: "AgriGPT Chat",     icon: <Bot className="w-4 h-4" /> },
      { id: "agents",      label: "ASO Swarm",        icon: <Cpu className="w-4 h-4" /> },
      { id: "mlops",       label: "MLOps Control",    icon: <Cpu className="w-4 h-4" /> },
      { id: "agriops",     label: "AgriOps Hub",      icon: <Layers className="w-4 h-4" />, badge: "NEW" },
      { id: "settings",    label: "Settings",         icon: <Settings2 className="w-4 h-4" /> },
    ]
  }
];

export default function App() {
  const { t, language, setLanguage } = useTranslation();
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("agrisense_token"));
  const [profile, setProfile] = useState<{ email: string; role: string } | null>(() => {
    try {
      const cached = localStorage.getItem("agrisense_profile");
      return cached ? JSON.parse(cached) : null;
    } catch {
      return null;
    }
  });

  const [activePage, setActivePage] = useState<string>("dashboard");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [farms, setFarms] = useState<Farm[]>([]);
  const [selectedFarm, setSelectedFarm] = useState<Farm | null>(null);
  const [farmDropdownOpen, setFarmDropdownOpen] = useState(false);

  const [sensors, setSensors] = useState<SensorReading[]>(() => {
    try {
      const cached = localStorage.getItem("agrisense_cached_telemetry");
      return cached ? JSON.parse(cached) : [];
    } catch {
      return [];
    }
  });
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const fetchTelemetry = async () => {
    try {
      const res = await fetch("/api/sensors");
      const data = await res.json();
      if (data.readings) {
        setSensors(data.readings);
        setBackendOnline(true);
        localStorage.setItem("agrisense_cached_telemetry", JSON.stringify(data.readings));
      }
    } catch {
      setBackendOnline(false);
      try {
        const cached = localStorage.getItem("agrisense_cached_telemetry");
        if (cached) {
          setSensors(JSON.parse(cached));
        }
      } catch (e) {
        console.error("Failed to parse cached telemetry", e);
      }
    }
  };

  const fetchFarms = async () => {
    try {
      const res = await fetch("/api/farms");
      if (res.ok) {
        const data = await res.json();
        setFarms(data);
        if (data.length > 0) {
          setSelectedFarm(data[0]);
        }
      }
    } catch (e) {
      console.error("Failed to fetch farms", e);
    }
  };

  useEffect(() => {
    if (token) {
      fetchTelemetry();
      fetchFarms();
      const interval = setInterval(fetchTelemetry, 65000);
      return () => clearInterval(interval);
    }
  }, [token]);

  const handleSimulateIngest = async (fieldVals: Partial<SensorReading>) => {
    try {
      const response = await fetch("/api/sensors/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fieldVals)
      });
      if (response.ok) await fetchTelemetry();
    } catch (err) {
      console.error("Ingest failed", err);
    }
  };

  const handleNavigation = (id: string) => {
    setActivePage(id);
    setMobileMenuOpen(false);
  };

  const handleLoginSuccess = (newToken: string, newProfile: { email: string; role: string; preferred_language?: string }) => {
    setToken(newToken);
    setProfile(newProfile);
    if (newProfile.preferred_language) {
      setLanguage(newProfile.preferred_language as any);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("agrisense_token");
    localStorage.removeItem("agrisense_profile");
    setToken(null);
    setProfile(null);
  };

  if (!token) {
    return <Login onLoginSuccess={handleLoginSuccess} />;
  }

  const renderContent = () => {
    return (
      <Suspense fallback={<div className="p-10 text-center text-sm font-semibold text-emerald-800">Loading Module...</div>}>
        {(() => {
          switch (activePage) {
            case "dashboard":   return <Dashboard onNavigate={handleNavigation} sensors={sensors} />;
            case "twin":        return <DigitalTwin />;
            case "disease":     return <DiseaseDetection />;
            case "crop":        return <CropRecommendation />;
            case "crops":       return <CropDatabase />;
            case "irrigation":  return <IrrigationOptimization />;
            case "sensors":     return <SensorMonitoring sensors={sensors} onRefresh={fetchTelemetry} onSimulateIngest={handleSimulateIngest} />;
            case "weather":     return <WeatherIntelligence />;
            case "yield":       return <YieldPrediction />;
            case "marketplace": return <Marketplace />;
            case "market_intelligence": return <MarketIntelligence />;
            case "aihub":       return <LocalAIHub />;
            case "agrigpt":     return <AgriGPT sensors={sensors} />;
            case "agents":      return <AgentDashboard />;
            case "mlops":       return <MLOpsDashboard />;
            case "agriops":     return <AgriOpsDashboard />;
            case "settings":    return <Settings />;
            default:            return <Dashboard onNavigate={handleNavigation} sensors={sensors} />;
          }
        })()}
      </Suspense>
    );
  };

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-emerald-900/40">
        <div className="flex items-center gap-3">
          <img src="/src/assets/logo.png" alt="Logo" className="w-9 h-9 object-cover rounded-xl shadow-lg shadow-emerald-900/40 flex-shrink-0" />
          <div>
            <span className="text-white font-black text-base tracking-tight block leading-tight">AgriSense</span>
            <span className="text-amber-400 text-[9px] font-bold uppercase tracking-[0.12em] font-mono block">Edge AI Platform</span>
          </div>
        </div>
      </div>

      {/* Farm Switcher */}
      {selectedFarm && (
        <div className="px-4 py-3 border-b border-emerald-900/40 relative">
          <button
            onClick={() => setFarmDropdownOpen(!farmDropdownOpen)}
            className="w-full px-3 py-2 bg-emerald-950/50 hover:bg-emerald-950/80 border border-emerald-800/40 rounded-lg flex items-center justify-between text-left text-xs font-semibold text-emerald-100 cursor-pointer transition-colors"
          >
            <div className="truncate">
              <span className="text-[8px] font-mono text-emerald-400/70 block uppercase tracking-wider">Active Field Context</span>
              <span className="truncate block mt-0.5">{selectedFarm.name}</span>
            </div>
            <ChevronDown className="w-3.5 h-3.5 text-emerald-400/70" />
          </button>
          {farmDropdownOpen && (
            <div className="absolute left-4 right-4 mt-1 bg-emerald-950 border border-emerald-800/80 rounded-lg shadow-xl overflow-hidden z-40 max-h-40 overflow-y-auto">
              {farms.map((f) => (
                <button
                  key={f.id}
                  onClick={() => {
                    setSelectedFarm(f);
                    setFarmDropdownOpen(false);
                  }}
                  className={`w-full px-3 py-2.5 text-left text-xs transition-colors cursor-pointer border-b border-emerald-900/40 hover:bg-emerald-900/60 ${
                    selectedFarm.id === f.id ? "text-amber-400 font-bold bg-emerald-900/40" : "text-emerald-100"
                  }`}
                >
                  {f.name}
                  <span className="block text-[8px] text-emerald-400/50 font-mono mt-0.5">{f.location}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-5 overflow-y-auto scrollbar-hide">
        {navSections.map((section) => {
          const secTitleKey = "nav." + (
            section.title === "AI Vision" ? "aivision" : 
            section.title === "Field Intelligence" ? "fieldintel" : 
            section.title === "Commerce & Infrastructure" ? "commerce" : "overview"
          );
          return (
            <div key={section.title}>
              <p className="text-[9px] font-bold uppercase tracking-[0.14em] text-emerald-600/70 px-2.5 mb-1.5 font-mono">
                {t(secTitleKey)}
              </p>
              <div className="space-y-0.5">
                {section.items.map((item) => {
                  const isActive = activePage === item.id;
                  const itemLabelKey = "nav." + (
                    item.id === "crops" ? "catalog" : 
                    item.id === "disease" ? "disease" : 
                    item.id === "crop" ? "suitability" : 
                    item.id === "sensors" ? "sensors" : 
                    item.id === "weather" ? "weather" : 
                    item.id === "yield" ? "yield" : 
                    item.id === "marketplace" ? "marketplace" : 
                    item.id === "market_intelligence" ? "market_intel" : 
                    item.id === "aihub" ? "aihub" : 
                    item.id === "agrigpt" ? "chat" : 
                    item.id === "agents" ? "swarm" : 
                    item.id === "mlops" ? "mlops" : 
                    item.id === "agriops" ? "agriops" : 
                    item.id === "settings" ? "settings" : item.id
                  );
                  return (
                    <button
                      key={item.id}
                      id={`sidebar-link-${item.id}`}
                      onClick={() => handleNavigation(item.id)}
                      className={`w-full px-3 py-2.5 rounded-lg text-[0.78rem] font-medium flex items-center gap-3 cursor-pointer transition-all duration-200 group ${
                        isActive
                          ? "nav-item-active"
                          : "text-emerald-200/70 hover:text-white hover:bg-white/[0.06]"
                      }`}
                    >
                      <span className={isActive ? "text-amber-900" : "text-emerald-400/80 group-hover:text-emerald-300 transition-colors"}>
                        {item.icon}
                      </span>
                      <span className="flex-1 text-left">{t(itemLabelKey)}</span>
                      {item.badge && (
                        <span className="text-[8px] font-bold px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 rounded font-mono tracking-wider">
                          {item.badge}
                        </span>
                      )}
                      {isActive && <ChevronRight className="w-3.5 h-3.5 text-amber-900/60" />}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </nav>

      {/* Language Selector Dropdown */}
      <div className="px-4 py-2 border-t border-emerald-900/40">
        <div className="flex items-center gap-2 bg-black/10 px-2.5 py-1.5 rounded-lg">
          <Globe className="w-3.5 h-3.5 text-emerald-400/80" />
          <select
            id="app-language-picker"
            value={language}
            onChange={(e) => setLanguage(e.target.value as any)}
            className="flex-1 bg-transparent border-none text-[10px] text-emerald-200 font-bold focus:ring-0 cursor-pointer outline-none font-mono"
          >
            <option value="en" className="bg-emerald-950 text-white">English</option>
            <option value="ta" className="bg-emerald-950 text-white">தமிழ் (Tamil)</option>
            <option value="te" className="bg-emerald-950 text-white">తెలుగు (Telugu)</option>
            <option value="ml" className="bg-emerald-950 text-white">മലയാളം (Malayalam)</option>
            <option value="hi" className="bg-emerald-950 text-white">हिन्दी (Hindi)</option>
          </select>
        </div>
      </div>

      {/* Backend status */}
      <div className="px-4 py-3 border-t border-emerald-900/40">
        <div className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg bg-black/20">
          {backendOnline === null ? (
            <span className="status-dot-amber" />
          ) : backendOnline ? (
            <span className="status-dot-green" />
          ) : (
            <span className="status-dot-red" />
          )}
          <div className="flex-1 min-w-0">
            <p className="text-[10px] font-bold text-white/90 font-mono">
              {backendOnline === null ? "Connecting..." : backendOnline ? "Edge Node Online" : "Backend Offline"}
            </p>
            <p className="text-[9px] text-emerald-400/60 font-mono">localhost:8000</p>
          </div>
          {backendOnline ? <Wifi className="w-3 h-3 text-emerald-400/60" /> : <WifiOff className="w-3 h-3 text-red-400/60" />}
        </div>

        {/* User */}
        <div className="flex items-center justify-between mt-2.5 px-1">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-amber-400 to-amber-600 text-amber-950 font-black text-[10px] flex items-center justify-center flex-shrink-0">
              {profile?.email ? profile.email.slice(0, 2).toUpperCase() : "AG"}
            </div>
            <div>
              <p className="text-[10.5px] font-bold text-white/90 truncate max-w-[110px]">{profile?.email || "Operator"}</p>
              <p className="text-[8.5px] text-emerald-400/70 font-mono uppercase tracking-wider">{profile?.role || "Active Operator"}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            title="Log Out Profile"
            className="p-1.5 rounded-lg hover:bg-white/[0.06] text-emerald-400 hover:text-white transition-colors cursor-pointer"
          >
            <LogOut className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex flex-col md:flex-row relative bg-[#f0f4f0]" id="agrisense-app-root">
      {/* Ambient background */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div className="absolute -top-24 -left-24 w-96 h-96 bg-emerald-500/[0.04] rounded-full blur-3xl" />
        <div className="absolute -bottom-24 -right-24 w-96 h-96 bg-amber-500/[0.04] rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-600/[0.02] rounded-full blur-3xl" />
      </div>

      {/* Desktop Sidebar */}
      <aside className="w-60 shrink-0 bg-gradient-to-b from-[#0c1a0e] to-[#0f2e1e] sticky top-0 h-screen hidden md:flex flex-col z-30 border-r border-emerald-950/60 shadow-2xl">
        <SidebarContent />
      </aside>

      {/* Mobile Header */}
      <header className="md:hidden bg-gradient-to-r from-[#0c1a0e] to-[#0f2e1e] border-b border-emerald-900/40 flex items-center justify-between px-5 py-3.5 sticky top-0 z-40 shadow-md">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center">
            <Leaf className="w-4 h-4 text-white" />
          </div>
          <div>
            <span className="text-white font-black text-sm">AgriSense</span>
          </div>
        </div>
        <button
          id="btn-mobile-menu"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-2 rounded-lg bg-white/[0.06] hover:bg-white/[0.12] border border-emerald-800/60 text-emerald-200 cursor-pointer transition-colors"
        >
          {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
        </button>
      </header>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed inset-0 z-30 bg-gradient-to-b from-[#0c1a0e] to-[#0f2e1e] pt-14 flex flex-col animate-fade-in" id="mobile-menu-drawer">
          <SidebarContent />
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 min-h-screen relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-7 md:py-8">
          <div className="animate-fade-in">
            {renderContent()}
          </div>
        </div>
      </main>
    </div>
  );
}
