/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Settings2, RefreshCw, Check, Info, Cpu, Server, Leaf, Wifi, Database, Zap, Globus } from "lucide-react";
import { useTranslation } from "../hooks/useTranslation";

export default function Settings() {
  const { t, language, setLanguage } = useTranslation();
  const [farmName, setFarmName] = useState("AgriSense Central Sector");
  const [alertThreshold, setAlertThreshold] = useState(35);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaving(true);
    setTimeout(() => {
      setSaving(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    }, 900);
  };

  const systemInfo = [
    { label: "Edge Backend",       value: "FastAPI 0.110",   status: "green",  icon: <Server className="w-3.5 h-3.5" /> },
    { label: "LLM Engine",         value: "Ollama + Qwen2.5 1.5B-Instruct", status: "green", icon: <Cpu className="w-3.5 h-3.5" /> },
    { label: "Vision Model",       value: "riven/florence-2 (Florence-2 3B)", status: "green", icon: <Zap className="w-3.5 h-3.5" /> },
    { label: "Database",           value: "SQLite (agrisense.db)", status: "amber", icon: <Database className="w-3.5 h-3.5" /> },
    { label: "ML Runtime",         value: "scikit-learn + TabPFN", status: "green", icon: <Leaf className="w-3.5 h-3.5" /> },
  ];

  return (
    <div className="space-y-6 animate-fade-in" id="settings-viewport">

      {/* Header */}
      <div className="page-header-strip p-6 text-white">
        <div className="relative z-10 space-y-2">
          <span className="agri-badge">⚙️ {t("nav.settings")}</span>
          <h1 className="text-2xl font-black tracking-tight">
            {t("settings.title")}
          </h1>
          <p className="text-emerald-100/80 text-sm max-w-xl">
            {t("settings.subtitle")}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Farm Configuration */}
        <div className="lg:col-span-7 agri-card p-6 space-y-6">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-emerald-600" />
            <h3 className="text-sm font-bold text-gray-800">{t("settings.farm_config")}</h3>
          </div>

          <div className="space-y-5">
            <div className="space-y-1.5">
              <label className="text-xs font-bold font-mono text-gray-500 uppercase tracking-wider">
                {t("settings.farm_name")}
              </label>
              <input
                id="input-settings-farmnm"
                type="text"
                value={farmName}
                onChange={(e) => setFarmName(e.target.value)}
                className="agri-input"
              />
              <p className="text-[10px] text-gray-400 font-mono">Used in reports and alert notifications</p>
            </div>

            <div className="section-divider" />

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <div>
                  <label className="text-xs font-bold font-mono text-gray-500 uppercase tracking-wider">
                    {t("settings.moisture_label")}
                  </label>
                  <p className="text-[10px] text-gray-400 font-mono mt-0.5">Trigger warning when soil moisture drops below this level</p>
                </div>
                <span className="agri-chip chip-green text-base font-black px-3 py-1">{alertThreshold}%</span>
              </div>
              <input
                id="slider-settings-threshold"
                type="range"
                min="15" max="50" step="1"
                value={alertThreshold}
                onChange={(e) => setAlertThreshold(parseInt(e.target.value))}
                className="agri-range w-full"
              />
              <div className="flex justify-between text-[9px] font-mono text-gray-400">
                <span>15% (Critical)</span>
                <span>32% (Optimal)</span>
                <span>50% (Saturated)</span>
              </div>
            </div>
          </div>

          <button
            id="btn-settings-save"
            onClick={handleSave}
            disabled={saving}
            className="btn-primary cursor-pointer"
          >
            {saving ? (
              <><RefreshCw className="w-4 h-4 animate-spin" /> {t("settings.saving")}</>
            ) : saved ? (
              <><Check className="w-4 h-4" /> {t("settings.saved")}</>
            ) : (
              t("settings.save")
            )}
          </button>
        </div>

        {/* System Status */}
        <div className="lg:col-span-5 agri-card p-6 space-y-5">
          <div className="flex items-center gap-2">
            <Wifi className="w-4 h-4 text-emerald-600" />
            <h3 className="text-sm font-bold text-gray-800">{t("settings.status")}</h3>
          </div>

          <div className="space-y-3">
            {systemInfo.map((item) => (
              <div key={item.label} className="flex items-center gap-3 p-3 rounded-xl bg-gray-50 border border-gray-100">
                <span className="text-gray-500">{item.icon}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-[10px] font-bold font-mono text-gray-500 uppercase tracking-wider">{item.label}</p>
                  <p className="text-xs text-gray-800 font-mono truncate">{item.value}</p>
                </div>
                <span className={`status-dot-${item.status}`} />
              </div>
            ))}
          </div>

          <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-100">
            <div className="flex items-start gap-2.5">
              <Info className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-[10px] font-bold font-mono text-emerald-800 uppercase tracking-wider">{t("settings.edge_title")}</p>
                <p className="text-xs text-emerald-700 leading-relaxed mt-1">
                  {t("settings.edge_desc")}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
