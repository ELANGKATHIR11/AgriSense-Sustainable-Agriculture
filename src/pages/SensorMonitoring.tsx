/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Terminal, Radio, Download, Check, RefreshCw, Wifi, Zap, Thermometer, Droplets, FlaskConical } from "lucide-react";
import { SensorReading } from "../types";

interface SensorMonitoringProps {
  sensors: SensorReading[];
  onRefresh: () => void;
  onSimulateIngest: (data: Partial<SensorReading>) => Promise<any>;
}

export default function SensorMonitoring({ sensors, onRefresh, onSimulateIngest }: SensorMonitoringProps) {
  const [activeTab, setActiveTab] = useState<"logs" | "esp32">("logs");
  const [simulationFields, setSimulationFields] = useState({
    deviceId: "ESP32-S01",
    soilMoisture: 38,
    temperature: 28.5,
    humidity: 59,
    pH: 6.3,
    nitrogen: 45,
    phosphorus: 38,
    potassium: 42
  });
  const [ingesting, setIngesting] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleSimulate = async () => {
    setIngesting(true);
    try {
      await onSimulateIngest(simulationFields);
      // Wait for ingest action to sync
    } catch (err) {
      console.error(err);
    } finally {
      setIngesting(false);
    }
  };

  const handleInputChange = (field: string, val: any) => {
    setSimulationFields(v => ({ ...v, [field]: val }));
  };

  const esp32Code = `/**
 * AGRISENSE v3.0 - ESP32 Smart Remote Farming Ingest
 * ESP32 + DHT22 + Capacitive Soil Moisture Sensor + NPK Transceiver
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

// Wi-Fi Configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Server url targeting Agrisense API Ingestion Endpoint
// Injected by AI Studio container at runtime
const char* serverUrl = "https://ais-dev-v2pzrohnfo7zf2yfmkps63-212139308419.asia-east1.run.app/api/sensors/ingest";

// Pin Configurations
#define DHTPIN 15
#define DHTTYPE DHT22
#define SOIL_MOISTURE_PIN 34

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();
  
  // Initiate Wi-Fi Connection
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\\nWiFi connected successfully!");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // Read raw physical DHT22 vectors
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    
    // Read capacitive sub-surface moisture (calibrate analog ranges 0-4095 against 0-100%)
    int rawMoisture = analogRead(SOIL_MOISTURE_PIN);
    float soilMoistureRH = map(rawMoisture, 4095, 1200, 0, 100);
    soilMoistureRH = constrain(soilMoistureRH, 0.0, 100.0);

    // Mock representation of RS485 NPK modbus sensor values
    int nitrogen = random(40, 52); 
    int phosphorus = random(35, 42);
    int potassium = random(40, 46);
    float pH = 6.3 + (random(-2, 3) / 10.0);

    // Form JSON payload
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    String jsonPayload = "{\\"deviceId\\":\\"ESP32-S01\\",";
    jsonPayload += "\\"soilMoisture\\":" + String(soilMoistureRH) + ",";
    jsonPayload += "\\"temperature\\":" + String(temp) + ",";
    jsonPayload += "\\"humidity\\":" + String(hum) + ",";
    jsonPayload += "\\"pH\\":" + String(pH) + ",";
    jsonPayload += "\\"nitrogen\\":" + String(nitrogen) + ",";
    jsonPayload += "\\"phosphorus\\":" + String(phosphorus) + ",";
    jsonPayload += "\\"potassium\\":" + String(potassium) + "}";

    Serial.println("Transmitting payload: " + jsonPayload);
    int httpResponseCode = http.POST(jsonPayload);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("HTTP success. Response: " + response);
    } else {
      Serial.print("Error transmitting telemetry. HTTP: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  }
  
  // Deep sleep timer (e.g. transmit telemetry packet hourly or every 60s for demo)
  delay(60000); 
}`;

  const copyCodeToClipboard = () => {
    navigator.clipboard.writeText(esp32Code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6 animate-fade-in" id="sensors-viewport">
      {/* Header */}
      <div className="page-header-strip p-6 text-white">
        <div className="relative z-10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="agri-badge">📡 IoT Telemetry</span>
              <span className="agri-badge agri-badge-amber">⚡ ESP32</span>
            </div>
            <h1 className="text-2xl font-black tracking-tight">
              IoT Sensor <span className="text-amber-400">Control Hub</span>
            </h1>
            <p className="text-emerald-100/80 text-sm max-w-xl">
              Live ESP32 telemetry feeds, soil sensor logs, and firmware deployment.
            </p>
          </div>
          <div className="mode-toggle flex-shrink-0">
            <button
              id="tab-sensors-logs"
              onClick={() => setActiveTab("logs")}
              className={`mode-toggle-btn ${activeTab === "logs" ? "active" : ""}`}
            >
              Live Feed
            </button>
            <button
              id="tab-sensors-esp32"
              onClick={() => setActiveTab("esp32")}
              className={`mode-toggle-btn ${activeTab === "esp32" ? "active" : ""}`}
            >
              ESP32 Firmware
            </button>
          </div>
        </div>
      </div>

      {activeTab === "logs" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Simulation Injector - 4 cols */}
          <div className="lg:col-span-4 agri-card p-5 space-y-4">
            <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
              <Terminal className="w-4 h-4 text-emerald-600" /> Simulate ESP32 Packet
            </h3>
            <p className="text-xs text-gray-500 leading-relaxed font-sans">
              No physical hardware ready? Use this simulated telemetry injector block to emit WiFi data packets to SQLite.
            </p>

            <div className="space-y-4 pt-2">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-mono text-gray-500 uppercase font-semibold">Device ID</label>
                  <input
                    id="sim-device"
                    type="text"
                    value={simulationFields.deviceId}
                    onChange={(e) => handleInputChange("deviceId", e.target.value)}
                    className="agri-input w-full"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-[10px] font-mono text-gray-500 uppercase font-semibold">Soil Moisture (%)</label>
                  <input
                    id="sim-moisture"
                    type="number"
                    value={simulationFields.soilMoisture}
                    onChange={(e) => handleInputChange("soilMoisture", parseFloat(e.target.value) || 0)}
                    className="agri-input w-full"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-mono text-gray-500 uppercase font-semibold">Temp (°C)</label>
                  <input
                    id="sim-temp"
                    type="number"
                    value={simulationFields.temperature}
                    onChange={(e) => handleInputChange("temperature", parseFloat(e.target.value) || 0)}
                    className="agri-input w-full"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-[10px] font-mono text-gray-500 uppercase font-semibold">Humidity (%)</label>
                  <input
                    id="sim-humidity"
                    type="number"
                    value={simulationFields.humidity}
                    onChange={(e) => handleInputChange("humidity", parseFloat(e.target.value) || 0)}
                    className="agri-input w-full"
                  />
                </div>
              </div>

              <div className="border-t border-gray-100 pt-3 grid grid-cols-3 gap-2">
                <div className="space-y-1">
                  <label className="text-[9px] font-mono text-gray-500 font-semibold">N (ppm)</label>
                  <input
                    id="sim-n"
                    type="number"
                    value={simulationFields.nitrogen}
                    onChange={(e) => handleInputChange("nitrogen", parseInt(e.target.value) || 0)}
                    className="agri-input w-full"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[9px] font-mono text-gray-500 font-semibold">P (ppm)</label>
                  <input
                    id="sim-p"
                    type="number"
                    value={simulationFields.phosphorus}
                    onChange={(e) => handleInputChange("phosphorus", parseInt(e.target.value) || 0)}
                    className="agri-input w-full"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[9px] font-mono text-gray-500 font-semibold">K (ppm)</label>
                  <input
                    id="sim-k"
                    type="number"
                    value={simulationFields.potassium}
                    onChange={(e) => handleInputChange("potassium", parseInt(e.target.value) || 0)}
                    className="agri-input w-full"
                  />
                </div>
              </div>
            </div>

            <button
              id="btn-simulate-esp"
              onClick={handleSimulate}
              disabled={ingesting}
              className="btn-primary w-full mt-2"
            >
              {ingesting ? <><RefreshCw className="w-4 h-4 animate-spin" /> Transmitting...</> : <><Wifi className="w-4 h-4" /> Broadcast Packet</>}
            </button>
          </div>
          {/* Historical Logs - 7 cols */}
          <div className="lg:col-span-8 agri-card p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-100 pb-3">
              <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
                <Radio className="w-4 h-4 text-emerald-600" /> Database Live Logs
              </h3>
              <button
                id="btn-refresh-telemetry"
                onClick={onRefresh}
                className="btn-secondary"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Refresh
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-500 uppercase">
                    <th className="py-3 px-3 font-semibold">Age</th>
                    <th className="py-3 px-3 font-semibold">Device</th>
                    <th className="py-3 px-3 font-semibold">Moisture</th>
                    <th className="py-3 px-3 font-semibold">Temp/Humidity</th>
                    <th className="py-3 px-3 font-semibold">pH</th>
                    <th className="py-3 px-3 font-semibold">NPK Ratio</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 text-gray-700">
                  {sensors.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-gray-400 uppercase">
                        No telemetry logs registered in schema.
                      </td>
                    </tr>
                  ) : (
                    sensors.map((reading) => {
                      const elapsedSec = Math.round((Date.now() - new Date(reading.timestamp).getTime()) / 1000);
                      let ageLabel = "Just now";
                      if (elapsedSec >= 3600) {
                        ageLabel = `${Math.round(elapsedSec / 3600)}h ago`;
                      } else if (elapsedSec >= 60) {
                        ageLabel = `${Math.round(elapsedSec / 60)}m ago`;
                      } else if (elapsedSec > 0) {
                        ageLabel = `${elapsedSec}s ago`;
                      }

                      return (
                        <tr key={reading.id} className="hover:bg-gray-50/60">
                          <td className="py-3 px-3 text-gray-400">{ageLabel}</td>
                          <td className="py-3 px-3 text-gray-600">{reading.deviceId}</td>
                          <td className="py-3 px-3 font-semibold text-emerald-600">{reading.soilMoisture}%</td>
                          <td className="py-3 px-3">{reading.temperature}°C / {reading.humidity}%</td>
                          <td className="py-3 px-3 text-gray-500">{reading.pH}</td>
                          <td className="py-3 px-3 text-gray-500 font-mono">
                            {reading.nitrogen}-{reading.phosphorus}-{reading.potassium}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === "esp32" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Arduino Code Segment - 8 cols */}
          <div className="lg:col-span-8 p-6 rounded-2xl bg-white border border-gray-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-gray-200 pb-3">
              <h3 className="text-sm font-semibold text-gray-800">ESP32 Ingestion Firmware [C++]</h3>
              <button
                id="btn-copy-esp-firmware"
                onClick={copyCodeToClipboard}
                className="px-3 py-1.5 rounded-lg bg-gray-50 hover:bg-gray-100 text-xs font-semibold text-emerald-700 flex items-center gap-1.5 cursor-pointer border border-gray-200 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Download className="w-3.5 h-3.5" />}
                {copied ? "Firmware Copied!" : "Copy Code"}
              </button>
            </div>
            <pre className="p-4 rounded-xl bg-gray-50 border border-gray-200 text-[11px] font-mono text-gray-600 overflow-x-auto max-h-[420px] leading-relaxed">
              {esp32Code}
            </pre>
          </div>

          {/* Schematic Guides - 4 cols */}
          <div className="lg:col-span-4 p-6 rounded-2xl bg-white border border-gray-200 shadow-sm space-y-6">
            <h3 className="text-sm font-semibold text-gray-800">Wiring Pinouts Schematic</h3>

            <div className="space-y-4">
              <div className="p-3 bg-gray-50 rounded-xl border border-gray-200 space-y-1">
                <span className="text-[9px] font-mono text-emerald-700 font-semibold">01 // DHT22 SENSOR</span>
                <p className="text-xs text-gray-900 font-medium">Pin VCC &rarr; 3.3V, Pin OUT &rarr; Pin GPIO15</p>
                <p className="text-[10px] text-gray-400">Ensure a 10k resistor is soldered between OUT and VCC for long bus stability.</p>
              </div>

              <div className="p-3 bg-gray-50 rounded-xl border border-gray-200 space-y-1">
                <span className="text-[9px] font-mono text-emerald-700 font-semibold">02 // CAPACITIVE MOISTURE SENSOR</span>
                <p className="text-xs text-gray-900 font-medium">Pin VCC &rarr; 3.3V, Pin ANALOG &rarr; GPIO34</p>
                <p className="text-[10px] text-gray-400">Capacitive probes are immune to corrosion compared with physical resistive forks.</p>
              </div>

              <div className="p-3 bg-gray-50 rounded-xl border border-gray-200 space-y-1">
                <span className="text-[9px] font-mono text-emerald-700 font-semibold">03 // SOLENOID WATER VALVE (RELAY)</span>
                <p className="text-xs text-gray-900 font-medium">Relay Control &rarr; GPIO12</p>
                <p className="text-[10px] text-gray-400">Drives sub-surface micro-valving timers specified by the LightGBM regressors.</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
