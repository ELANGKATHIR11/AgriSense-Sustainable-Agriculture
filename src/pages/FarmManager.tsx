import React, { useState, useEffect } from "react";
import { Sprout, MapPin, Layers, Radio, Plus } from "lucide-react";

interface Farm {
  id: number;
  name: string;
  location: string;
  fields: Array<{ id: number; name: string; cropType: string; area: number }>;
}

export default function FarmManager() {
  const [farms, setFarms] = useState<Farm[]>([]);
  const [activeFarm, setActiveFarm] = useState<number | null>(null);
  const [newFarmName, setNewFarmName] = useState("");
  const [newFarmLocation, setNewFarmLocation] = useState("");

  const fetchFarms = async () => {
    try {
      const res = await fetch("/api/farms");
      const data = await res.json();
      setFarms(data);
      if (data.length > 0 && activeFarm === null) {
        setActiveFarm(data[0].id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchFarms();
  }, []);

  const handleCreateFarm = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch("/api/farms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newFarmName, location: newFarmLocation })
      });
      if (res.ok) {
        setNewFarmName("");
        setNewFarmLocation("");
        await fetchFarms();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const selectedFarm = farms.find(f => f.id === activeFarm);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black text-[#0f2e1e]">Multi-Farm Context Selector</h1>
          <p className="text-xs text-gray-500 uppercase tracking-wider font-mono">Scope predictions to selected farm grid</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left list */}
        <div className="lg:col-span-1 space-y-4">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider font-mono">Farms Registry</h3>
          <div className="space-y-2">
            {farms.map(f => (
              <button
                key={f.id}
                onClick={() => setActiveFarm(f.id)}
                className={`w-full p-4 rounded-xl text-left border flex items-start gap-3 transition-all ${
                  f.id === activeFarm
                    ? "bg-white border-emerald-500 shadow-md ring-2 ring-emerald-500/10"
                    : "bg-white border-gray-100 hover:border-gray-300"
                }`}
              >
                <div className="p-2 rounded-lg bg-emerald-50 text-emerald-600 shrink-0">
                  <Sprout className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="font-bold text-[#0f2e1e] text-sm">{f.name}</h4>
                  <div className="flex items-center gap-1 text-[11px] text-gray-400 mt-1">
                    <MapPin className="w-3 h-3" />
                    <span>{f.location}</span>
                  </div>
                </div>
              </button>
            ))}
          </div>

          {/* Add farm form */}
          <div className="bg-white p-5 rounded-xl border border-gray-100 space-y-3">
            <h4 className="font-bold text-[#0f2e1e] text-xs uppercase tracking-wider text-gray-400 font-mono">Add New Farm Context</h4>
            <form onSubmit={handleCreateFarm} className="space-y-2">
              <input
                type="text"
                placeholder="Farm Name"
                className="w-full px-3 py-2 text-xs border rounded-lg focus:ring-emerald-500 focus:outline-none"
                required
                value={newFarmName}
                onChange={e => setNewFarmName(e.target.value)}
              />
              <input
                type="text"
                placeholder="Location"
                className="w-full px-3 py-2 text-xs border rounded-lg focus:ring-emerald-500 focus:outline-none"
                required
                value={newFarmLocation}
                onChange={e => setNewFarmLocation(e.target.value)}
              />
              <button
                type="submit"
                className="w-full py-2 bg-emerald-600 text-white rounded-lg font-bold text-xs flex items-center justify-center gap-1"
              >
                <Plus className="w-3.5 h-3.5" /> Add Farm
              </button>
            </form>
          </div>
        </div>

        {/* Right view details */}
        <div className="lg:col-span-2">
          {selectedFarm ? (
            <div className="bg-white rounded-2xl border border-gray-100 p-6 space-y-6">
              <div className="border-b pb-4">
                <span className="text-[10px] font-mono font-bold text-emerald-600 uppercase tracking-widest bg-emerald-50 px-2 py-0.5 rounded">Active Focus</span>
                <h2 className="text-xl font-bold text-[#0f2e1e] mt-2">{selectedFarm.name}</h2>
                <p className="text-xs text-gray-400">{selectedFarm.location}</p>
              </div>

              {/* Fields List */}
              <div className="space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-400 font-mono flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5" /> Fields & Crop Zoning ({selectedFarm.fields.length})
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {selectedFarm.fields.map(fl => (
                    <div key={fl.id} className="p-4 bg-emerald-50/20 border border-emerald-900/10 rounded-xl space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-sm text-[#0f2e1e]">{fl.name}</span>
                        <span className="text-[10px] font-bold bg-amber-50 text-amber-600 px-2 py-0.5 rounded">{fl.cropType}</span>
                      </div>
                      <p className="text-xs text-gray-500">Surface Area: {fl.area} Acres</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Devices list */}
              <div className="space-y-3 pt-4 border-t">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-400 font-mono flex items-center gap-1.5">
                  <Radio className="w-3.5 h-3.5" /> Smart ESP32 Nodes / Actuators
                </h3>
                <div className="p-4 bg-gray-50 rounded-xl flex items-center justify-between text-xs">
                  <div>
                    <p className="font-bold text-[#0f2e1e]">ESP32-S01 Telemetry Hub</p>
                    <p className="text-gray-400">GPIO pins configured: GPIO34 (ADC1), GPIO15 (DHT22)</p>
                  </div>
                  <span className="agri-chip chip-green">Active / Connected</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-2xl border border-gray-100 p-8 text-center text-gray-400">
              Select or register a farm context.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
