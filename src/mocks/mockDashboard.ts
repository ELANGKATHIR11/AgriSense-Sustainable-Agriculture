/**
 * AGRISENSE Dynamic Dashboard Mock Data
 * High-fidelity, edge-optimized simulated farm records
 */

export const mockDashboardData = {
  farmSummary: {
    name: "North Grid Sect-A",
    acresActive: 45.8,
    soilHealthScore: 89.4,
    companionPlanted: true,
    irrigationSource: "Underground Borewell B-4"
  },
  waterUsage: [
    { day: "Mon", liters: 6400, baseline: 6200 },
    { day: "Tue", liters: 6800, baseline: 6200 },
    { day: "Wed", liters: 5200, baseline: 6200 },
    { day: "Thu", liters: 4100, baseline: 6200 },
    { day: "Fri", liters: 7200, baseline: 6200 },
    { day: "Sat", liters: 6100, baseline: 6200 },
    { day: "Sun", liters: 4900, baseline: 6200 }
  ],
  cropSummary: [
    { name: "Rice (Basmati)", stage: "Tillering Phase", condition: "Optimal", coverage: "35%" },
    { name: "Sweet Maize", stage: "Tasseling Stage", condition: "Satisfactory", coverage: "30%" },
    { name: "Chickpeas", stage: "Vegetative Branching", condition: "Critical Moisture", coverage: "15%" },
    { name: "Golden Melon", stage: "Flowering & Fruit Set", condition: "Optimal", coverage: "20%" }
  ],
  recentAlerts: [
    { id: "alt-01", timestamp: new Date(Date.now() - 3600000).toISOString(), type: "warning", message: "Sector Chickpea moisture level checked under critical limit (19.4%)" },
    { id: "alt-02", timestamp: new Date(Date.now() - 7200000).toISOString(), type: "info", message: "Automated solenoid micro-flush completed on Grid-E" },
    { id: "alt-03", timestamp: new Date(Date.now() - 86400000).toISOString(), type: "critical", message: "Possible Tomato Late Blight risk flagged due to sustained microclimate moisture" }
  ]
};
