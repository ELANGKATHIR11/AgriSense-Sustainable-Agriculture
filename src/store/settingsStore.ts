/**
 * AGRISENSE System Settings Parameter Store
 */

import { create } from "zustand";

interface SettingsState {
  farmName: string;
  moistureAlertThreshold: number;
  language: string;
  systemStatus: "Healthy" | "Maintenance" | "Offline";
  setFarmName: (name: string) => void;
  setMoistureAlertThreshold: (val: number) => void;
  setLanguage: (lang: string) => void;
  setSystemStatus: (status: "Healthy" | "Maintenance" | "Offline") => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  farmName: "North Grid Sector-A",
  moistureAlertThreshold: 35,
  language: "English (US)",
  systemStatus: "Healthy",
  setFarmName: (farmName) => set({ farmName }),
  setMoistureAlertThreshold: (moistureAlertThreshold) => set({ moistureAlertThreshold }),
  setLanguage: (language) => set({ language }),
  setSystemStatus: (systemStatus) => set({ systemStatus })
}));
