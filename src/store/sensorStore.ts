/**
 * AGRISENSE Sensor Logs Global Store
 */

import { create } from "zustand";
import { SensorReading } from "../types";
import { initialMockSensors } from "../mocks/mockSensors";

interface SensorState {
  readings: SensorReading[];
  lastUpdated: string | null;
  setReadings: (readings: SensorReading[]) => void;
  appendReading: (reading: SensorReading) => void;
}

export const useSensorStore = create<SensorState>((set) => ({
  readings: initialMockSensors,
  lastUpdated: new Date().toISOString(),
  setReadings: (readings) => set({ readings, lastUpdated: new Date().toISOString() }),
  appendReading: (reading) => set((state) => ({
    readings: [reading, ...state.readings].slice(0, 40),
    lastUpdated: new Date().toISOString()
  }))
}));
