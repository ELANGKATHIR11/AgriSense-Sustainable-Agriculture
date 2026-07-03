/**
 * AGRISENSE Zustand User Session Store
 */

import { create } from "zustand";

interface UserProfile {
  name: string;
  email: string;
  role: string;
  farmName: string;
}

interface AuthState {
  isAuthenticated: boolean;
  user: UserProfile | null;
  login: (userData: UserProfile) => void;
  logout: () => void;
}

export const useAuthStore = create((set) => ({
  isAuthenticated: true, // Default to true in Sandbox
  user: {
    name: "Alex Agronomist",
    email: "alex@agrisense.io",
    role: "Senior Farm Operator",
    farmName: "North Grid Sector-A",
  },
  login: (user) => set({ isAuthenticated: true, user }),
  logout: () => set({ isAuthenticated: false, user: null })
}));
