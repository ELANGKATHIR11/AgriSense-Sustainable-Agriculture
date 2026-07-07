/**
 * License: GNU Affero General Public License v3.0 (AGPL-3.0)
 * This file is part of AgriSense.
 * 
 * TERMS OF USE:
 * This project is licensed under the AGPL-3.0. Private modifications or private use
 * without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
 * AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
 * Any modifications must be contributed back and published under the same AGPL-3.0 license.
 */

import React, { useState } from "react";
import { User, Lock, Mail, ShieldAlert } from "lucide-react";

export default function AuthPages({ onLoginSuccess }: { onLoginSuccess: (email: string) => void }) {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("farmer");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onLoginSuccess(email);
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center bg-gradient-to-br from-[#f3f8f5] to-[#f3e7db] px-4">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-xl p-8 border border-emerald-100 animate-fade-in">
        <h2 className="text-2xl font-black text-[#0f2e1e] text-center mb-2">
          {isRegister ? "Create AgriSense Account" : "Access AgriSense Desktop V2"}
        </h2>
        <p className="text-center text-xs text-gray-500 mb-8 uppercase tracking-wider font-mono font-bold">
          {isRegister ? "Select profile role for RBAC setup" : "Offline-first single click platform"}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <Mail className="absolute left-3 top-3.5 w-4 h-4 text-emerald-600/70" />
            <input
              type="email"
              placeholder="Enter email address"
              className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="relative">
            <Lock className="absolute left-3 top-3.5 w-4 h-4 text-emerald-600/70" />
            <input
              type="password"
              placeholder="Enter secure password"
              className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {isRegister && (
            <div className="space-y-1">
              <label className="text-[10px] font-bold text-gray-400 uppercase tracking-wider block">Profile Role</label>
              <select
                className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white"
                value={role}
                onChange={(e) => setRole(e.target.value)}
              >
                <option value="farmer">Farmer (Default)</option>
                <option value="consultant">Agri Consultant</option>
                <option value="researcher">Researcher</option>
                <option value="enterprise">Enterprise Owner</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            className="w-full py-3 bg-gradient-to-r from-emerald-500 to-emerald-700 text-white rounded-lg font-bold shadow-lg shadow-emerald-950/20 hover:opacity-95 transition-opacity"
          >
            {isRegister ? "Complete Registration" : "Sign In to Platform"}
          </button>
        </form>

        <p className="text-center text-xs text-gray-500 mt-6">
          {isRegister ? "Already have an account?" : "Don't have a local profile?"}{" "}
          <button
            onClick={() => setIsRegister(!isRegister)}
            className="text-emerald-600 font-bold hover:underline"
          >
            {isRegister ? "Sign In" : "Register Profile"}
          </button>
        </p>
      </div>
    </div>
  );
}
