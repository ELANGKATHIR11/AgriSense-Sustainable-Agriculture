import React, { useState } from "react";
import { Leaf, Lock, Mail, UserPlus, LogIn, ShieldCheck } from "lucide-react";

interface LoginProps {
  onLoginSuccess: (token: string, profile: { email: string; role: string }) => void;
}

export default function Login({ onLoginSuccess }: LoginProps) {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("farmer");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const endpoint = isRegister ? "/api/auth/register" : "/api/auth/login";
      const payload = isRegister ? { email, password, role } : { email, password };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Authentication failed");
      }

      const data = await response.json();

      if (isRegister) {
        setIsRegister(false);
        setError("Account created! Please sign in.");
      } else {
        localStorage.setItem("agrisense_token", data.accessToken);
        localStorage.setItem("agrisense_profile", JSON.stringify(data.profile));
        onLoginSuccess(data.accessToken, data.profile);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#f0f4f0] px-4 relative overflow-hidden">
      {/* Background radial overlays */}
      <div className="absolute -top-32 -left-32 w-96 h-96 bg-emerald-500/[0.06] rounded-full blur-3xl" />
      <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-amber-500/[0.06] rounded-full blur-3xl" />

      <div className="w-full max-w-md bg-white/90 backdrop-blur-md rounded-2xl border border-emerald-900/10 shadow-2xl p-8 relative z-10">
        <div className="text-center mb-8">
          <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-emerald-500/20">
            <Leaf className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-black text-[#0f2e1e] tracking-tight">
            {isRegister ? "Join AgriSense" : "Welcome Back"}
          </h2>
          <p className="text-xs text-emerald-800/60 mt-1 font-medium">
            {isRegister ? "Register your new operator hub account" : "Sign in to access your offline farm intelligence"}
          </p>
        </div>

        {error && (
          <div className={`p-3 rounded-lg text-xs font-mono mb-5 ${error.includes("created") ? "bg-emerald-50 border border-emerald-200 text-emerald-800" : "bg-red-50 border border-red-200 text-red-800"}`}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-[10px] font-bold text-emerald-800 uppercase tracking-wider mb-1.5 font-mono">
              Email Address
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-2.5 w-4 h-4 text-emerald-600/50" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="farmer@agrisense.io"
                className="w-full pl-10 pr-4 py-2 border border-emerald-900/15 rounded-lg bg-[#fcfdfc] text-sm outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/10 transition-all font-mono"
              />
            </div>
          </div>

          <div>
            <label className="block text-[10px] font-bold text-emerald-800 uppercase tracking-wider mb-1.5 font-mono">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-2.5 w-4 h-4 text-emerald-600/50" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-10 pr-4 py-2 border border-emerald-900/15 rounded-lg bg-[#fcfdfc] text-sm outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/10 transition-all font-mono"
              />
            </div>
          </div>

          {isRegister && (
            <div>
              <label className="block text-[10px] font-bold text-emerald-800 uppercase tracking-wider mb-1.5 font-mono">
                Operator Role
              </label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="w-full px-3 py-2 border border-emerald-900/15 rounded-lg bg-[#fcfdfc] text-sm outline-none focus:border-emerald-500 transition-all font-medium text-emerald-900"
              >
                <option value="farmer">Farmer (Default)</option>
                <option value="consultant">Consultant</option>
                <option value="researcher">Researcher</option>
                <option value="enterprise">Enterprise Admin</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary mt-6 py-2.5 flex items-center justify-center gap-2 cursor-pointer font-bold text-sm tracking-tight"
          >
            {loading ? (
              <span className="animate-spin text-sm">⌛</span>
            ) : isRegister ? (
              <>
                <UserPlus className="w-4 h-4" />
                Register Operator
              </>
            ) : (
              <>
                <LogIn className="w-4 h-4" />
                Access Console
              </>
            )}
          </button>
        </form>

        <div className="mt-6 pt-5 border-t border-emerald-900/5 text-center">
          <button
            onClick={() => setIsRegister(!isRegister)}
            className="text-xs text-emerald-600 hover:text-emerald-800 font-semibold cursor-pointer transition-colors"
          >
            {isRegister ? "Already registered? Access here" : "Need a new local profile? Create one here"}
          </button>
        </div>

        <div className="mt-8 flex items-center justify-center gap-2 text-[9px] font-mono text-emerald-700/40 uppercase tracking-wider">
          <ShieldCheck className="w-3.5 h-3.5" />
          Offline Edge Engine Sandbox
        </div>
      </div>
    </div>
  );
}
