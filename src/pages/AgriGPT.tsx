/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Trash2, Sparkles, Leaf, Cpu, Activity } from "lucide-react";
import { ChatMessage, SensorReading } from "../types";

interface AgriGPTProps {
  sensors: SensorReading[];
}

const SAMPLE_PROMPTS = [
  "Best NPK ratio for sandy dry soil?",
  "Suggest cover crops for 45 ppm Nitrogen.",
  "Explain tomato late blight treatment.",
  "Optimal pH range for paddy rice?",
];

export default function AgriGPT({ sensors }: AgriGPTProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "init-01",
      role: "model",
      content: "Hello! I'm AgriGPT — your edge-native agricultural intelligence assistant powered by **Qwen2.5 1.5B-Instruct** running locally on Ollama.\n\nI'm synced with your live field data. Ask me anything about soil health, crop pathology, irrigation scheduling, or ML model metrics.",
      timestamp: new Date().toISOString()
    }
  ]);
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;

    const userMsg: ChatMessage = {
      id: "usr-" + Math.random().toString(36).slice(4, 9),
      role: "user",
      content: text,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMsg]);
    setInputText("");
    setLoading(true);

    try {
      const activeSensor = sensors[0] || { soilMoisture: 38, temperature: 28, humidity: 60, pH: 6.3, nitrogen: 45, phosphorus: 38, potassium: 42 };

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...messages, userMsg].map(m => ({ role: m.role, content: m.content })),
          sensorContext: activeSensor
        })
      });

      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const data = await response.json();

      setMessages(prev => [...prev, {
        id: "gpt-" + Math.random().toString(36).slice(4, 9),
        role: "model",
        content: data.text || "Unable to generate advisory for this request.",
        timestamp: new Date().toISOString()
      }]);
    } catch (err: any) {
      setMessages(prev => [...prev, {
        id: "err-" + Math.random().toString(36).slice(4, 9),
        role: "model",
        content: `⚠️ Chat error: ${err.message}. The backend may still be starting up — please wait a moment and try again.`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => setMessages([{
    id: "init-" + Date.now(),
    role: "model",
    content: "Chat cleared. AgriGPT ready for new agronomic inquiries.",
    timestamp: new Date().toISOString()
  }]);

  const activeSensor = sensors[0];

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] gap-4 animate-fade-in" id="agrigpt-viewport">

      {/* Header */}
      <div className="page-header-strip p-5 text-white flex-shrink-0">
        <div className="relative z-10 flex items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="agri-badge">🤖 Qwen2.5 1.5B-Instruct</span>
              <span className="agri-badge agri-badge-amber">⚡ Ollama Local</span>
            </div>
            <h1 className="text-xl font-black tracking-tight">
              AgriGPT <span className="text-amber-400">Intelligence Chat</span>
            </h1>
          </div>
          <button
            id="btn-clear-chat"
            onClick={clearChat}
            className="btn-secondary flex-shrink-0"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear Chat
          </button>
        </div>
      </div>

      {/* Sensor context bar */}
      {activeSensor && (
        <div className="flex flex-wrap items-center gap-2 px-1 flex-shrink-0">
          <span className="text-[9px] font-bold font-mono text-gray-400 uppercase tracking-widest">Live Context:</span>
          {[
            { label: "Moisture", value: `${activeSensor.soilMoisture}%` },
            { label: "Temp", value: `${activeSensor.temperature}°C` },
            { label: "N", value: `${activeSensor.nitrogen} ppm` },
            { label: "pH", value: activeSensor.pH?.toString() }
          ].filter(i => i.value).map(item => (
            <span key={item.label} className="agri-chip chip-green">{item.label}: {item.value}</span>
          ))}
        </div>
      )}

      {/* Message Thread */}
      <div className="flex-1 min-h-0 overflow-y-auto agri-card p-4 space-y-4">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex gap-3 max-w-2xl ${m.role === "user" ? "ml-auto flex-row-reverse" : ""}`}
          >
            <div className={`w-8 h-8 rounded-xl flex-shrink-0 flex items-center justify-center border ${
              m.role === "user"
                ? "bg-emerald-600 border-emerald-500 text-white"
                : "bg-white border-emerald-100 text-emerald-600"
            }`}>
              {m.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>
            <div className={`px-4 py-3 rounded-xl text-sm leading-relaxed whitespace-pre-wrap ${
              m.role === "user"
                ? "bg-emerald-600 text-white rounded-tr-none"
                : "bg-white border border-gray-100 text-gray-800 rounded-tl-none shadow-sm"
            }`}>
              {m.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-3 max-w-xs animate-pulse">
            <div className="w-8 h-8 rounded-xl flex-shrink-0 bg-white border border-emerald-100 text-emerald-600 flex items-center justify-center">
              <Bot className="w-4 h-4" />
            </div>
            <div className="px-4 py-3 rounded-xl bg-white border border-gray-100 shadow-sm rounded-tl-none flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-bounce" />
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-bounce [animation-delay:0.15s]" />
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-bounce [animation-delay:0.3s]" />
              <span className="text-[10px] font-mono text-gray-400 ml-1">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Prompt Suggestions */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-hide flex-shrink-0">
        {SAMPLE_PROMPTS.map((p, idx) => (
          <button
            key={idx}
            id={`preset-prompt-${idx}`}
            onClick={() => sendMessage(p)}
            className="flex-shrink-0 px-3 py-1.5 rounded-lg bg-white border border-emerald-100 hover:border-emerald-300 hover:bg-emerald-50 text-xs text-gray-600 hover:text-emerald-700 font-mono cursor-pointer transition-all"
          >
            {p}
          </button>
        ))}
      </div>

      {/* Input */}
      <form
        onSubmit={(e) => { e.preventDefault(); sendMessage(inputText); }}
        className="flex gap-2 items-center bg-white border border-gray-200 rounded-xl px-4 py-2.5 shadow-sm flex-shrink-0"
      >
        <Leaf className="w-4 h-4 text-emerald-400 flex-shrink-0" />
        <input
          id="input-agrigpt-chat"
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask AgriGPT anything about your crops, soil, or system..."
          className="flex-1 bg-transparent text-sm text-gray-800 placeholder-gray-400 outline-none"
        />
        <button
          id="btn-send-agrigpt"
          type="submit"
          disabled={loading || !inputText.trim()}
          className="w-9 h-9 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-100 disabled:text-gray-400 text-white flex items-center justify-center flex-shrink-0 transition-all cursor-pointer"
        >
          {loading ? <Activity className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        </button>
      </form>

      {/* Trace */}
      <div className="trace-bar flex-shrink-0">
        <span>MODEL: Qwen2.5-Coder 3B · BACKEND: Ollama Edge Node</span>
        <span className="text-emerald-600 font-bold">LOCAL · NO CLOUD</span>
      </div>
    </div>
  );
}
