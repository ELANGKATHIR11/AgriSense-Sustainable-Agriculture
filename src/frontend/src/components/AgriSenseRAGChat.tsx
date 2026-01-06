/**
 * AgriSense RAG Chatbot Component
 * Integrates Retrieval-Augmented Generation for intelligent crop recommendations
 */

import React, { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import axios from 'axios';

interface RAGResponse {
  query: string;
  intent: string;
  confidence: number;
  response_text: string;
  data: {
    recommendations: Array<any>;
    weather_info: Record<string, any>;
    disease_info: Record<string, any>;
    soil_info: Record<string, any>;
    pricing_info: Record<string, any>;
  };
  timestamp: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  intent?: string;
  confidence?: number;
  data?: Record<string, any>;
}

export const AgriSenseRAGChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [season, setSeason] = useState<string>('Kharif');
  const [cropType, setCropType] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Mutation for RAG query
  const ragQueryMutation = useMutation({
    mutationFn: async (query: string) => {
      const response = await axios.post(`${API_BASE_URL}/api/v1/ml/rag/query`, {
        query,
        season,
        crop_type: cropType || undefined,
      });
      return response.data as RAGResponse;
    },
  });

  // Handle sending message
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const result = await ragQueryMutation.mutateAsync(input);

      // Add assistant response
      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: result.response_text,
        timestamp: new Date(),
        intent: result.intent,
        confidence: result.confidence,
        data: result.data,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      // Add error message
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your query. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-green-50 to-blue-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-4 shadow-lg">
        <h1 className="text-2xl font-bold">🌾 AgriSense AI Assistant</h1>
        <p className="text-sm opacity-90">
          Ask about crop recommendations, weather, diseases, soil management, or pricing
        </p>
      </div>

      {/* Context Settings */}
      <div className="bg-white border-b p-4 flex gap-4 flex-wrap">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Season
          </label>
          <select
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="Kharif">Kharif (Jun-Oct)</option>
            <option value="Rabi">Rabi (Oct-Mar)</option>
            <option value="Zaid">Zaid (Mar-Jun)</option>
            <option value="Perennial">Perennial</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Crop Type (Optional)
          </label>
          <select
            value={cropType}
            onChange={(e) => setCropType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="">All Types</option>
            <option value="Cereal">Cereal</option>
            <option value="Pulse">Pulse</option>
            <option value="Cash">Cash Crops</option>
            <option value="Fruit">Fruit</option>
            <option value="Vegetable">Vegetable</option>
            <option value="Spice">Spice</option>
            <option value="Oilseed">Oilseed</option>
          </select>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="text-6xl mb-4">🌱</div>
              <h2 className="text-2xl font-bold text-gray-700 mb-2">
                Welcome to AgriSense AI
              </h2>
              <p className="text-gray-600 max-w-md">
                I can help you with crop recommendations, weather advice, disease
                prevention, soil management, and market pricing. Just ask!
              </p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-md lg:max-w-2xl rounded-lg p-4 ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-800 border border-gray-200'
              }`}
            >
              <p className="text-sm">{message.content}</p>

              {/* Display metadata for assistant messages */}
              {message.role === 'assistant' && message.intent && (
                <div className="mt-3 pt-3 border-t border-gray-200 text-xs">
                  <div className="flex gap-2">
                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded">
                      Intent: {message.intent}
                    </span>
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">
                      {(message.confidence! * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                </div>
              )}

              {/* Display recommendations */}
              {message.data?.recommendations &&
                message.data.recommendations.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <h4 className="font-semibold text-sm mb-2">
                      Recommended Crops:
                    </h4>
                    <ul className="space-y-1 text-xs">
                      {message.data.recommendations
                        .slice(0, 3)
                        .map((crop: any, idx: number) => (
                          <li
                            key={idx}
                            className="bg-green-50 p-2 rounded flex justify-between"
                          >
                            <span>{crop.crop_name}</span>
                            <span className="text-green-700">
                              {crop.crop_type}
                            </span>
                          </li>
                        ))}
                    </ul>
                  </div>
                )}

              <p className="mt-2 text-xs opacity-50">
                {new Date(message.timestamp).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white text-gray-800 rounded-lg p-4 border border-gray-200">
              <div className="flex gap-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        <form onSubmit={handleSendMessage} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about crops, weather, diseases, soil, or pricing..."
            disabled={loading}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-gray-100"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-gradient-to-r from-green-600 to-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:shadow-lg disabled:opacity-50 transition"
          >
            {loading ? '...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default AgriSenseRAGChat;
