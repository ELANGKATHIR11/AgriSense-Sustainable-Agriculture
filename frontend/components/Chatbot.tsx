import React, { useState, useRef, useEffect } from 'react';
import { sendChatMessage, analyzeCropImage } from '../services/api';
import { ChatMessage } from '../types';
import { Send, Bot, Loader2, Image as ImageIcon, Upload, X } from 'lucide-react';

const Chatbot: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: '1', sender: 'bot', text: "Namaste! I am AgriSense, your farming assistant. How can I help with your crops today?", timestamp: new Date() }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() && !selectedImage) return;

    if (selectedImage) {
        // Image Analysis Flow
        const userMsg: ChatMessage = { 
            id: Date.now().toString(), 
            sender: 'user', 
            text: input || "Analyze this crop image", 
            timestamp: new Date(),
            image: imagePreview 
        };
        setMessages(prev => [...prev, userMsg]);
        setSelectedImage(null);
        setImagePreview(null);
        setInput('');
        setLoading(true);

        try {
            const result = await analyzeCropImage(selectedImage);
            // Format VLM Result as a structured message with enhanced data
            let responseText = `**Diagnostic Report**\n\n`;
            responseText += `**Crop:** ${result.crop_identified}\n`;
            responseText += `**Diagnosis:** ${result.diagnosis} (${result.confidence.toFixed(1)}%)\n`;
            if (result.scientific_name) responseText += `*Pathogen: ${result.scientific_name}*\n`;
            responseText += `**Severity:** ${result.severity || 'N/A'}\n\n`;
            
            responseText += `**💊 Cure:**\n`;
            result.cure.immediate_actions.forEach((a: string) => responseText += `- ${a}\n`);
            
            if (result.cure.biological_treatments?.length > 0) {
                responseText += `\n**🍃 Bio-Control:**\n`;
                result.cure.biological_treatments.forEach((a: string) => responseText += `- ${a}\n`);
            }

            responseText += `\n**🛡️ Prevention:**\n`;
            result.prevention.long_term_strategy.forEach((a: string) => responseText += `- ${a}\n`);
            
            setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), sender: 'bot', text: responseText, timestamp: new Date() }]);
        } catch (err) {
             setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), sender: 'bot', text: "Sorry, I encountered an error analyzing the image.", timestamp: new Date() }]);
        } finally {
            setLoading(false);
        }
        return;
    }

    // Text Chat Flow
    const userMsg: ChatMessage = { id: Date.now().toString(), sender: 'user', text: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const responseText = await sendChatMessage(userMsg.text);
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), sender: 'bot', text: responseText, timestamp: new Date() }]);
    } catch { /* fallback in API */ } finally { setLoading(false); }
  };

  return (
    <div className="h-[calc(100vh-140px)] flex flex-col bg-white rounded-xl shadow-sm border border-agri-100 overflow-hidden">
      <div className="bg-agri-50 p-4 border-b border-agri-100 flex items-center space-x-3">
        <div className="p-2 bg-agri-100 rounded-full"><Bot className="w-6 h-6 text-agri-700" /></div>
        <div>
          <h2 className="font-semibold text-agri-900">AgriSense Assistant</h2>
          <p className="text-xs text-agri-600">Powered by LLM & RAG</p>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] md:max-w-[70%] rounded-2xl px-4 py-3 shadow-sm ${
              msg.sender === 'user' ? 'bg-agri-600 text-white rounded-br-none' : 'bg-white text-gray-800 border border-gray-100 rounded-bl-none'
            }`}>
              <div className="flex items-start space-x-2">
                {msg.sender === 'bot' && <Bot className="w-4 h-4 mt-1 opacity-50 flex-shrink-0" />}
                <div className="flex flex-col">
                    {/* Display User Uploaded Image if exists */}
                    {/* @ts-ignore - Check if image property exists on message type extension */}
                    {msg.image && (
                        <div className="mb-2 rounded-lg overflow-hidden border border-white/20">
                            <img src={msg.image} alt="Uploaded crop" className="w-full h-auto max-h-48 object-cover" />
                        </div>
                    )}
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                </div>
              </div>
              <p className={`text-[10px] mt-1 text-right ${msg.sender === 'user' ? 'text-agri-200' : 'text-gray-400'}`}>
                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </p>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-2xl px-4 py-3 shadow-sm border border-agri-100">
              <div className="flex flex-col space-y-2">
                 <div className="flex items-center space-x-2">
                    <Loader2 className="w-4 h-4 text-agri-600 animate-spin" />
                    <span className="text-sm text-gray-500 font-medium">Yield Guardian Reasoning...</span>
                 </div>
                 <div className="text-xs text-gray-400 pl-6">
                    Checking soil health • Optimizing yield • Verifying sustainability
                 </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="p-4 bg-white border-t border-gray-100">
        <form onSubmit={handleSend} className="flex items-center space-x-3">
          <input 
            type="file" 
            accept="image/*" 
            className="hidden" 
            ref={fileInputRef}
            onChange={handleImageSelect}
            title="Upload Image"
          />
          
          {/* Image Preview Area in Input */}
          {imagePreview && (
            <div className="relative">
                <div className="w-12 h-12 rounded-lg overflow-hidden border border-agri-200">
                    <img src={imagePreview} alt="Preview" className="w-full h-full object-cover" />
                </div>
                <button 
                    type="button"
                    onClick={() => { setImagePreview(null); setSelectedImage(null); }}
                    className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-0.5"
                    title="Remove Image"
                    aria-label="Remove Image"
                >
                    <X className="w-3 h-3" />
                </button>
            </div>
          )}

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-2 text-gray-400 hover:text-agri-600 transition-colors"
            title="Upload Crop Image"
          >
            <ImageIcon className="w-6 h-6" />
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={selectedImage ? "Add a note..." : "Ask about crops, diseases..."}
            className="flex-1 px-4 py-3 border border-gray-200 rounded-full focus:outline-none focus:ring-2 focus:ring-agri-400 focus:border-transparent bg-gray-50"
          />
          <button 
            type="submit" 
            disabled={loading || (!input.trim() && !selectedImage)} 
            title="Send Message"
            className="p-3 bg-agri-600 text-white rounded-full hover:bg-agri-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
};

export default Chatbot;
