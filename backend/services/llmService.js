const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');
const { pythonPath } = require('../config/pythonConfig');

class LLMService {
  constructor() {
    this.apiKey = process.env.OPENAI_API_KEY || process.env.GEMINI_API_KEY;
    this.baseURL = process.env.OPENAI_API_URL || 'https://api.openai.com/v1';
    this.model = process.env.LLM_MODEL || 'gpt-3.5-turbo';
    this.useEdgeAI = process.env.USE_EDGE_AI !== 'false'; // Default to true
    this.edgeAIServiceUrl = process.env.EDGE_AI_SERVICE_URL || 'http://localhost:5002';
  }

  async generateAgriculturalAdvice(query, context = {}) {
    try {
      // Try edge AI first if enabled
      if (this.useEdgeAI) {
        try {
          const edgeAIResponse = await this.callEdgeAIChatbot(query, context);
          if (edgeAIResponse && edgeAIResponse.success) {
            return edgeAIResponse;
          }
        } catch (edgeError) {
          console.warn('Edge AI unavailable, using fallback:', edgeError.message);
          // Fall through to fallback
        }
      }
      
      // Fallback to rule-based response
      const mockResponse = await this.mockLLMResponse(query, context);
      
      return {
        success: true,
        advice: mockResponse.advice,
        recommendations: mockResponse.recommendations,
        confidence: mockResponse.confidence,
        sources: mockResponse.sources,
        timestamp: new Date().toISOString(),
        source: 'rule_based'
      };
    } catch (error) {
      console.error('LLM Service Error:', error);
      return {
        success: false,
        error: 'Failed to generate agricultural advice',
        message: error.message
      };
    }
  }

  async callEdgeAIChatbot(query, context = {}) {
    try {
      // Try HTTP API first (if edge AI service is running)
      try {
        const response = await axios.post(`${this.edgeAIServiceUrl}/chatbot/query`, {
          query: query,
          crop_name: context.crop_name || null,
          context: context
        }, { timeout: 5000 });
        
        if (response.data && response.data.success) {
          const result = response.data.result;
          return {
            success: true,
            advice: result.response?.answer || result.response?.overview || 'Agricultural advice provided',
            recommendations: result.response?.tips || result.response?.water_efficiency_tips || [],
            optimization_tips: result.optimization_tips || [],
            cultivation_guide: result.type === 'cultivation_guide' ? result.response : null,
            confidence: 0.9,
            sources: ['Edge AI Agricultural Knowledge Base'],
            timestamp: new Date().toISOString(),
            source: 'edge_ai'
          };
        }
      } catch (httpError) {
        // Fall through to Python subprocess
        console.log('Edge AI HTTP service not available, trying Python subprocess...');
      }
      
      // Fallback: Use Python subprocess
      const cropName = context.crop_name || null;
      const mlPath = path.join(__dirname, '../ml');
      
      return new Promise((resolve, reject) => {
        const python = spawn(pythonPath, [
          '-c',
          `import sys, json, os; sys.path.insert(0, r'${mlPath.replace(/\\/g, '/')}'); from edge_ai_chatbot import get_chatbot; chatbot = get_chatbot(); result = chatbot.process_query(r'${query.replace(/'/g, "\\'").replace(/"/g, '\\"')}', ${cropName ? `r'${cropName.replace(/'/g, "\\'")}'` : 'None'}); print(json.dumps(result, ensure_ascii=False))`
        ], {
          cwd: mlPath,
          shell: true
        });
        
        let output = '';
        let errorOutput = '';
        
        python.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
          errorOutput += data.toString();
        });
        
        python.on('close', (code) => {
          if (code === 0 && output.trim()) {
            try {
              const result = JSON.parse(output.trim());
              resolve({
                success: true,
                advice: result.response?.answer || result.response?.overview || 'Agricultural advice provided',
                recommendations: result.response?.tips || result.response?.water_efficiency_tips || [],
                optimization_tips: result.optimization_tips || [],
                cultivation_guide: result.type === 'cultivation_guide' ? result.response : null,
                confidence: 0.9,
                sources: ['Edge AI Agricultural Knowledge Base'],
                timestamp: new Date().toISOString(),
                source: 'edge_ai'
              });
            } catch (e) {
              reject(new Error(`Failed to parse edge AI response: ${e.message}`));
            }
          } else {
            reject(new Error(errorOutput || 'Edge AI service failed'));
          }
        });
        
        python.on('error', (error) => {
          reject(new Error(`Failed to start Python process: ${error.message}`));
        });
      });
    } catch (error) {
      throw new Error(`Edge AI call failed: ${error.message}`);
    }
  }

  buildAgriculturalPrompt(query, context) {
    const contextStr = Object.keys(context).length > 0 
      ? `Context: ${JSON.stringify(context, null, 2)}\n\n` 
      : '';

    return `${contextStr}You are an expert agricultural AI assistant with deep knowledge of sustainable farming practices, crop management, soil science, and modern agricultural techniques. 

User Query: ${query}

Please provide:
1. Specific, actionable advice
2. Scientific explanation where relevant
3. Sustainability considerations
4. Risk factors or warnings
5. Recommended next steps

Format your response as JSON with keys: advice, recommendations, confidence, sources`;
  }

  async mockLLMResponse(query, context) {
    // Simulate LLM processing time
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

    const queryLower = query.toLowerCase();
    
    // Generate contextual responses based on query keywords
    if (queryLower.includes('rice') || queryLower.includes('paddy')) {
      return {
        advice: "For rice cultivation, ensure proper water management and maintain 2-3 inches of standing water during the vegetative stage. Use balanced NPK fertilizer with ratio 4:2:1.",
        recommendations: [
          "Maintain proper water level (2-3 inches) during vegetative stage",
          "Apply balanced NPK fertilizer (4:2:1 ratio)",
          "Monitor for pests like brown planthopper and rice blast",
          "Consider integrated pest management (IPM) practices"
        ],
        confidence: 0.89,
        sources: ["International Rice Research Institute", "FAO Guidelines"]
      };
    } else if (queryLower.includes('wheat')) {
      return {
        advice: "Wheat requires well-drained soil with pH 6.0-7.0. Apply nitrogen in split doses - 50% at sowing, 25% at tillering, and 25% at booting stage.",
        recommendations: [
          "Ensure soil pH between 6.0-7.0 with proper drainage",
          "Split nitrogen application: 50% at sowing, 25% at tillering, 25% at booting",
          "Monitor for rust diseases and aphids",
          "Consider crop rotation with legumes for soil health"
        ],
        confidence: 0.92,
        sources: ["CIMMYT Wheat Guidelines", "Agricultural Universities"]
      };
    } else if (queryLower.includes('organic') || queryLower.includes('sustainable')) {
      return {
        advice: "Organic farming focuses on soil health, biodiversity, and natural inputs. Use compost, green manures, and biological pest control for sustainable yields.",
        recommendations: [
          "Prepare compost from farm waste and kitchen scraps",
          "Use green manures like dhaincha or sunhemp",
          "Implement biological pest control using neem oil and beneficial insects",
          "Practice crop rotation to break pest cycles and improve soil fertility"
        ],
        confidence: 0.87,
        sources: ["Organic Farming Research Institute", "Sustainable Agriculture Standards"]
      };
    } else if (queryLower.includes('pest') || queryLower.includes('disease')) {
      return {
        advice: "Integrated Pest Management (IPM) combines cultural, biological, and chemical methods. Start with resistant varieties, monitor pest populations regularly, and use pesticides only when economic thresholds are reached.",
        recommendations: [
          "Plant pest-resistant crop varieties",
          "Regular field scouting to monitor pest populations",
          "Use biological controls (beneficial insects, neem-based products)",
          "Apply chemical pesticides only when economic thresholds are reached",
          "Maintain proper field sanitation"
        ],
        confidence: 0.85,
        sources: ["IPM Guidelines", "Entomology Research Centers"]
      };
    } else {
      return {
        advice: "Modern sustainable agriculture combines traditional wisdom with scientific innovations. Focus on soil health, water efficiency, and integrated pest management for optimal yields.",
        recommendations: [
          "Test soil regularly and amend based on results",
          "Implement drip irrigation for water efficiency",
          "Use crop rotation to maintain soil fertility",
          "Adopt conservation agriculture practices",
          "Monitor weather forecasts for timely operations"
        ],
        confidence: 0.78,
        sources: ["General Agricultural Best Practices", "Sustainable Farming Guidelines"]
      };
    }
  }

  async generateSoilAnalysis(soilData) {
    try {
      const { ph, nitrogen, phosphorus, potassium, organic_matter, texture } = soilData;
      
      const analysis = await this.mockSoilAnalysis(soilData);
      
      return {
        success: true,
        soil_type: analysis.soil_type,
        quality: analysis.quality,
        recommendations: analysis.recommendations,
        suitable_crops: analysis.suitable_crops,
        amendments: analysis.amendments,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Soil Analysis Error:', error);
      return {
        success: false,
        error: 'Failed to analyze soil data',
        message: error.message
      };
    }
  }

  async mockSoilAnalysis(soilData) {
    await new Promise(resolve => setTimeout(resolve, 800));

    const { ph, nitrogen, phosphorus, potassium } = soilData;
    
    let soilType = 'Loamy';
    let quality = 'Good';
    let recommendations = [];
    let suitableCrops = [];
    let amendments = [];

    // Analyze pH
    if (ph < 6.0) {
      recommendations.push('Soil is acidic - consider adding lime to raise pH');
      amendments.push('Agricultural lime: 2-3 tons/ha');
    } else if (ph > 7.5) {
      recommendations.push('Soil is alkaline - consider adding sulfur or organic matter');
      amendments.push('Elemental sulfur: 100-200 kg/ha');
    } else {
      recommendations.push('Soil pH is optimal for most crops');
    }

    // Analyze nutrients
    if (nitrogen < 50) {
      recommendations.push('Low nitrogen - add well-composted manure or nitrogen fertilizer');
      suitableCrops.push('Legumes (for nitrogen fixation)');
    }
    if (phosphorus < 20) {
      recommendations.push('Low phosphorus - add rock phosphate or bone meal');
      amendments.push('Rock phosphate: 200-300 kg/ha');
    }
    if (potassium < 30) {
      recommendations.push('Low potassium - add wood ash or potash');
      amendments.push('Muriate of potash: 100-150 kg/ha');
    }

    // Determine suitable crops based on analysis
    if (ph >= 6.0 && ph <= 7.5 && nitrogen > 50) {
      suitableCrops.push('Rice', 'Wheat', 'Maize', 'Sugarcane');
    }
    if (ph >= 6.5 && ph <= 7.0) {
      suitableCrops.push('Tomato', 'Potato', 'Onion');
    }

    return {
      soil_type: soilType,
      quality: quality,
      recommendations: recommendations,
      suitable_crops: [...new Set(suitableCrops)], // Remove duplicates
      amendments: amendments
    };
  }
}

module.exports = new LLMService();
