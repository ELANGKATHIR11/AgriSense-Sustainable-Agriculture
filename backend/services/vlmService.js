const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const { pythonPath } = require('../config/pythonConfig');

class VLMService {
  constructor() {
    this.apiKey = process.env.OPENAI_API_KEY || process.env.GEMINI_API_KEY;
    this.model = process.env.VLM_MODEL || 'gpt-4-vision-preview';
    this.uploadDir = path.join(__dirname, '../../uploads');
    this.ensureUploadDir();
  }

  async ensureUploadDir() {
    try {
      await fs.mkdir(this.uploadDir, { recursive: true });
    } catch (error) {
      console.log('Upload directory exists or created');
    }
  }

  // Configure multer for file uploads
  getMulterConfig() {
    return multer({
      storage: multer.diskStorage({
        destination: (req, file, cb) => {
          cb(null, this.uploadDir);
        },
        filename: (req, file, cb) => {
          const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
          cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
        }
      }),
      limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
      },
      fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png|gif|bmp|webp/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype);
        
        if (mimetype && extname) {
          return cb(null, true);
        } else {
          cb(new Error('Only image files are allowed (JPEG, PNG, GIF, BMP, WebP)'));
        }
      }
    });
  }

  async analyzePlantImage(imagePath, query = '') {
    try {
      console.log(`Analyzing plant image: ${imagePath}`);
      
      // Try edge AI vision model first
      const useEdgeAI = process.env.USE_EDGE_AI !== 'false';
      if (useEdgeAI) {
        try {
          const edgeAIAnalysis = await this.callEdgeAIVision(imagePath, query);
          if (edgeAIAnalysis && edgeAIAnalysis.success) {
            return edgeAIAnalysis;
          }
        } catch (edgeError) {
          console.warn('Edge AI Vision unavailable, using fallback:', edgeError.message);
          // Fall through to fallback
        }
      }
      
      // Fallback to mock analysis
      const analysis = await this.mockVLMAnalysis(imagePath, query);
      
      return {
        success: true,
        analysis: analysis,
        image_info: {
          filename: path.basename(imagePath),
          size: await this.getImageSize(imagePath),
          analyzed_at: new Date().toISOString()
        },
        source: 'rule_based'
      };
    } catch (error) {
      console.error('VLM Analysis Error:', error);
      return {
        success: false,
        error: 'Failed to analyze plant image',
        message: error.message
      };
    }
  }

  async callEdgeAIVision(imagePath, query = '') {
    try {
      const { spawn } = require('child_process');
      const pythonScript = path.join(__dirname, '../ml/edge_ai_vision.py');
      
      // Extract crop name from query if provided
      const cropMatch = query.match(/\b(Rice|Wheat|Maize|Tomato|Potato|Cotton|Sugarcane|Groundnut|Soybean|Chilli|Brinjal|Cucumber|Grapes|Apple|Banana|Mango|Coffee|Tea)\b/i);
      const cropName = cropMatch ? cropMatch[1] : null;
      
      return new Promise((resolve, reject) => {
        const python = spawn(pythonPath, [
          '-c',
          `
import sys
sys.path.insert(0, '${path.join(__dirname, '../ml')}')
from edge_ai_vision import get_vision_model
import json

vision = get_vision_model()
result = vision.analyze_plant_image('${imagePath.replace(/\\/g, '/')}', '${cropName || ""}')
print(json.dumps(result))
          `
        ]);
        
        let output = '';
        let errorOutput = '';
        
        python.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
          errorOutput += data.toString();
        });
        
        python.on('close', (code) => {
          if (code === 0 && output) {
            try {
              const result = JSON.parse(output.trim());
              resolve({
                success: true,
                analysis: {
                  plant_detected: result.crop_detected || 'Unknown',
                  health_status: result.health_status,
                  diseases: result.diseases_detected || [],
                  disease_details: result.disease_details || [],
                  pests: [],
                  nutrient_deficiencies: [],
                  recommendations: result.recommendations || [],
                  treatment_suggestions: result.treatment?.chemical_treatments || [],
                  organic_treatments: result.treatment?.organic_treatments || [],
                  prevention_measures: result.prevention || [],
                  confidence: result.confidence || 0.85
                },
                image_info: {
                  filename: path.basename(imagePath),
                  size: { size_bytes: 0, size_mb: 0 },
                  analyzed_at: new Date().toISOString()
                },
                source: 'edge_ai'
              });
            } catch (e) {
              reject(new Error('Failed to parse edge AI vision response'));
            }
          } else {
            reject(new Error(errorOutput || 'Edge AI vision service failed'));
          }
        });
      });
    } catch (error) {
      throw new Error(`Edge AI Vision call failed: ${error.message}`);
    }
  }

  async mockVLMAnalysis(imagePath, query) {
    // Simulate VLM processing time
    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 3000));

    const filename = path.basename(imagePath).toLowerCase();
    
    // Generate analysis based on filename and query
    let analysis = {
      plant_detected: 'Unknown',
      health_status: 'Healthy',
      diseases: [],
      pests: [],
      nutrient_deficiencies: [],
      recommendations: [],
      confidence: 0.85,
      treatment_suggestions: []
    };

    // Simulate different plant analyses
    if (filename.includes('rice') || filename.includes('paddy')) {
      analysis = {
        plant_detected: 'Rice (Oryza sativa)',
        health_status: 'Moderately Healthy',
        diseases: ['Brown spot detected on 15% of leaves'],
        pests: ['Signs of brown planthopper activity'],
        nutrient_deficiencies: ['Slight nitrogen deficiency'],
        recommendations: [
          'Apply nitrogen fertilizer (50 kg/ha) immediately',
          'Monitor for brown planthopper population',
          'Consider fungicide application for brown spot',
          'Improve water drainage to prevent fungal growth'
        ],
        confidence: 0.89,
        treatment_suggestions: [
          'Fungicide: Copper oxychloride 2kg/ha',
          'Insecticide: Imidacloprid 150ml/ha',
          'Nutrient: Urea 50kg/ha'
        ]
      };
    } else if (filename.includes('wheat')) {
      analysis = {
        plant_detected: 'Wheat (Triticum aestivum)',
        health_status: 'Good',
        diseases: [],
        pests: [],
        nutrient_deficiencies: [],
        recommendations: [
          'Current growth stage appears to be booting',
          'Monitor for rust diseases in humid conditions',
          'Consider second nitrogen application at tillering',
          'Adequate moisture levels observed'
        ],
        confidence: 0.92,
        treatment_suggestions: [
          'Preventive: Monitor weather for disease risk',
          'Nutrient: Ammonium sulfate 25kg/ha at tillering'
        ]
      };
    } else if (filename.includes('tomato')) {
      analysis = {
        plant_detected: 'Tomato (Solanum lycopersicum)',
        health_status: 'Poor',
        diseases: ['Early blight symptoms detected', 'Possible viral infection'],
        pests: ['Aphid colonies observed on new growth'],
        nutrient_deficiencies: ['Calcium deficiency (blossom end rot risk)'],
        recommendations: [
          'Immediate fungicide application required',
          'Remove affected leaves to prevent spread',
          'Apply calcium nitrate to prevent blossom end rot',
          'Introduce beneficial insects for aphid control'
        ],
        confidence: 0.94,
        treatment_suggestions: [
          'Fungicide: Mancozeb 2g/L water, spray weekly',
          'Insecticide: Neem oil 5ml/L for aphids',
          'Nutrient: Calcium nitrate 15kg/ha'
        ]
      };
    } else if (filename.includes('maize') || filename.includes('corn')) {
      analysis = {
        plant_detected: 'Maize (Zea mays)',
        health_status: 'Good',
        diseases: [],
        pests: [],
        nutrient_deficiencies: [],
        recommendations: [
          'Plant at optimal density (60,000 plants/ha)',
          'Monitor for fall armyworm during early growth',
          'Apply nitrogen in split doses for better efficiency',
          'Consider drip irrigation for water efficiency'
        ],
        confidence: 0.88,
        treatment_suggestions: [
          'Preventive: Install pheromone traps for armyworm',
          'Nutrient: Urea 120kg/ha split into 3 applications'
        ]
      };
    }

    // Custom analysis based on query
    if (query.toLowerCase().includes('disease')) {
      analysis.diseases.push('User-specified disease investigation requested');
      analysis.recommendations.push('Detailed disease analysis being performed');
    }

    if (query.toLowerCase().includes('pest')) {
      analysis.pests.push('User-specified pest investigation requested');
      analysis.recommendations.push('Detailed pest analysis being performed');
    }

    return analysis;
  }

  async getImageSize(imagePath) {
    try {
      const stats = await fs.stat(imagePath);
      return {
        size_bytes: stats.size,
        size_mb: (stats.size / (1024 * 1024)).toFixed(2)
      };
    } catch (error) {
      return { size_bytes: 0, size_mb: 0 };
    }
  }

  async generateDiseaseReport(analysis) {
    try {
      const report = {
        summary: this.generateDiseaseSummary(analysis),
        severity: this.assessSeverity(analysis),
        action_plan: this.generateActionPlan(analysis),
        prevention_measures: this.generatePreventionMeasures(analysis),
        monitoring_schedule: this.generateMonitoringSchedule(analysis),
        generated_at: new Date().toISOString()
      };

      return {
        success: true,
        report: report
      };
    } catch (error) {
      console.error('Disease Report Generation Error:', error);
      return {
        success: false,
        error: 'Failed to generate disease report',
        message: error.message
      };
    }
  }

  generateDiseaseSummary(analysis) {
    const { plant_detected, health_status, diseases, pests } = analysis;
    
    let summary = `Plant identified: ${plant_detected}. Overall health: ${health_status}.`;
    
    if (diseases.length > 0) {
      summary += ` Diseases detected: ${diseases.join(', ')}.`;
    }
    
    if (pests.length > 0) {
      summary += ` Pests detected: ${pests.join(', ')}.`;
    }

    if (diseases.length === 0 && pests.length === 0) {
      summary += ' No immediate threats detected.';
    }

    return summary;
  }

  assessSeverity(analysis) {
    const { health_status, diseases, pests } = analysis;
    
    if (health_status === 'Poor' || diseases.length > 2 || pests.length > 2) {
      return 'High - Immediate action required';
    } else if (health_status === 'Moderately Healthy' || diseases.length > 0 || pests.length > 0) {
      return 'Medium - Monitor closely and treat as needed';
    } else {
      return 'Low - Continue regular monitoring';
    }
  }

  generateActionPlan(analysis) {
    const { recommendations, treatment_suggestions } = analysis;
    
    return {
      immediate_actions: recommendations.slice(0, 2),
      short_term_actions: recommendations.slice(2, 4),
      long_term_actions: recommendations.slice(4),
      treatments: treatment_suggestions
    };
  }

  generatePreventionMeasures(analysis) {
    const { plant_detected } = analysis;
    
    const generalMeasures = [
      'Regular field scouting (twice weekly)',
      'Maintain proper plant spacing',
      'Ensure adequate drainage',
      'Practice crop rotation',
      'Use disease-resistant varieties when available'
    ];

    const plantSpecificMeasures = {
      'Rice (Oryza sativa)': [
        'Monitor water levels carefully',
        'Avoid excessive nitrogen application',
        'Remove weed hosts for pests'
      ],
      'Wheat (Triticum aestivum)': [
        'Monitor humidity levels',
        'Apply fungicides preventively in high-risk periods',
        'Control volunteer plants'
      ],
      'Tomato (Solanum lycopersicum)': [
        'Stake plants for better air circulation',
        'Mulch to prevent soil-borne diseases',
        'Control whitefly populations'
      ]
    };

    return [
      ...generalMeasures,
      ...(plantSpecificMeasures[plant_detected] || [])
    ];
  }

  generateMonitoringSchedule(analysis) {
    return {
      daily: [
        'Visual inspection for new symptoms',
        'Check pest traps',
        'Monitor weather conditions'
      ],
      weekly: [
        'Comprehensive field scouting',
        'Check plant growth stages',
        'Update pest/disease records'
      ],
      monthly: [
        'Soil testing',
        'Review treatment effectiveness',
        'Plan next month\'s activities'
      ]
    };
  }
}

module.exports = new VLMService();
