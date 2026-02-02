const path = require('path');
const fs = require('fs').promises;

const DEFAULT_MODEL_PATH = path.join(__dirname, '../ml/models/water_requirement_model.json');

const CROP_COEFFICIENTS = {
  // Cereals
  'Rice': { ini: 1.05, mid: 1.20, end: 0.90 }, // Higher baseline for paddy rice
  'Wheat': { ini: 0.3, mid: 1.15, end: 0.25 },
  'Maize': { ini: 0.3, mid: 1.20, end: 0.35 },
  'Corn': { ini: 0.3, mid: 1.20, end: 0.35 },
  'Barley': { ini: 0.3, mid: 1.15, end: 0.25 },
  'Millet': { ini: 0.3, mid: 1.00, end: 0.30 },
  'Sorghum': { ini: 0.3, mid: 1.00, end: 0.55 },
  
  // Pulses
  'Chickpea': { ini: 0.4, mid: 1.00, end: 0.35 },
  'Lentil': { ini: 0.4, mid: 1.10, end: 0.30 },
  'Pigeonpeas': { ini: 0.4, mid: 1.10, end: 0.35 },
  'Kidneybeans': { ini: 0.4, mid: 1.15, end: 0.35 },
  'Mothbeans': { ini: 0.3, mid: 1.05, end: 0.35 },
  'Mungbean': { ini: 0.4, mid: 1.05, end: 0.35 },
  'Blackgram': { ini: 0.4, mid: 1.05, end: 0.35 },
  
  // Vegetables
  'Watermelon': { ini: 0.4, mid: 1.00, end: 0.75 },
  'Muskmelon': { ini: 0.4, mid: 1.05, end: 0.8 },
  'Cotton': { ini: 0.35, mid: 1.15, end: 0.70 },
  'Jute': { ini: 0.4, mid: 1.10, end: 0.8 },
  
  // Fruits
  'Apple': { ini: 0.45, mid: 0.95, end: 0.70 },
  'Orange': { ini: 0.5, mid: 0.80, end: 0.70 },
  'Papaya': { ini: 0.5, mid: 1.10, end: 1.0 },
  'Coconut': { ini: 0.8, mid: 1.00, end: 1.0 }, // Tropical perennial
  'Pomegranate': { ini: 0.4, mid: 0.85, end: 0.65 },
  'Banana': { ini: 0.50, mid: 1.10, end: 1.00 },
  'Mango': { ini: 0.45, mid: 0.90, end: 0.65 },
  'Grapes': { ini: 0.3, mid: 0.85, end: 0.45 },
  
  // Others
  'Coffee': { ini: 0.9, mid: 1.05, end: 0.95 },
  'Jute': { ini: 0.4, mid: 1.10, end: 0.80 }
};

class WaterRequirementService {
  constructor() {
    this.modelPath = DEFAULT_MODEL_PATH;
    this.modelParams = null;
    this.isModelLoaded = false;
  }

  async loadModel() {
    try {
      console.log('Loading water requirement model parameters...');
      const data = await fs.readFile(this.modelPath, 'utf-8');
      this.modelParams = JSON.parse(data);
      this.isModelLoaded = true;
      console.log('Water requirement model loaded successfully');
    } catch (error) {
      console.warn(`Water requirement model parameters not found at ${this.modelPath}. Using heuristic fallback. Error: ${error.message}`);
      this.modelParams = null;
      this.isModelLoaded = true;
    }
  }

  async predictWaterRequirement(inputData) {
    if (!this.isModelLoaded) {
      await this.loadModel(); // Still load for fallback params if needed
    }

    return new Promise((resolve, reject) => { 
        // 1. Try Native Python Inference
        try {
            const { spawn } = require('child_process');
            const { pythonPath } = require('../config/pythonConfig');
            const path = require('path');
            const pythonScript = path.join(__dirname, '../ml/predict_water.py');
            
            const pythonProcess = spawn(pythonPath, [pythonScript]);

            let dataString = '';
            let errorString = '';

            pythonProcess.stdin.write(JSON.stringify(inputData));
            pythonProcess.stdin.end();

            pythonProcess.stdout.on('data', (data) => {
                dataString += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorString += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0 && dataString) {
                    try {
                        const result = JSON.parse(dataString);
                        if (result.error) throw new Error(result.error);
                        
                        // Add helper details
                        const totalLiters = (result.predicted_water_requirement || 0) * (inputData.area || 1) * 10000;
                        
                        resolve({
                            ...result,
                            total_liters_per_day: totalLiters,
                            input_data: inputData,
                            details: `ML Model Prediction for ${inputData.crop || 'Unknown Crop'}.`,
                            prediction_timestamp: new Date().toISOString()
                        });
                    } catch (e) {
                         console.warn('Water ML parsing failed, using fallback:', e);
                         resolve(this._predictHeuristic(inputData));
                    }
                } else {
                    console.warn('Water ML process failed:', errorString);
                    resolve(this._predictHeuristic(inputData));
                }
            });

            // Timeout 10s
            setTimeout(() => {
                if(!pythonProcess.killed) pythonProcess.kill();
                resolve(this._predictHeuristic(inputData));
            }, 10000);

        } catch (err) {
            console.warn('Failed to spawn Water ML process:', err);
            resolve(this._predictHeuristic(inputData));
        }
    });
  }

  _predictHeuristic(inputData) {
      // Original Heuristic Logic moved here as fallback
      console.log('Using Heuristic Fallback for Water Requirement');
      const { temperature, humidity, rainfall, growth_duration, crop, area = 1 } = inputData;
      
      let waterRequirement;
      let confidence;
      let kcUsed = 1.0;
      
      const tempFactor = Math.max(0, 0.005 * Math.pow(temperature, 1.8)); 
      const humidityFactor = 1.2 - (humidity / 100) * 0.6;
      const baseETo = (2 + tempFactor) * humidityFactor;

      const normalizedCrop = crop ? crop.charAt(0).toUpperCase() + crop.slice(1).toLowerCase() : '';
      const matchedCropKey = Object.keys(CROP_COEFFICIENTS).find(k => 
        normalizedCrop.includes(k) || k.includes(normalizedCrop)
      );

      if (matchedCropKey) {
        kcUsed = CROP_COEFFICIENTS[matchedCropKey].mid; 
      }

      let predictedRequirement = baseETo * kcUsed;
      waterRequirement = Math.max(1.5, Math.round(predictedRequirement * 100) / 100);
      confidence = this.calculateConfidence(inputData, null);
      
      if (matchedCropKey) confidence = Math.min(0.98, confidence + 0.1);

      const totalLitersPerDay = waterRequirement * area * 10000;

      return {
        predicted_water_requirement: waterRequirement,
        unit: 'mm per day',
        confidence,
        requirement_range: {
          minimum: Math.max(0.1, Math.round(waterRequirement * 0.9 * 100) / 100),
          maximum: Math.round(waterRequirement * 1.1 * 100) / 100
        },
        monthly_requirement: Math.round(waterRequirement * 30 * 100) / 100,
        total_liters_per_day: totalLitersPerDay,
        input_data: { ...inputData, area },
        details: `Fallback Calculation (ETo ~${Math.round(baseETo*100)/100}) for ${matchedCropKey || 'generic crop'}.`,
        model_r2_score: this.modelParams?.r2 || null,
        model_rmse: this.modelParams?.rmse || null,
        prediction_timestamp: new Date().toISOString()
      };
  }

  scaleFeatures(values) {
    const means = this.modelParams.scaler_mean;
    const scales = this.modelParams.scaler_scale;
    if (!means || !scales || means.length !== values.length || scales.length !== values.length) {
      throw new Error('Scaler parameters missing or mismatched');
    }
    return values.map((v, idx) => (v - means[idx]) / (scales[idx] || 1));
  }

  calculateConfidence(inputData, params) {
    const { temperature, humidity, rainfall, growth_duration } = inputData;
    let confidence = 0.7;
    if (params && params.r2) {
      confidence = Math.min(0.9, 0.7 + params.r2 * 0.2);
    }
    if (temperature >= 10 && temperature <= 40) confidence += 0.02;
    if (humidity >= 30 && humidity <= 90) confidence += 0.02;
    if (rainfall >= 50 && rainfall <= 400) confidence += 0.02;
    if (growth_duration >= 60 && growth_duration <= 300) confidence += 0.02;
    return Math.min(confidence, 0.95);
  }

  async getModelInfo() {
    const info = {
      name: 'Water Requirement Prediction Model',
      water_unit: 'mm per day',
      features_required: ['temperature', 'humidity', 'rainfall', 'growth_duration', 'crop', 'area'],
      last_updated: new Date().toISOString()
    };
    if (this.modelParams) {
      info.r2_score = this.modelParams.r2 || null;
      info.rmse = this.modelParams.rmse || null;
    }
    return info;
  }
}

module.exports = new WaterRequirementService();
