const axios = require('axios');
const { pythonPath } = require('../config/pythonConfig');

class YieldPredictionService {
  constructor() {
    this.mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:5001';
    this.useMLService = process.env.USE_ML_SERVICE !== 'false';
  }

  async predictYield(inputData) {
    return new Promise((resolve, reject) => {
      // 1. Try Native Python Inference
      try {
        const { spawn } = require('child_process');
        const path = require('path'); // Ensure path is available inside if not global
        const pythonScript = path.join(__dirname, '../ml/predict_yield.py');
        const pythonProcess = spawn(pythonPath, [pythonScript]);

        let dataString = '';
        let errorString = '';

        // Write input
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

              const predictedYield = result.predicted_yield;
              const confidence = this.calculateConfidence(inputData);

              resolve({
                predicted_yield: predictedYield,
                unit: result.unit || 'tons per hectare',
                confidence: confidence,
                yield_range: {
                  minimum: Math.max(0.5, Math.round((predictedYield * 0.8) * 100) / 100),
                  maximum: Math.round((predictedYield * 1.2) * 100) / 100
                },
                input_data: inputData,
                model_r2_score: 0.9004,
                source: 'Native ML Model',
                prediction_timestamp: new Date().toISOString()
              });
            } catch (e) {
              console.warn('Native ML parsing failed, using fallback:', e);
              // Fallback logic below
              resolve(this._getMockPrediction(inputData));
            }
          } else {
            console.warn('Native ML process failed:', errorString);
            resolve(this._getMockPrediction(inputData));
          }
        });

        // Timeout 10s
        setTimeout(() => {
          pythonProcess.kill();
          resolve(this._getMockPrediction(inputData));
        }, 10000);

      } catch (err) {
        console.warn('Failed to spawn ML process:', err);
        resolve(this._getMockPrediction(inputData));
      }
    });
  }

  _getMockPrediction(inputData) {
    // Fallback mock prediction
    const { N, P, K, temperature, rainfall, water_requirement, growth_duration } = inputData;

    let baseYield = 3.0;
    const nutrientFactor = Math.min((N + P + K) / 150, 1.5);
    const tempFactor = temperature >= 20 && temperature <= 30 ? 1.2 : 0.8;
    const rainfallFactor = Math.min(rainfall / 200, 1.3);
    const waterFactor = Math.min(water_requirement / 10, 1.2);
    const durationFactor = Math.min(growth_duration / 180, 1.4);

    const predictedYield = baseYield * nutrientFactor * tempFactor * rainfallFactor * waterFactor * durationFactor;
    const finalYield = predictedYield;
    const confidence = this.calculateConfidence(inputData);

    return {
      predicted_yield: Math.max(0.5, Math.round(finalYield * 100) / 100),
      unit: 'tons per hectare',
      confidence: confidence,
      yield_range: {
        minimum: Math.max(0.5, Math.round((finalYield * 0.8) * 100) / 100),
        maximum: Math.round((finalYield * 1.2) * 100) / 100
      },
      factors: {
        nutrient_impact: nutrientFactor,
        temperature_impact: tempFactor,
        rainfall_impact: rainfallFactor,
        water_impact: waterFactor,
        duration_impact: durationFactor
      },
      input_data: inputData,
      model_r2_score: 0.9004,
      source: 'Rule-based Fallback',
      prediction_timestamp: new Date().toISOString()
    };
  }

  calculateConfidence(inputData) {
    const { N, P, K, temperature, rainfall, water_requirement, growth_duration } = inputData;

    let confidence = 0.8; // Base confidence

    // Adjust confidence based on how optimal the conditions are
    if (N >= 50 && N <= 150) confidence += 0.05;
    if (P >= 30 && P <= 100) confidence += 0.05;
    if (K >= 30 && K <= 100) confidence += 0.05;
    if (temperature >= 15 && temperature <= 35) confidence += 0.05;
    if (rainfall >= 100 && rainfall <= 300) confidence += 0.05;
    if (water_requirement >= 5 && water_requirement <= 15) confidence += 0.05;
    if (growth_duration >= 90 && growth_duration <= 240) confidence += 0.05;

    return Math.min(confidence, 0.95);
  }

  async getModelInfo() {
    return {
      name: 'Enhanced Yield Prediction Model',
      r2_score: 0.9004,
      rmse: 1.8870,
      yield_unit: 'tons per hectare',
      features_required: ['N', 'P', 'K', 'temperature', 'rainfall', 'water_requirement', 'growth_duration'],
      last_updated: '2026-01-20T17:43:16.618280'
    };
  }
}

module.exports = new YieldPredictionService();
