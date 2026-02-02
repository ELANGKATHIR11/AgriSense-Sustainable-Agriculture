const express = require('express');
const { body, validationResult } = require('express-validator');
const yieldPredictionService = require('../services/yieldPredictionService');

const router = express.Router();

// Validation middleware
const validateYieldInput = [
  body('crop').notEmpty().withMessage('Crop name is required'),
  body('area').isFloat({ min: 0.1, max: 1000 }).withMessage('Area must be between 0.1 and 1000 hectares'),
  body('humidity').isFloat({ min: 0, max: 100 }).withMessage('Humidity must be between 0% and 100%'),
  body('N').isFloat({ min: 0, max: 200 }).withMessage('Nitrogen must be between 0 and 200 kg/ha'),
  body('P').isFloat({ min: 0, max: 200 }).withMessage('Phosphorus must be between 0 and 200 kg/ha'),
  body('K').isFloat({ min: 0, max: 200 }).withMessage('Potassium must be between 0 and 200 kg/ha'),
  body('temperature').isFloat({ min: -10, max: 50 }).withMessage('Temperature must be between -10°C and 50°C'),
  body('rainfall').isFloat({ min: 0, max: 500 }).withMessage('Rainfall must be between 0 and 500 mm'),
  body('water_requirement').isFloat({ min: 0, max: 20 }).withMessage('Water requirement must be between 0 and 20 mm/day'),
  body('growth_duration').isFloat({ min: 30, max: 365 }).withMessage('Growth duration must be between 30 and 365 days'),
];

// POST /api/yield-prediction/predict
router.post('/predict', validateYieldInput, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const { crop, area, humidity, N, P, K, temperature, rainfall, water_requirement, growth_duration } = req.body;

    const prediction = await yieldPredictionService.predictYield({
      crop, area, humidity, N, P, K, temperature, rainfall, water_requirement, growth_duration
    });

    res.json({
      success: true,
      data: prediction,
      model_info: {
        name: 'Enhanced Yield Prediction Model',
        r2_score: 0.9004,
        rmse: 1.8870,
        version: '1.0'
      }
    });

  } catch (error) {
    console.error('Yield prediction error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to predict yield',
      message: error.message
    });
  }
});

// GET /api/yield-prediction/info
router.get('/info', (req, res) => {
  res.json({
    model: 'Enhanced Yield Prediction',
    description: 'Advanced ensemble model for agricultural yield prediction using real agricultural data',
    r2_score: 0.9004,
    rmse: 1.8870,
    features: [
      'Nitrogen (N) in kg/ha',
      'Phosphorus (P) in kg/ha',
      'Potassium (K) in kg/ha',
      'Temperature in °C',
      'Rainfall in mm',
      'Water requirement in mm/day',
      'Growth duration in days'
    ],
    yield_unit: 'tons per hectare',
    last_updated: '2026-01-20T17:43:16.618280'
  });
});

module.exports = router;
