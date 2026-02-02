const express = require('express');
const { body, validationResult } = require('express-validator');
const cropTypeService = require('../services/cropTypeService');

const router = express.Router();

// Validation middleware
const validateCropTypeInput = [
  body('N').isFloat({ min: 0, max: 200 }).withMessage('Nitrogen must be between 0 and 200 kg/ha'),
  body('P').isFloat({ min: 0, max: 200 }).withMessage('Phosphorus must be between 0 and 200 kg/ha'),
  body('K').isFloat({ min: 0, max: 200 }).withMessage('Potassium must be between 0 and 200 kg/ha'),
  body('temperature').isFloat({ min: -10, max: 50 }).withMessage('Temperature must be between -10°C and 50°C'),
  body('humidity').isFloat({ min: 0, max: 100 }).withMessage('Humidity must be between 0% and 100%'),
  body('ph').isFloat({ min: 0, max: 14 }).withMessage('pH must be between 0 and 14'),
  body('rainfall').isFloat({ min: 0, max: 500 }).withMessage('Rainfall must be between 0 and 500 mm'),
  body('growth_duration').optional().isFloat({ min: 30, max: 365 }).withMessage('Growth duration must be between 30 and 365 days'),
];

// POST /api/crop-type/predict
router.post('/predict', validateCropTypeInput, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const { N, P, K, temperature, humidity, ph, rainfall, growth_duration } = req.body;

    const prediction = await cropTypeService.predictCropType({
      N, P, K, temperature, humidity, ph, rainfall, growth_duration
    });

    res.json({
      success: true,
      data: prediction,
      model_info: {
        name: 'Enhanced Crop Type Classification Model',
        accuracy: 0.55,
        f1_score: 0.4633,
        version: '1.0'
      }
    });

  } catch (error) {
    console.error('Crop type classification error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to classify crop type',
      message: error.message
    });
  }
});

// GET /api/crop-type/info
router.get('/info', (req, res) => {
  res.json({
    model: 'Enhanced Crop Type Classification',
    description: 'Model for classifying crops into agricultural categories',
    accuracy: 0.55,
    f1_score: 0.4633,
    features: [
      'Nitrogen (N) in kg/ha',
      'Phosphorus (P) in kg/ha',
      'Potassium (K) in kg/ha',
      'Temperature in °C',
      'Humidity in %',
      'pH level',
      'Rainfall in mm',
      'Growth duration in days (optional)'
    ],
    crop_types: [
      'Cereal', 'Pulse', 'Oilseed', 'Cash Crop', 
      'Vegetable', 'Fruit', 'Spice', 'Plantation', 'Tuber', 'Fiber'
    ],
    last_updated: '2026-01-20T17:43:16.618280'
  });
});

module.exports = router;
