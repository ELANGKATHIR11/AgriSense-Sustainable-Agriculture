const express = require('express');
const { body, validationResult } = require('express-validator');
const waterRequirementService = require('../services/waterRequirementService');

const router = express.Router();

// Validation middleware
const validateWaterInput = [
  body('temperature').isFloat({ min: -10, max: 50 }).withMessage('Temperature must be between -10°C and 50°C'),
  body('humidity').isFloat({ min: 0, max: 100 }).withMessage('Humidity must be between 0% and 100%'),
  body('rainfall').isFloat({ min: 0, max: 500 }).withMessage('Rainfall must be between 0 and 500 mm'),
  body('growth_duration').isFloat({ min: 30, max: 365 }).withMessage('Growth duration must be between 30 and 365 days'),
  body('area').optional().isFloat({ min: 0.1, max: 1000 }).withMessage('Area must be between 0.1 and 1000 hectares'),
];

// POST /api/water-requirement/predict
router.post('/predict', validateWaterInput, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const { temperature, humidity, rainfall, growth_duration, crop, area } = req.body;

    const prediction = await waterRequirementService.predictWaterRequirement({
      temperature, humidity, rainfall, growth_duration, crop, area
    });

    res.json({
      success: true,
      data: prediction,
      model_info: {
        name: 'Water Requirement Prediction Model',
        r2_score: 0.2793,
        rmse: 2.3104,
        version: '1.0'
      }
    });

  } catch (error) {
    console.error('Water requirement prediction error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to predict water requirement',
      message: error.message
    });
  }
});

// GET /api/water-requirement/info
router.get('/info', (req, res) => {
  res.json({
    model: 'Water Requirement Prediction',
    description: 'Model for predicting daily water requirements for crops',
    r2_score: 0.2793,
    rmse: 2.3104,
    features: [
      'Temperature in °C',
      'Humidity in %',
      'Rainfall in mm',
      'Growth duration in days'
    ],
    water_unit: 'mm per day',
    last_updated: '2026-01-20T17:43:16.618280'
  });
});

module.exports = router;
