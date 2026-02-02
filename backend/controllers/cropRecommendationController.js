const express = require('express');
const { body, validationResult } = require('express-validator');
const cropRecommendationService = require('../services/cropRecommendationService');

const router = express.Router();

// Validation middleware
const validateCropInput = [
  body('N').isFloat({ min: 0, max: 200 }).withMessage('Nitrogen must be between 0 and 200 kg/ha'),
  body('P').isFloat({ min: 0, max: 200 }).withMessage('Phosphorus must be between 0 and 200 kg/ha'),
  body('K').isFloat({ min: 0, max: 200 }).withMessage('Potassium must be between 0 and 200 kg/ha'),
  body('temperature').isFloat({ min: -10, max: 50 }).withMessage('Temperature must be between -10°C and 50°C'),
  body('humidity').isFloat({ min: 0, max: 100 }).withMessage('Humidity must be between 0% and 100%'),
  body('ph').isFloat({ min: 0, max: 14 }).withMessage('pH must be between 0 and 14'),
  body('rainfall').isFloat({ min: 0, max: 500 }).withMessage('Rainfall must be between 0 and 500 mm'),
];

// POST /api/crop-recommendation/predict
router.post('/predict', validateCropInput, async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const { N, P, K, temperature, humidity, ph, rainfall } = req.body;

    // Get prediction from enhanced ML model
    const prediction = await cropRecommendationService.predictCrop({
      N, P, K, temperature, humidity, ph, rainfall
    });

    res.json({
      success: true,
      data: prediction,
      model_info: {
        name: 'Enhanced Crop Recommendation Model',
        accuracy: 0.9245,
        f1_score: 0.9247,
        version: '1.0'
      }
    });

  } catch (error) {
    console.error('Crop recommendation error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to make crop recommendation',
      message: error.message
    });
  }
});

// GET /api/crop-recommendation/info
router.get('/info', (req, res) => {
  res.json({
    model: 'Enhanced Crop Recommendation',
    description: 'Advanced ensemble model for crop recommendation using data augmentation and multiple ML algorithms',
    accuracy: 0.9245,
    f1_score: 0.9247,
    features: [
      'Nitrogen (N) in kg/ha',
      'Phosphorus (P) in kg/ha',
      'Potassium (K) in kg/ha',
      'Temperature in °C',
      'Humidity in %',
      'pH level',
      'Rainfall in mm'
    ],
    supported_crops: 96,
    last_updated: '2026-01-20T17:43:16.618280'
  });
});

// GET /api/crop-recommendation/crops (New endpoint)
router.get('/crops', async (req, res) => {
  try {
    const crops = await cropRecommendationService.getAllCrops();
    res.json({
      success: true,
      count: crops.length,
      data: crops
    });
  } catch (error) {
    console.error('Error fetching crops:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch crop list'
    });
  }
});

module.exports = router;
