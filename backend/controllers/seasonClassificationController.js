const express = require('express');
const { body, validationResult } = require('express-validator');
const seasonClassificationService = require('../services/seasonClassificationService');

const router = express.Router();

// Validation middleware
const validateSeasonInput = [
  body('temperature').isFloat({ min: -10, max: 50 }).withMessage('Temperature must be between -10°C and 50°C'),
  body('rainfall').isFloat({ min: 0, max: 500 }).withMessage('Rainfall must be between 0 and 500 mm'),
  body('humidity').isFloat({ min: 0, max: 100 }).withMessage('Humidity must be between 0% and 100%'),
  body('growth_duration').isFloat({ min: 30, max: 365 }).withMessage('Growth duration must be between 30 and 365 days'),
];

// POST /api/season-classification/predict
router.post('/predict', validateSeasonInput, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const { temperature, rainfall, humidity, growth_duration } = req.body;

    const prediction = await seasonClassificationService.predictSeason({
      temperature, rainfall, humidity, growth_duration
    });

    res.json({
      success: true,
      data: prediction,
      model_info: {
        name: 'Season Classification Model',
        accuracy: 0.80,
        f1_score: 0.80,
        version: '1.0'
      }
    });

  } catch (error) {
    console.error('Season classification error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to classify season',
      message: error.message
    });
  }
});

// GET /api/season-classification/info
router.get('/info', (req, res) => {
  res.json({
    model: 'Season Classification',
    description: 'Model for classifying optimal growing seasons',
    accuracy: 0.80,
    f1_score: 0.80,
    features: [
      'Temperature in °C',
      'Rainfall in mm',
      'Humidity in %',
      'Growth duration in days'
    ],
    seasons: [
      'Kharif (Monsoon)',
      'Rabi (Winter)', 
      'Zaid (Summer)',
      'Summer',
      'Winter'
    ],
    last_updated: '2026-01-20T17:43:16.618280'
  });
});

module.exports = router;
