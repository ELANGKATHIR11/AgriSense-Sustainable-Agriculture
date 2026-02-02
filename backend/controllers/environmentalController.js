const { spawn } = require('child_process');
const path = require('path');
const logger = require('../utils/logger');
const { validationResult } = require('express-validator');

/**
 * Predict environmental disease risk
 * POST /api/environmental-assessment
 * Body: { temperature, humidity, rainfall, ph }
 */
exports.predictDiseaseRisk = async (req, res) => {
  // Validate input
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ success: false, errors: errors.array() });
  }

  const { temperature, humidity, rainfall, ph } = req.body;

  logger.info('Environmental risk prediction requested', { temperature, humidity, rainfall, ph });

  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, '../ml/predict_environmental.py');
    const params = [
        pythonScript,
        '--temp', temperature,
        '--humidity', humidity,
        '--rainfall', rainfall,
        '--ph', ph
    ];

    const python = spawn('python', params);

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout);
          res.json({
            success: true,
            data: result
          });
        } catch (e) {
          logger.error('Failed to parse Python output', { error: e.message, stdout });
          res.status(500).json({ success: false, error: 'Failed to process prediction results' });
        }
      } else {
        logger.error('Python script failed', { code, stderr });
        res.status(500).json({ success: false, error: 'Prediction failed' });
      }
    });
  });
};
