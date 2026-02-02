const express = require('express');
const multer = require('multer');
const { body, validationResult } = require('express-validator');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const logger = require('../utils/logger');

const router = express.Router();

// Configure multer for image upload
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../uploads/vlm');
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1E9)}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|bmp|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);

    if (extname && mimetype) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

// Helper function to run Python VLM inference
async function runVLMInference(imagePath) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, '../ml/train_native_vlm.py');
    const modelPath = path.join(__dirname, '../ml/models/best_vlm_model.pth');

    const python = spawn('python', [
      pythonScript,
      '--mode', 'infer',
      '--image', imagePath,
      '--model', modelPath
    ]);

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
          // Parse JSON output from Python script
          const results = JSON.parse(stdout);
          resolve(results);
        } catch (error) {
          logger.error('Failed to parse VLM output:', error);
          reject(new Error('Invalid VLM output format'));
        }
      } else {
        logger.error(`VLM inference failed with code ${code}:`, stderr);
        reject(new Error(stderr || 'VLM inference failed'));
      }
    });
  });
}

// Helper function to run Targeted VLM inference
async function runTargetedInference(imagePath) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, '../ml/predict_targeted_vlm.py');
    const modelDir = path.join(__dirname, '../ml/models');

    const python = spawn('python', [
      pythonScript,
      '--image', imagePath,
      '--model_dir', modelDir
    ]);

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
          const results = JSON.parse(stdout);
          resolve(results);
        } catch (error) {
          logger.error('Failed to parse Targeted VLM output:', error);
          reject(new Error('Invalid Targeted VLM output format'));
        }
      } else {
        logger.error(`Targeted VLM inference failed with code ${code}:`, stderr);
        reject(new Error(stderr || 'Targeted VLM inference failed'));
      }
    });
  });
}

/**
 * POST /api/vlm/analyze-plant
 * Analyze plant image using native VLM model
 */
router.post('/analyze-plant', upload.single('image'), [
  body('query').optional().isLength({ min: 1, max: 500 }).withMessage('Query must be between 1 and 500 characters'),
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No image file provided'
      });
    }

    const { query = '' } = req.body;
    const imagePath = req.file.path;

    logger.info(`VLM analysis requested for: ${imagePath}`);

    // Run native VLM inference
    const analysis = await runVLMInference(imagePath);

    // Clean up uploaded file
    try {
      await fs.unlink(imagePath);
    } catch (err) {
      logger.warn(`Failed to delete temporary file: ${imagePath}`);
    }

    // Format response
    res.json({
      success: true,
      data: {
        predictions: analysis,
        query: query || 'Disease detection',
        model_info: {
          name: 'Native Plant Disease VLM',
          version: '1.0.0',
          type: 'Vision-Language Model',
          backend: 'PyTorch',
          framework: 'CLIP-style contrastive learning'
        },
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    logger.error('VLM Plant Analysis Error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to analyze plant image',
      message: error.message
    });
  }
});

/**
 * POST /api/vlm/analyze-targeted
 * Analyze specific plant disease (Healthy/Rust/Powdery)
 */
router.post('/analyze-targeted', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No image file provided'
      });
    }

    const imagePath = req.file.path;
    logger.info(`Targeted VLM analysis requested for: ${imagePath}`);

    const analysis = await runTargetedInference(imagePath);

    // Clean up
    try {
      await fs.unlink(imagePath);
    } catch (err) {
      logger.warn(`Failed to delete temporary file: ${imagePath}`);
    }

    res.json({
      success: true,
      data: {
        ...analysis,
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    logger.error('Targeted VLM Analysis Error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to analyze image with targeted model',
      message: error.message
    });
  }
});

/**
 * POST /api/vlm/disease-report
 * Generate disease report from analysis  
 */
router.post('/disease-report', async (req, res) => {
  try {
    const { analysis } = req.body;

    if (!analysis || !analysis.predictions) {
      return res.status(400).json({
        success: false,
        error: 'Analysis data with predictions is required'
      });
    }

    // Generate report from VLM predictions
    const topPrediction = analysis.predictions[0];

    const report = {
      disease_detected: topPrediction.disease,
      confidence: topPrediction.confidence,
      severity: topPrediction.severity,
      treatment_recommendations: [
        topPrediction.treatment,
        'Monitor the plant regularly for disease progression',
        'Ensure proper ventilation and avoid overcrowding',
        'Remove and destroy infected leaves to prevent spread',
        'Consider applying appropriate organic or chemical treatments'
      ],
      prevention_measures: [
        'Practice crop rotation to reduce disease buildup',
        'Maintain proper plant spacing for air circulation',
        'Water at the base of plants to keep foliage dry',
        'Use disease-resistant varieties when available',
        'Keep the growing area clean and remove plant debris'
      ],
      alternative_diagnoses: analysis.predictions.slice(1, 3).map(pred => ({
        disease: pred.disease,
        confidence: pred.confidence
      })),
      expert_consultation: topPrediction.confidence < 0.7 ?
        'Low confidence detection - recommend expert verification' :
        'Consult with an agricultural extension officer for confirmation',
      generated_at: new Date().toISOString()
    };

    res.json({
      success: true,
      data: report,
      model_info: {
        name: 'Disease Report Generator',
        capabilities: [
          'severity_assessment',
          'treatment_planning',
          'prevention_measures',
          'alternative_diagnoses'
        ]
      }
    });

  } catch (error) {
    logger.error('VLM Disease Report Error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to generate disease report',
      message: error.message
    });
  }
});

/**
 * GET /api/vlm/info
 * Get information about the native VLM model
 */
router.get('/info', (req, res) => {
  res.json({
    model: 'Native Plant Disease Vision-Language Model',
    version: '1.0.0',
    description: 'PyTorch-based VLM for plant disease detection using contrastive learning',
    architecture: {
      image_encoder: 'ResNet50',
      text_encoder: 'LSTM',
      embedding_dim: 512,
      training_framework: 'PyTorch'
    },
    capabilities: [
      'Plant disease identification',
      'Multi-class classification (38+ diseases)',
      'Confidence scoring',
      'Treatment recommendations',
      'Health status assessment',
      'Alternative diagnoses'
    ],
    supported_formats: ['JPEG', 'PNG', 'GIF', 'BMP', 'WebP'],
    max_file_size: '10MB',
    supported_plants: [
      'Tomato', 'Potato', 'Pepper', 'Corn (Maize)', 'Apple',
      'Grape', 'Cherry', 'Peach', 'Strawberry', 'Orange',
      'Squash', 'Raspberry', 'Soybean', 'Blueberry'
    ],
    detectable_diseases: [
      'Bacterial spot', 'Early blight', 'Late blight',
      'Leaf mold', 'Septoria leaf spot', 'Spider mites',
      'Target spot', 'Mosaic virus', 'Yellow leaf curl virus',
      'Powdery mildew', 'Black rot', 'Rust', 'Scab',
      'And 25+ more diseases'
    ],
    training_dataset: 'PlantVillage (54,000+ images, 38 classes)',
    accuracy: '90%+ on validation set',
    inference_time: '< 1 second',
    deployment: 'Local CPU/GPU inference',
    cost: '$0 (no API calls)',
    last_updated: '2026-01-25T21:00:00.000Z'
  });
});

/**
 * GET /api/vlm/health
 * Health check for VLM service
 */
router.get('/health', async (req, res) => {
  try {
    const modelPath = path.join(__dirname, '../ml/models/best_vlm_model.pth');

    // Check if model exists
    const modelExists = await fs.access(modelPath)
      .then(() => true)
      .catch(() => false);

    res.json({
      status: modelExists ? 'operational' : 'model_not_found',
      model_loaded: modelExists,
      model_path: modelPath,
      message: modelExists ?
        'Native VLM service is ready' :
        'VLM model not found - run training first',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

module.exports = router;
