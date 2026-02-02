const express = require('express');
const { body, validationResult } = require('express-validator');
const mlService = require('../services/mlService');

const router = express.Router();

const validateAnalysisInput = [
    body('N').isFloat({ min: 0, max: 500 }),
    body('P').isFloat({ min: 0, max: 500 }),
    body('K').isFloat({ min: 0, max: 500 }),
    body('temperature').isFloat({ min: -10, max: 60 }),
    body('humidity').isFloat({ min: 0, max: 100 }),
    body('ph').isFloat({ min: 0, max: 14 }),
    body('rainfall').isFloat({ min: 0, max: 1000 }),
    body('area').isFloat({ min: 0.1, max: 1000 })
];

router.post('/analyze', validateAnalysisInput, async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ success: false, errors: errors.array() });
        }

        const result = await mlService.analyze(req.body);
        
        res.json({
            success: true,
            data: result,
            metadata: {
                timestamp: new Date().toISOString(),
                engine: "AgriSense Unified ML v2.0",
                stages: ["Water", "Season", "Group", "Species", "Yield"]
            }
        });

    } catch (error) {
        console.error('ML Analysis error:', error);
        res.status(500).json({
            success: false,
            message: 'Agri-Intelligence Engine Error',
            error: error.message
        });
    }
});

module.exports = router;
