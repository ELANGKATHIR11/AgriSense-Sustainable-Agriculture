const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { pythonPath } = require('../config/pythonConfig');
const llmService = require('../services/llmService');

/**
 * @route POST /api/llm/chat
 * @desc Get agricultural advice from native Phi-2 model or rule-based fallback
 * @access Public
 */
router.post('/chat', async (req, res) => {
  try {
    const { message, conversation_history, query: bodyQuery, context: bodyContext } = req.body;
    const query = message || bodyQuery;
    const context = conversation_history || bodyContext;

    if (!query) {
      return res.status(400).json({
        error: 'Query is required',
        message: 'Please provide a question or query'
      });
    }

    const pythonScript = path.join(__dirname, '..', 'ml', 'native_agricultural_advisor.py');
    const scriptExists = fs.existsSync(pythonScript);

    // Use Node.js rule-based fallback if Python script missing (no Py/torch dependency)
    if (!scriptExists) {
      const result = await llmService.generateAgriculturalAdvice(query, context || {});
      if (result.success && result.advice) {
        return res.json({
          success: true,
          reply: result.advice,
          advice: result.advice,
          recommendations: result.recommendations || [],
          model: result.source || 'rule-based',
          timestamp: new Date().toISOString()
        });
      }
    }

    // Spawn Python process
    const pythonProcess = spawn(pythonPath, [pythonScript]);

    let dataString = '';
    let errorString = '';
    let responseSent = false;

    // Set timeout (120 seconds) to prevent hanging requests but allow model load time
    const timeoutId = setTimeout(() => {
      if (!responseSent) {
        pythonProcess.kill();
        responseSent = true;
        res.status(504).json({
          error: 'Request timeout',
          message: 'ML service took too long to respond (Model loading)'
        });
      }
    }, 120000);

    // Send input data
    pythonProcess.stdin.write(JSON.stringify({ query, context }));
    pythonProcess.stdin.end();

    // Collect output
    pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorString += data.toString();
    });

    // Handle completion
    pythonProcess.on('close', (code) => {
      clearTimeout(timeoutId);

      if (responseSent) return;

      if (code !== 0) {
        console.error('Python process error:', errorString);
        llmService.generateAgriculturalAdvice(query, context || {})
          .then((result) => {
            if (responseSent) return;
            if (result.success && result.advice) {
              responseSent = true;
              return res.json({
                success: true,
                reply: result.advice,
                advice: result.advice,
                recommendations: result.recommendations || [],
                model: 'rule-based-fallback',
                timestamp: new Date().toISOString()
              });
            }
            throw new Error('No advice');
          })
          .catch(() => {
            if (responseSent) return;
            responseSent = true;
            res.json({
              success: true,
              reply: "For most crops, maintaining proper soil moisture and watching for early signs of disease like discoloration is key. Check our Crop Library for crop-specific advice!",
              model: 'fallback',
              timestamp: new Date().toISOString()
            });
          });
        return;
      }

      try {
        // Parse the response
        let result;
        try {
          result = JSON.parse(dataString);
        } catch (e) {
          // Sometimes python might output extra logs before json, try to find json
          const jsonMatch = dataString.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            result = JSON.parse(jsonMatch[0]);
          } else {
            throw e;
          }
        }

        responseSent = true;
        res.json({
          success: true,
          model: result.model || 'native',
          reply: result.advice, // Map 'advice' to 'reply' for frontend
          advice: result.advice,
          recommendations: result.recommendations || [],
          confidence: result.confidence || 0.75,
          sources: result.sources || ['Native Agricultural Advisor'],
          timestamp: new Date().toISOString()
        });
      } catch (parseError) {
        console.error('Parse error:', parseError, 'Data:', dataString);
        llmService.generateAgriculturalAdvice(query, context || {}).then((r) => {
          if (responseSent) return;
          responseSent = true;
          if (r.success && r.advice) res.json({ success: true, reply: r.advice, advice: r.advice, model: 'rule-based', timestamp: new Date().toISOString() });
          else res.status(500).json({ error: 'Failed to parse response', message: 'ML service returned invalid data' });
        }).catch(() => {
          if (responseSent) return;
          responseSent = true;
          res.status(500).json({ error: 'Failed to parse response', message: 'ML service returned invalid data' });
        });
      }
    });

    pythonProcess.on('error', (err) => {
      clearTimeout(timeoutId);
      if (responseSent) return;
      console.error('Python spawn error:', err);
      llmService.generateAgriculturalAdvice(query, context || {}).then((r) => {
        if (responseSent) return;
        responseSent = true;
        if (r.success && r.advice) res.json({ success: true, reply: r.advice, advice: r.advice, model: 'rule-based', timestamp: new Date().toISOString() });
        else res.status(500).json({ error: 'LLM unavailable', message: err.message });
      }).catch(() => {
        if (responseSent) return;
        responseSent = true;
        res.status(500).json({ error: 'LLM unavailable', message: err.message });
      });
    });

  } catch (error) {
    console.error('LLM Controller Error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

/**
 * @route GET /api/llm/health
 * @desc Check LLM service health
 * @access Public
 */
router.get('/health', async (req, res) => {
  try {
    const pythonScript = path.join(__dirname, '..', 'ml', 'native_agricultural_advisor.py');
    const { exec } = require('child_process');

    exec(`"${pythonPath}" "${pythonScript}" --health`, (error, stdout, stderr) => {
      if (error) {
        return res.status(503).json({
          status: 'unhealthy',
          message: 'Native advisor not accessible',
          error: stderr
        });
      }

      res.json({
        status: 'healthy',
        message: 'Native agricultural advisor is operational',
        model: 'phi-2 or rule-based',
        timestamp: new Date().toISOString()
      });
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message
    });
  }
});

module.exports = router;
