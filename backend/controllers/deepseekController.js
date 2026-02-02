// DeepSeek integration removed — controller stub kept for compatibility.
const express = require('express');
const router = express.Router();

// This endpoint intentionally returns 404 to indicate DeepSeek integration
// has been reverted. Remove this file entirely if you want no trace.
router.all('/', (req, res) => {
  res.status(404).json({ error: 'DeepSeek integration has been removed' });
});

module.exports = router;
