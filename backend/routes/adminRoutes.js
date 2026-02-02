const express = require('express');
const router = express.Router();

// In-memory activity log (in production, use DB)
let activityLog = [];

const addLog = (action, details, status = 'success') => {
    const log = {
        id: Date.now(),
        action,
        details,
        status,
        timestamp: new Date()
    };
    activityLog.unshift(log);
    if (activityLog.length > 50) activityLog.pop(); // Keep last 50
    return log;
};

// GET /api/admin/activities
router.get('/activities', (req, res) => {
    res.json(activityLog);
});

// GET /api/admin/summary (System metrics)
router.get('/summary', (req, res) => {
    const memoryUsage = process.memoryUsage();
    res.json({
        uptime: process.uptime(),
        memory: {
            heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024) + 'MB',
            heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024) + 'MB',
        },
        cpuLoad: Math.round(Math.random() * 20) + '%', // Mock CPU load
        apiStatus: 'Active',
        activeConnections: 0 // connect to socket service to get real count
    });
});

// POST /api/admin/reset
router.post('/reset', (req, res) => {
    try {
        // Logic to clear database or reset state
        // For now, we just clear logs and mock success
        activityLog = [];
        addLog('System Reset', 'All system data erased by admin', 'warning');
        res.json({ success: true, message: 'System reset successfully' });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
});

// POST /api/admin/action (Generic actions)
router.post('/action', (req, res) => {
    const { action } = req.body;

    let details = '';
    switch (action) {
        case 'Reload':
            details = 'ML Models reloaded successfully';
            break;
        case 'Dataset':
            details = 'Training datasets refreshed';
            break;
        case 'Weather':
            details = 'Weather data synced with OpenWeatherMap';
            break;
        default:
            details = `Action ${action} executed`;
    }

    addLog(action, details, 'success');
    res.json({ success: true, message: details });
});

addLog('System Startup', 'All services initialized', 'success');

module.exports = router;
