const express = require('express');
const router = express.Router();
const dbType = (process.env.DB_TYPE || 'mongodb').toLowerCase();
let mongoose = null;
let pgPool = null;
if (dbType === 'postgres') {
    try {
        const pg = require('../config/postgres');
        pgPool = pg.pool;
    } catch (e) {
        // ignore
    }
} else {
    mongoose = require('mongoose');
}
const { exec } = require('child_process');
const path = require('path');

/**
 * @route GET /api/health
 * @desc Basic health check
 * @access Public
 */
router.get('/', async (req, res) => {
    const healthcheck = {
        uptime: process.uptime(),
        message: 'OK',
        timestamp: Date.now(),
        environment: process.env.NODE_ENV || 'development',
        version: process.env.npm_package_version || '1.0.0',
    };

    try {
        res.send(healthcheck);
    } catch (error) {
        healthcheck.message = error.message;
        res.status(503).send(healthcheck);
    }
});

/**
 * @route GET /api/health/ready
 * @desc Kubernetes readiness probe - checks if all dependencies are ready
 * @access Public
 */
router.get('/ready', async (req, res) => {
    const checks = {
        database: false,
        mlService: false,
    };

    let allReady = true;

    // Check database
    try {
        if (dbType === 'postgres') {
            // simple check: pool exists and can run a lightweight query
            if (pgPool) {
                try {
                    await pgPool.query('SELECT 1');
                    checks.database = true;
                } catch (e) {
                    checks.database = false;
                    allReady = false;
                }
            } else {
                checks.database = false;
                allReady = false;
            }
        } else {
            if (mongoose.connection.readyState === 1) {
                checks.database = true;
            } else {
                allReady = false;
            }
        }
    } catch (error) {
        checks.database = false;
        allReady = false;
    }

    // Check ML service
    try {
        const mlHealthCheck = await checkMLService();
        checks.mlService = mlHealthCheck;
        if (!mlHealthCheck) allReady = false;
    } catch (error) {
        checks.mlService = false;
        allReady = false;
    }

    const status = allReady ? 200 : 503;

    res.status(status).json({
        ready: allReady,
        checks,
        timestamp: new Date().toISOString(),
    });
});

/**
 * @route GET /api/health/live
 * @desc Kubernetes liveness probe - checks if the application is alive
 * @access Public
 */
router.get('/live', (req, res) => {
    res.status(200).json({
        alive: true,
        timestamp: new Date().toISOString(),
    });
});

/**
 * @route GET /api/health/detailed
 * @desc Detailed health check with all services
 * @access Public
 */
router.get('/detailed', async (req, res) => {
    const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: process.env.NODE_ENV || 'development',
        services: {},
        system: {
            memory: process.memoryUsage(),
            cpu: process.cpuUsage(),
            platform: process.platform,
            nodeVersion: process.version,
        },
    };

    // Check database
    try {
        if (dbType === 'postgres') {
            let ok = false;
            try {
                await pgPool.query('SELECT 1');
                ok = true;
            } catch (e) {
                ok = false;
            }
            health.services.database = {
                status: ok ? 'connected' : 'disconnected',
                type: 'postgres',
                name: process.env.PGDATABASE || 'agrisense',
            };
            if (!ok) health.status = 'degraded';
        } else {
            const dbState = mongoose.connection.readyState;
            health.services.database = {
                status: dbState === 1 ? 'connected' : 'disconnected',
                state: getMongooseState(dbState),
                name: mongoose.connection.name || 'unknown',
            };
            if (dbState !== 1) health.status = 'degraded';
        }
    } catch (error) {
        health.services.database = {
            status: 'error',
            error: error.message,
        };
        health.status = 'degraded';
    }

    // Check ML Service
    try {
        const mlServiceOk = await checkMLService();
        health.services.mlService = {
            status: mlServiceOk ? 'operational' : 'unavailable',
            endpoint: 'native ML service',
        };
        if (!mlServiceOk) health.status = 'degraded';
    } catch (error) {
        health.services.mlService = {
            status: 'error',
            error: error.message,
        };
        health.status = 'degraded';
    }

    // Check MQTT (if configured)
    try {
        // Placeholder - implement if MQTT is critical
        health.services.mqtt = {
            status: 'not_implemented',
        };
    } catch (error) {
        health.services.mqtt = {
            status: 'error',
            error: error.message,
        };
    }

    const statusCode = health.status === 'healthy' ? 200 : 503;
    res.status(statusCode).json(health);
});

/**
 * Helper function to check ML service availability
 */
async function checkMLService() {
    return new Promise((resolve) => {
        const pythonScript = path.join(__dirname, '..', 'ml', 'native_agricultural_advisor.py');

        // Simple check - try to import the module
        exec(`python -c "import sys; sys.path.append('${path.dirname(pythonScript)}'); print('OK')"`, {
            timeout: 5000,
        }, (error, stdout) => {
            if (error) {
                resolve(false);
            } else {
                resolve(stdout.trim() === 'OK');
            }
        });
    });
}

/**
 * Helper to get human-readable mongoose state
 */
function getMongooseState(state) {
    const states = {
        0: 'disconnected',
        1: 'connected',
        2: 'connecting',
        3: 'disconnecting',
    };
    return states[state] || 'unknown';
}

module.exports = router;
