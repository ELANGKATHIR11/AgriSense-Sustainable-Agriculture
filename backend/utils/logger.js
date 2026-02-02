const winston = require('winston');
const path = require('path');

// Define log format
const logFormat = winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    winston.format.errors({ stack: true }),
    winston.format.splat(),
    winston.format.json()
);

// Console format for development
const consoleFormat = winston.format.combine(
    winston.format.colorize(),
    winston.format.timestamp({ format: 'HH:mm:ss' }),
    winston.format.printf(({ level, message, timestamp, ...metadata }) => {
        let msg = `${timestamp} [${level}]: ${message}`;
        if (Object.keys(metadata).length > 0) {
            msg += ` ${JSON.stringify(metadata)}`;
        }
        return msg;
    })
);

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, '..', 'logs');
const fs = require('fs');
if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir);
}

// Create logger
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: logFormat,
    defaultMeta: { service: 'agrisense-backend' },
    transports: [
        // Error logs
        new winston.transports.File({
            filename: path.join(logsDir, 'error.log'),
            level: 'error',
            maxsize: 5242880, // 5MB
            maxFiles: 5,
        }),
        // Combined logs
        new winston.transports.File({
            filename: path.join(logsDir, 'combined.log'),
            maxsize: 5242880, // 5MB
            maxFiles: 5,
        }),
        // ML prediction logs (separate for analytics)
        new winston.transports.File({
            filename: path.join(logsDir, 'ml-predictions.log'),
            level: 'info',
            maxsize: 10485760, // 10MB
            maxFiles: 10,
        }),
    ],
    // Handle uncaught exceptions and rejections
    exceptionHandlers: [
        new winston.transports.File({
            filename: path.join(logsDir, 'exceptions.log'),
        }),
    ],
    rejectionHandlers: [
        new winston.transports.File({
            filename: path.join(logsDir, 'rejections.log'),
        }),
    ],
});

// Add console transport in development
if (process.env.NODE_ENV !== 'production') {
    logger.add(
        new winston.transports.Console({
            format: consoleFormat,
        })
    );
}

// Helper methods for structured logging
logger.logMLPrediction = (modelName, input, output, duration) => {
    logger.info('ML Prediction', {
        type: 'ml_prediction',
        model: modelName,
        input_hash: require('crypto').createHash('md5').update(JSON.stringify(input)).digest('hex'),
        output_summary: typeof output === 'object' ? Object.keys(output) : 'string',
        duration_ms: duration,
        timestamp: new Date().toISOString(),
    });
};

logger.logAPIRequest = (method, path, statusCode, duration, userId = null) => {
    logger.info('API Request', {
        type: 'api_request',
        method,
        path,
        status_code: statusCode,
        duration_ms: duration,
        user_id: userId,
        timestamp: new Date().toISOString(),
    });
};

logger.logError = (error, context = {}) => {
    logger.error('Application Error', {
        type: 'application_error',
        message: error.message,
        stack: error.stack,
        ...context,
        timestamp: new Date().toISOString(),
    });
};

// Export logger
module.exports = logger;
