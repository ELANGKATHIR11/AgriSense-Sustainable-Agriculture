const rateLimit = require('express-rate-limit');

// General API rate limiter
const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // 100 requests per window per IP
    message: {
        error: 'Too many requests',
        message: 'You have exceeded the 100 requests in 15 minutes limit. Please try again later.',
        retryAfter: '15 minutes'
    },
    standardHeaders: true, // Return rate limit info in headers
    legacyHeaders: false, // Disable X-RateLimit-* headers
});

// Stricter limiter for ML prediction endpoints
const mlLimiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 10, // 10 predictions per minute
    message: {
        error: 'Too many ML requests',
        message: 'ML predictions are limited to 10 per minute. Please wait before trying again.',
        retryAfter: '1 minute'
    },
    standardHeaders: true,
    legacyHeaders: false,
    // Skip successful requests from counting (optional)
    skip: (req, res) => res.statusCode < 400,
});

// Very strict limiter for expensive operations (like fine-tuning, batch predictions)
const heavyLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 5, // 5 requests per hour
    message: {
        error: 'Resource limit exceeded',
        message: 'Heavy operations are limited to 5 per hour.',
        retryAfter: '1 hour'
    },
    standardHeaders: true,
    legacyHeaders: false,
});

// Authentication endpoint limiter (prevent brute force)
const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 failed attempts
    message: {
        error: 'Too many login attempts',
        message: 'Too many authentication attempts. Please try again after 15 minutes.',
        retryAfter: '15 minutes'
    },
    skipSuccessfulRequests: true, // Don't count successful logins
});

module.exports = {
    apiLimiter,
    mlLimiter,
    heavyLimiter,
    authLimiter
};
