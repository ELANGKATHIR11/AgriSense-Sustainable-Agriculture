const logger = require('../utils/logger');
const path = require('path');

/**
 * Database connector. Supports `mongodb` (legacy) and `postgres` via env var `DB_TYPE`.
 * Default behavior: if DB_TYPE=postgres uses Postgres, otherwise falls back to previous Mongo behaviour if available.
 */
const connectDB = async () => {
    const dbType = (process.env.DB_TYPE || 'mongodb').toLowerCase();

    if (dbType === 'postgres') {
        try {
            // lazy-require to avoid adding pg when not used
            const { ensurePostgresInitialized } = require('./postgres');
            await ensurePostgresInitialized();
            logger.info('✅ PostgreSQL initialized and ready');
            return;
        } catch (err) {
            logger.error('Postgres initialization failed:', err.message);
            logger.warn('⚠️  Continuing without DB connection');
            return;
        }
    }

    // Legacy MongoDB support (if still present)
    try {
        const mongoose = require('mongoose');
        mongoose.set('strictQuery', false);

        const options = {
            maxPoolSize: 10,
            minPoolSize: 2,
            socketTimeoutMS: 45000,
            serverSelectionTimeoutMS: 5000,
            family: 4,
        };

        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/agrisense', options);
        logger.info('✅ MongoDB connected successfully');
        return;
    } catch (error) {
        logger.warn('MongoDB not available or failed to connect; skipping.');
        logger.warn('⚠️  Continuing without MongoDB connection');
        return;
    }
};

module.exports = connectDB;
