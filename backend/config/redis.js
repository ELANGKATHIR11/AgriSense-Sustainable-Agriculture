const redis = require('redis');
const { promisify } = require('util');

// Create Redis client
const client = redis.createClient({
    host: process.env.REDIS_HOST || 'localhost',
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD || undefined,
    retry_strategy: (options) => {
        if (options.error && options.error.code === 'ECONNREFUSED') {
            // End reconnecting on a specific error and flush all commands with an error
            return new Error('The server refused the connection');
        }
        if (options.total_retry_time > 1000 * 60 * 60) {
            // End reconnecting after a specific timeout and flush all commands with an error
            return new Error('Retry time exhausted');
        }
        if (options.attempt > 10) {
            // End reconnecting with built in error
            return undefined;
        }
        // Reconnect after
        return Math.min(options.attempt * 100, 3000);
    }
});

// Connection event handlers
client.on('connect', () => {
    console.log('✅ Redis client connected');
});

client.on('ready', () => {
    console.log('✅ Redis client ready');
});

client.on('error', (err) => {
    console.error('❌ Redis Client Error:', err);
});

client.on('end', () => {
    console.log('⚠️  Redis client disconnected');
});

// Promisify Redis methods
const getAsync = promisify(client.get).bind(client);
const setAsync = promisify(client.set).bind(client);
const delAsync = promisify(client.del).bind(client);
const existsAsync = promisify(client.exists).bind(client);
const expireAsync = promisify(client.expire).bind(client);

/**
 * Caching middleware for ML predictions
 * @param {number} duration - Cache duration in seconds (default: 1 hour)
 */
const cacheMLPrediction = (duration = 3600) => {
    return async (req, res, next) => {
        // Only cache GET requests and POST requests with consistent inputs
        if (req.method !== 'GET' && req.method !== 'POST') {
            return next();
        }

        // Create cache key from request
        const cacheKey = `ml:${req.path}:${JSON.stringify(req.body || req.query)}`;

        try {
            // Check if cached response exists
            const cachedResponse = await getAsync(cacheKey);

            if (cachedResponse) {
                console.log(`🎯 Cache HIT: ${req.path}`);
                return res.json(JSON.parse(cachedResponse));
            }

            console.log(`⚠️  Cache MISS: ${req.path}`);

            // Store original res.json
            const originalJson = res.json.bind(res);

            // Override res.json to cache the response
            res.json = (data) => {
                // Cache the response
                setAsync(cacheKey, JSON.stringify(data), 'EX', duration)
                    .catch(err => console.error('Cache set error:', err));

                // Send response
                return originalJson(data);
            };

            next();
        } catch (error) {
            console.error('Cache middleware error:', error);
            // Continue without caching on error
            next();
        }
    };
};

/**
 * General purpose cache functions
 */
const cache = {
    /**
     * Get value from cache
     * @param {string} key
     * @returns {Promise<any>}
     */
    get: async (key) => {
        try {
            const value = await getAsync(key);
            return value ? JSON.parse(value) : null;
        } catch (error) {
            console.error('Cache get error:', error);
            return null;
        }
    },

    /**
     * Set value in cache
     * @param {string} key
     * @param {any} value
     * @param {number} ttl - Time to live in seconds
     */
    set: async (key, value, ttl = 3600) => {
        try {
            await setAsync(key, JSON.stringify(value), 'EX', ttl);
            return true;
        } catch (error) {
            console.error('Cache set error:', error);
            return false;
        }
    },

    /**
     * Delete key from cache
     * @param {string} key
     */
    del: async (key) => {
        try {
            await delAsync(key);
            return true;
        } catch (error) {
            console.error('Cache del error:', error);
            return false;
        }
    },

    /**
     * Check if key exists
     * @param {string} key
     */
    exists: async (key) => {
        try {
            const result = await existsAsync(key);
            return result === 1;
        } catch (error) {
            console.error('Cache exists error:', error);
            return false;
        }
    },

    /**
     * Clear all cache matching pattern
     * @param {string} pattern
     */
    clearPattern: async (pattern) => {
        return new Promise((resolve, reject) => {
            client.keys(pattern, async (err, keys) => {
                if (err) return reject(err);

                if (keys.length === 0) return resolve(0);

                client.del(keys, (err, count) => {
                    if (err) return reject(err);
                    resolve(count);
                });
            });
        });
    }
};

module.exports = {
    client,
    cache,
    cacheMLPrediction
};
