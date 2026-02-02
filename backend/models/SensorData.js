// SensorData model: supports Mongo (legacy) or Postgres (preferred when DB_TYPE=postgres)
const dbType = (process.env.DB_TYPE || 'mongodb').toLowerCase();

if (dbType === 'postgres') {
    const { pool } = require('../config/postgres');

    module.exports = {
        // Find latest sensor data for a device
        findOne: async (query = {}) => {
            const deviceId = query.deviceId || query.device_id || query.device;
            if (!deviceId) return null;
            const q = `SELECT * FROM sensor_data WHERE device_id = $1 ORDER BY timestamp DESC LIMIT 1`;
            const res = await pool.query(q, [deviceId]);
            return res.rows[0] || null;
        },
        // Generic find with optional filters (simple implementation)
        find: async (filter = {}) => {
            const clauses = [];
            const values = [];
            let idx = 1;
            if (filter.deviceId) {
                clauses.push(`device_id = $${idx++}`);
                values.push(filter.deviceId);
            }
            if (filter.limit) {
                // handled separately
            }
            const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
            const limit = filter.limit ? `LIMIT ${parseInt(filter.limit)}` : '';
            const q = `SELECT * FROM sensor_data ${where} ORDER BY timestamp DESC ${limit}`;
            const res = await pool.query(q, values);
            return res.rows;
        }
    };

} else {
    const mongoose = require('mongoose');

    const sensorDataSchema = new mongoose.Schema({
        deviceId: {
            type: String,
            required: true,
            ref: 'Device',
            index: true
        },
        timestamp: {
            type: Date,
            default: Date.now,
            index: true
        },
        // Environmental
        temperature: Number,
        humidity: Number,
        pressure: Number,
        lightIntensity: Number,

        // Soil
        soilMoisture: Number,
        soilTemperature: Number,
        phLevel: Number,
        nitrogen: Number,
        phosphorus: Number,
        potassium: Number,

        // Other
        batteryLevel: Number,
        signalStrength: Number
    }, {
        timestamps: true // adds createdAt, updatedAt
    });

    // Compound index for efficient time-range queries per device
    sensorDataSchema.index({ deviceId: 1, timestamp: -1 });

    module.exports = mongoose.model('SensorData', sensorDataSchema);
}
