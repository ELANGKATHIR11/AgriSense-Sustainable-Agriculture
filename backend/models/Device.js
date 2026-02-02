// Device model: supports Mongo (legacy) or Postgres (preferred when DB_TYPE=postgres)
const dbType = (process.env.DB_TYPE || 'mongodb').toLowerCase();

if (dbType === 'postgres') {
    const { pool } = require('../config/postgres');

    module.exports = {
        // filter: simple object { status: 'active' }
        find: async (filter = {}) => {
            const clauses = [];
            const values = [];
            let idx = 1;
            if (filter.status) {
                clauses.push(`status = $${idx++}`);
                values.push(filter.status);
            }
            const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
            const q = `SELECT device_id as "deviceId", name, type, location, status, last_active as "lastActive", configuration, metadata FROM devices ${where}`;
            const res = await pool.query(q, values);
            return res.rows.map(r => ({ deviceId: r.deviceId || r.device_id || r.device_id, ...r }));
        },
        findOne: async (filter = {}) => {
            const clauses = [];
            const values = [];
            let idx = 1;
            if (filter.deviceId || filter.device_id) {
                clauses.push(`device_id = $${idx++}`);
                values.push(filter.deviceId || filter.device_id);
            }
            const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
            const q = `SELECT device_id as "deviceId", name, type, location, status, last_active as "lastActive", configuration, metadata FROM devices ${where} ORDER BY id DESC LIMIT 1`;
            const res = await pool.query(q, values);
            return res.rows[0] || null;
        }
    };
} else {
    const mongoose = require('mongoose');

    const deviceSchema = new mongoose.Schema({
        deviceId: {
            type: String,
            required: true,
            unique: true,
            trim: true
        },
        name: {
            type: String,
            required: true
        },
        type: {
            type: String,
            enum: ['sensor', 'actuator', 'hybrid', 'environmental', 'soil'],
            default: 'sensor'
        },
        location: String,
        status: {
            type: String,
            enum: ['active', 'inactive', 'maintenance', 'error'],
            default: 'active'
        },
        lastActive: {
            type: Date,
            default: Date.now
        },
        configuration: {
            interval: { type: Number, default: 30000 }, // ms
            sensors: [String]
        },
        metadata: {
            firmwareVersion: String,
            ipAddress: String
        }
    }, {
        timestamps: true
    });

    module.exports = mongoose.model('Device', deviceSchema);
}
