const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');
const logger = require('../utils/logger');

// Read connection from ENV or construct
const connectionString = process.env.POSTGRES_URL || (process.env.PGHOST ?
  `postgresql://${process.env.PGUSER || 'postgres'}:${process.env.PGPASSWORD || ''}@${process.env.PGHOST}:${process.env.PGPORT || 5432}/${process.env.PGDATABASE || 'agrisense'}`
  : null);

const pool = new Pool(connectionString ? { connectionString } : {
  host: process.env.PGHOST || 'localhost',
  port: process.env.PGPORT ? parseInt(process.env.PGPORT) : 5432,
  user: process.env.PGUSER || 'postgres',
  password: process.env.PGPASSWORD || '',
  database: process.env.PGDATABASE || 'agrisense',
  max: 10
});

// Consolidated table creation for all required tables
async function ensureTables() {
  const client = await pool.connect();
  try {
    // devices
    await client.query(`
      CREATE TABLE IF NOT EXISTS devices (
        id SERIAL PRIMARY KEY,
        device_id VARCHAR(128) UNIQUE,
        name VARCHAR(255),
        type VARCHAR(64),
        location VARCHAR(255),
        status VARCHAR(32),
        last_active TIMESTAMP,
        configuration JSONB,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT now(),
        updated_at TIMESTAMP DEFAULT now()
      );
    `);

    // sensor_data
    await client.query(`
      CREATE TABLE IF NOT EXISTS sensor_data (
        id SERIAL PRIMARY KEY,
        device_id VARCHAR(128),
        timestamp TIMESTAMP DEFAULT now(),
        temperature DOUBLE PRECISION,
        humidity DOUBLE PRECISION,
        pressure DOUBLE PRECISION,
        light_intensity DOUBLE PRECISION,
        soil_moisture DOUBLE PRECISION,
        soil_temperature DOUBLE PRECISION,
        ph_level DOUBLE PRECISION,
        nitrogen DOUBLE PRECISION,
        phosphorus DOUBLE PRECISION,
        potassium DOUBLE PRECISION,
        battery_level DOUBLE PRECISION,
        signal_strength DOUBLE PRECISION,
        created_at TIMESTAMP DEFAULT now()
      );
    `);

    // datasets table for ML/LLM/VLM
    await client.query(`
      CREATE TABLE IF NOT EXISTS datasets (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type VARCHAR(32) DEFAULT 'CSV',
        size_bytes BIGINT DEFAULT 0,
        records INTEGER DEFAULT 0,
        uploaded_at TIMESTAMP DEFAULT now(),
        status VARCHAR(32) DEFAULT 'Ready',
        file_data BYTEA,
        file_path TEXT
      );
    `);

    // ml_models table
    await client.query(`
      CREATE TABLE IF NOT EXISTS ml_models (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        version TEXT,
        type VARCHAR(32),
        status VARCHAR(32) DEFAULT 'Trained',
        accuracy DOUBLE PRECISION,
        last_trained TIMESTAMP,
        metadata JSONB
      );
    `);

    // ml_jobs table to track training jobs
    await client.query(`
      CREATE TABLE IF NOT EXISTS ml_jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        model_name TEXT,
        dataset_id INTEGER,
        status VARCHAR(32) DEFAULT 'queued',
        progress INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT now(),
        updated_at TIMESTAMP DEFAULT now(),
        details JSONB
      );
    `);

    // Indexes
    await client.query('CREATE INDEX IF NOT EXISTS idx_sensor_device_time ON sensor_data(device_id, timestamp DESC);');
    await client.query('CREATE INDEX IF NOT EXISTS idx_devices_device_id ON devices(device_id);');
    await client.query('CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);');
  } finally {
    client.release();
  }
}

async function ensurePostgresInitialized() {
  try {
    await pool.query('SELECT 1');
    await ensureTables();
    logger.info('Postgres connection test succeeded');
  } catch (err) {
    logger.error('Postgres init error', { error: err && err.message ? err.message : err });
    throw err;
  }
}

async function ensureTables() {
  const client = await pool.connect();
  try {
    // datasets table to store ML/LLM/VLM dataset metadata and optional file blobs
    await client.query(`
      CREATE TABLE IF NOT EXISTS datasets (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type VARCHAR(32) DEFAULT 'CSV',
        size_bytes BIGINT DEFAULT 0,
        records INTEGER DEFAULT 0,
        uploaded_at TIMESTAMP DEFAULT now(),
        status VARCHAR(32) DEFAULT 'Ready',
        file_data BYTEA,
        file_path TEXT
      );
    `);

    // models table to store trained model metadata
    await client.query(`
      CREATE TABLE IF NOT EXISTS ml_models (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        version TEXT,
        type VARCHAR(32),
        status VARCHAR(32) DEFAULT 'Trained',
        accuracy DOUBLE PRECISION,
        last_trained TIMESTAMP,
        metadata JSONB
      );
    `);

    await client.query('CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);');
  } finally {
    client.release();
  }
}

module.exports = {
  pool,
  ensurePostgresInitialized
};
