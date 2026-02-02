const express = require('express');
const router = express.Router();
const dbType = (process.env.DB_TYPE || 'mongodb').toLowerCase();
let mongoose = null;
if (dbType !== 'postgres') {
  mongoose = require('mongoose');
}
const Device = require('../models/Device');
const SensorData = require('../models/SensorData');
let { pool } = {};
if (dbType === 'postgres') {
  try {
    ({ pool } = require('../config/postgres'));
  } catch (e) {
    // ignore
  }
}

// Mock Data for Fallback
const mockDevices = [
  { id: 'device-001', name: 'Field A Environmental', type: 'environmental', location: 'Field A', status: 'active' },
  { id: 'device-002', name: 'Field B Soil', type: 'soil', location: 'Field B', status: 'active' }
];
const mockSensorData = {};

const isDbConnected = () => {
  if (dbType === 'postgres') {
    // Assume initialized during app startup; presence of pool indicates configured Postgres
    return !!pool;
  }
  return mongoose && mongoose.connection && mongoose.connection.readyState === 1;
};

// Normalize sensor payload for frontend (SensorData shape)
const sensorPayload = (o) => ({
  success: true,
  temperature: Number(o.temperature) || 24,
  humidity: Number(o.humidity) || 60,
  soilMoisture: Number(o.soilMoisture) || 45,
  soilTemperature: Number(o.soilTemperature) || 22,
  phLevel: Number(o.phLevel) || 6.5,
  nitrogen: Number(o.nitrogen) || 40,
  phosphorus: Number(o.phosphorus) || 20,
  potassium: Number(o.potassium) || 30,
  lightIntensity: Number(o.lightIntensity) || 800,
  timestamp: (o.timestamp instanceof Date ? o.timestamp : new Date()).toISOString(),
  ...(o.isMock && { isMock: true })
});

// GET latest sensor data (aggregated)
router.get('/sensors/latest', async (req, res) => {
  try {
    let aggregated = {
      temperature: 0, humidity: 0, soilMoisture: 0, soilTemperature: 22, phLevel: 6.5,
      nitrogen: 40, phosphorus: 20, potassium: 30, lightIntensity: 800, timestamp: new Date()
    };
    let count = 0;

    if (isDbConnected()) {
      const devices = await Device.find({ status: 'active' });
      for (const device of devices) {
        const latestInfo = await SensorData.findOne({ deviceId: device.deviceId }).sort({ timestamp: -1 });
        if (latestInfo) {
          if (latestInfo.temperature != null) aggregated.temperature += latestInfo.temperature;
          if (latestInfo.humidity != null) aggregated.humidity += latestInfo.humidity;
          if (latestInfo.soilMoisture != null) aggregated.soilMoisture += latestInfo.soilMoisture;
          if (latestInfo.soilTemperature != null) aggregated.soilTemperature = latestInfo.soilTemperature;
          if (latestInfo.phLevel != null) aggregated.phLevel = latestInfo.phLevel;
          if (latestInfo.nitrogen != null) aggregated.nitrogen = latestInfo.nitrogen;
          if (latestInfo.phosphorus != null) aggregated.phosphorus = latestInfo.phosphorus;
          if (latestInfo.potassium != null) aggregated.potassium = latestInfo.potassium;
          if (latestInfo.lightIntensity != null) aggregated.lightIntensity = latestInfo.lightIntensity;
          count++;
        }
      }
    } else {
      Object.values(mockSensorData).forEach(readings => {
        if (readings.length > 0) {
          const latest = readings[readings.length - 1];
          aggregated.temperature += latest.temperature || 0;
          aggregated.humidity += latest.humidity || 0;
          count++;
        }
      });
    }

    if (count > 0) {
      aggregated.temperature = aggregated.temperature / count;
      aggregated.humidity = aggregated.humidity / count;
      aggregated.soilMoisture = aggregated.soilMoisture / count;
    } else {
      return res.json(sensorPayload({
        temperature: 24, humidity: 60, soilMoisture: 45, soilTemperature: 22, phLevel: 6.5,
        nitrogen: 40, phosphorus: 20, potassium: 30, lightIntensity: 800, timestamp: new Date(), isMock: true
      }));
    }

    return res.json(sensorPayload(aggregated));
  } catch (error) {
    console.error('Sensor error:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

// GET all devices
router.get('/devices', async (req, res) => {
  if (isDbConnected()) {
    try {
      const devices = await Device.find();
      return res.json({ success: true, data: devices });
    } catch (e) { return res.status(500).json({ error: e.message }); }
  } else {
    return res.json({ success: true, data: mockDevices, source: 'memory-fallback' });
  }
});

// Water tank status (irrigation) - in-memory state
let waterTankState = { level: 78, pumpStatus: 'OFF', lastUpdated: new Date().toISOString() };

router.get('/water-tank/status', (req, res) => {
  res.json({
    tankId: 'tank-001',
    level: waterTankState.level,
    capacity: 10000,
    currentVolume: Math.round(waterTankState.level * 100),
    lastUpdated: waterTankState.lastUpdated,
    pumpStatus: waterTankState.pumpStatus,
    alerts: []
  });
});

router.post('/water-tank/pump', (req, res) => {
  const action = (req.body?.action || 'OFF').toUpperCase();
  waterTankState.pumpStatus = action === 'ON' ? 'ON' : 'OFF';
  waterTankState.lastUpdated = new Date().toISOString();
  if (action === 'ON') waterTankState.level = Math.max(0, waterTankState.level - 0.5);
  res.json({ success: true, action: waterTankState.pumpStatus, timestamp: waterTankState.lastUpdated });
});

router.get('/water-tank/history', (req, res) => {
  const history = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    history.push({ date: d.toISOString().split('T')[0], usage: Math.floor(500 + Math.random() * 500) });
  }
  res.json(history);
});

module.exports = router;
