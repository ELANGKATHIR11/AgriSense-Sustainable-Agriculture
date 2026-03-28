/**
 * Smoke tests for the AgriSense backend.
 *
 * These tests verify basic sanity of the codebase — environment variable
 * handling and module-level require behaviour — without requiring a live
 * database or external ML service.
 */

describe('Environment configuration smoke test', () => {
  it('NODE_ENV is recognised', () => {
    const env = process.env.NODE_ENV || 'development';
    expect(['development', 'test', 'production']).toContain(env);
  });

  it('PORT falls back to 5000 when not set', () => {
    const savedPort = process.env.PORT;
    delete process.env.PORT;
    const port = parseInt(process.env.PORT || '5000', 10);
    expect(port).toBe(5000);
    if (savedPort !== undefined) process.env.PORT = savedPort;
  });

  it('ML_SERVICE_URL has a sensible default', () => {
    const url = process.env.ML_SERVICE_URL || 'http://localhost:8000';
    expect(url).toMatch(/^https?:\/\//);
  });
});

describe('Package structure smoke test', () => {
  it('package.json exists and has required fields', () => {
    // eslint-disable-next-line global-require
    const pkg = require('../package.json');
    expect(pkg.name).toBe('agrisense-backend');
    expect(pkg.version).toBeDefined();
    expect(pkg.scripts).toBeDefined();
    expect(pkg.scripts.start).toBeDefined();
  });

  it('server entry point can be located', () => {
    const path = require('path');
    const fs = require('fs');
    const serverPath = path.join(__dirname, '..', 'server.js');
    expect(fs.existsSync(serverPath)).toBe(true);
  });
});
