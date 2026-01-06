/**
 * Local CouchDB/PouchDB Sync Server
 * Runs natively on your device for local-first database
 *
 * Install dependencies:
 * npm install express cors pouchdb pouchdb-express-router body-parser
 *
 * Run:
 * node pouchdb-server.js
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const PouchDB = require('pouchdb');
const path = require('path');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 5984;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

// Initialize PouchDB instance
const dbPath = path.join(__dirname, '.pouchdb-data');
const db = new PouchDB('agrisense', {
  prefix: dbPath,
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'PouchDB Sync Server' });
});

// Database info endpoint
app.get('/db/info', async (req, res) => {
  try {
    const info = await db.info();
    res.json(info);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get all documents
app.get('/db/docs', async (req, res) => {
  try {
    const result = await db.allDocs({ include_docs: true });
    res.json(result.rows.map((row) => row.doc));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get document by ID
app.get('/db/docs/:id', async (req, res) => {
  try {
    const doc = await db.get(req.params.id);
    res.json(doc);
  } catch (error) {
    if (error.status === 404) {
      res.status(404).json({ error: 'Document not found' });
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Save document
app.post('/db/docs', async (req, res) => {
  try {
    const { _id, ...docData } = req.body;

    let result;
    if (_id) {
      // Update existing
      const existing = await db.get(_id).catch(() => null);
      const doc = {
        ...docData,
        _id,
        ...(existing && { _rev: existing._rev }),
      };
      result = await db.put(doc);
    } else {
      // Create new
      result = await db.post(docData);
    }

    res.status(201).json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update document
app.put('/db/docs/:id', async (req, res) => {
  try {
    const existing = await db.get(req.params.id);
    const doc = {
      ...req.body,
      _id: req.params.id,
      _rev: existing._rev,
    };
    const result = await db.put(doc);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Delete document
app.delete('/db/docs/:id', async (req, res) => {
  try {
    const doc = await db.get(req.params.id);
    const result = await db.remove(doc);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Sync endpoint (for CouchDB replication protocol)
app.all('/agrisense', async (req, res) => {
  try {
    const info = await db.info();
    res.json(info);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Bulk docs endpoint for sync
app.post('/agrisense/_bulk_docs', async (req, res) => {
  try {
    const result = await db.bulkDocs(req.body.docs || []);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Changes feed for sync
app.get('/agrisense/_changes', async (req, res) => {
  try {
    const options = {
      since: req.query.since || 0,
      include_docs: req.query.include_docs === 'true',
      limit: parseInt(req.query.limit) || 100
    };
    const changes = await db.changes(options);
    res.json(changes);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// All docs endpoint
app.get('/agrisense/_all_docs', async (req, res) => {
  try {
    const result = await db.allDocs({ include_docs: req.query.include_docs === 'true' });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: err.message });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 PouchDB Sync Server running on http://localhost:${PORT}`);
  console.log(`📊 Database: agrisense`);
  console.log(`🔄 Sync endpoint: http://localhost:${PORT}/agrisense`);
  console.log(`💾 Data stored in: ${dbPath}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n✓ Server shutting down gracefully...');
  db.close();
  process.exit(0);
});
