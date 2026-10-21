import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '15mb' }));

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8004';

// Real HTTP latency and request telemetry ring buffer
interface ApiMetricSample {
  id: string;
  method: string;
  route: string;
  statusCode: number;
  latencyMs: number;
  timestamp: string;
}

const apiMetricsBuffer: ApiMetricSample[] = [];
let totalApiRequests = 0;
let totalApiErrors = 0;

// Reverse proxy forwarding helper to Python FastAPI Backend
async function forwardToPython(req: express.Request, res: express.Response) {
  const startTime = Date.now();
  totalApiRequests++;
  
  const targetUrl = `${PYTHON_BACKEND_URL}${req.originalUrl}`;
  try {
    const fetchOptions: RequestInit = {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    };

    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOptions.body = JSON.stringify(req.body);
    }

    const response = await fetch(targetUrl, fetchOptions);
    const latency = Date.now() - startTime;
    const contentType = response.headers.get('content-type') || '';

    apiMetricsBuffer.unshift({
      id: `req-${Math.random().toString(16).slice(2, 10)}`,
      method: req.method,
      route: req.path,
      statusCode: response.status,
      latencyMs: latency,
      timestamp: new Date().toISOString(),
    });
    if (apiMetricsBuffer.length > 100) apiMetricsBuffer.pop();

    if (contentType.includes('application/json')) {
      const data = await response.json();
      return res.status(response.status).json(data);
    } else {
      const text = await response.text();
      return res.status(response.status).send(text);
    }
  } catch (error: any) {
    totalApiErrors++;
    const latency = Date.now() - startTime;
    console.error(`[Gateway Proxy Error] ${req.method} ${req.originalUrl} -> ${error?.message}`);
    
    return res.status(503).json({
      error: `AgriSense Core Engine unreachable at ${PYTHON_BACKEND_URL}. Ensure FastAPI is running on port 8004.`,
      details: error?.message,
      gatewayLatencyMs: latency,
    });
  }
}

// All /api/* routes forwarded to FastAPI backend
app.use('/api', (req, res) => {
  forwardToPython(req, res);
});

async function startServer() {
  const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

  if (process.env.NODE_ENV === 'production') {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`================================================================`);
    console.log(`  AGRISENSE OFFLINE GATEWAY & PLATFORM`);
    console.log(`  Frontend: http://localhost:${PORT}`);
    console.log(`  FastAPI Target: ${PYTHON_BACKEND_URL}`);
    console.log(`================================================================`);
  });
}

startServer();
