const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");
const compression = require("compression");
require("dotenv").config();

// Import logger
const logger = require("./utils/logger");

// Import database
const connectDB = require("./config/database");

// Import ML model controllers
const cropRecommendationController = require("./controllers/cropRecommendationController");
const yieldPredictionController = require("./controllers/yieldPredictionController");
const cropTypeController = require("./controllers/cropTypeController");
const waterRequirementController = require("./controllers/waterRequirementController");
const seasonClassificationController = require("./controllers/seasonClassificationController");

// Import IoT routes
const iotRoutes = require("./routes/iotRoutes");
const mlRoutes = require("./routes/mlRoutes");

// Import AI controllers
const llmController = require("./controllers/llmController");
const vlmController = require("./controllers/vlmController");
const environmentalController = require("./controllers/environmentalController");

// Import middleware
const { apiLimiter, mlLimiter } = require("./middleware/rateLimiter");

// Import health routes
const healthRoutes = require("./routes/health");

const app = express();
const PORT = process.env.PORT || 5000;

// Connect to database
connectDB();

// Middleware
app.use(helmet());
app.use(compression());

// Custom Morgan logger using Winston
app.use(
  morgan("combined", {
    stream: { write: (message) => logger.info(message.trim()) },
  }),
);

// CORS configuration - allow multiple origins for development
const allowedOrigins = [
  "http://localhost:3001",
  "http://127.0.0.1:3001",
  "http://192.168.1.26:3001", // Your network IP
  process.env.FRONTEND_URL,
].filter(Boolean);

app.use(
  cors({
    origin: function (origin, callback) {
      // Allow requests with no origin (like mobile apps or curl requests)
      if (!origin) return callback(null, true);

      if (allowedOrigins.indexOf(origin) !== -1) {
        callback(null, true);
      } else {
        // In development, allow all origins
        if (process.env.NODE_ENV === "development" || !process.env.NODE_ENV) {
          callback(null, true);
        } else {
          callback(new Error("Not allowed by CORS"));
        }
      }
    },
    credentials: true,
  }),
);
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Serve uploaded files
app.use("/uploads", express.static("uploads"));

// Serve frontend static files in production
const path = require("path");
if (process.env.NODE_ENV === "production") {
  const frontendBuildPath = path.join(__dirname, "../frontend/dist");
  app.use(express.static(frontendBuildPath));
  logger.info(`📦 Serving frontend from: ${frontendBuildPath}`);
}

// Apply rate limiting to all API routes
app.use("/api/", apiLimiter);

// Health check routes
app.use("/api/health", healthRoutes);

// Swagger API Documentation (wrapped in try-catch to prevent crashes)
try {
  const { swaggerUi, swaggerSpec } = require("./swagger/config");
  app.use(
    "/api/docs",
    swaggerUi.serve,
    swaggerUi.setup(swaggerSpec, {
      customCss: ".swagger-ui .topbar { display: none }",
      customSiteTitle: "AgriSense API Docs",
    }),
  );

  // Swagger JSON endpoint
  app.get("/api/docs.json", (req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.send(swaggerSpec);
  });
  logger.info("✅ Swagger API documentation loaded", {
    service: "agrisense-backend",
  });
} catch (error) {
  logger.warn("⚠️  Swagger documentation not available", {
    service: "agrisense-backend",
    error: error.message,
  });
}

// Legacy health endpoint
app.get("/health", (req, res) => {
  res.status(200).json({
    status: "OK",
    message: "AgriSense Backend API is running",
    timestamp: new Date().toISOString(),
    version: "1.0.0",
  });
});

// ML Model API Routes - with ML rate limiter
app.use("/api/crop-recommendation", mlLimiter, cropRecommendationController);
app.use("/api/yield-prediction", mlLimiter, yieldPredictionController);
app.use("/api/crop-type", mlLimiter, cropTypeController);
app.use("/api/water-requirement", mlLimiter, waterRequirementController);
app.use(
  "/api/season-classification",
  mlLimiter,
  seasonClassificationController,
);
app.use("/api/ml", mlLimiter, mlRoutes);

// IoT Routes
app.use("/api/iot", iotRoutes);

// Admin Routes
const adminRoutes = require("./routes/adminRoutes");
app.use("/api/admin", adminRoutes);

// AI Model API Routes - with ML rate limiter
app.use("/api/llm", mlLimiter, llmController);
app.use("/api/vlm", mlLimiter, vlmController);

// Environmental Disease Risk Route
const { body } = require("express-validator");
app.post(
  "/api/environmental-assessment",
  mlLimiter,
  [
    body("temperature").isFloat(),
    body("humidity").isFloat(),
    body("rainfall").isFloat(),
    body("ph").isFloat(),
  ],
  environmentalController.predictDiseaseRisk
);

// Model information endpoint
app.get("/api/models", (req, res) => {
  res.json({
    models: {
      // ML Models
      crop_recommendation: {
        accuracy: 0.9245,
        f1_score: 0.9247,
        type: "classification",
        description: "Enhanced ensemble model for crop recommendation",
      },
      yield_prediction: {
        r2_score: 0.9004,
        rmse: 1.887,
        type: "regression",
        description: "Enhanced ensemble model for yield prediction",
      },
      crop_type_classification: {
        accuracy: 0.55,
        f1_score: 0.4633,
        type: "classification",
        description: "Enhanced model for crop type classification",
      },
      water_requirement: {
        r2_score: 0.2793,
        rmse: 2.3104,
        type: "regression",
        description: "Model for water requirement prediction",
      },
      season_classification: {
        accuracy: 0.8,
        f1_score: 0.8,
        type: "classification",
        description: "Model for growing season classification",
      },
      // AI Models
      llm_agricultural_advisor: {
        type: "llm",
        provider: "OpenAI",
        model: "gpt-3.5-turbo",
        description:
          "Large Language Model for agricultural advice and recommendations",
      },
      vlm_plant_analysis: {
        type: "vlm",
        provider: "OpenAI",
        model: "gpt-4-vision-preview",
        description:
          "Vision Language Model for plant disease detection and health assessment",
      },
    },
    last_updated: "2026-01-20T18:00:00.000Z",
    total_models: 7,
    ml_models: 5,
    ai_models: 2,
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: "Something went wrong!",
    message:
      process.env.NODE_ENV === "development"
        ? err.message
        : "Internal server error",
  });
});

// Serve frontend index.html for all non-API routes (client-side routing)
if (process.env.NODE_ENV === 'production') {
  app.get('*', (req, res) => {
    const frontendBuildPath = path.join(__dirname, '../frontend/dist');
    res.sendFile(path.join(frontendBuildPath, 'index.html'));
  });
} else {
  // 404 handler for development mode
  app.use('*', (req, res) => {
    res.status(404).json({
      error: 'Route not found',
      message: `Cannot ${req.method} ${req.originalUrl}`
    });
  });
}

// Start server with Socket.IO support
const http = require("http");
const httpServer = http.createServer(app);

// Initialize Socket.IO
try {
  const { initializeSocket } = require("./services/socketService");
  const io = initializeSocket(httpServer);
  logger.info("✅ Socket.IO service loaded");
} catch (err) {
  logger.warn("⚠️  Socket.IO service not available:", err.message);
}

const server = httpServer.listen(PORT, () => {
  logger.info(`🚀 AgriSense Backend API running on port ${PORT}`);
  logger.info(`📊 ML Models loaded and ready`);
  logger.info(`🔌 Socket.IO enabled for real-time IoT`);
  logger.info(`🌐 Environment: ${process.env.NODE_ENV || "development"}`);
  logger.info(`📚 API Docs: http://localhost:${PORT}/api/docs`);
});

// Graceful shutdown
const gracefulShutdown = async (signal) => {
  logger.info(`\n${signal} received. Starting graceful shutdown...`);

  // Stop accepting new requests
  server.close(() => {
    logger.info("✅ HTTP server closed");
  });

  // Close database connection
  const mongoose = require("mongoose");
  if (mongoose.connection.readyState !== 0) {
    await mongoose.connection.close();
    logger.info("✅ MongoDB connection closed");
  }

  // Close Redis connection if exists
  try {
    const { client } = require("./config/redis");
    if (client && client.quit) {
      client.quit();
      logger.info("✅ Redis connection closed");
    }
  } catch (err) {
    // Redis not configured, skip
  }

  logger.info("👋 Shutdown complete. Goodbye!");
  process.exit(0);
};

// Handle shutdown signals
process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

// Handle uncaught exceptions
process.on("uncaughtException", (error) => {
  logger.error("💥 Uncaught Exception:", error);
  gracefulShutdown("uncaughtException");
});

// Handle unhandled promise rejections
process.on("unhandledRejection", (reason, promise) => {
  logger.error("💥 Unhandled Rejection at:", promise, "reason:", reason);
  gracefulShutdown("unhandledRejection");
});

module.exports = app;
