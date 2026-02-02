const socketIo = require('socket.io');
const logger = require('../utils/logger');

let io;

/**
 * Initialize Socket.IO server
 */
const initializeSocket = (server) => {
    io = socketIo(server, {
        cors: {
            origin: process.env.FRONTEND_URL || 'http://localhost:3001',
            methods: ['GET', 'POST']
        },
        pingTimeout: 60000,
        pingInterval: 25000
    });

    // Connection handler
    io.on('connection', (socket) => {
        logger.info(`Client connected: ${socket.id}`);

        // Join IoT room
        socket.on('join-iot', (deviceId) => {
            socket.join(`iot-${deviceId}`);
            logger.info(`Client ${socket.id} joined IoT device: ${deviceId}`);

            socket.emit('iot-joined', {
                deviceId,
                message: 'Successfully connected to IoT device stream'
            });
        });

        // Leave IoT room
        socket.on('leave-iot', (deviceId) => {
            socket.leave(`iot-${deviceId}`);
            logger.info(`Client ${socket.id} left IoT device: ${deviceId}`);
        });

        // Disconnect
        socket.on('disconnect', () => {
            logger.info(`Client disconnected: ${socket.id}`);
        });

        // Error handling
        socket.on('error', (error) => {
            logger.error(`Socket error for ${socket.id}:`, error);
        });
    });

    logger.info('✅ Socket.IO initialized');
    return io;
};

/**
 * Broadcast IoT sensor data to connected clients
 */
const broadcastIoTData = (deviceId, sensorData) => {
    if (!io) {
        logger.warn('Socket.IO not initialized');
        return;
    }

    const room = `iot-${deviceId}`;
    const payload = {
        deviceId,
        timestamp: new Date().toISOString(),
        ...sensorData
    };

    io.to(room).emit('iot-data', payload);
    logger.info(`IoT data broadcasted to room: ${room}`);
};

/**
 * Send alert to all connected clients
 */
const sendAlert = (type, message, data = {}) => {
    if (!io) return;

    io.emit('alert', {
        type, // 'info', 'warning', 'error', 'critical'
        message,
        data,
        timestamp: new Date().toISOString()
    });

    logger.info(`Alert sent: ${type} - ${message}`);
};

/**
 * Get current connection count
 */
const getConnectionCount = () => {
    if (!io) return 0;
    return io.sockets.sockets.size;
};

module.exports = {
    initializeSocket,
    broadcastIoTData,
    sendAlert,
    getConnectionCount
};
