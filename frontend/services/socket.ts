import { io, Socket } from 'socket.io-client';

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || 'http://localhost:5000';

class SocketService {
    private socket: Socket | null = null;
    private listeners: Map<string, Set<Function>> = new Map();

    connect(): void {
        if (this.socket?.connected) {
            console.log('Socket already connected');
            return;
        }

        this.socket = io(SOCKET_URL, {
            transports: ['websocket', 'polling'],
            reconnectionAttempts: 5,
            reconnectionDelay: 1000,
        });

        this.socket.on('connect', () => {
            console.log('✅ Socket.IO connected');
        });

        this.socket.on('disconnect', (reason) => {
            console.log('❌ Socket.IO disconnected:', reason);
        });

        this.socket.on('connect_error', (error) => {
            console.error('Socket.IO connection error:', error);
        });

        // Listen for sensor updates from backend
        this.socket.on('sensorUpdate', (data) => {
            console.log('Sensor update received:', data);
            this.notifyListeners('sensorUpdate', data);
        });

        // Listen for IoT data updates
        this.socket.on('iotData', (data) => {
            console.log('IoT data received:', data);
            this.notifyListeners('iotData', data);
        });
    }

    disconnect(): void {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            console.log('Socket.IO disconnected');
        }
    }

    on(event: string, callback: Function): void {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)?.add(callback);
    }

    off(event: string, callback: Function): void {
        this.listeners.get(event)?.delete(callback);
    }

    private notifyListeners(event: string, data: any): void {
        this.listeners.get(event)?.forEach((callback) => {
            try {
                callback(data);
            } catch (error) {
                console.error('Error in socket listener:', error);
            }
        });
    }

    emit(event: string, data: any): void {
        if (this.socket?.connected) {
            this.socket.emit(event, data);
        } else {
            console.warn('Socket not connected, cannot emit event:', event);
        }
    }

    isConnected(): boolean {
        return this.socket?.connected || false;
    }
}

// Export singleton instance
const socketService = new SocketService();
export default socketService;
