/**
 * Firebase Configuration
 * Update these with your Firebase project credentials
 */

export const firebaseConfig = {
  apiKey: process.env.VITE_FIREBASE_API_KEY || 'YOUR_API_KEY',
  authDomain: process.env.VITE_FIREBASE_AUTH_DOMAIN || 'your-project.firebaseapp.com',
  projectId: process.env.VITE_FIREBASE_PROJECT_ID || 'your-project-id',
  storageBucket: process.env.VITE_FIREBASE_STORAGE_BUCKET || 'your-project.appspot.com',
  messagingSenderId: process.env.VITE_FIREBASE_MESSAGING_SENDER_ID || 'your-sender-id',
  appId: process.env.VITE_FIREBASE_APP_ID || 'your-app-id',
};

/**
 * Local PouchDB Sync Server Configuration
 */
export const pouchdbConfig = {
  localServerUrl: process.env.VITE_POUCHDB_SERVER_URL || 'http://localhost:5984',
  databaseName: 'agrisense',
  syncInterval: 30000, // 30 seconds
};

/**
 * Backend API Configuration
 */
export const backendConfig = {
  apiUrl: process.env.VITE_BACKEND_API_URL || 'http://localhost:8004/api/v1',
  wsUrl: process.env.VITE_BACKEND_WS_URL || 'ws://localhost:8004',
};

/**
 * App Configuration
 */
export const appConfig = {
  enableOfflineMode: process.env.VITE_ENABLE_OFFLINE_MODE !== 'false',
  enablePouchDBSync: process.env.VITE_ENABLE_POUCHDB_SYNC !== 'false',
  logLevel: (process.env.VITE_LOG_LEVEL || 'info') as 'debug' | 'info' | 'warn' | 'error',
};
