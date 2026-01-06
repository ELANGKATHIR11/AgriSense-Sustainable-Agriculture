import { StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { i18nPromise } from './i18n' // Initialize i18n before rendering

// Debug: Log that we're starting
console.log('main.tsx: Starting initialization...');

// Wait for i18n to be ready before rendering
const renderApp = () => {
  console.log('main.tsx: Rendering app...');
  const root = createRoot(document.getElementById("root")!);
  root.render(
    <StrictMode>
      <Suspense fallback={<div className="flex items-center justify-center h-screen">Loading...</div>}>
        <App />
      </Suspense>
    </StrictMode>
  );
  console.log('main.tsx: App rendered successfully');
};

i18nPromise.then(() => {
  console.log('main.tsx: i18n initialized successfully');
  renderApp();
}).catch((error) => {
  console.error('main.tsx: Failed to initialize i18n:', error);
  // Render app anyway with fallback language
  renderApp();
});

// Service worker temporarily disabled to fix caching issues
// if ('serviceWorker' in navigator && import.meta.env.PROD) {
// 	window.addEventListener('load', () => {
// 		const swUrl = `${import.meta.env.BASE_URL}sw.js`;
// 		navigator.serviceWorker.register(swUrl).catch(() => {
// 			// ignore
// 		});
// 	});
// }
