import React, { useState, useEffect } from 'react';
import { Wifi, WifiOff, X, RefreshCw, Download } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { isOnline, onNetworkChange, checkForUpdates, activateUpdate, isPWAInstalled } from '../services/pwa';

// Offline Indicator Banner
export const OfflineBanner: React.FC = () => {
  const [online, setOnline] = useState(isOnline());

  useEffect(() => {
    const unsubscribe = onNetworkChange(setOnline);
    return unsubscribe;
  }, []);

  if (online) return null;

  return (
    <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-80 bg-amber-500 text-white p-4 rounded-xl shadow-lg z-50 animate-slide-up">
      <div className="flex items-center gap-3">
        <WifiOff className="w-5 h-5 flex-shrink-0" />
        <div className="flex-1">
          <p className="font-medium">You're offline</p>
          <p className="text-sm opacity-90">Some features may be unavailable</p>
        </div>
      </div>
    </div>
  );
};

// Update Available Banner
export const UpdateBanner: React.FC = () => {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);

  useEffect(() => {
    const handleUpdate = () => setUpdateAvailable(true);
    window.addEventListener('swUpdate', handleUpdate);
    return () => window.removeEventListener('swUpdate', handleUpdate);
  }, []);

  const handleApplyUpdate = async () => {
    setIsUpdating(true);
    await activateUpdate();
  };

  if (!updateAvailable) return null;

  return (
    <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 bg-agri-600 text-white p-4 rounded-xl shadow-lg z-50 animate-slide-up">
      <div className="flex items-start gap-3">
        <RefreshCw className="w-5 h-5 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="font-medium">Update Available</p>
          <p className="text-sm opacity-90 mb-3">A new version is ready to install</p>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={handleApplyUpdate}
              isLoading={isUpdating}
              className="bg-white text-agri-700 hover:bg-gray-100"
            >
              Update Now
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setUpdateAvailable(false)}
              className="text-white hover:bg-white/20"
            >
              Later
            </Button>
          </div>
        </div>
        <button onClick={() => setUpdateAvailable(false)} className="hover:opacity-70" aria-label="Close update banner">
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

// Install PWA Prompt
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

export const InstallPrompt: React.FC = () => {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    // Don't show if already installed
    if (isPWAInstalled()) return;

    const handleBeforeInstall = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
      setShowPrompt(true);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstall);
    return () => window.removeEventListener('beforeinstallprompt', handleBeforeInstall);
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;

    await deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    
    if (outcome === 'accepted') {
      console.log('[PWA] User accepted install');
    }
    
    setDeferredPrompt(null);
    setShowPrompt(false);
  };

  if (!showPrompt) return null;

  return (
    <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 bg-gradient-to-r from-agri-600 to-water-600 text-white p-4 rounded-xl shadow-lg z-50 animate-slide-up">
      <div className="flex items-start gap-3">
        <Download className="w-5 h-5 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="font-medium">Install AgriSense</p>
          <p className="text-sm opacity-90 mb-3">Add to home screen for quick access</p>
          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={handleInstall}
              className="bg-white text-agri-700 hover:bg-gray-100"
            >
              Install
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setShowPrompt(false)}
              className="text-white hover:bg-white/20"
            >
              Not Now
            </Button>
          </div>
        </div>
        <button onClick={() => setShowPrompt(false)} className="hover:opacity-70" aria-label="Close install prompt">
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

// Network Status Indicator (for header/sidebar)
export const NetworkStatus: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  const [online, setOnline] = useState(isOnline());

  useEffect(() => {
    const unsubscribe = onNetworkChange(setOnline);
    return unsubscribe;
  }, []);

  if (compact) {
    return (
      <div className={`w-2 h-2 rounded-full ${online ? 'bg-green-500' : 'bg-red-500'}`} />
    );
  }

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
      online 
        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' 
        : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
    }`}>
      {online ? <Wifi size={14} /> : <WifiOff size={14} />}
      <span>{online ? 'Online' : 'Offline'}</span>
    </div>
  );
};
