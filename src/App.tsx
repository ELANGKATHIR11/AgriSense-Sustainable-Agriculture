import React, { startTransition, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  LayoutDashboard,
  Cpu,
  Droplets,
  Microscope,
  LineChart,
  Bot,
  Sliders,
  Sprout,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  RefreshCw,
  Bell,
  Globe,
  AlertTriangle,
} from 'lucide-react';
import {
  SUPPORTED_LANGUAGES,
  SupportedLanguageCode,
  setAppLanguage,
} from './i18n';
import {
  SensorsResponse,
  SystemStatusResponse,
  agriApi,
} from './api/client';
import { DashboardPage } from './pages/DashboardPage';
import { IotMonitoringPage } from './pages/IotMonitoringPage';
import { IrrigationPage } from './pages/IrrigationPage';
import { DiseaseDetectionPage } from './pages/DiseaseDetectionPage';
import { AnalyticsTwinPage } from './pages/AnalyticsTwinPage';
import { AssistantSettingsPage } from './pages/AssistantSettingsPage';
import { OperationsCenterPage } from './pages/OperationsCenterPage';
import { formatDateTime } from './utils/format';

export type PageRoute =
  | 'dashboard'
  | 'iot'
  | 'irrigation'
  | 'disease'
  | 'analytics'
  | 'assistant'
  | 'operations';

const VALID_ROUTES: PageRoute[] = [
  'dashboard',
  'iot',
  'irrigation',
  'disease',
  'analytics',
  'assistant',
  'operations',
];

function getRouteFromLocation(): PageRoute {
  if (typeof window === 'undefined') return 'dashboard';
  const cleanPath = window.location.pathname.replace(/^\/+|\/+$/g, '').toLowerCase();
  if ((VALID_ROUTES as string[]).includes(cleanPath)) {
    return cleanPath as PageRoute;
  }
  return 'dashboard';
}

export default function App() {
  const { t, i18n } = useTranslation(['common', 'dashboard']);
  const currentLang = (i18n.language || 'en') as SupportedLanguageCode;

  const [activePage, setActivePage] = useState<PageRoute>(getRouteFromLocation);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [alertsOpen, setAlertsOpen] = useState(false);

  const [darkMode, setDarkMode] = useState<boolean>(() => {
    try {
      return localStorage.getItem('agrisense_theme') === 'dark';
    } catch {
      return false;
    }
  });

  const [sensorsData, setSensorsData] = useState<SensorsResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    try {
      localStorage.setItem('agrisense_theme', darkMode ? 'dark' : 'light');
    } catch {
      // ignore storage errors
    }
  }, [darkMode]);

  const navigateTo = useCallback((page: PageRoute) => {
    setMobileMenuOpen(false);
    setAlertsOpen(false);
    const targetPath = page === 'dashboard' ? '/dashboard' : `/${page}`;
    if (window.location.pathname !== targetPath) {
      window.history.pushState({ page }, '', targetPath);
    }
    startTransition(() => {
      setActivePage(page);
    });
  }, []);

  useEffect(() => {
    const onPopState = () => {
      startTransition(() => {
        setActivePage(getRouteFromLocation());
      });
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const fetchTelemetry = useCallback(async (isBackground = false) => {
    if (!isBackground) setSyncing(true);
    try {
      const [sensors, status] = await Promise.all([
        agriApi.getSensors(),
        agriApi.getStatus(),
      ]);
      setSensorsData(sensors);
      setSystemStatus(status);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Telemetry sync failed');
    } finally {
      setLoading(false);
      setSyncing(false);
    }
  }, []);

  const handleManualRefresh = useCallback(() => {
    return fetchTelemetry(false);
  }, [fetchTelemetry]);

  useEffect(() => {
    fetchTelemetry(false);
    const timer = setInterval(() => {
      if (!document.hidden) {
        fetchTelemetry(true);
      }
    }, 4000);
    return () => clearInterval(timer);
  }, [fetchTelemetry]);

  const navItems: {
    id: PageRoute;
    labelKey: string;
    icon: React.ComponentType<{ className?: string }>;
    badge?: string;
  }[] = [
    { id: 'dashboard', labelKey: 'common:nav.dashboard', icon: LayoutDashboard, badge: '3D' },
    { id: 'iot', labelKey: 'common:nav.iot', icon: Cpu, badge: '3D' },
    { id: 'irrigation', labelKey: 'common:nav.irrigation', icon: Droplets, badge: '3D' },
    { id: 'disease', labelKey: 'common:nav.disease', icon: Microscope },
    { id: 'analytics', labelKey: 'common:nav.analytics', icon: LineChart, badge: '3D' },
    { id: 'assistant', labelKey: 'common:nav.assistant', icon: Bot },
    { id: 'operations', labelKey: 'common:nav.operations', icon: Sliders, badge: 'OPS' },
  ];

  const recentEvents = sensorsData?.irrigation.recentEvents ?? [];

  return (
    <div className="min-h-screen flex bg-[var(--agri-bg)] text-[var(--agri-text)]">
      {/* Desktop Sidebar */}
      <aside
        className={`hidden md:flex flex-col justify-between bg-[#142F22] text-white shrink-0 transition-all duration-150 border-r border-white/10 ${
          sidebarCollapsed ? 'w-20' : 'w-68'
        }`}
      >
        <div>
          {/* Brand Header */}
          <div className="h-16 px-5 flex items-center justify-between border-b border-white/10">
            <button
              type="button"
              onClick={() => navigateTo('dashboard')}
              className="flex items-center gap-3 text-left cursor-pointer overflow-hidden"
            >
              <div className="w-9 h-9 rounded-lg bg-[#4A8C55] flex items-center justify-center shrink-0">
                <Sprout className="w-5 h-5 text-white" />
              </div>
              {!sidebarCollapsed && (
                <div className="min-w-0">
                  <div className="text-base font-bold tracking-tight text-white whitespace-nowrap">
                    {t('common:brand.name')}
                  </div>
                  <div className="text-[10px] font-mono-tabular text-[#DCE8D5]/70 truncate">
                    {t('common:brand.subtitle')}
                  </div>
                </div>
              )}
            </button>
          </div>

          {/* 7 Main Page Navigation Links */}
          <nav className="p-3 space-y-1.5" aria-label="Main Navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => navigateTo(item.id)}
                  title={sidebarCollapsed ? t(item.labelKey) : undefined}
                  className={`w-full flex items-center justify-between gap-2 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                    isActive
                      ? 'bg-[#4A8C55] text-white font-semibold'
                      : 'text-[#DCE8D5]/85 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <Icon className="w-4 h-4 shrink-0" />
                    {!sidebarCollapsed && (
                      <span className="truncate">{t(item.labelKey)}</span>
                    )}
                  </div>
                  {!sidebarCollapsed && item.badge && (
                    <span
                      className={`text-[9px] font-mono-tabular px-1.5 py-0.5 rounded shrink-0 ${
                        isActive
                          ? 'bg-black/25 text-white'
                          : 'bg-white/10 text-[#DCE8D5]/80'
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Collapse Sidebar Toggle */}
        <div className="p-3 border-t border-white/10">
          <button
            type="button"
            onClick={() => setSidebarCollapsed((c) => !c)}
            aria-label={
              sidebarCollapsed ? t('common:nav.expand') : t('common:nav.collapse')
            }
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs text-[#DCE8D5]/80 hover:bg-white/10 hover:text-white transition-colors cursor-pointer"
          >
            {sidebarCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <>
                <ChevronLeft className="w-4 h-4" />
                <span className="truncate">{t('common:nav.collapse')}</span>
              </>
            )}
          </button>
        </div>
      </aside>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex">
          <div
            className="fixed inset-0 bg-black/50"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="relative z-10 w-72 max-w-[85vw] bg-[#142F22] text-white flex flex-col justify-between p-4">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-white/10">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-[#4A8C55] flex items-center justify-center">
                    <Sprout className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-base font-bold">{t('common:brand.name')}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setMobileMenuOpen(false)}
                  aria-label={t('common:actions.close')}
                  className="p-1 text-white/80 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <nav className="mt-4 space-y-1.5">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = activePage === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => navigateTo(item.id)}
                      className={`w-full flex items-center justify-between gap-2 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-colors ${
                        isActive
                          ? 'bg-[#4A8C55] text-white font-semibold'
                          : 'text-[#DCE8D5]/85 hover:bg-white/10'
                      }`}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <Icon className="w-4 h-4 shrink-0" />
                        <span className="truncate">{t(item.labelKey)}</span>
                      </div>
                      {item.badge && (
                        <span className="text-[9px] font-mono-tabular px-1.5 py-0.5 rounded bg-white/10 text-[#DCE8D5]">
                          {item.badge}
                        </span>
                      )}
                    </button>
                  );
                })}
              </nav>
            </div>

            <div className="pt-4 border-t border-white/10 text-[11px] text-[#DCE8D5]/70">
              {t('common:topbar.operatorProfile')}
            </div>
          </aside>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="h-16 bg-[var(--agri-card)] border-b border-[var(--agri-border)] px-4 lg:px-8 flex items-center justify-between gap-4 sticky top-0 z-30">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileMenuOpen(true)}
              aria-label={t('common:nav.mobileMenu')}
              className="md:hidden p-2 rounded-lg text-[var(--agri-text)] hover:bg-[var(--agri-bg)]"
            >
              <Menu className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-2 text-sm min-w-0">
              <span className="text-[var(--agri-muted)] hidden sm:inline whitespace-nowrap">
                {t('common:brand.name')}
              </span>
              <span className="text-[var(--agri-muted)] hidden sm:inline">/</span>
              <span className="font-semibold text-[var(--agri-text)] truncate">
                {t(`common:nav.${activePage}`)}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <button
              type="button"
              onClick={() => navigateTo('operations')}
              className="hidden lg:flex items-center gap-1.5 text-xs text-[var(--agri-muted)] hover:text-[var(--agri-text)] cursor-pointer"
              title={t('common:nav.operations')}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  error ? 'bg-[var(--agri-red)]' : 'bg-[var(--agri-leaf)]'
                }`}
              />
              <span className="whitespace-nowrap">
                {error
                  ? t('common:topbar.backendDegraded')
                  : t('common:topbar.backendConnected')}
              </span>
            </button>

            {/* 7-Language Native Selector */}
            <div className="relative flex items-center">
              <Globe className="w-3.5 h-3.5 text-[var(--agri-muted)] absolute left-2.5 pointer-events-none" />
              <select
                aria-label={t('common:topbar.language')}
                value={currentLang}
                onChange={(e) => {
                  const nextLang = e.target.value as SupportedLanguageCode;
                  startTransition(() => {
                    setAppLanguage(nextLang);
                  });
                }}
                className="pl-8 pr-3 py-1.5 text-xs font-medium bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)] focus:outline-none focus:border-[var(--agri-leaf)] cursor-pointer"
              >
                {SUPPORTED_LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.nativeLabel} ({l.code.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>

            {/* Notifications Popover */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setAlertsOpen((o) => !o)}
                aria-label={t('common:topbar.notifications')}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)] cursor-pointer"
              >
                <Bell className="w-3.5 h-3.5 text-[var(--agri-amber)]" />
                <span className="font-mono-tabular">{recentEvents.length}</span>
              </button>

              {alertsOpen && (
                <div className="absolute right-0 mt-2 w-80 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl shadow-lg p-4 z-50">
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-[var(--agri-border)]">
                    <span className="text-xs font-semibold text-[var(--agri-text)]">
                      {t('common:topbar.notifications')}
                    </span>
                    <button
                      type="button"
                      onClick={() => setAlertsOpen(false)}
                      className="text-xs text-[var(--agri-muted)] hover:text-[var(--agri-text)] cursor-pointer"
                    >
                      {t('common:actions.close')}
                    </button>
                  </div>
                  {recentEvents.length === 0 ? (
                    <p className="text-xs text-[var(--agri-muted)] py-3">
                      {t('common:topbar.noAlerts')}
                    </p>
                  ) : (
                    <div className="divide-y divide-[var(--agri-border)] max-h-64 overflow-y-auto">
                      {recentEvents.map((ev) => (
                        <div key={ev.id} className="py-2.5 first:pt-0 last:pb-0 text-xs">
                          <div className="font-medium text-[var(--agri-text)]">
                            {t(`dashboard:alerts.${ev.code}`, { defaultValue: ev.code })}
                          </div>
                          <div className="text-[11px] text-[var(--agri-muted)] font-mono-tabular mt-0.5">
                            {ev.zoneId} · {ev.value} ·{' '}
                            {formatDateTime(ev.timestamp, currentLang)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Sync Telemetry Action */}
            <button
              type="button"
              disabled={syncing}
              onClick={handleManualRefresh}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${syncing ? 'animate-spin' : ''}`} />
              <span className="hidden sm:inline">
                {syncing ? t('common:topbar.syncing') : t('common:topbar.refreshData')}
              </span>
            </button>
          </div>
        </header>

        {/* Main Viewport Content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          {error && (
            <div className="mb-6 bg-[var(--agri-card)] border border-[var(--agri-red)] rounded-xl p-4 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 text-xs text-[var(--agri-red)]">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>
                  {t('common:states.errorTitle')}: {error}
                </span>
              </div>
              <button
                type="button"
                onClick={handleManualRefresh}
                className="px-3 py-1.5 text-xs font-semibold text-white bg-[var(--agri-red)] rounded-lg cursor-pointer whitespace-nowrap shrink-0"
              >
                {t('common:actions.retry')}
              </button>
            </div>
          )}

          {activePage === 'dashboard' && (
            <DashboardPage
              data={sensorsData}
              loading={loading}
              onNavigate={navigateTo}
            />
          )}

          {activePage === 'iot' && (
            <IotMonitoringPage
              data={sensorsData}
              loading={loading}
              onRefresh={handleManualRefresh}
            />
          )}

          {activePage === 'irrigation' && (
            <IrrigationPage
              data={sensorsData}
              onRefresh={handleManualRefresh}
            />
          )}

          {activePage === 'disease' && <DiseaseDetectionPage />}

          {activePage === 'analytics' && (
            <AnalyticsTwinPage sensorsData={sensorsData} />
          )}

          {activePage === 'assistant' && (
            <AssistantSettingsPage
              systemStatus={systemStatus}
              darkMode={darkMode}
              onToggleDarkMode={setDarkMode}
            />
          )}

          {activePage === 'operations' && (
            <OperationsCenterPage
              devices={sensorsData?.devices ?? []}
              darkMode={darkMode}
              onToggleDarkMode={setDarkMode}
              onRefreshTelemetry={handleManualRefresh}
            />
          )}
        </main>
      </div>
    </div>
  );
}
