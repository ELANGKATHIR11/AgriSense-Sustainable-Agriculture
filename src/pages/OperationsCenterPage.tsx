import React, { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Activity,
  Bot,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  GitBranch,
  Layers,
  Moon,
  Network,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sliders,
  Sun,
  Terminal,
  Wrench,
} from 'lucide-react';
import {
  DependencyNode,
  DeviceTelemetry,
  OperationsOverviewResponse,
  PlatformSettings,
  agriApi,
} from '../api/client';
import {
  SUPPORTED_LANGUAGES,
  SupportedLanguageCode,
  setAppLanguage,
} from '../i18n';
import { formatDateTime, formatNumber } from '../utils/format';

export type OperationsDomainKey =
  | 'overview'
  | 'software'
  | 'hardware'
  | 'mlops'
  | 'dataops'
  | 'llmops'
  | 'cicd'
  | 'security'
  | 'architecture';

interface OperationsCenterPageProps {
  devices: DeviceTelemetry[];
  darkMode: boolean;
  onToggleDarkMode: (dark: boolean) => void;
  onRefreshTelemetry: () => Promise<void>;
}

export const OperationsCenterPage: React.FC<OperationsCenterPageProps> = ({
  devices,
  darkMode,
  onToggleDarkMode,
  onRefreshTelemetry,
}) => {
  const { t, i18n } = useTranslation(['operations', 'common', 'assistant']);
  const lang = i18n.language as SupportedLanguageCode;

  const [activeDomain, setActiveDomain] = useState<OperationsDomainKey>('overview');
  const [opsData, setOpsData] = useState<OperationsOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionBanner, setActionBanner] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  // Local editable settings form
  const [settingsForm, setSettingsForm] = useState<PlatformSettings | null>(null);
  const [selectedGraphNode, setSelectedGraphNode] = useState<DependencyNode | null>(null);

  const loadOperations = useCallback(async () => {
    try {
      const data = await agriApi.getOperationsOverview();
      setOpsData(data);
      setSettingsForm(data.settings);
      if (!selectedGraphNode && data.dependencyGraph.length > 0) {
        setSelectedGraphNode(data.dependencyGraph[1] || data.dependencyGraph[0]);
      }
      setActionError(null);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to load operations overview');
    } finally {
      setLoading(false);
    }
  }, [selectedGraphNode]);

  useEffect(() => {
    loadOperations();
    const timer = setInterval(() => {
      if (!document.hidden && !busyAction) {
        loadOperations();
      }
    }, 5000);
    return () => clearInterval(timer);
  }, [loadOperations, busyAction]);

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!settingsForm) return;
    setBusyAction('save-settings');
    setActionError(null);
    try {
      const res = await agriApi.updatePlatformSettings(settingsForm);
      setSettingsForm(res.settings);
      setActionBanner(t('operations:software.savedBanner'));
      await loadOperations();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Unable to save settings');
    } finally {
      setBusyAction(null);
    }
  };

  const handleHardwareCommand = async (
    deviceId: string,
    command: 'wake_ping' | 'zero_calibrate' | 'gateway_selftest'
  ) => {
    setBusyAction(`${deviceId}-${command}`);
    setActionError(null);
    try {
      await agriApi.sendHardwareCommand({ deviceId, command });
      setActionBanner(`${t('operations:hardware.commandConfirmed')} (${deviceId} · ${command})`);
      await Promise.all([loadOperations(), onRefreshTelemetry()]);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Hardware command failed');
    } finally {
      setBusyAction(null);
    }
  };

  const handleMlOpsDeploy = async (modelId: string, action: 'rollback' | 'promote') => {
    setBusyAction(`${modelId}-${action}`);
    setActionError(null);
    try {
      const res = await agriApi.deployMlModel({ modelId, action });
      setActionBanner(
        `MLOps ${action.toUpperCase()} confirmed: ${res.model.name} -> ${res.model.version} (${res.model.stage})`
      );
      await loadOperations();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'MLOps deployment update failed');
    } finally {
      setBusyAction(null);
    }
  };

  const handleRunDataValidation = async () => {
    setBusyAction('dataops-validate');
    setActionError(null);
    try {
      const res = await agriApi.runDataOpsValidation();
      setActionBanner(
        `DataOps validation sweep complete: ${res.checkedNodes} streams verified, ${res.staleNodes} stale stream(s) quarantined.`
      );
      await Promise.all([loadOperations(), onRefreshTelemetry()]);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Validation sweep failed');
    } finally {
      setBusyAction(null);
    }
  };

  const handleExportAuditCsv = () => {
    if (!opsData) return;
    const headers = ['ID', 'Timestamp', 'Actor', 'Role', 'Category', 'Action', 'Target', 'Status', 'CorrelationID', 'Details'];
    const rows = opsData.security.auditTrail.map((entry) => [
      entry.id,
      entry.timestamp,
      entry.actor,
      entry.role,
      entry.category,
      entry.action,
      `"${entry.target.replace(/"/g, '""')}"`,
      entry.status,
      entry.correlationId,
      `"${entry.details.replace(/"/g, '""')}"`,
    ]);
    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `agrisense-audit-trail-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const domainNav: {
    id: OperationsDomainKey;
    icon: React.ComponentType<{ className?: string }>;
  }[] = [
    { id: 'overview', icon: Activity },
    { id: 'software', icon: Sliders },
    { id: 'hardware', icon: Cpu },
    { id: 'mlops', icon: Layers },
    { id: 'dataops', icon: Database },
    { id: 'llmops', icon: Bot },
    { id: 'cicd', icon: GitBranch },
    { id: 'security', icon: ShieldCheck },
    { id: 'architecture', icon: Network },
  ];

  if (loading && !opsData) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-80 rounded bg-[var(--agri-sage)] animate-pulse" />
        <div className="h-72 rounded-xl bg-[var(--agri-card)] border border-[var(--agri-border)] animate-pulse" />
      </div>
    );
  }

  if (!opsData) return null;

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[var(--agri-border)] pb-4">
        <div>
          <div className="text-xs font-mono-tabular text-[var(--agri-leaf)] font-semibold">
            AGRISENSE INDUSTRIAL CONTROL &amp; OBSERVABILITY · PAGE 07
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--agri-text)] mt-1">
            {t('operations:title')}
          </h1>
          <p className="text-sm text-[var(--agri-muted)] mt-1 max-w-3xl">
            {t('operations:subtitle')}
          </p>
        </div>

        <button
          type="button"
          onClick={loadOperations}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-[var(--agri-text)] bg-[var(--agri-card)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg cursor-pointer self-start md:self-auto"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>{t('common:topbar.refreshData')}</span>
        </button>
      </div>

      {/* 9-Domain Structured Sub-Navigation */}
      <div className="flex flex-wrap items-center gap-1.5 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-2">
        {domainNav.map((item) => {
          const Icon = item.icon;
          const isSelected = activeDomain === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setActiveDomain(item.id);
                setActionBanner(null);
              }}
              className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
                isSelected
                  ? 'bg-[#173D2B] text-white font-semibold'
                  : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)] hover:bg-[var(--agri-bg)]'
              }`}
            >
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span>{t(`operations:domains.${item.id}`)}</span>
            </button>
          );
        })}
      </div>

      {actionBanner && (
        <div className="bg-[var(--agri-card)] border border-[var(--agri-leaf)] rounded-xl px-4 py-3 flex items-center justify-between text-xs text-[var(--agri-text)]">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-[var(--agri-leaf)] shrink-0" />
            <span>{actionBanner}</span>
          </div>
          <button
            type="button"
            onClick={() => setActionBanner(null)}
            className="text-[var(--agri-muted)] hover:text-[var(--agri-text)] cursor-pointer"
          >
            {t('common:actions.close')}
          </button>
        </div>
      )}

      {actionError && (
        <div className="bg-[var(--agri-card)] border border-[var(--agri-red)] rounded-xl px-4 py-3 text-xs text-[var(--agri-red)]">
          {actionError}
        </div>
      )}

      {/* DOMAIN A: SYSTEM HEALTH OVERVIEW */}
      {activeDomain === 'overview' && (
        <div className="space-y-6">
          {/* Hairline Metric Strip */}
          <div className="grid grid-cols-2 lg:grid-cols-4 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl divide-y sm:divide-y-0 sm:divide-x divide-[var(--agri-border)]">
            <div className="p-5">
              <span className="text-xs text-[var(--agri-muted)] block">
                {t('operations:overview.uptime')}
              </span>
              <span className="font-mono-tabular text-2xl font-bold text-[var(--agri-text)] mt-1 block">
                {formatNumber(Math.floor(opsData.systemHealth.uptimeSeconds / 60), lang)}m{' '}
                {formatNumber(opsData.systemHealth.uptimeSeconds % 60, lang)}s
              </span>
              <span className="text-[11px] font-mono-tabular text-[var(--agri-leaf)]">
                PID Active · Node.js Runtime
              </span>
            </div>

            <div className="p-5">
              <span className="text-xs text-[var(--agri-muted)] block">
                {t('operations:overview.memoryRss')}
              </span>
              <span className="font-mono-tabular text-2xl font-bold text-[var(--agri-text)] mt-1 block">
                {formatNumber(opsData.systemHealth.memoryRssMb, lang)} MB
              </span>
              <span className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
                Heap Used: {formatNumber(opsData.systemHealth.heapUsedMb, lang)} MB
              </span>
            </div>

            <div className="p-5">
              <span className="text-xs text-[var(--agri-muted)] block">
                {t('operations:overview.apiRequests')}
              </span>
              <span className="font-mono-tabular text-2xl font-bold text-[var(--agri-text)] mt-1 block">
                {formatNumber(opsData.systemHealth.totalApiRequests, lang)}
              </span>
              <span className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
                Errors: {opsData.systemHealth.totalApiErrors}
              </span>
            </div>

            <div className="p-5">
              <span className="text-xs text-[var(--agri-muted)] block">
                {t('operations:overview.avgLatency')}
              </span>
              <span className="font-mono-tabular text-2xl font-bold text-[var(--agri-water)] mt-1 block">
                {formatNumber(opsData.systemHealth.avgApiLatencyMs, lang)} ms
              </span>
              <span className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
                Ring Buffer (Last 80 reqs)
              </span>
            </div>
          </div>

          {/* Live Service Health Probes Table */}
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-[var(--agri-border)] flex items-center justify-between">
              <h2 className="text-sm font-semibold text-[var(--agri-text)]">
                {t('operations:overview.servicesTitle')}
              </h2>
              <span className="text-xs font-mono-tabular text-[var(--agri-muted)]">
                {formatDateTime(opsData.timestamp, lang)}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[var(--agri-border)] text-[var(--agri-muted)] bg-[var(--agri-bg)]/50">
                    <th className="py-3 px-6 font-medium">{t('operations:overview.serviceName')}</th>
                    <th className="py-3 px-4 font-medium">{t('operations:overview.status')}</th>
                    <th className="py-3 px-4 font-medium">{t('operations:overview.latency')}</th>
                    <th className="py-3 px-4 font-medium">{t('operations:overview.version')}</th>
                    <th className="py-3 px-6 font-medium">{t('operations:overview.dependency')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--agri-border)]">
                  {opsData.systemHealth.services.map((svc) => (
                    <tr key={svc.id} className="hover:bg-[var(--agri-bg)]/40">
                      <td className="py-3.5 px-6 font-medium text-[var(--agri-text)]">
                        <div>{svc.name}</div>
                        <div className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
                          {svc.id}
                        </div>
                      </td>
                      <td className="py-3.5 px-4 font-mono-tabular">
                        <span
                          className={`font-semibold ${
                            svc.status === 'healthy'
                              ? 'text-[var(--agri-leaf)]'
                              : 'text-[var(--agri-amber)]'
                          }`}
                        >
                          {svc.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 font-mono-tabular text-[var(--agri-text)]">
                        {formatNumber(svc.latencyMs, lang)} ms
                      </td>
                      <td className="py-3.5 px-4 font-mono-tabular text-[var(--agri-muted)]">
                        {svc.version}
                      </td>
                      <td className="py-3.5 px-6 font-mono-tabular text-[var(--agri-muted)]">
                        {svc.dependency}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* DOMAIN B: SOFTWARE & APPLICATION SETTINGS */}
      {activeDomain === 'software' && settingsForm && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Language & Visual Theme */}
          <div className="lg:col-span-5 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-6">
            <div>
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('assistant:settings.languageSection')}
              </h2>
              <p className="text-xs text-[var(--agri-muted)] mt-1">
                {t('assistant:settings.languageDesc')}
              </p>

              <div className="grid grid-cols-2 gap-2.5 mt-4">
                {SUPPORTED_LANGUAGES.map((item) => {
                  const isCurrent = lang === item.code;
                  return (
                    <button
                      key={item.code}
                      type="button"
                      onClick={() => setAppLanguage(item.code)}
                      className={`px-3.5 py-2.5 rounded-lg border text-left transition-colors cursor-pointer ${
                        isCurrent
                          ? 'bg-[#173D2B] text-white border-[#173D2B]'
                          : 'bg-[var(--agri-bg)] text-[var(--agri-text)] border-[var(--agri-border)] hover:bg-[var(--agri-sage)]'
                      }`}
                    >
                      <div className="text-sm font-semibold">{item.nativeLabel}</div>
                      <div
                        className={`text-[11px] font-mono-tabular ${
                          isCurrent ? 'text-white/80' : 'text-[var(--agri-muted)]'
                        }`}
                      >
                        {item.label} ({item.code.toUpperCase()})
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="pt-5 border-t border-[var(--agri-border)]">
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('assistant:settings.themeSection')}
              </h2>
              <p className="text-xs text-[var(--agri-muted)] mt-1">
                {t('assistant:settings.themeDesc')}
              </p>
              <div className="flex items-center gap-3 mt-3">
                <button
                  type="button"
                  onClick={() => onToggleDarkMode(false)}
                  className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg border cursor-pointer ${
                    !darkMode
                      ? 'bg-[#173D2B] text-white border-[#173D2B]'
                      : 'bg-[var(--agri-bg)] text-[var(--agri-text)] border-[var(--agri-border)]'
                  }`}
                >
                  <Sun className="w-4 h-4" />
                  <span>{t('assistant:settings.lightMode')}</span>
                </button>
                <button
                  type="button"
                  onClick={() => onToggleDarkMode(true)}
                  className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg border cursor-pointer ${
                    darkMode
                      ? 'bg-[#173D2B] text-white border-[#173D2B]'
                      : 'bg-[var(--agri-bg)] text-[var(--agri-text)] border-[var(--agri-border)]'
                  }`}
                >
                  <Moon className="w-4 h-4" />
                  <span>{t('assistant:settings.darkMode')}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Right: Operational Thresholds & Interlock Policy Form */}
          <form
            onSubmit={handleSaveSettings}
            className="lg:col-span-7 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-5"
          >
            <div>
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('operations:software.title')}
              </h2>
              <p className="text-xs text-[var(--agri-muted)] mt-1">
                {t('operations:software.desc')}
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('operations:software.moistureAlertLabel')}
                </label>
                <input
                  type="number"
                  min={15}
                  max={60}
                  value={settingsForm.moistureCriticalAlertPct}
                  onChange={(e) =>
                    setSettingsForm({
                      ...settingsForm,
                      moistureCriticalAlertPct: Number(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                />
              </div>

              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('operations:software.staleTimeoutLabel')}
                </label>
                <input
                  type="number"
                  min={5}
                  max={120}
                  value={settingsForm.staleNodeTimeoutMin}
                  onChange={(e) =>
                    setSettingsForm({
                      ...settingsForm,
                      staleNodeTimeoutMin: Number(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                />
              </div>

              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('operations:software.maxIrrigationLabel')}
                </label>
                <input
                  type="number"
                  min={10}
                  max={180}
                  value={settingsForm.maxContinuousIrrigationMin}
                  onChange={(e) =>
                    setSettingsForm({
                      ...settingsForm,
                      maxContinuousIrrigationMin: Number(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                />
              </div>
            </div>

            <div className="space-y-3 pt-3 border-t border-[var(--agri-border)] text-xs">
              <label className="flex items-center justify-between gap-4 cursor-pointer">
                <span className="text-[var(--agri-text)] font-medium">
                  {t('operations:software.safetyInterlockLabel')}
                </span>
                <input
                  type="checkbox"
                  checked={settingsForm.safetyInterlockRequired}
                  onChange={(e) =>
                    setSettingsForm({
                      ...settingsForm,
                      safetyInterlockRequired: e.target.checked,
                    })
                  }
                  className="w-4 h-4 accent-[#173D2B]"
                />
              </label>

              <label className="flex items-center justify-between gap-4 cursor-pointer">
                <span className="text-[var(--agri-text)] font-medium">
                  {t('operations:software.quarantineStaleLabel')}
                </span>
                <input
                  type="checkbox"
                  checked={settingsForm.quarantineStaleNodes}
                  onChange={(e) =>
                    setSettingsForm({
                      ...settingsForm,
                      quarantineStaleNodes: e.target.checked,
                    })
                  }
                  className="w-4 h-4 accent-[#173D2B]"
                />
              </label>

              <label className="flex items-center justify-between gap-4 cursor-pointer">
                <span className="text-[var(--agri-text)] font-medium">
                  {t('operations:software.shadows3dLabel')}
                </span>
                <input
                  type="checkbox"
                  checked={settingsForm.enable3DHardwareShadows}
                  onChange={(e) =>
                    setSettingsForm({
                      ...settingsForm,
                      enable3DHardwareShadows: e.target.checked,
                    })
                  }
                  className="w-4 h-4 accent-[#173D2B]"
                />
              </label>
            </div>

            <div className="pt-3 border-t border-[var(--agri-border)] flex items-center justify-between gap-4">
              <div className="text-xs">
                <span className="text-[var(--agri-muted)] mr-2">
                  {t('operations:software.logLevelLabel')}:
                </span>
                <select
                  value={settingsForm.logLevel}
                  onChange={(e) =>
                    setSettingsForm({
                      ...settingsForm,
                      logLevel: e.target.value as PlatformSettings['logLevel'],
                    })
                  }
                  className="px-2.5 py-1.5 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                >
                  <option value="DEBUG">DEBUG</option>
                  <option value="INFO">INFO</option>
                  <option value="WARN">WARN</option>
                  <option value="ERROR">ERROR</option>
                </select>
              </div>

              <button
                type="submit"
                disabled={busyAction === 'save-settings'}
                className="px-5 py-2.5 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer disabled:opacity-50"
              >
                {t('common:actions.save')}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* DOMAIN C: HARDWARE & IOT OPERATIONS */}
      {activeDomain === 'hardware' && (
        <div className="space-y-4">
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6">
            <h2 className="text-base font-semibold text-[var(--agri-text)]">
              {t('operations:hardware.title')}
            </h2>
            <p className="text-xs text-[var(--agri-muted)] mt-1">
              {t('operations:hardware.desc')}
            </p>

            <div className="mt-5 divide-y divide-[var(--agri-border)]">
              {devices.map((dev) => {
                const isStale = dev.status === 'stale';
                return (
                  <div
                    key={dev.id}
                    className="py-4 first:pt-0 last:pb-0 flex flex-col lg:flex-row lg:items-center justify-between gap-4"
                  >
                    <div>
                      <div className="flex items-center gap-2 text-xs font-mono-tabular">
                        <span className="font-bold text-[var(--agri-text)]">{dev.id}</span>
                        <span className="text-[var(--agri-muted)]">·</span>
                        <span className="text-[var(--agri-muted)]">{dev.protocol}</span>
                        <span className="text-[var(--agri-muted)]">·</span>
                        <span className="text-[var(--agri-muted)]">{dev.firmware}</span>
                        <span className="text-[var(--agri-muted)]">·</span>
                        <span
                          className={
                            isStale
                              ? 'text-[var(--agri-amber)] font-semibold'
                              : 'text-[var(--agri-leaf)] font-semibold'
                          }
                        >
                          {dev.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-sm font-semibold text-[var(--agri-text)] mt-1">
                        {dev.name} — {dev.zoneName}
                      </div>
                      <div className="text-xs text-[var(--agri-muted)] font-mono-tabular mt-0.5">
                        Battery: {dev.batteryPct}% · RSSI: {dev.signalRssi} dBm · Interval:{' '}
                        {dev.samplingIntervalSec}s · Offset: {dev.calibrationOffsetPct ?? 0}%
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        disabled={busyAction === `${dev.id}-wake_ping`}
                        onClick={() => handleHardwareCommand(dev.id, 'wake_ping')}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer disabled:opacity-50"
                      >
                        <Play className="w-3 h-3" />
                        <span>{t('operations:hardware.wakePingBtn')}</span>
                      </button>

                      <button
                        type="button"
                        disabled={busyAction === `${dev.id}-zero_calibrate`}
                        onClick={() => handleHardwareCommand(dev.id, 'zero_calibrate')}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[var(--agri-text)] bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg cursor-pointer disabled:opacity-50"
                      >
                        <Wrench className="w-3 h-3" />
                        <span>{t('operations:hardware.zeroCalibrateBtn')}</span>
                      </button>

                      {dev.type === 'RPI_EDGE_GATEWAY' && (
                        <button
                          type="button"
                          disabled={busyAction === `${dev.id}-gateway_selftest`}
                          onClick={() => handleHardwareCommand(dev.id, 'gateway_selftest')}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[var(--agri-water)] bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg cursor-pointer disabled:opacity-50"
                        >
                          <Activity className="w-3 h-3" />
                          <span>{t('operations:hardware.selfTestBtn')}</span>
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* DOMAIN D: MLOPS MODEL REGISTRY & DRIFT GOVERNANCE */}
      {activeDomain === 'mlops' && (
        <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-5">
          <div>
            <h2 className="text-base font-semibold text-[var(--agri-text)]">
              {t('operations:mlops.title')}
            </h2>
            <p className="text-xs text-[var(--agri-muted)] mt-1">
              {t('operations:mlops.desc')}
            </p>
          </div>

          <div className="divide-y divide-[var(--agri-border)]">
            {opsData.mlOps.models.map((m) => (
              <div
                key={m.id}
                className="py-4 first:pt-0 last:pb-0 flex flex-col lg:flex-row lg:items-center justify-between gap-4"
              >
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2 text-xs font-mono-tabular">
                    <span className="font-bold text-[var(--agri-text)]">{m.id}</span>
                    <span>·</span>
                    <span className="text-[var(--agri-leaf)] font-semibold">{m.version}</span>
                    <span>·</span>
                    <span
                      className={
                        m.stage === 'production'
                          ? 'text-[var(--agri-leaf)] font-semibold'
                          : 'text-[var(--agri-amber)] font-semibold'
                      }
                    >
                      {t(`operations:mlops.stage_${m.stage}`, { defaultValue: m.stage })}
                    </span>
                    <span>·</span>
                    <span className="text-[var(--agri-muted)]">Dataset: {m.datasetVersion}</span>
                  </div>

                  <div className="text-sm font-semibold text-[var(--agri-text)]">{m.name}</div>
                  <div className="text-xs text-[var(--agri-muted)]">
                    {m.task} · {m.framework}
                  </div>

                  <div className="flex flex-wrap items-center gap-4 pt-1 text-xs font-mono-tabular">
                    <span>
                      <strong className="text-[var(--agri-text)]">{m.primaryMetricName}:</strong>{' '}
                      {m.primaryMetricValue}
                    </span>
                    <span>·</span>
                    <span>{m.secondaryMetricLabel}</span>
                    <span>·</span>
                    <span>Latency: {m.avgLatencyMs} ms</span>
                    <span>·</span>
                    <span
                      className={
                        m.driftStatus === 'watch'
                          ? 'text-[var(--agri-amber)] font-semibold'
                          : 'text-[var(--agri-leaf)]'
                      }
                    >
                      PSI Drift: {m.driftScorePsi} (
                      {t(`operations:mlops.drift_${m.driftStatus}`, {
                        defaultValue: m.driftStatus,
                      })}
                      )
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  {m.stage === 'staging' && (
                    <button
                      type="button"
                      disabled={busyAction === `${m.id}-promote`}
                      onClick={() => handleMlOpsDeploy(m.id, 'promote')}
                      className="px-3 py-1.5 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer"
                    >
                      {t('operations:mlops.promoteBtn')}
                    </button>
                  )}
                  <button
                    type="button"
                    disabled={busyAction === `${m.id}-rollback`}
                    onClick={() => handleMlOpsDeploy(m.id, 'rollback')}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[var(--agri-text)] bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>
                      {t('operations:mlops.rollbackBtn')} ({m.previousVersion})
                    </span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* DOMAIN E: DATAOPS CONTROL CENTER */}
      {activeDomain === 'dataops' && (
        <div className="space-y-6">
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('operations:dataops.title')}
              </h2>
              <p className="text-xs text-[var(--agri-muted)] mt-1">
                {t('operations:dataops.desc')}
              </p>
              <div className="flex flex-wrap items-center gap-4 mt-3 text-xs font-mono-tabular">
                <span>
                  {t('operations:dataops.freshnessSla')}:{' '}
                  <strong className="text-[var(--agri-leaf)]">
                    {formatNumber(opsData.dataOps.freshnessPct, lang)}%
                  </strong>
                </span>
                <span>·</span>
                <span>
                  {t('operations:dataops.schemaCompliance')}:{' '}
                  <strong className="text-[var(--agri-leaf)]">
                    {formatNumber(opsData.dataOps.schemaCompliancePct, lang)}%
                  </strong>
                </span>
                <span>·</span>
                <span>Stale Streams Quarantined: {opsData.dataOps.staleNodeCount}</span>
              </div>
            </div>

            <button
              type="button"
              disabled={busyAction === 'dataops-validate'}
              onClick={handleRunDataValidation}
              className="px-4 py-2 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer whitespace-nowrap self-start sm:self-auto"
            >
              {t('operations:dataops.runValidationBtn')}
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Datasets Lineage */}
            <div className="lg:col-span-7 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4">
              <h3 className="text-sm font-semibold text-[var(--agri-text)]">
                {t('operations:dataops.datasetsTitle')}
              </h3>
              <div className="divide-y divide-[var(--agri-border)] text-xs">
                {opsData.dataOps.datasets.map((ds) => (
                  <div key={ds.id} className="py-3.5 first:pt-0 last:pb-0 space-y-1">
                    <div className="flex items-center justify-between font-mono-tabular">
                      <span className="font-bold text-[var(--agri-text)]">
                        {ds.id} ({ds.version})
                      </span>
                      <span className="text-[var(--agri-leaf)] font-semibold">
                        {ds.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-sm font-medium text-[var(--agri-text)]">{ds.name}</div>
                    <div className="text-[var(--agri-muted)] font-mono-tabular">
                      Records: {formatNumber(ds.records, lang)} · Freshness: {ds.freshness} ·
                      Missingness: {ds.missingnessPct}% · {ds.license}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Automated Validation Rules */}
            <div className="lg:col-span-5 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4">
              <h3 className="text-sm font-semibold text-[var(--agri-text)]">
                {t('operations:dataops.rulesTitle')}
              </h3>
              <div className="divide-y divide-[var(--agri-border)] text-xs">
                {opsData.dataOps.validationRules.map((rule) => (
                  <div key={rule.id} className="py-3.5 first:pt-0 last:pb-0 space-y-1">
                    <div className="flex items-center justify-between font-mono-tabular">
                      <span className="font-semibold text-[var(--agri-text)]">{rule.id}</span>
                      <span
                        className={
                          rule.passed
                            ? 'text-[var(--agri-leaf)] font-semibold'
                            : 'text-[var(--agri-amber)] font-semibold'
                        }
                      >
                        {rule.passed ? 'PASSED' : `${rule.violationCount} VIOLATION(S)`}
                      </span>
                    </div>
                    <div className="text-[var(--agri-text)]">{rule.name}</div>
                    <div className="text-[var(--agri-muted)] font-mono-tabular">
                      Checked Streams: {rule.checkedCount}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DOMAIN F: LLMOPS & RAG OPERATIONS */}
      {activeDomain === 'llmops' && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl divide-y sm:divide-y-0 sm:divide-x divide-[var(--agri-border)]">
            <div className="p-5">
              <span className="text-xs text-[var(--agri-muted)] block">
                {t('operations:llmops.totalInferences')}
              </span>
              <span className="font-mono-tabular text-2xl font-bold text-[var(--agri-text)] mt-1 block">
                {formatNumber(opsData.llmOps.totalRequests, lang)}
              </span>
              <span className="text-[11px] font-mono-tabular text-[var(--agri-leaf)]">
                Fallbacks: {opsData.llmOps.fallbackCount}
              </span>
            </div>

            <div className="p-5">
              <span className="text-xs text-[var(--agri-muted)] block">
                {t('operations:llmops.visionVsChat')}
              </span>
              <span className="font-mono-tabular text-2xl font-bold text-[var(--agri-text)] mt-1 block">
                {opsData.llmOps.visionRequests} / {opsData.llmOps.assistantRequests}
              </span>
              <span className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
                Vision Leaf Lab vs RAG Chat
              </span>
            </div>

            <div className="p-5">
              <span className="text-xs text-[var(--agri-muted)] block">
                {t('operations:llmops.avgLatency')}
              </span>
              <span className="font-mono-tabular text-2xl font-bold text-[var(--agri-water)] mt-1 block">
                {formatNumber(opsData.llmOps.avgLatencyMs, lang)} ms
              </span>
              <span className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
                Model: {opsData.llmOps.providerStatus || 'AgriSense-Local-RTX5060'}
              </span>
            </div>

            <div className="p-5">
              <span className="text-xs text-[var(--agri-muted)] block">
                {t('operations:llmops.tokenVolume')}
              </span>
              <span className="font-mono-tabular text-2xl font-bold text-[var(--agri-text)] mt-1 block">
                {formatNumber(opsData.llmOps.estimatedInputTokens, lang)} /{' '}
                {formatNumber(opsData.llmOps.estimatedOutputTokens, lang)}
              </span>
              <span className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
                Prompt: {opsData.llmOps.activeSystemPromptVersion}
              </span>
            </div>
          </div>

          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold text-[var(--agri-text)]">
                  {t('operations:llmops.ragCorpusTitle')}
                </h3>
                <p className="text-xs text-[var(--agri-muted)] mt-0.5">
                  Provider Status: {opsData.llmOps.providerStatus} · Guardrail:{' '}
                  <span className="font-mono-tabular">{opsData.llmOps.guardrailPolicy}</span>
                </p>
              </div>
              <span className="text-xs font-mono-tabular text-[var(--agri-leaf)] font-semibold">
                {opsData.llmOps.ragChunksIndexed} Total Vector Chunks Indexed
              </span>
            </div>

            <div className="divide-y divide-[var(--agri-border)] text-xs">
              {opsData.llmOps.ragDocuments.map((doc) => (
                <div
                  key={doc.id}
                  className="py-3 first:pt-0 last:pb-0 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                >
                  <div>
                    <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                      {doc.id}
                    </span>
                    <span className="mx-2 text-[var(--agri-muted)]">·</span>
                    <span className="font-medium text-[var(--agri-text)]">{doc.title}</span>
                  </div>
                  <div className="font-mono-tabular text-[var(--agri-muted)]">
                    {doc.chunks} chunks · {doc.embeddingModel} ·{' '}
                    <span className="text-[var(--agri-leaf)] font-semibold">
                      {doc.status.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* DOMAIN G: CI/CD & RELEASE ENGINEERING */}
      {activeDomain === 'cicd' && (
        <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-5">
          <div>
            <h2 className="text-base font-semibold text-[var(--agri-text)]">
              {t('operations:cicd.title')}
            </h2>
            <p className="text-xs text-[var(--agri-muted)] mt-1">
              {t('operations:cicd.desc')}
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-[var(--agri-bg)] border border-[var(--agri-border)] text-xs font-mono-tabular text-[var(--agri-muted)]">
            Branch: <strong className="text-[var(--agri-text)]">{opsData.cicd.branch}</strong> ·
            Runtime: <strong className="text-[var(--agri-text)]">{opsData.cicd.runtimeEnv}</strong>{' '}
            · {opsData.cicd.externalProviderNote}
          </div>

          <div className="divide-y divide-[var(--agri-border)] text-xs">
            {opsData.cicd.checks.map((chk) => (
              <div
                key={chk.id}
                className="py-3.5 first:pt-0 last:pb-0 flex items-center justify-between gap-4"
              >
                <div>
                  <div className="font-semibold text-[var(--agri-text)]">{chk.name}</div>
                  <div className="font-mono-tabular text-[var(--agri-muted)] mt-0.5">
                    {chk.id} · {formatDateTime(chk.lastRunIso, lang)}
                  </div>
                </div>
                <div className="font-mono-tabular flex items-center gap-3">
                  <span className="text-[var(--agri-muted)]">{chk.durationSec}s</span>
                  <span className="text-[var(--agri-leaf)] font-semibold">
                    {chk.status.toUpperCase()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* DOMAIN H: SECURITY, RBAC & IMMUTABLE AUDIT TRAIL */}
      {activeDomain === 'security' && (
        <div className="space-y-6">
          {/* RBAC Matrix */}
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4">
            <h2 className="text-base font-semibold text-[var(--agri-text)]">
              {t('operations:security.rbacTitle')}
            </h2>
            <div className="divide-y divide-[var(--agri-border)] text-xs">
              {opsData.security.rbacRoles.map((r) => (
                <div
                  key={r.role}
                  className="py-3 first:pt-0 last:pb-0 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                >
                  <div>
                    <span className="font-semibold text-[var(--agri-text)]">{r.role}</span>
                    <p className="text-[var(--agri-muted)] mt-0.5">{r.permissions}</p>
                  </div>
                  <span className="font-mono-tabular text-[var(--agri-leaf)] font-semibold shrink-0">
                    Scope: {r.scope}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Audit Trail with CSV Export */}
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4">
            <div className="flex items-center justify-between gap-4">
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('operations:security.auditTitle')}
              </h2>
              <button
                type="button"
                onClick={handleExportAuditCsv}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-[var(--agri-text)] bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                <span>{t('common:actions.exportCsv')}</span>
              </button>
            </div>

            <div className="divide-y divide-[var(--agri-border)] text-xs max-h-80 overflow-y-auto">
              {opsData.security.auditTrail.map((aud) => (
                <div key={aud.id} className="py-3 first:pt-0 last:pb-0 space-y-1">
                  <div className="flex flex-wrap items-center justify-between gap-2 font-mono-tabular">
                    <div>
                      <span className="font-bold text-[var(--agri-text)]">{aud.id}</span>
                      <span className="mx-1.5 text-[var(--agri-muted)]">·</span>
                      <span className="text-[var(--agri-leaf)] font-semibold">{aud.category}</span>
                      <span className="mx-1.5 text-[var(--agri-muted)]">·</span>
                      <span className="font-semibold text-[var(--agri-text)]">{aud.action}</span>
                    </div>
                    <span className="text-[var(--agri-muted)]">
                      {aud.correlationId} · {formatDateTime(aud.timestamp, lang)}
                    </span>
                  </div>
                  <div className="text-[var(--agri-text)]">
                    <strong>{aud.target}:</strong> {aud.details}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Structured Service Logs */}
          <div className="bg-[#0D1912] text-[#DCE8D5] border border-[var(--agri-border)] rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-xs font-mono-tabular text-[#5CA668]">
              <Terminal className="w-4 h-4" />
              <span className="font-semibold">{t('operations:security.logsTitle')}</span>
            </div>
            <div className="space-y-2 font-mono-tabular text-xs max-h-56 overflow-y-auto">
              {opsData.security.serviceLogs.map((log) => (
                <div key={log.id} className="border-b border-white/10 pb-1.5 last:border-b-0">
                  <span className="text-white/50">{log.timestamp.slice(11, 19)}</span>{' '}
                  <span
                    className={
                      log.level === 'WARN'
                        ? 'text-[#E5B55C] font-bold'
                        : 'text-[#5CA668] font-bold'
                    }
                  >
                    [{log.level}]
                  </span>{' '}
                  <span className="text-[#55A3D1]">{log.service}</span>{' '}
                  <span className="text-white/40">({log.correlationId})</span>: {log.message}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* DOMAIN I: INTERACTIVE ARCHITECTURE & DEPENDENCY GRAPH */}
      {activeDomain === 'architecture' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-8 bg-[#0D1912] text-white border border-[var(--agri-border)] rounded-xl p-6 space-y-4">
            <div>
              <h2 className="text-base font-semibold text-white">
                {t('operations:architecture.title')}
              </h2>
              <p className="text-xs text-[#DCE8D5]/75 mt-1">
                {t('operations:architecture.desc')}
              </p>
            </div>

            {/* Interactive Pipeline Node Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
              {opsData.dependencyGraph.map((node) => {
                const isSelected = selectedGraphNode?.id === node.id;
                const isDownstreamOfSelected =
                  selectedGraphNode?.downstream.includes(node.id) ?? false;

                return (
                  <button
                    key={node.id}
                    type="button"
                    onClick={() => setSelectedGraphNode(node)}
                    className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-[#173D2B] border-[#5CA668] shadow-lg'
                        : isDownstreamOfSelected
                        ? 'bg-[#13291E] border-[#55A3D1]'
                        : 'bg-white/5 border-white/15 hover:bg-white/10'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[11px] font-mono-tabular">
                      <span className="text-[#55A3D1]">{node.layer}</span>
                      <span
                        className={
                          node.status === 'warning' ? 'text-[#E5B55C]' : 'text-[#5CA668]'
                        }
                      >
                        ● {node.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-xs font-bold text-white mt-1.5">{node.label}</div>
                    <div className="text-[11px] font-mono-tabular text-[#DCE8D5]/75 mt-1">
                      {node.metrics}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Right: Selected Pipeline Node Diagnostic Inspector */}
          <div className="lg:col-span-4 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4">
            <h3 className="text-sm font-semibold text-[var(--agri-text)]">
              {t('operations:architecture.selectedNodeTitle')}
            </h3>

            {selectedGraphNode && (
              <div className="space-y-3 text-xs">
                <div className="pb-3 border-b border-[var(--agri-border)]">
                  <div className="font-mono-tabular text-[var(--agri-muted)]">
                    {selectedGraphNode.id} · {selectedGraphNode.layer}
                  </div>
                  <div className="text-base font-bold text-[var(--agri-text)] mt-1">
                    {selectedGraphNode.label}
                  </div>
                  <div className="font-mono-tabular text-[var(--agri-leaf)] font-semibold mt-1">
                    {selectedGraphNode.metrics}
                  </div>
                </div>

                <div>
                  <span className="text-[var(--agri-muted)] block mb-1.5">
                    {t('operations:architecture.downstreamLabel')}:
                  </span>
                  {selectedGraphNode.downstream.length === 0 ? (
                    <span className="font-mono-tabular text-[var(--agri-muted)]">
                      Terminal Presentation / Actuation Layer
                    </span>
                  ) : (
                    <div className="flex flex-wrap gap-1.5">
                      {selectedGraphNode.downstream.map((dId) => (
                        <span
                          key={dId}
                          className="px-2.5 py-1 rounded bg-[var(--agri-bg)] border border-[var(--agri-border)] font-mono-tabular text-[var(--agri-text)]"
                        >
                          → {dId}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
