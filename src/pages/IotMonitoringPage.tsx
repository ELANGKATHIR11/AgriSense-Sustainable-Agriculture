import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertTriangle,
  BatteryCharging,
  CheckCircle2,
  Radio,
  Search,
  Settings2,
  Wifi,
  X,
} from 'lucide-react';
import { DeviceTelemetry, SensorsResponse, agriApi } from '../api/client';
import { Farm3DScene } from '../components/Farm3DScene';
import { TelemetryChart } from '../components/TelemetryChart';
import { formatNumber } from '../utils/format';

interface IotMonitoringPageProps {
  data: SensorsResponse | null;
  loading: boolean;
  onRefresh: () => Promise<void>;
}

export const IotMonitoringPage: React.FC<IotMonitoringPageProps> = ({
  data,
  loading,
  onRefresh,
}) => {
  const { t, i18n } = useTranslation(['iot', 'common', 'dashboard', 'operations']);
  const lang = i18n.language;

  const [statusFilter, setStatusFilter] = useState<'all' | 'online' | 'stale'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedZoneId, setSelectedZoneId] = useState<string>('ZONE-A');
  const [editingDevice, setEditingDevice] = useState<DeviceTelemetry | null>(null);
  const [intervalInput, setIntervalInput] = useState('30');
  const [thresholdInput, setThresholdInput] = useState('42');
  const [offsetInput, setOffsetInput] = useState('0.0');
  const [savingConfig, setSavingConfig] = useState(false);
  const [configFeedback, setConfigFeedback] = useState<string | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [pingingId, setPingingId] = useState<string | null>(null);

  if (!data && loading) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-64 rounded bg-[var(--agri-sage)] animate-pulse" />
        <div className="h-64 rounded-xl bg-[var(--agri-card)] border border-[var(--agri-border)] animate-pulse" />
      </div>
    );
  }

  if (!data) return null;

  const filteredDevices = data.devices.filter((d) => {
    if (statusFilter === 'online' && d.status !== 'online') return false;
    if (statusFilter === 'stale' && d.status === 'online') return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return (
        d.id.toLowerCase().includes(q) ||
        d.name.toLowerCase().includes(q) ||
        d.zoneName.toLowerCase().includes(q) ||
        d.cropType.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const openConfigModal = (dev: DeviceTelemetry) => {
    setEditingDevice(dev);
    setIntervalInput(String(dev.samplingIntervalSec));
    setThresholdInput(String(dev.moistureThresholdPct));
    setOffsetInput(String(dev.calibrationOffsetPct ?? 0));
    setConfigError(null);
    setConfigFeedback(null);
  };

  const handleSaveConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingDevice) return;
    setSavingConfig(true);
    setConfigError(null);
    try {
      await agriApi.updateDeviceConfig(editingDevice.id, {
        samplingIntervalSec: Number(intervalInput),
        moistureThresholdPct: Number(thresholdInput),
        calibrationOffsetPct: Number(offsetInput),
      });
      setConfigFeedback(t('iot:configModal.savedSuccess'));
      await onRefresh();
      setTimeout(() => {
        setEditingDevice(null);
        setConfigFeedback(null);
      }, 900);
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : 'Error updating device');
    } finally {
      setSavingConfig(false);
    }
  };

  const handleWakePing = async (devId: string) => {
    setPingingId(devId);
    try {
      await agriApi.sendHardwareCommand({ deviceId: devId, command: 'wake_ping' });
      await onRefresh();
    } finally {
      setPingingId(null);
    }
  };

  const chartPoints = data.history24h.map((pt) => ({
    label: pt.hourLabel,
    primary: pt.soilMoisturePct,
    secondary: pt.airHumidityPct,
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-[var(--agri-border)] pb-4">
        <div className="text-xs font-mono-tabular text-[var(--agri-leaf)] font-semibold">
          MQTT/TLS 1.3 &amp; LORAWAN FIELD TOPOLOGY · PAGE 02
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-[var(--agri-text)] mt-1">
          {t('iot:title')}
        </h1>
        <p className="text-sm text-[var(--agri-muted)] mt-1 max-w-3xl">
          {t('iot:subtitle')}
        </p>
      </div>

      {/* 3D Sensor Tower & Reservoir Topology Viewport */}
      <Farm3DScene
        devices={data.devices}
        zones={data.irrigation.zones}
        tankLevelPct={data.irrigation.tankLevelPct}
        selectedZoneId={selectedZoneId}
        onSelectZone={(zId) => setSelectedZoneId(zId)}
        onSelectDevice={(dev) => openConfigModal(dev)}
        heightClass="h-[360px]"
      />

      {/* Stale data integrity banner */}
      <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl px-4 py-3 flex items-center gap-3 text-xs text-[var(--agri-text)]">
        <AlertTriangle className="w-4 h-4 text-[var(--agri-amber)] shrink-0" />
        <span>{t('iot:staleBanner')}</span>
      </div>

      {/* Filter & Search Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="inline-flex items-center gap-1 p-1 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-lg self-start">
          {(['all', 'online', 'stale'] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setStatusFilter(tab)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                statusFilter === tab
                  ? 'bg-[#173D2B] text-white'
                  : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
              }`}
            >
              {t(`iot:filters.${tab}`)}
            </button>
          ))}
        </div>

        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-[var(--agri-muted)] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t('iot:filters.searchPlaceholder')}
            className="w-full pl-9 pr-3 py-2 text-xs bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)] placeholder:text-[var(--agri-muted)] focus:outline-none focus:border-[var(--agri-leaf)]"
          />
        </div>
      </div>

      {/* IoT Nodes Grid */}
      {filteredDevices.length === 0 ? (
        <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-8 text-center text-sm text-[var(--agri-muted)]">
          {t('common:states.empty')}
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {filteredDevices.map((dev) => {
            const isStale = dev.status === 'stale';
            return (
              <div
                key={dev.id}
                onClick={() => setSelectedZoneId(dev.zoneId)}
                className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-5 flex flex-col justify-between space-y-4 cursor-pointer hover:border-[var(--agri-leaf)] transition-colors"
              >
                {/* Top Row: Name, ID, Status, Configure button */}
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 text-xs font-mono-tabular text-[var(--agri-muted)]">
                      <span>{dev.id}</span>
                      <span>·</span>
                      <span>{dev.protocol}</span>
                      <span>·</span>
                      <span
                        className={
                          isStale
                            ? 'text-[var(--agri-amber)] font-semibold'
                            : 'text-[var(--agri-leaf)] font-semibold'
                        }
                      >
                        {isStale ? t('common:status.stale') : t('common:status.online')}
                      </span>
                    </div>
                    <h2 className="text-base font-semibold text-[var(--agri-text)] mt-1">
                      {dev.name}
                    </h2>
                    <div className="text-xs text-[var(--agri-muted)] mt-0.5">
                      <span>{dev.zoneName}</span>
                      <span className="mx-1.5">·</span>
                      <span>{dev.cropType}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {isStale && (
                      <button
                        type="button"
                        disabled={pingingId === dev.id}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleWakePing(dev.id);
                        }}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer"
                      >
                        <Radio className="w-3.5 h-3.5" />
                        <span>{t('operations:hardware.wakePingBtn')}</span>
                      </button>
                    )}

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        openConfigModal(dev);
                      }}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[var(--agri-text)] bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Settings2 className="w-3.5 h-3.5" />
                      <span>{t('iot:deviceCard.configure')}</span>
                    </button>
                  </div>
                </div>

                {/* Stale warning if applicable */}
                {isStale && (
                  <div className="px-3 py-2 rounded-lg bg-[var(--agri-bg)] border border-[var(--agri-amber)] text-xs text-[var(--agri-text)] flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-[var(--agri-amber)] shrink-0" />
                    <span>
                      {t('iot:deviceCard.staleWarning', {
                        minutes: formatNumber(dev.lastSeenMinutesAgo, lang),
                      })}
                    </span>
                  </div>
                )}

                {/* Sensor Readings Table-like Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-3 gap-x-4 pt-3 border-t border-[var(--agri-border)] text-xs">
                  <div>
                    <span className="text-[var(--agri-muted)] block">
                      {t('iot:deviceCard.soilMoisture')}
                    </span>
                    <span className="font-mono-tabular font-semibold text-sm text-[var(--agri-text)]">
                      {formatNumber(dev.readings.soilMoisturePct, lang)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[var(--agri-muted)] block">
                      {t('iot:deviceCard.soilTemp')}
                    </span>
                    <span className="font-mono-tabular font-semibold text-sm text-[var(--agri-text)]">
                      {formatNumber(dev.readings.soilTempC, lang)}°C
                    </span>
                  </div>
                  <div>
                    <span className="text-[var(--agri-muted)] block">
                      {t('iot:deviceCard.airHumidity')}
                    </span>
                    <span className="font-mono-tabular font-semibold text-sm text-[var(--agri-text)]">
                      {formatNumber(dev.readings.airHumidityPct, lang)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[var(--agri-muted)] block">
                      {t('iot:deviceCard.soilPh')}
                    </span>
                    <span className="font-mono-tabular font-semibold text-sm text-[var(--agri-text)]">
                      {formatNumber(dev.readings.soilPh, lang)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[var(--agri-muted)] block">
                      {t('iot:deviceCard.npk')}
                    </span>
                    <span className="font-mono-tabular font-semibold text-sm text-[var(--agri-text)]">
                      {dev.readings.nitrogenMgKg}-{dev.readings.phosphorusMgKg}-
                      {dev.readings.potassiumMgKg}
                    </span>
                  </div>
                  <div>
                    <span className="text-[var(--agri-muted)] block">
                      {t('iot:deviceCard.ec')}
                    </span>
                    <span className="font-mono-tabular font-semibold text-sm text-[var(--agri-text)]">
                      {formatNumber(dev.readings.ecDsM, lang)} dS/m
                    </span>
                  </div>
                </div>

                {/* Footer Hardware Telemetry Metadata */}
                <div className="flex flex-wrap items-center justify-between gap-2 pt-3 border-t border-[var(--agri-border)] text-xs text-[var(--agri-muted)] font-mono-tabular">
                  <div className="flex items-center gap-3">
                    <span className="inline-flex items-center gap-1">
                      <BatteryCharging className="w-3.5 h-3.5" />
                      {formatNumber(dev.batteryPct, lang)}%
                    </span>
                    <span>·</span>
                    <span className="inline-flex items-center gap-1">
                      <Wifi className="w-3.5 h-3.5" />
                      {dev.signalRssi} dBm
                    </span>
                    <span>·</span>
                    <span>{dev.firmware}</span>
                  </div>
                  <div>
                    {t('iot:deviceCard.lastSeen')}:{' '}
                    {dev.lastSeenMinutesAgo === 0
                      ? t('iot:deviceCard.justNow')
                      : t('iot:deviceCard.minutesAgo', {
                          count: dev.lastSeenMinutesAgo,
                        })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Water Tank & Sensor History Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Water Tank Telemetry */}
        <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-semibold text-[var(--agri-text)] mb-4">
              {t('iot:waterTank.title')}
            </h2>
            <div className="mb-4">
              <div className="flex items-baseline justify-between text-xs mb-1.5">
                <span className="text-[var(--agri-muted)]">{t('iot:waterTank.level')}</span>
                <span className="font-mono-tabular font-bold text-lg text-[var(--agri-water)]">
                  {formatNumber(data.irrigation.tankLevelPct, lang)}%
                </span>
              </div>
              <div className="w-full h-3 bg-[var(--agri-bg)] rounded-full overflow-hidden border border-[var(--agri-border)]">
                <div
                  className="h-full bg-[var(--agri-water)] transition-all duration-200"
                  style={{ width: `${data.irrigation.tankLevelPct}%` }}
                />
              </div>
            </div>

            <div className="space-y-2.5 text-xs pt-2 border-t border-[var(--agri-border)]">
              <div className="flex justify-between">
                <span className="text-[var(--agri-muted)]">{t('iot:waterTank.volume')}</span>
                <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                  {formatNumber(data.irrigation.currentVolumeLiters, lang, {
                    maximumFractionDigits: 0,
                  })}{' '}
                  / {formatNumber(data.irrigation.tankCapacityLiters, lang, { maximumFractionDigits: 0 })} L
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--agri-muted)]">{t('iot:waterTank.pumpState')}</span>
                <span className="font-medium text-[var(--agri-text)]">
                  {data.irrigation.pumpStatus === 'active'
                    ? t('common:status.active')
                    : t('common:status.idle')}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--agri-muted)]">{t('iot:waterTank.pressure')}</span>
                <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                  {formatNumber(data.irrigation.pumpPressureBar, lang)} bar
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--agri-muted)]">{t('iot:waterTank.waterPh')}</span>
                <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                  {formatNumber(data.irrigation.waterQualityPh, lang)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--agri-muted)]">{t('iot:waterTank.waterTds')}</span>
                <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                  {formatNumber(data.irrigation.waterQualityTdsPpm, lang)} ppm
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Soil Moisture & Air Humidity Chart */}
        <div className="lg:col-span-2 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6">
          <h2 className="text-base font-semibold text-[var(--agri-text)] mb-1">
            {t('dashboard:charts.historyTitle')}
          </h2>
          <p className="text-xs text-[var(--agri-muted)] mb-4">
            {t('dashboard:charts.historySubtitle')}
          </p>
          <TelemetryChart
            data={chartPoints}
            primaryLabel={t('dashboard:charts.moistureLegend')}
            primaryUnit="%"
            primaryColor="#4A8C55"
            secondaryLabel={t('dashboard:charts.humidityLegend')}
            secondaryUnit="%"
            secondaryColor="#438EBA"
            lang={lang}
          />
        </div>
      </div>

      {/* Node Configuration Modal */}
      {editingDevice && (
        <div
          className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
        >
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl max-w-md w-full p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-[var(--agri-text)]">
                {t('iot:configModal.title')} · {editingDevice.id}
              </h3>
              <button
                type="button"
                onClick={() => setEditingDevice(null)}
                className="p-1 text-[var(--agri-muted)] hover:text-[var(--agri-text)] cursor-pointer"
                aria-label={t('common:actions.close')}
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleSaveConfig} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-[var(--agri-text)] mb-1">
                  {t('iot:configModal.samplingInterval')}
                </label>
                <input
                  type="number"
                  min={10}
                  max={3600}
                  value={intervalInput}
                  onChange={(e) => setIntervalInput(e.target.value)}
                  className="w-full px-3 py-2 text-sm font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
                <p className="text-[11px] text-[var(--agri-muted)] mt-1">
                  {t('iot:configModal.intervalHelp')}
                </p>
              </div>

              <div>
                <label className="block text-xs font-medium text-[var(--agri-text)] mb-1">
                  {t('iot:configModal.moistureThreshold')}
                </label>
                <input
                  type="number"
                  min={15}
                  max={85}
                  value={thresholdInput}
                  onChange={(e) => setThresholdInput(e.target.value)}
                  className="w-full px-3 py-2 text-sm font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
                <p className="text-[11px] text-[var(--agri-muted)] mt-1">
                  {t('iot:configModal.thresholdHelp')}
                </p>
              </div>

              <div>
                <label className="block text-xs font-medium text-[var(--agri-text)] mb-1">
                  Dielectric Calibration Offset (% VWC, -10 to +10)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min={-10}
                  max={10}
                  value={offsetInput}
                  onChange={(e) => setOffsetInput(e.target.value)}
                  className="w-full px-3 py-2 text-sm font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
              </div>

              {configError && (
                <div className="text-xs text-[var(--agri-red)] font-medium">
                  {configError}
                </div>
              )}

              {configFeedback && (
                <div className="text-xs text-[var(--agri-leaf)] font-medium flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>{configFeedback}</span>
                </div>
              )}

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setEditingDevice(null)}
                  className="px-4 py-2 text-xs font-medium text-[var(--agri-text)] bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg cursor-pointer"
                >
                  {t('common:actions.cancel')}
                </button>
                <button
                  type="submit"
                  disabled={savingConfig}
                  className="px-4 py-2 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer disabled:opacity-50"
                >
                  {t('common:actions.save')}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
