import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertTriangle,
  ArrowRight,
  CloudSun,
  Cpu,
  Droplets,
  Gauge,
  Sliders,
  Sprout,
  Thermometer,
  Wind,
} from 'lucide-react';
import { SensorsResponse } from '../api/client';
import { Farm3DScene } from '../components/Farm3DScene';
import { TelemetryChart } from '../components/TelemetryChart';
import { formatDateTime, formatNumber } from '../utils/format';

interface DashboardPageProps {
  data: SensorsResponse | null;
  loading: boolean;
  onNavigate: (
    page: 'iot' | 'irrigation' | 'disease' | 'analytics' | 'assistant' | 'operations'
  ) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  data,
  loading,
  onNavigate,
}) => {
  const { t, i18n } = useTranslation(['dashboard', 'common']);
  const lang = i18n.language;

  const [focusedZoneId, setFocusedZoneId] = useState<string>('ZONE-A');

  if (loading && !data) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-64 rounded bg-[var(--agri-sage)] animate-pulse" />
        <div className="h-96 rounded-xl bg-[var(--agri-card)] border border-[var(--agri-border)] animate-pulse" />
      </div>
    );
  }

  if (!data) return null;

  const { summary, history24h, irrigation, devices } = data;

  const focusedDevice =
    devices.find((d) => d.zoneId === focusedZoneId) || devices[0];

  const chartPoints = history24h.map((pt) => ({
    label: pt.hourLabel,
    primary: pt.soilMoisturePct,
    secondary: pt.soilTempC,
  }));

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[var(--agri-border)] pb-4">
        <div>
          <div className="text-xs font-mono-tabular text-[var(--agri-leaf)] font-semibold">
            CAUVERY DELTA RESEARCH FARM · 6.8 HA SPATIAL COMMAND
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--agri-text)] mt-1">
            {t('dashboard:title')}
          </h1>
          <p className="text-sm text-[var(--agri-muted)] mt-1 max-w-3xl">
            {t('dashboard:subtitle')}
          </p>
        </div>
        <div className="text-xs text-[var(--agri-muted)] font-mono-tabular">
          <span className="text-[var(--agri-leaf)] font-semibold">
            {t('common:status.live')}
          </span>
          <span className="mx-1.5">·</span>
          <span>{formatDateTime(data.timestamp, lang)}</span>
        </div>
      </div>

      {/* Asymmetric Spatial Hero: 3D Farm Terrain (8 cols) + Live Zone & Micro-Climate Ledger (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-8">
          <Farm3DScene
            devices={devices}
            zones={irrigation.zones}
            tankLevelPct={irrigation.tankLevelPct}
            selectedZoneId={focusedZoneId}
            onSelectZone={(zId) => setFocusedZoneId(zId)}
            heightClass="h-[430px]"
          />
        </div>

        {/* Right Column: Focused Block Telemetry & Weather Station Ledger */}
        <div className="lg:col-span-4 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col justify-between space-y-5">
          <div>
            <div className="flex items-center justify-between text-xs font-mono-tabular text-[var(--agri-muted)] pb-3 border-b border-[var(--agri-border)]">
              <span>SELECTED SPATIAL SECTOR</span>
              <span className="text-[var(--agri-leaf)] font-semibold">
                {focusedDevice.id}
              </span>
            </div>

            <h2 className="text-base font-bold text-[var(--agri-text)] mt-3">
              {focusedDevice.zoneName}
            </h2>
            <p className="text-xs text-[var(--agri-muted)] mt-0.5">
              {focusedDevice.cropType} · {focusedDevice.protocol}
            </p>

            {/* Focused Block Readings */}
            <div className="grid grid-cols-2 gap-3 mt-4 py-3 border-y border-[var(--agri-border)] text-xs">
              <div>
                <span className="text-[var(--agri-muted)] block">
                  {t('dashboard:kpi.soilMoisture')}
                </span>
                <span className="font-mono-tabular text-lg font-bold text-[var(--agri-text)]">
                  {formatNumber(focusedDevice.readings.soilMoisturePct, lang)}%
                </span>
              </div>
              <div>
                <span className="text-[var(--agri-muted)] block">
                  {t('dashboard:kpi.soilTemperature')}
                </span>
                <span className="font-mono-tabular text-lg font-bold text-[var(--agri-text)]">
                  {formatNumber(focusedDevice.readings.soilTempC, lang)}°C
                </span>
              </div>
              <div>
                <span className="text-[var(--agri-muted)] block">NPK (mg/kg)</span>
                <span className="font-mono-tabular text-sm font-semibold text-[var(--agri-text)]">
                  {focusedDevice.readings.nitrogenMgKg}-
                  {focusedDevice.readings.phosphorusMgKg}-
                  {focusedDevice.readings.potassiumMgKg}
                </span>
              </div>
              <div>
                <span className="text-[var(--agri-muted)] block">pH · EC</span>
                <span className="font-mono-tabular text-sm font-semibold text-[var(--agri-text)]">
                  {formatNumber(focusedDevice.readings.soilPh, lang)} ·{' '}
                  {formatNumber(focusedDevice.readings.ecDsM, lang)} dS/m
                </span>
              </div>
            </div>

            {/* Micro-Climate Station Readout */}
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-[var(--agri-text)]">
                  {t('dashboard:weather.title')}
                </span>
                <CloudSun className="w-4 h-4 text-[var(--agri-amber)]" />
              </div>
              <p className="text-xs text-[var(--agri-muted)] mb-3">
                {t('dashboard:weather.condition_PARTLY_CLOUDY_HUMID')}
              </p>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('dashboard:weather.windSpeed')}
                  </span>
                  <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                    {formatNumber(summary.weatherSummary.windSpeedKmh, lang)} km/h
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('dashboard:weather.rainProbability')}
                  </span>
                  <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                    {formatNumber(summary.weatherSummary.rainProbabilityPct, lang)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('dashboard:weather.solarRadiation')}
                  </span>
                  <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                    {formatNumber(summary.weatherSummary.solarRadiationWm2, lang)} W/m²
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-[var(--agri-border)] flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => onNavigate('analytics')}
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-[var(--agri-leaf)] hover:underline cursor-pointer"
            >
              <span>4D Digital Twin</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => onNavigate('operations')}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-[var(--agri-muted)] hover:text-[var(--agri-text)] cursor-pointer"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>{t('common:nav.operations')}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Editorial 6-Column Hairline Telemetry Ledger (Replaces repetitive card clutter) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl divide-y sm:divide-y-0 sm:divide-x divide-[var(--agri-border)]">
        {/* 1. Connected Sensors */}
        <div className="p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[var(--agri-muted)]">
            <span>{t('dashboard:kpi.connectedSensors')}</span>
            <Cpu className="w-3.5 h-3.5 text-[var(--agri-leaf)]" />
          </div>
          <div className="my-2 font-mono-tabular text-2xl font-bold text-[var(--agri-text)]">
            {formatNumber(summary.onlineSensors, lang)} / {formatNumber(summary.totalSensors, lang)}
          </div>
          <div className="text-[11px] text-[var(--agri-muted)]">
            {summary.staleSensors > 0 ? (
              <span className="text-[var(--agri-amber)] font-medium">
                {formatNumber(summary.staleSensors, lang)} {t('common:status.stale')}
              </span>
            ) : (
              t('common:status.online')
            )}
          </div>
        </div>

        {/* 2. Mean Soil Moisture */}
        <div className="p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[var(--agri-muted)]">
            <span>{t('dashboard:kpi.soilMoisture')}</span>
            <Droplets className="w-3.5 h-3.5 text-[var(--agri-water)]" />
          </div>
          <div className="my-2 font-mono-tabular text-2xl font-bold text-[var(--agri-text)]">
            {formatNumber(summary.avgSoilMoisturePct, lang)}%
          </div>
          <div className="text-[11px] text-[var(--agri-muted)]">
            {t('dashboard:kpi.optimalRange')}
          </div>
        </div>

        {/* 3. Mean Soil Temperature */}
        <div className="p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[var(--agri-muted)]">
            <span>{t('dashboard:kpi.soilTemperature')}</span>
            <Thermometer className="w-3.5 h-3.5 text-[var(--agri-amber)]" />
          </div>
          <div className="my-2 font-mono-tabular text-2xl font-bold text-[var(--agri-text)]">
            {formatNumber(summary.avgSoilTempC, lang)}°C
          </div>
          <div className="text-[11px] text-[var(--agri-muted)]">
            {t('dashboard:kpi.vaporBalance', {
              temp: formatNumber(summary.avgAirTempC, lang),
            })}
          </div>
        </div>

        {/* 4. Air Humidity */}
        <div className="p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[var(--agri-muted)]">
            <span>{t('dashboard:kpi.airHumidity')}</span>
            <Wind className="w-3.5 h-3.5 text-[var(--agri-water)]" />
          </div>
          <div className="my-2 font-mono-tabular text-2xl font-bold text-[var(--agri-text)]">
            {formatNumber(summary.avgAirHumidityPct, lang)}%
          </div>
          <div className="text-[11px] text-[var(--agri-muted)]">RPI-GW-01 Hygrometer</div>
        </div>

        {/* 5. Irrigation Status */}
        <div className="p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[var(--agri-muted)]">
            <span>{t('dashboard:kpi.irrigationStatus')}</span>
            <Gauge className="w-3.5 h-3.5 text-[var(--agri-water)]" />
          </div>
          <div className="my-2 text-lg font-bold text-[var(--agri-text)]">
            {irrigation.pumpStatus === 'active'
              ? t('common:status.active')
              : t('common:status.idle')}
          </div>
          <div className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
            Tank {formatNumber(irrigation.tankLevelPct, lang)}% ·{' '}
            {formatNumber(irrigation.pumpPressureBar, lang)} bar
          </div>
        </div>

        {/* 6. Crop Health Summary */}
        <div className="p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[var(--agri-muted)]">
            <span>{t('dashboard:kpi.cropHealth')}</span>
            <Sprout className="w-3.5 h-3.5 text-[var(--agri-leaf)]" />
          </div>
          <div className="my-2 text-lg font-bold text-[var(--agri-text)]">
            {summary.cropHealthSummary.overallStatus === 'healthy'
              ? t('common:status.healthy')
              : t('common:status.attention_needed')}
          </div>
          <div className="text-[11px] text-[var(--agri-muted)]">
            {t('dashboard:kpi.plotsSummary', {
              healthy: formatNumber(summary.cropHealthSummary.healthyPlotsCount, lang),
              attention: formatNumber(summary.cropHealthSummary.attentionPlotsCount, lang),
            })}
          </div>
        </div>
      </div>

      {/* 24-Hour Sensor History Chart + Priority Agronomic Recommendations & Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-7 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6">
          <div className="mb-4">
            <h2 className="text-base font-semibold text-[var(--agri-text)]">
              {t('dashboard:charts.historyTitle')}
            </h2>
            <p className="text-xs text-[var(--agri-muted)] mt-0.5">
              {t('dashboard:charts.historySubtitle')}
            </p>
          </div>
          <TelemetryChart
            data={chartPoints}
            primaryLabel={t('dashboard:charts.moistureLegend')}
            primaryUnit="%"
            primaryColor="#4A8C55"
            secondaryLabel={t('dashboard:charts.tempLegend')}
            secondaryUnit="°C"
            secondaryColor="#438EBA"
            lang={lang}
          />
        </div>

        {/* Priority Agronomic Recommendations & Alerts */}
        <div className="lg:col-span-5 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col justify-between space-y-5">
          <div>
            <h2 className="text-base font-semibold text-[var(--agri-text)] mb-3">
              {t('dashboard:recommendations.title')}
            </h2>
            <div className="divide-y divide-[var(--agri-border)] text-xs">
              <div className="py-2.5 first:pt-0">
                <div className="font-semibold text-[var(--agri-text)]">
                  01. {t('dashboard:recommendations.rec1Title')}
                </div>
                <p className="text-[var(--agri-muted)] mt-0.5">
                  {t('dashboard:recommendations.rec1Desc')}
                </p>
              </div>
              <div className="py-2.5">
                <div className="font-semibold text-[var(--agri-text)]">
                  02. {t('dashboard:recommendations.rec2Title')}
                </div>
                <p className="text-[var(--agri-muted)] mt-0.5">
                  {t('dashboard:recommendations.rec2Desc')}
                </p>
              </div>
              <div className="py-2.5 last:pb-0">
                <div className="font-semibold text-[var(--agri-text)]">
                  03. {t('dashboard:recommendations.rec3Title')}
                </div>
                <p className="text-[var(--agri-muted)] mt-0.5">
                  {t('dashboard:recommendations.rec3Desc')}
                </p>
              </div>
            </div>

            {/* Active Alert Stream */}
            <div className="mt-4 pt-3 border-t border-[var(--agri-border)]">
              <div className="text-xs font-semibold text-[var(--agri-text)] mb-2">
                {t('dashboard:alerts.title')}
              </div>
              <div className="space-y-2">
                {irrigation.recentEvents.slice(0, 2).map((evt) => {
                  const isWarn = evt.type === 'warning' || evt.type === 'stale';
                  return (
                    <div
                      key={evt.id}
                      className="flex items-start gap-2 text-xs bg-[var(--agri-bg)] px-3 py-2 rounded-lg border border-[var(--agri-border)]"
                    >
                      <AlertTriangle
                        className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${
                          isWarn ? 'text-[var(--agri-amber)]' : 'text-[var(--agri-leaf)]'
                        }`}
                      />
                      <div className="min-w-0">
                        <div className="font-medium text-[var(--agri-text)] truncate">
                          {t(`dashboard:alerts.${evt.code}`, { defaultValue: evt.code })}
                        </div>
                        <div className="text-[11px] font-mono-tabular text-[var(--agri-muted)]">
                          {evt.zoneId} · {evt.value}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 pt-3 border-t border-[var(--agri-border)]">
            <button
              type="button"
              onClick={() => onNavigate('irrigation')}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer"
            >
              <span>{t('dashboard:recommendations.goToIrrigation')}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => onNavigate('iot')}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-[var(--agri-text)] bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg cursor-pointer"
            >
              <span>{t('dashboard:recommendations.goToIot')}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
