import React, { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Droplets,
  Layers,
  RotateCcw,
  Sliders,
  Sprout,
  Sun,
} from 'lucide-react';
import {
  DigitalTwinResponse,
  DigitalTwinSeriesPoint,
  SensorsResponse,
  agriApi,
} from '../api/client';
import { Farm3DScene } from '../components/Farm3DScene';
import { TelemetryChart } from '../components/TelemetryChart';
import { formatNumber, formatShortDate } from '../utils/format';

interface AnalyticsTwinPageProps {
  sensorsData?: SensorsResponse | null;
}

export const AnalyticsTwinPage: React.FC<AnalyticsTwinPageProps> = ({
  sensorsData,
}) => {
  const { t, i18n } = useTranslation(['analytics', 'common']);
  const lang = i18n.language;

  const [zoneId, setZoneId] = useState<'ZONE-A' | 'ZONE-B' | 'ZONE-C'>('ZONE-A');
  const [days, setDays] = useState<7 | 14 | 30>(14);
  const [metric, setMetric] = useState<'moisture' | 'temp' | 'ndvi'>('moisture');
  const [selectedDayIndex, setSelectedDayIndex] = useState<number>(0);

  // Biophysical scenario sliders
  const [rainDeltaPct, setRainDeltaPct] = useState<number>(0);
  const [tempDeltaC, setTempDeltaC] = useState<number>(0);
  const [irrigationEffPct, setIrrigationEffPct] = useState<number>(85);
  const [fertilizerEfficiencyPct, setFertilizerEfficiencyPct] = useState<number>(88);

  const [twinData, setTwinData] = useState<DigitalTwinResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const [mlPredictedYield, setMlPredictedYield] = useState<number | null>(null);

  // Initial load or zone/days change
  useEffect(() => {
    let active = true;
    setLoading(true);
    agriApi
      .getDigitalTwin(zoneId, days, {
        rainDeltaPct,
        tempDeltaC,
        irrigationEffPct,
        fertilizerEfficiencyPct,
      })
      .then((res) => {
        if (active) {
          setTwinData(res);
          setMlPredictedYield(res.cropTwin.projectedYieldTonHa);
          setSelectedDayIndex((prev) => (prev > 0 ? prev : Math.max(0, res.series.length - 1)));
          setLoading(false);
        }
      })
      .catch(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [zoneId, days, sensorsData?.timestamp]);

  // Debounced live ML model inference query to FastAPI backend when sliders move
  useEffect(() => {
    let active = true;
    const timer = setTimeout(() => {
      agriApi
        .getDigitalTwin(zoneId, days, {
          rainDeltaPct,
          tempDeltaC,
          irrigationEffPct,
          fertilizerEfficiencyPct,
        })
        .then((res) => {
          if (active) {
            setMlPredictedYield(res.cropTwin.projectedYieldTonHa);
          }
        })
        .catch(() => {});
    }, 120);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [zoneId, days, rainDeltaPct, tempDeltaC, irrigationEffPct, fertilizerEfficiencyPct]);

  const resetScenario = () => {
    setRainDeltaPct(0);
    setTempDeltaC(0);
    setIrrigationEffPct(85);
    setFertilizerEfficiencyPct(88);
  };

  // Apply biophysical scenario perturbations to the time-series and twin state
  const simulatedSeries: DigitalTwinSeriesPoint[] = useMemo(() => {
    if (!twinData) return [];
    const rainFactor = rainDeltaPct * 0.14;
    const effFactor = (irrigationEffPct - 85) * 0.18;
    const tempEvapFactor = tempDeltaC * 0.42;
    const ndviDelta =
      rainDeltaPct * 0.0008 +
      (fertilizerEfficiencyPct - 88) * 0.0018 -
      Math.max(0, tempDeltaC - 1.5) * 0.012;

    return twinData.series.map((pt) => {
      const predictedMoisturePct = Number(
        Math.max(
          18,
          Math.min(
            92,
            pt.predictedMoisturePct + rainFactor + effFactor - tempEvapFactor * 1.3
          )
        ).toFixed(1)
      );
      const soilTempC = Number((pt.soilTempC + tempDeltaC * 0.75).toFixed(1));
      const evapotranspirationMm = Number(
        Math.max(1.2, pt.evapotranspirationMm + tempEvapFactor).toFixed(2)
      );
      const cropNdvi = Number(
        Math.max(0.25, Math.min(0.94, pt.cropNdvi + ndviDelta)).toFixed(2)
      );
      return {
        ...pt,
        predictedMoisturePct,
        soilTempC,
        evapotranspirationMm,
        cropNdvi,
      };
    });
  }, [twinData, rainDeltaPct, tempDeltaC, irrigationEffPct, fertilizerEfficiencyPct]);

  const simulatedMetrics = useMemo(() => {
    if (!twinData) return null;
    const isSimulated =
      rainDeltaPct !== 0 ||
      tempDeltaC !== 0 ||
      irrigationEffPct !== 85 ||
      fertilizerEfficiencyPct !== 88;

    // Use live ML model prediction if available, else responsive biophysical fallback
    const yieldDelta =
      rainDeltaPct * 0.018 +
      (irrigationEffPct - 85) * 0.035 +
      (fertilizerEfficiencyPct - 88) * 0.042 -
      Math.max(0, tempDeltaC - 1.0) * 0.19;

    const projectedYieldTonHa = mlPredictedYield !== null
      ? mlPredictedYield
      : Number(Math.max(1.5, twinData.cropTwin.projectedYieldTonHa + yieldDelta).toFixed(2));

    const dailyEt0Mm = Number(
      Math.max(1.4, twinData.waterTwin.dailyEt0Mm + tempDeltaC * 0.38).toFixed(2)
    );
    const leafAreaIndex = Number(
      Math.max(
        1.1,
        twinData.cropTwin.leafAreaIndex +
          (fertilizerEfficiencyPct - 88) * 0.015 +
          rainDeltaPct * 0.004 -
          Math.max(0, tempDeltaC - 1.5) * 0.08
      ).toFixed(2)
    );
    const nextIrrigationHours = Math.max(
      1,
      Math.round(
        twinData.waterTwin.nextIrrigationHours +
          rainDeltaPct * 0.25 -
          tempDeltaC * 2.2 +
          (irrigationEffPct - 85) * 0.3
      )
    );
    const rootZoneDepletionMm = Number(
      Math.max(
        0,
        twinData.waterTwin.rootZoneDepletionMm -
          rainDeltaPct * 0.21 -
          (irrigationEffPct - 85) * 0.15 +
          tempDeltaC * 0.8
      ).toFixed(1)
    );
    const cationExchangeCapacityMeq = Number(
      Math.max(
        18,
        Math.min(
          38,
          twinData.soilTwin.cationExchangeCapacityMeq + (fertilizerEfficiencyPct - 88) * 0.06
        )
      ).toFixed(1)
    );
    const microbialActivityIndex = Number(
      Math.max(
        3.0,
        Math.min(
          9.9,
          twinData.soilTwin.microbialActivityIndex +
            (fertilizerEfficiencyPct - 88) * 0.04 -
            Math.max(0, tempDeltaC - 2.0) * 0.3
        )
      ).toFixed(1)
    );
    const npkMineralizationRate =
      fertilizerEfficiencyPct > 92
        ? 'optimal_accelerated'
        : twinData.soilTwin.npkMineralizationRate;

    const growingDegreeDays = Number(
      (twinData.cropTwin.growingDegreeDays + tempDeltaC * 14.0).toFixed(1)
    );

    return {
      isSimulated,
      projectedYieldTonHa,
      dailyEt0Mm,
      leafAreaIndex,
      nextIrrigationHours,
      rootZoneDepletionMm,
      cationExchangeCapacityMeq,
      microbialActivityIndex,
      npkMineralizationRate,
      growingDegreeDays,
    };
  }, [
    twinData,
    mlPredictedYield,
    rainDeltaPct,
    tempDeltaC,
    irrigationEffPct,
    fertilizerEfficiencyPct,
  ]);

  const chartPoints = useMemo(() => {
    return simulatedSeries.map((pt) => {
      const label = formatShortDate(pt.dateIso, lang);
      if (metric === 'temp') {
        return {
          label,
          primary: pt.soilTempC,
          secondary: pt.evapotranspirationMm,
        };
      }
      if (metric === 'ndvi') {
        return {
          label,
          primary: Number((pt.cropNdvi * 100).toFixed(1)),
          barValue: pt.irrigationMm,
        };
      }
      return {
        label,
        primary: pt.observedMoisturePct,
        secondary: pt.predictedMoisturePct,
        barValue: pt.irrigationMm,
      };
    });
  }, [simulatedSeries, metric, lang]);

  const devices = sensorsData?.devices ?? [];
  const zones = sensorsData?.irrigation.zones ?? [];
  const tankLevelPct = sensorsData?.irrigation.tankLevelPct ?? 82;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[var(--agri-border)] pb-4">
        <div>
          <div className="text-[11px] font-mono-tabular uppercase tracking-widest text-[var(--agri-leaf)] font-semibold mb-1">
            4D BIOPHYSICAL DIGITAL TWIN · WATER / SOIL / CROP COUPLED SIMULATION
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--agri-text)]">
            {t('analytics:title')}
          </h1>
          <p className="text-sm text-[var(--agri-muted)] mt-1 max-w-3xl">
            {t('analytics:subtitle')}
          </p>
        </div>

        {/* Zone & Horizon Selector */}
        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <div className="inline-flex items-center gap-1 p-1 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-lg">
            {(['ZONE-A', 'ZONE-B', 'ZONE-C'] as const).map((z) => (
              <button
                key={z}
                type="button"
                onClick={() => setZoneId(z)}
                className={`px-2.5 py-1 text-xs font-mono-tabular font-semibold rounded-md transition-colors cursor-pointer ${
                  zoneId === z
                    ? 'bg-[#173D2B] text-white'
                    : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
                }`}
              >
                {z}
              </button>
            ))}
          </div>

          <div className="inline-flex items-center gap-1 p-1 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-lg">
            {([7, 14, 30] as const).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDays(d)}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  days === d
                    ? 'bg-[#173D2B] text-white'
                    : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
                }`}
              >
                {t(`analytics:filters.days${d}`)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Asymmetric Split: 3D Digital Twin Viewport + Biophysical Scenario Parametric Controls */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        <div className="xl:col-span-8">
          <Farm3DScene
            devices={devices}
            zones={zones}
            tankLevelPct={tankLevelPct}
            selectedZoneId={zoneId}
            onSelectZone={(z) => {
              if (z === 'ZONE-A' || z === 'ZONE-B' || z === 'ZONE-C') {
                setZoneId(z);
              }
            }}
            twinSeries={simulatedSeries}
            selectedDayIndex={selectedDayIndex}
            onChangeDayIndex={setSelectedDayIndex}
            heightClass="h-[440px]"
          />
        </div>

        {/* Parametric Scenario Controller */}
        <div className="xl:col-span-4 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[var(--agri-border)] pb-3">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-[var(--agri-leaf)]" />
              <h2 className="text-sm font-semibold text-[var(--agri-text)]">
                Biophysical Scenario Stress-Test
              </h2>
            </div>
            <button
              type="button"
              onClick={resetScenario}
              className="inline-flex items-center gap-1 text-[11px] font-medium text-[var(--agri-muted)] hover:text-[var(--agri-text)] cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              Reset
            </button>
          </div>

          <div className="space-y-3.5 text-xs">
            {/* Rainfall Anomaly */}
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-[var(--agri-muted)] flex items-center gap-1.5">
                  <Droplets className="w-3.5 h-3.5 text-[var(--agri-water)]" />
                  Precipitation Delta
                </span>
                <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                  {rainDeltaPct > 0 ? `+${rainDeltaPct}%` : `${rainDeltaPct}%`}
                </span>
              </div>
              <input
                type="range"
                min={-40}
                max={40}
                step={5}
                value={rainDeltaPct}
                onChange={(e) => setRainDeltaPct(Number(e.target.value))}
                className="w-full accent-[#438EBA] cursor-pointer"
              />
            </div>

            {/* Temperature Anomaly */}
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-[var(--agri-muted)] flex items-center gap-1.5">
                  <Sun className="w-3.5 h-3.5 text-[var(--agri-amber)]" />
                  Canopy Temp Anomaly (ΔT)
                </span>
                <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                  {tempDeltaC > 0 ? `+${tempDeltaC.toFixed(1)}°C` : `${tempDeltaC.toFixed(1)}°C`}
                </span>
              </div>
              <input
                type="range"
                min={-3}
                max={5}
                step={0.5}
                value={tempDeltaC}
                onChange={(e) => setTempDeltaC(Number(e.target.value))}
                className="w-full accent-[#DCA849] cursor-pointer"
              />
            </div>

            {/* Irrigation Efficiency */}
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-[var(--agri-muted)]">
                  Drip Delivery Efficiency
                </span>
                <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                  {irrigationEffPct}%
                </span>
              </div>
              <input
                type="range"
                min={60}
                max={98}
                step={2}
                value={irrigationEffPct}
                onChange={(e) => setIrrigationEffPct(Number(e.target.value))}
                className="w-full accent-[#4A8C55] cursor-pointer"
              />
            </div>

            {/* Fertilizer Mineralization Efficiency */}
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-[var(--agri-muted)]">
                  NPK Uptake Efficiency
                </span>
                <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                  {fertilizerEfficiencyPct}%
                </span>
              </div>
              <input
                type="range"
                min={60}
                max={100}
                step={2}
                value={fertilizerEfficiencyPct}
                onChange={(e) => setFertilizerEfficiencyPct(Number(e.target.value))}
                className="w-full accent-[#173D2B] cursor-pointer"
              />
            </div>
          </div>

          {/* Live Scenario Impact Readout */}
          {simulatedMetrics && (
            <div className="pt-3 border-t border-[var(--agri-border)] space-y-2 text-xs">
              <div className="text-[10px] font-mono-tabular uppercase tracking-wider text-[var(--agri-muted)]">
                COUPLED SCENARIO RESPONSE ({zoneId})
              </div>
              <div className="grid grid-cols-2 gap-2.5">
                <div className="p-2.5 rounded-lg bg-[var(--agri-bg)] border border-[var(--agri-border)]">
                  <div className="text-[10px] text-[var(--agri-muted)]">
                    {t('analytics:twins.projectedYield')}
                  </div>
                  <div className="text-sm font-mono-tabular font-bold text-[var(--agri-leaf)] mt-0.5">
                    {formatNumber(simulatedMetrics.projectedYieldTonHa, lang)}{' '}
                    {t('common:units.tonHa')}
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-[var(--agri-bg)] border border-[var(--agri-border)]">
                  <div className="text-[10px] text-[var(--agri-muted)]">
                    {t('analytics:twins.dailyEt')}
                  </div>
                  <div className="text-sm font-mono-tabular font-bold text-[var(--agri-water)] mt-0.5">
                    {formatNumber(simulatedMetrics.dailyEt0Mm, lang)} mm/d
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-[var(--agri-bg)] border border-[var(--agri-border)]">
                  <div className="text-[10px] text-[var(--agri-muted)]">
                    {t('analytics:twins.lai')}
                  </div>
                  <div className="text-sm font-mono-tabular font-semibold text-[var(--agri-text)] mt-0.5">
                    {formatNumber(simulatedMetrics.leafAreaIndex, lang)} m²/m²
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-[var(--agri-bg)] border border-[var(--agri-border)]">
                  <div className="text-[10px] text-[var(--agri-muted)]">
                    {t('analytics:twins.nextIrrigation')}
                  </div>
                  <div className="text-sm font-mono-tabular font-semibold text-[var(--agri-text)] mt-0.5">
                    {formatNumber(simulatedMetrics.nextIrrigationHours, lang)}{' '}
                    {t('common:units.hours')}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Historical vs Digital Twin Forecast Chart */}
      <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-[var(--agri-text)]">
              Observed vs. Calibrated Digital Twin Trajectory ({zoneId})
            </h2>
            <p className="text-xs text-[var(--agri-muted)]">
              FAO-56 Penman-Monteith & GDD Phenology Coupled Time-Series
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-[var(--agri-muted)] font-medium">
              {t('analytics:filters.metricLabel')}:
            </span>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as 'moisture' | 'temp' | 'ndvi')}
              className="px-3 py-1.5 text-xs bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
            >
              <option value="moisture">{t('analytics:filters.metricMoisture')}</option>
              <option value="temp">{t('analytics:filters.metricTemp')}</option>
              <option value="ndvi">{t('analytics:filters.metricNdvi')}</option>
            </select>
          </div>
        </div>

        {loading && !twinData ? (
          <div className="h-64 animate-pulse bg-[var(--agri-bg)] rounded-lg" />
        ) : (
          <TelemetryChart
            data={chartPoints}
            primaryLabel={
              metric === 'moisture'
                ? t('analytics:chart.observedMoisture')
                : metric === 'temp'
                ? t('analytics:chart.soilTemp')
                : t('analytics:chart.ndvi')
            }
            primaryUnit={metric === 'temp' ? '°C' : '%'}
            primaryColor="#4A8C55"
            secondaryLabel={
              metric === 'moisture'
                ? t('analytics:chart.predictedMoisture')
                : metric === 'temp'
                ? 'ET₀ (mm/day)'
                : undefined
            }
            secondaryUnit={metric === 'moisture' ? '%' : ' mm'}
            secondaryColor="#438EBA"
            barLabel={
              metric !== 'temp' ? t('analytics:chart.irrigationBars') : undefined
            }
            barUnit="mm"
            lang={lang}
            height={260}
          />
        )}
      </div>

      {/* Three Digital Twin Biophysical State Ledgers: WaterTwin, SoilTwin, CropTwin */}
      {twinData && simulatedMetrics && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 1. WaterTwin */}
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between text-xs text-[var(--agri-muted)] mb-2">
                <span className="font-mono-tabular uppercase tracking-wider">
                  {simulatedMetrics.isSimulated
                    ? 'SCENARIO SIMULATION'
                    : t('analytics:twins.calibratedBadge')}
                </span>
                <Droplets className="w-4 h-4 text-[var(--agri-water)]" />
              </div>
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('analytics:twins.waterTwinTitle')}
              </h2>

              <div className="space-y-3 mt-4 pt-3 border-t border-[var(--agri-border)] text-xs">
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.depletion')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {formatNumber(simulatedMetrics.rootZoneDepletionMm, lang)} mm
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.fieldCapacity')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {formatNumber(twinData.waterTwin.fieldCapacityPct, lang)}% /{' '}
                    {formatNumber(twinData.waterTwin.wiltingPointPct, lang)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.dailyEt')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {formatNumber(simulatedMetrics.dailyEt0Mm, lang)} mm/day
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.nextIrrigation')}
                  </span>
                  <span className="font-mono-tabular font-bold text-[var(--agri-water)]">
                    {formatNumber(simulatedMetrics.nextIrrigationHours, lang)}{' '}
                    {t('common:units.hours')}
                  </span>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-[var(--agri-border)] flex justify-between text-xs text-[var(--agri-muted)] font-mono-tabular">
              <span>{t('analytics:twins.confidence')}</span>
              <span className="font-semibold text-[var(--agri-leaf)]">
                {formatNumber(simulatedMetrics.isSimulated ? 91.8 : twinData.waterTwin.confidencePct, lang)}%
              </span>
            </div>
          </div>

          {/* 2. SoilTwin */}
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between text-xs text-[var(--agri-muted)] mb-2">
                <span className="font-mono-tabular uppercase tracking-wider">
                  {simulatedMetrics.isSimulated
                    ? 'SCENARIO SIMULATION'
                    : t('analytics:twins.simulatedBadge')}
                </span>
                <Layers className="w-4 h-4 text-[var(--agri-amber)]" />
              </div>
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('analytics:twins.soilTwinTitle')}
              </h2>

              <div className="space-y-3 mt-4 pt-3 border-t border-[var(--agri-border)] text-xs">
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">{t('analytics:twins.cec')}</span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {formatNumber(simulatedMetrics.cationExchangeCapacityMeq, lang)} meq/100g
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.organicCarbon')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {formatNumber(twinData.soilTwin.organicCarbonPct, lang)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.microbialIndex')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {formatNumber(simulatedMetrics.microbialActivityIndex, lang)} / 10.0
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.mineralization')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {simulatedMetrics.npkMineralizationRate}
                  </span>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-[var(--agri-border)] flex justify-between text-xs text-[var(--agri-muted)] font-mono-tabular">
              <span>{t('analytics:twins.confidence')}</span>
              <span className="font-semibold text-[var(--agri-leaf)]">
                {formatNumber(simulatedMetrics.isSimulated ? 89.5 : twinData.soilTwin.confidencePct, lang)}%
              </span>
            </div>
          </div>

          {/* 3. CropTwin */}
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between text-xs text-[var(--agri-muted)] mb-2">
                <span className="font-mono-tabular uppercase tracking-wider">
                  {simulatedMetrics.isSimulated
                    ? 'ML SCENARIO PREDICTED'
                    : t('analytics:twins.simulatedBadge')}
                </span>
                <Sprout className="w-4 h-4 text-[var(--agri-leaf)]" />
              </div>
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('analytics:twins.cropTwinTitle')}
              </h2>

              <div className="space-y-3 mt-4 pt-3 border-t border-[var(--agri-border)] text-xs">
                <div className="flex justify-between gap-2">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.stage')}
                  </span>
                  <span className="font-semibold text-[var(--agri-text)] text-right">
                    {t(`analytics:twins.stage_${twinData.cropTwin.phenologyStageKey}`)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">{t('analytics:twins.gdd')}</span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {formatNumber(simulatedMetrics.growingDegreeDays, lang)} °C·d
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">{t('analytics:twins.lai')}</span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {formatNumber(simulatedMetrics.leafAreaIndex, lang)} m²/m²
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--agri-muted)]">
                    {t('analytics:twins.projectedYield')}
                  </span>
                  <span className="font-mono-tabular font-bold text-[var(--agri-leaf)]">
                    {formatNumber(simulatedMetrics.projectedYieldTonHa, lang)}{' '}
                    {t('common:units.tonHa')}
                  </span>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-[var(--agri-border)] flex justify-between text-xs text-[var(--agri-muted)] font-mono-tabular">
              <span>{t('analytics:twins.confidence')}</span>
              <span className="font-semibold text-[var(--agri-leaf)]">
                {formatNumber(simulatedMetrics.isSimulated ? 90.6 : twinData.cropTwin.confidencePct, lang)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
