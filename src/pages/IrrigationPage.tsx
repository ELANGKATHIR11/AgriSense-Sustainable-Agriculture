import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertTriangle,
  CheckCircle2,
  Droplets,
  FlaskConical,
  Play,
  Sprout,
  Square,
} from 'lucide-react';
import {
  CropRecommendationItem,
  FertilizerResponse,
  IrrigationZoneState,
  SensorsResponse,
  agriApi,
} from '../api/client';
import { RootZone3DProfile } from '../components/RootZone3DProfile';
import { CropDirectory200 } from '../components/CropDirectory200';
import { CropDirectoryItem } from '../data/cropsData';
import { formatDateTime, formatNumber } from '../utils/format';

interface IrrigationPageProps {
  data: SensorsResponse | null;
  onRefresh: () => Promise<void>;
}

export const IrrigationPage: React.FC<IrrigationPageProps> = ({ data, onRefresh }) => {
  const { t, i18n } = useTranslation(['irrigation', 'common']);
  const lang = i18n.language;

  const [activeTab, setActiveTab] = useState<'irrigation' | 'crop' | 'fertilizer'>('irrigation');
  const [selectedZoneId, setSelectedZoneId] = useState<string>('ZONE-B');

  // Confirmation modal state for consequential irrigation valve actions
  const [pendingAction, setPendingAction] = useState<{
    zone: IrrigationZoneState;
    action: 'start' | 'stop';
  } | null>(null);
  const [actuating, setActuating] = useState(false);
  const [confirmationBanner, setConfirmationBanner] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // Crop Recommendation Form State
  const [cropInputs, setCropInputs] = useState({
    nitrogen: '118',
    phosphorus: '46',
    potassium: '62',
    temperature: '30.8',
    humidity: '74.0',
    ph: '6.5',
    rainfall: '185',
  });
  const [cropResults, setCropResults] = useState<CropRecommendationItem[] | null>(null);
  const [cropLoading, setCropLoading] = useState(false);
  const [cropError, setCropError] = useState<string | null>(null);

  // Fertilizer Recommendation Form State
  const [fertInputs, setFertInputs] = useState({
    cropType: 'paddy',
    nitrogen: '94',
    phosphorus: '42',
    potassium: '58',
    soilPh: '6.3',
    soilMoisture: '33.8',
  });
  const [fertResult, setFertResult] = useState<FertilizerResponse | null>(null);
  const [fertLoading, setFertLoading] = useState(false);
  const [fertError, setFertError] = useState<string | null>(null);

  const handleConfirmIrrigation = async () => {
    if (!pendingAction) return;
    setActuating(true);
    setActionError(null);
    try {
      const res = await agriApi.controlIrrigation({
        zoneId: pendingAction.zone.zoneId,
        action: pendingAction.action,
        durationMin: pendingAction.zone.recommendedDurationMin,
      });
      if (res.confirmed) {
        setConfirmationBanner(
          t('irrigation:zones.backendConfirmed', { zone: pendingAction.zone.zoneName })
        );
        await onRefresh();
      }
      setPendingAction(null);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Irrigation command failed');
    } finally {
      setActuating(false);
    }
  };

  const handleToggleAuto = async (zone: IrrigationZoneState) => {
    try {
      await agriApi.controlIrrigation({
        zoneId: zone.zoneId,
        action: 'toggle_auto',
      });
      await onRefresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Unable to toggle schedule');
    }
  };

  const loadLiveBlockATelemetry = () => {
    const nodeA = data?.devices[0];
    if (!nodeA) return;
    setCropInputs({
      nitrogen: String(nodeA.readings.nitrogenMgKg),
      phosphorus: String(nodeA.readings.phosphorusMgKg),
      potassium: String(nodeA.readings.potassiumMgKg),
      temperature: String(nodeA.readings.airTempC),
      humidity: String(nodeA.readings.airHumidityPct),
      ph: String(nodeA.readings.soilPh),
      rainfall: '190',
    });
  };

  const handleSelectCropFromDirectory = (crop: CropDirectoryItem) => {
    const tempNum = crop.temperatureRangeC.split('-')[0].replace('°C', '').trim() || '25';
    const rainNum = crop.rainfallRangeMm.split('-')[0].replace('mm', '').trim() || '750';
    const phNum = crop.optimalPh.split('-')[0].trim() || '6.5';
    setCropInputs({
      nitrogen: String(crop.optN),
      phosphorus: String(crop.optP),
      potassium: String(crop.optK),
      ph: phNum,
      temperature: tempNum,
      humidity: '68',
      rainfall: rainNum,
    });
    setConfirmationBanner(`Loaded agronomic requirements for ${crop.name} (${crop.category}) into AI Recommendation Model.`);
    window.scrollTo({ top: 120, behavior: 'smooth' });
  };

  const handleRunCropModel = async (e: React.FormEvent) => {
    e.preventDefault();
    setCropLoading(true);
    setCropError(null);
    try {
      const res = await agriApi.recommendCrops({
        nitrogen: Number(cropInputs.nitrogen),
        phosphorus: Number(cropInputs.phosphorus),
        potassium: Number(cropInputs.potassium),
        temperature: Number(cropInputs.temperature),
        humidity: Number(cropInputs.humidity),
        ph: Number(cropInputs.ph),
        rainfall: Number(cropInputs.rainfall),
      });
      setCropResults(res.recommendations);
    } catch (err) {
      setCropError(err instanceof Error ? err.message : 'Failed to compute crop recommendations');
    } finally {
      setCropLoading(false);
    }
  };

  const handleRunFertilizerModel = async (e: React.FormEvent) => {
    e.preventDefault();
    setFertLoading(true);
    setFertError(null);
    try {
      const res = await agriApi.recommendFertilizer({
        cropType: fertInputs.cropType,
        nitrogen: Number(fertInputs.nitrogen),
        phosphorus: Number(fertInputs.phosphorus),
        potassium: Number(fertInputs.potassium),
        soilPh: Number(fertInputs.soilPh),
        soilMoisture: Number(fertInputs.soilMoisture),
      });
      setFertResult(res);
    } catch (err) {
      setFertError(err instanceof Error ? err.message : 'Failed to compute fertilizer dosage');
    } finally {
      setFertLoading(false);
    }
  };

  const selectedZone =
    data?.irrigation.zones.find((z) => z.zoneId === selectedZoneId) ||
    data?.irrigation.zones[0];
  const selectedDevice =
    data?.devices.find((d) => d.zoneId === selectedZone?.zoneId) || data?.devices[0];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-[var(--agri-border)] pb-4">
        <div className="text-xs font-mono-tabular text-[var(--agri-leaf)] font-semibold">
          HYDROLOGICAL ACTUATION &amp; NPK DECISION ENGINE · PAGE 03
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-[var(--agri-text)] mt-1">
          {t('irrigation:title')}
        </h1>
        <p className="text-sm text-[var(--agri-muted)] mt-1 max-w-3xl">
          {t('irrigation:subtitle')}
        </p>
      </div>

      {/* Internal Section Tabs */}
      <div className="inline-flex flex-wrap items-center gap-1 p-1 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-lg">
        <button
          type="button"
          onClick={() => setActiveTab('irrigation')}
          className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            activeTab === 'irrigation'
              ? 'bg-[#173D2B] text-white'
              : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
          }`}
        >
          <Droplets className="w-3.5 h-3.5" />
          <span>{t('irrigation:tabs.irrigation')}</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('crop')}
          className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            activeTab === 'crop'
              ? 'bg-[#173D2B] text-white'
              : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
          }`}
        >
          <Sprout className="w-3.5 h-3.5" />
          <span>{t('irrigation:tabs.crop')}</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('fertilizer')}
          className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            activeTab === 'fertilizer'
              ? 'bg-[#173D2B] text-white'
              : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
          }`}
        >
          <FlaskConical className="w-3.5 h-3.5" />
          <span>{t('irrigation:tabs.fertilizer')}</span>
        </button>
      </div>

      {confirmationBanner && (
        <div className="bg-[var(--agri-card)] border border-[var(--agri-leaf)] rounded-xl px-4 py-3 flex items-center justify-between text-xs text-[var(--agri-text)]">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-[var(--agri-leaf)] shrink-0" />
            <span>{confirmationBanner}</span>
          </div>
          <button
            type="button"
            onClick={() => setConfirmationBanner(null)}
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

      {/* TAB 1: IRRIGATION CONTROL + 3D ROOT-ZONE HYDROLOGY */}
      {activeTab === 'irrigation' && data && selectedZone && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left 5 Cols: 3D Root-Zone Soil Hydrology Cross-Section */}
          <div className="lg:col-span-5 space-y-3">
            <RootZone3DProfile
              zone={selectedZone}
              nitrogenMgKg={selectedDevice?.readings.nitrogenMgKg ?? 110}
              phosphorusMgKg={selectedDevice?.readings.phosphorusMgKg ?? 48}
              potassiumMgKg={selectedDevice?.readings.potassiumMgKg ?? 65}
            />
            <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-4 text-xs flex items-center justify-between">
              <span className="text-[var(--agri-muted)]">
                Inspecting Root Horizon: <strong className="text-[var(--agri-text)]">{selectedZone.zoneName}</strong>
              </span>
              <span className="font-mono-tabular text-[var(--agri-leaf)] font-semibold">
                Target {selectedZone.targetMoisturePct}% VWC
              </span>
            </div>
          </div>

          {/* Right 7 Cols: Solenoid Valve Actuation Cards */}
          <div className="lg:col-span-7 space-y-4">
            {data.irrigation.zones.map((zone) => {
              const isActive = zone.valveStatus === 'active';
              const needsWater = zone.soilMoisturePct < zone.targetMoisturePct;
              const isFocused = zone.zoneId === selectedZone.zoneId;

              return (
                <div
                  key={zone.zoneId}
                  onClick={() => setSelectedZoneId(zone.zoneId)}
                  className={`bg-[var(--agri-card)] border rounded-xl p-5 flex flex-col justify-between space-y-4 cursor-pointer transition-colors ${
                    isFocused
                      ? 'border-[var(--agri-leaf)]'
                      : 'border-[var(--agri-border)] hover:border-[var(--agri-leaf)]/60'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between text-xs text-[var(--agri-muted)] font-mono-tabular">
                      <span>{zone.zoneId} · {zone.cropType}</span>
                      <span
                        className={
                          isActive
                            ? 'text-[var(--agri-water)] font-semibold'
                            : needsWater
                            ? 'text-[var(--agri-amber)] font-semibold'
                            : 'text-[var(--agri-leaf)] font-semibold'
                        }
                      >
                        {isActive
                          ? t('common:status.active')
                          : needsWater
                          ? t('common:status.attention_needed')
                          : t('common:status.idle')}
                      </span>
                    </div>
                    <h2 className="text-base font-semibold text-[var(--agri-text)] mt-1">
                      {zone.zoneName}
                    </h2>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-4 pt-3 border-t border-[var(--agri-border)] text-xs">
                      <div>
                        <span className="text-[var(--agri-muted)] block">
                          {t('irrigation:zones.currentMoisture')}
                        </span>
                        <span className="font-mono-tabular font-bold text-base text-[var(--agri-text)]">
                          {formatNumber(zone.soilMoisturePct, lang)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-[var(--agri-muted)] block">
                          {t('irrigation:zones.targetMoisture')}
                        </span>
                        <span className="font-mono-tabular font-bold text-base text-[var(--agri-text)]">
                          {formatNumber(zone.targetMoisturePct, lang)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-[var(--agri-muted)] block">
                          {t('irrigation:zones.flowRate')}
                        </span>
                        <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                          {formatNumber(zone.flowRateLpm, lang)} {t('common:units.lpm')}
                        </span>
                      </div>
                      <div>
                        <span className="text-[var(--agri-muted)] block">
                          {t('irrigation:zones.recommendedDuration')}
                        </span>
                        <span className="font-mono-tabular font-medium text-[var(--agri-text)]">
                          {formatNumber(zone.recommendedDurationMin, lang)}{' '}
                          {t('common:units.minutes')}
                        </span>
                      </div>
                      <div>
                        <span className="text-[var(--agri-muted)] block">
                          {t('irrigation:zones.waterSaved')}
                        </span>
                        <span className="font-mono-tabular font-medium text-[var(--agri-leaf)]">
                          {formatNumber(zone.waterSavedLitersToday, lang)} L
                        </span>
                      </div>
                      <div>
                        <span className="text-[var(--agri-muted)] block">
                          {t('irrigation:zones.autoSchedule')}
                        </span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleToggleAuto(zone);
                          }}
                          className="font-mono-tabular font-semibold text-[var(--agri-water)] hover:underline cursor-pointer"
                        >
                          {zone.autoScheduleEnabled ? 'ON (ET₀)' : 'OFF'}
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-[var(--agri-border)] flex items-center justify-between gap-2">
                    <span className="text-[11px] text-[var(--agri-muted)] font-mono-tabular">
                      {formatDateTime(zone.lastIrrigatedIso, lang)}
                    </span>
                    {isActive ? (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setPendingAction({ zone, action: 'stop' });
                        }}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-[var(--agri-red)] hover:opacity-90 rounded-lg cursor-pointer whitespace-nowrap shrink-0"
                      >
                        <Square className="w-3.5 h-3.5" />
                        <span>{t('irrigation:zones.stopIrrigation')}</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setPendingAction({ zone, action: 'start' });
                        }}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer whitespace-nowrap shrink-0"
                      >
                        <Play className="w-3.5 h-3.5" />
                        <span>{t('irrigation:zones.startIrrigation')}</span>
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* TAB 2: CROP RECOMMENDATION MODEL & 200 CROPS DIRECTORY */}
      {activeTab === 'crop' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <form
            onSubmit={handleRunCropModel}
            className="lg:col-span-5 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4"
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <h2 className="text-base font-semibold text-[var(--agri-text)]">
                  {t('irrigation:cropForm.title')}
                </h2>
                <p className="text-xs text-[var(--agri-muted)] mt-0.5">
                  {t('irrigation:cropForm.desc')}
                </p>
              </div>
              <button
                type="button"
                onClick={loadLiveBlockATelemetry}
                className="px-2.5 py-1.5 text-xs font-medium text-[var(--agri-text)] bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg whitespace-nowrap shrink-0 cursor-pointer"
              >
                {t('irrigation:cropForm.loadLiveSensor')}
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('irrigation:cropForm.nitrogen')}
                </label>
                <input
                  type="number"
                  step="any"
                  value={cropInputs.nitrogen}
                  onChange={(e) => setCropInputs({ ...cropInputs, nitrogen: e.target.value })}
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
              </div>
              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('irrigation:cropForm.phosphorus')}
                </label>
                <input
                  type="number"
                  step="any"
                  value={cropInputs.phosphorus}
                  onChange={(e) => setCropInputs({ ...cropInputs, phosphorus: e.target.value })}
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
              </div>
              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('irrigation:cropForm.potassium')}
                </label>
                <input
                  type="number"
                  step="any"
                  value={cropInputs.potassium}
                  onChange={(e) => setCropInputs({ ...cropInputs, potassium: e.target.value })}
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
              </div>
              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('irrigation:cropForm.ph')}
                </label>
                <input
                  type="number"
                  step="0.1"
                  min={3.5}
                  max={9.5}
                  value={cropInputs.ph}
                  onChange={(e) => setCropInputs({ ...cropInputs, ph: e.target.value })}
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
              </div>
              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('irrigation:cropForm.temperature')}
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={cropInputs.temperature}
                  onChange={(e) => setCropInputs({ ...cropInputs, temperature: e.target.value })}
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
              </div>
              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('irrigation:cropForm.humidity')}
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={cropInputs.humidity}
                  onChange={(e) => setCropInputs({ ...cropInputs, humidity: e.target.value })}
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
              </div>
              <div className="col-span-2">
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('irrigation:cropForm.rainfall')}
                </label>
                <input
                  type="number"
                  step="1"
                  value={cropInputs.rainfall}
                  onChange={(e) => setCropInputs({ ...cropInputs, rainfall: e.target.value })}
                  className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                  required
                />
              </div>
            </div>

            {cropError && (
              <div className="text-xs text-[var(--agri-red)] font-medium">{cropError}</div>
            )}

            <button
              type="submit"
              disabled={cropLoading}
              className="w-full py-2.5 px-4 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg transition-colors cursor-pointer disabled:opacity-50"
            >
              {t('irrigation:cropForm.submit')}
            </button>
          </form>

          <div className="lg:col-span-7 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6">
            <h2 className="text-base font-semibold text-[var(--agri-text)] mb-4">
              {t('irrigation:cropForm.resultsTitle')}
            </h2>

            {!cropResults ? (
              <div className="py-12 text-center text-xs text-[var(--agri-muted)]">
                {t('irrigation:cropForm.desc')}
              </div>
            ) : (
              <div className="divide-y divide-[var(--agri-border)]">
                {cropResults.map((item) => (
                  <div key={item.cropKey} className="py-4 first:pt-0 last:pb-0 space-y-2">
                    <div className="flex items-baseline justify-between gap-2">
                      <div>
                        <span className="text-sm font-bold text-[var(--agri-text)]">
                          0{item.rank}. {item.cropName}
                        </span>
                        <span className="mx-2 text-xs text-[var(--agri-muted)]">·</span>
                        <span className="text-xs italic text-[var(--agri-muted)]">
                          {item.scientificName}
                        </span>
                      </div>
                      <span className="font-mono-tabular text-sm font-bold text-[var(--agri-leaf)]">
                        {formatNumber(item.confidencePct, lang)}% {t('irrigation:cropForm.confidence')}
                      </span>
                    </div>

                    <p className="text-xs text-[var(--agri-muted)]">{item.limitingFactor}</p>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1 text-xs font-mono-tabular">
                      <div>
                        <span className="text-[var(--agri-muted)] block font-sans">
                          {t('irrigation:cropForm.waterReq')}
                        </span>
                        <span className="text-[var(--agri-text)] font-semibold">
                          {formatNumber(item.waterRequirementMm, lang)} mm
                        </span>
                      </div>
                      <div>
                        <span className="text-[var(--agri-muted)] block font-sans">
                          {t('irrigation:cropForm.duration')}
                        </span>
                        <span className="text-[var(--agri-text)] font-semibold">
                          {formatNumber(item.growthDurationDays, lang)} {t('common:units.days')}
                        </span>
                      </div>
                      <div>
                        <span className="text-[var(--agri-muted)] block font-sans">
                          {t('irrigation:cropForm.yield')}
                        </span>
                        <span className="text-[var(--agri-text)] font-semibold">
                          {formatNumber(item.expectedYieldTonHa, lang)} {t('common:units.tonHa')}
                        </span>
                      </div>
                      <div>
                        <span className="text-[var(--agri-muted)] block font-sans">
                          {t('irrigation:cropForm.dripSavings')}
                        </span>
                        <span className="text-[var(--agri-water)] font-semibold">
                          {formatNumber(item.dripSavingsPct, lang)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* 200 Indian Crops Directory with Search & Category Filtering */}
        <CropDirectory200 onSelectCrop={handleSelectCropFromDirectory} />
      </div>
    )}

      {/* TAB 3: FERTILIZER DOSAGE CALCULATOR */}
      {activeTab === 'fertilizer' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <form
            onSubmit={handleRunFertilizerModel}
            className="lg:col-span-5 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-4"
          >
            <div>
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('irrigation:fertilizerForm.title')}
              </h2>
              <p className="text-xs text-[var(--agri-muted)] mt-0.5">
                {t('irrigation:fertilizerForm.desc')}
              </p>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-[var(--agri-muted)] mb-1">
                  {t('irrigation:fertilizerForm.selectCrop')}
                </label>
                <select
                  value={fertInputs.cropType}
                  onChange={(e) => setFertInputs({ ...fertInputs, cropType: e.target.value })}
                  className="w-full px-3 py-2 bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                >
                  {(['paddy', 'tomato', 'millet', 'turmeric', 'cotton'] as const).map((c) => (
                    <option key={c} value={c}>
                      {t(`irrigation:fertilizerForm.crops.${c}`)}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-[var(--agri-muted)] mb-1">N (mg/kg)</label>
                  <input
                    type="number"
                    value={fertInputs.nitrogen}
                    onChange={(e) => setFertInputs({ ...fertInputs, nitrogen: e.target.value })}
                    className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                    required
                  />
                </div>
                <div>
                  <label className="block text-[var(--agri-muted)] mb-1">P (mg/kg)</label>
                  <input
                    type="number"
                    value={fertInputs.phosphorus}
                    onChange={(e) => setFertInputs({ ...fertInputs, phosphorus: e.target.value })}
                    className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                    required
                  />
                </div>
                <div>
                  <label className="block text-[var(--agri-muted)] mb-1">K (mg/kg)</label>
                  <input
                    type="number"
                    value={fertInputs.potassium}
                    onChange={(e) => setFertInputs({ ...fertInputs, potassium: e.target.value })}
                    className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                    required
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[var(--agri-muted)] mb-1">
                    {t('irrigation:cropForm.ph')}
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={fertInputs.soilPh}
                    onChange={(e) => setFertInputs({ ...fertInputs, soilPh: e.target.value })}
                    className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                    required
                  />
                </div>
                <div>
                  <label className="block text-[var(--agri-muted)] mb-1">
                    {t('irrigation:fertilizerForm.soilMoisture')}
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={fertInputs.soilMoisture}
                    onChange={(e) => setFertInputs({ ...fertInputs, soilMoisture: e.target.value })}
                    className="w-full px-3 py-2 font-mono-tabular bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                    required
                  />
                </div>
              </div>
            </div>

            {fertError && (
              <div className="text-xs text-[var(--agri-red)] font-medium">{fertError}</div>
            )}

            <button
              type="submit"
              disabled={fertLoading}
              className="w-full py-2.5 px-4 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg transition-colors cursor-pointer disabled:opacity-50"
            >
              {t('irrigation:fertilizerForm.submit')}
            </button>
          </form>

          <div className="lg:col-span-7 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-5">
            <h2 className="text-base font-semibold text-[var(--agri-text)]">
              {t('irrigation:fertilizerForm.scheduleTitle')}
            </h2>

            {!fertResult ? (
              <div className="py-12 text-center text-xs text-[var(--agri-muted)]">
                {t('irrigation:fertilizerForm.desc')}
              </div>
            ) : (
              <div className="space-y-5">
                {fertResult.moistureWarning && (
                  <div className="p-3.5 rounded-lg bg-[var(--agri-bg)] border border-[var(--agri-amber)] text-xs text-[var(--agri-text)] flex items-start gap-2.5">
                    <AlertTriangle className="w-4 h-4 text-[var(--agri-amber)] shrink-0 mt-0.5" />
                    <span>{fertResult.moistureWarning}</span>
                  </div>
                )}

                <div className="grid grid-cols-3 gap-4 pb-4 border-b border-[var(--agri-border)] text-xs">
                  <div>
                    <span className="text-[var(--agri-muted)] block">N Deficit</span>
                    <span className="font-mono-tabular text-base font-bold text-[var(--agri-text)]">
                      {formatNumber(fertResult.deficitsKgHa.nitrogen, lang)} {t('common:units.kgHa')}
                    </span>
                  </div>
                  <div>
                    <span className="text-[var(--agri-muted)] block">P₂O₅ Deficit</span>
                    <span className="font-mono-tabular text-base font-bold text-[var(--agri-text)]">
                      {formatNumber(fertResult.deficitsKgHa.phosphorus, lang)}{' '}
                      {t('common:units.kgHa')}
                    </span>
                  </div>
                  <div>
                    <span className="text-[var(--agri-muted)] block">K₂O Deficit</span>
                    <span className="font-mono-tabular text-base font-bold text-[var(--agri-text)]">
                      {formatNumber(fertResult.deficitsKgHa.potassium, lang)}{' '}
                      {t('common:units.kgHa')}
                    </span>
                  </div>
                </div>

                <div className="divide-y divide-[var(--agri-border)]">
                  {fertResult.schedule.map((item) => (
                    <div key={item.productKey} className="py-3 first:pt-0 last:pb-0">
                      <div className="flex items-baseline justify-between">
                        <span className="text-sm font-semibold text-[var(--agri-text)]">
                          {item.productName}
                        </span>
                        <span className="font-mono-tabular text-sm font-bold text-[var(--agri-leaf)]">
                          {formatNumber(item.dosageKgHa, lang)} {t('common:units.kgHa')}
                        </span>
                      </div>
                      <p className="text-xs text-[var(--agri-muted)] mt-1">
                        {item.applicationNote}
                      </p>
                    </div>
                  ))}
                </div>

                <div className="pt-4 border-t border-[var(--agri-border)]">
                  <div className="text-xs font-semibold text-[var(--agri-text)]">
                    {t('irrigation:fertilizerForm.organicTitle')}
                  </div>
                  <p className="text-xs text-[var(--agri-muted)] mt-1">
                    {fertResult.organicAmendment}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Confirmation Dialog for Irrigation Valve Actuation */}
      {pendingAction && (
        <div
          className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
        >
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl max-w-md w-full p-6 space-y-4">
            <h3 className="text-base font-semibold text-[var(--agri-text)]">
              {t('irrigation:zones.confirmTitle')}
            </h3>
            <p className="text-xs text-[var(--agri-muted)] leading-relaxed">
              {pendingAction.action === 'start'
                ? t('irrigation:zones.confirmStartDesc', {
                    zone: pendingAction.zone.zoneName,
                    crop: pendingAction.zone.cropType,
                    duration: formatNumber(pendingAction.zone.recommendedDurationMin, lang),
                  })
                : t('irrigation:zones.confirmStopDesc', {
                    zone: pendingAction.zone.zoneName,
                  })}
            </p>
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setPendingAction(null)}
                className="px-4 py-2 text-xs font-medium text-[var(--agri-text)] bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg cursor-pointer"
              >
                {t('common:actions.cancel')}
              </button>
              <button
                type="button"
                disabled={actuating}
                onClick={handleConfirmIrrigation}
                className="px-4 py-2 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg cursor-pointer disabled:opacity-50"
              >
                {t('common:actions.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
