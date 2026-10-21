import React, { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Upload,
  Microscope,
  AlertCircle,
  CheckCircle2,
  Trash2,
  ShieldCheck,
} from 'lucide-react';
import { DiseaseDiagnosisResult, agriApi } from '../api/client';
import { LEAF_SPECIMEN_PRESETS, LeafSpecimenPreset } from '../utils/leafSpecimens';
import { formatDateTime, formatNumber } from '../utils/format';

export const DiseaseDetectionPage: React.FC = () => {
  const { t, i18n } = useTranslation(['disease', 'common']);
  const lang = i18n.language;
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(
    LEAF_SPECIMEN_PRESETS[0].dataUrl
  );
  const [selectedPreset, setSelectedPreset] = useState<string | undefined>(
    LEAF_SPECIMEN_PRESETS[0].id
  );
  const [uploadedBase64, setUploadedBase64] = useState<string | undefined>(undefined);
  const [uploadedMime, setUploadedMime] = useState<string | undefined>(undefined);
  const [cropHint, setCropHint] = useState<string>('Tomato (Solanum lycopersicum)');
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<DiseaseDiagnosisResult | null>(null);
  const [history, setHistory] = useState<DiseaseDiagnosisResult[]>([]);

  const handleFileValidationAndRead = (file: File) => {
    setValidationError(null);
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type) || file.size > 8 * 1024 * 1024) {
      setValidationError(t('disease:upload.invalidFile'));
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result);
      setPreviewUrl(dataUrl);
      setUploadedBase64(dataUrl);
      setUploadedMime(file.type);
      setSelectedPreset(undefined);
    };
    reader.readAsDataURL(file);
  };

  const handleSelectPreset = (preset: LeafSpecimenPreset) => {
    setValidationError(null);
    setPreviewUrl(preset.dataUrl);
    setSelectedPreset(preset.id);
    setUploadedBase64(undefined);
    setUploadedMime(undefined);
    setCropHint(preset.cropHint);
  };

  const handleRunDiagnosis = async () => {
    if (!previewUrl) return;
    setAnalyzing(true);
    setValidationError(null);
    try {
      const diag = await agriApi.analyzeDisease({
        imageBase64: uploadedBase64,
        mimeType: uploadedMime,
        cropHint,
        samplePreset: selectedPreset,
        language: lang,
      });
      setResult(diag);
      setHistory((prev) => [diag, ...prev.slice(0, 4)]);
    } catch (err) {
      setValidationError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleClearImage = () => {
    setPreviewUrl(null);
    setUploadedBase64(undefined);
    setUploadedMime(undefined);
    setSelectedPreset(undefined);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-[var(--agri-text)]">
          {t('disease:title')}
        </h1>
        <p className="text-sm text-[var(--agri-muted)] mt-1 max-w-3xl">
          {t('disease:subtitle')}
        </p>
      </div>

      {/* Model Prediction Honesty Disclaimer */}
      <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl px-4 py-3 flex items-start gap-3 text-xs text-[var(--agri-muted)]">
        <ShieldCheck className="w-4 h-4 text-[var(--agri-leaf)] shrink-0 mt-0.5" />
        <span>{t('disease:disclaimer')}</span>
      </div>

      {/* Main Split: Upload & Specimen Controls (Left) + Diagnostic Report (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Image Upload & Reference Specimens */}
        <div className="lg:col-span-5 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-5">
          {/* Drag-and-Drop Zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              const f = e.dataTransfer.files?.[0];
              if (f) handleFileValidationAndRead(f);
            }}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-5 text-center transition-colors cursor-pointer ${
              isDragging
                ? 'border-[var(--agri-leaf)] bg-[var(--agri-sage)]/30'
                : 'border-[var(--agri-border)] hover:border-[var(--agri-leaf)]'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFileValidationAndRead(f);
              }}
              className="hidden"
            />
            <Upload className="w-6 h-6 text-[var(--agri-leaf)] mx-auto mb-2" />
            <p className="text-xs font-semibold text-[var(--agri-text)]">
              {t('disease:upload.dropzoneTitle')}
            </p>
            <p className="text-[11px] text-[var(--agri-muted)] mt-1">
              {t('disease:upload.dropzoneFormats')}
            </p>
          </div>

          {/* Reference Field Specimens */}
          <div>
            <p className="text-xs font-medium text-[var(--agri-muted)] mb-2">
              {t('disease:upload.sampleTitle')}
            </p>
            <div className="flex flex-col gap-2">
              {LEAF_SPECIMEN_PRESETS.map((preset) => {
                const isSelected = selectedPreset === preset.id;
                return (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => handleSelectPreset(preset)}
                    className={`w-full text-left px-3 py-2 text-xs font-medium rounded-lg border transition-colors cursor-pointer ${
                      isSelected
                        ? 'bg-[#173D2B] text-white border-[#173D2B]'
                        : 'bg-[var(--agri-bg)] text-[var(--agri-text)] border-[var(--agri-border)] hover:bg-[var(--agri-sage)]'
                    }`}
                  >
                    {t(`disease:upload.${preset.labelKey}`)}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Image Preview */}
          {previewUrl && (
            <div className="space-y-3">
              <div className="rounded-xl overflow-hidden border border-[var(--agri-border)] bg-[#173D2B]">
                <img
                  src={previewUrl}
                  alt="Plant leaf diagnostic specimen"
                  referrerPolicy="no-referrer"
                  className="w-full h-48 object-cover"
                />
              </div>

              <div>
                <label className="block text-xs text-[var(--agri-muted)] mb-1">
                  {t('disease:upload.cropHintLabel')}
                </label>
                <input
                  type="text"
                  value={cropHint}
                  onChange={(e) => setCropHint(e.target.value)}
                  className="w-full px-3 py-2 text-xs bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)]"
                />
              </div>

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  disabled={analyzing}
                  onClick={handleRunDiagnosis}
                  className="flex-1 inline-flex items-center justify-center gap-2 py-2.5 px-4 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg transition-colors cursor-pointer disabled:opacity-50"
                >
                  <Microscope className="w-4 h-4" />
                  <span>
                    {analyzing ? t('disease:upload.analyzing') : t('disease:upload.analyzeBtn')}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={handleClearImage}
                  className="p-2.5 text-xs text-[var(--agri-muted)] hover:text-[var(--agri-red)] bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg cursor-pointer"
                  aria-label={t('disease:upload.clearBtn')}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {validationError && (
            <div className="text-xs text-[var(--agri-red)] flex items-center gap-1.5">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{validationError}</span>
            </div>
          )}
        </div>

        {/* Right Column: Classification Results & Treatment Report */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6">
            <div className="flex items-center justify-between border-b border-[var(--agri-border)] pb-4 mb-5">
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('disease:results.title')}
              </h2>
              {result && (
                <div className="text-xs text-[var(--agri-muted)] font-mono-tabular">
                  <span>{t('disease:results.predictionBadge')}</span>
                  <span className="mx-1.5">·</span>
                  <span>{formatDateTime(result.analyzedAt, lang)}</span>
                </div>
              )}
            </div>

            {!result ? (
              <div className="py-16 text-center text-xs text-[var(--agri-muted)]">
                {t('disease:results.emptyPrompt')}
              </div>
            ) : (
              <div className="space-y-6">
                {/* Summary Header */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pb-5 border-b border-[var(--agri-border)]">
                  <div>
                    <span className="text-xs text-[var(--agri-muted)] block">
                      {t('disease:results.diagnosis')}
                    </span>
                    <div className="flex items-center gap-2 mt-1">
                      {result.isHealthy ? (
                        <CheckCircle2 className="w-5 h-5 text-[var(--agri-leaf)] shrink-0" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-[var(--agri-amber)] shrink-0" />
                      )}
                      <span className="text-lg font-bold text-[var(--agri-text)]">
                        {result.diseaseName}
                      </span>
                    </div>
                    <p className="text-xs italic text-[var(--agri-muted)] mt-1">
                      {result.scientificName} · {result.cropIdentified}
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="text-[var(--agri-muted)] block">
                        {t('disease:results.confidence')}
                      </span>
                      <span className="font-mono-tabular text-xl font-bold text-[var(--agri-leaf)] mt-1 block">
                        {formatNumber(result.confidencePct, lang)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-[var(--agri-muted)] block">
                        {t('disease:results.severity')}
                      </span>
                      <span
                        className={`text-sm font-semibold mt-1.5 block ${
                          result.severity === 'high'
                            ? 'text-[var(--agri-red)]'
                            : result.severity === 'moderate'
                            ? 'text-[var(--agri-amber)]'
                            : 'text-[var(--agri-leaf)]'
                        }`}
                      >
                        {t(`disease:results.severity_${result.severity}`, {
                          defaultValue: result.severity,
                        })}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Symptoms */}
                <div>
                  <h3 className="text-xs font-semibold text-[var(--agri-text)] mb-2">
                    01. {t('disease:results.symptomsTitle')}
                  </h3>
                  <ul className="space-y-1.5 text-xs text-[var(--agri-muted)] list-disc list-inside">
                    {result.symptomsObserved.map((s, idx) => (
                      <li key={idx}>{s}</li>
                    ))}
                  </ul>
                </div>

                {/* Organic Treatment */}
                <div className="pt-4 border-t border-[var(--agri-border)]">
                  <h3 className="text-xs font-semibold text-[var(--agri-text)] mb-2">
                    02. {t('disease:results.organicTitle')}
                  </h3>
                  <ul className="space-y-1.5 text-xs text-[var(--agri-muted)] list-disc list-inside">
                    {result.organicTreatment.map((s, idx) => (
                      <li key={idx}>{s}</li>
                    ))}
                  </ul>
                </div>

                {/* Chemical Intervention */}
                <div className="pt-4 border-t border-[var(--agri-border)]">
                  <h3 className="text-xs font-semibold text-[var(--agri-text)] mb-2">
                    03. {t('disease:results.chemicalTitle')}
                  </h3>
                  <ul className="space-y-1.5 text-xs text-[var(--agri-muted)] list-disc list-inside">
                    {result.chemicalTreatment.map((s, idx) => (
                      <li key={idx}>{s}</li>
                    ))}
                  </ul>
                </div>

                {/* Prevention Steps */}
                <div className="pt-4 border-t border-[var(--agri-border)]">
                  <h3 className="text-xs font-semibold text-[var(--agri-text)] mb-2">
                    04. {t('disease:results.preventionTitle')}
                  </h3>
                  <ul className="space-y-1.5 text-xs text-[var(--agri-muted)] list-disc list-inside">
                    {result.preventionSteps.map((s, idx) => (
                      <li key={idx}>{s}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* Analysis History */}
          {history.length > 0 && (
            <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6">
              <h3 className="text-sm font-semibold text-[var(--agri-text)] mb-3">
                {t('disease:results.historyTitle')}
              </h3>
              <div className="divide-y divide-[var(--agri-border)] text-xs">
                {history.map((item, idx) => (
                  <div key={idx} className="py-2.5 first:pt-0 last:pb-0 flex items-center justify-between gap-4">
                    <div>
                      <span className="font-semibold text-[var(--agri-text)]">
                        {item.diseaseName}
                      </span>
                      <span className="mx-1.5 text-[var(--agri-muted)]">·</span>
                      <span className="text-[var(--agri-muted)]">{item.cropIdentified}</span>
                    </div>
                    <div className="font-mono-tabular text-[var(--agri-muted)] shrink-0">
                      {formatNumber(item.confidencePct, lang)}% ·{' '}
                      {formatDateTime(item.analyzedAt, lang)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
