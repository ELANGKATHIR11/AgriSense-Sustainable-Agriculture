import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import enCommon from './locales/en/common.json';
import enDashboard from './locales/en/dashboard.json';
import enIot from './locales/en/iot.json';
import enIrrigation from './locales/en/irrigation.json';
import enDisease from './locales/en/disease.json';
import enAnalytics from './locales/en/analytics.json';
import enAssistant from './locales/en/assistant.json';
import enOperations from './locales/en/operations.json';

import taCommon from './locales/ta/common.json';
import taDashboard from './locales/ta/dashboard.json';
import taIot from './locales/ta/iot.json';
import taIrrigation from './locales/ta/irrigation.json';
import taDisease from './locales/ta/disease.json';
import taAnalytics from './locales/ta/analytics.json';
import taAssistant from './locales/ta/assistant.json';
import taOperations from './locales/ta/operations.json';

import teCommon from './locales/te/common.json';
import teDashboard from './locales/te/dashboard.json';
import teIot from './locales/te/iot.json';
import teIrrigation from './locales/te/irrigation.json';
import teDisease from './locales/te/disease.json';
import teAnalytics from './locales/te/analytics.json';
import teAssistant from './locales/te/assistant.json';
import teOperations from './locales/te/operations.json';

import knCommon from './locales/kn/common.json';
import knDashboard from './locales/kn/dashboard.json';
import knIot from './locales/kn/iot.json';
import knIrrigation from './locales/kn/irrigation.json';
import knDisease from './locales/kn/disease.json';
import knAnalytics from './locales/kn/analytics.json';
import knAssistant from './locales/kn/assistant.json';
import knOperations from './locales/kn/operations.json';

import mlCommon from './locales/ml/common.json';
import mlDashboard from './locales/ml/dashboard.json';
import mlIot from './locales/ml/iot.json';
import mlIrrigation from './locales/ml/irrigation.json';
import mlDisease from './locales/ml/disease.json';
import mlAnalytics from './locales/ml/analytics.json';
import mlAssistant from './locales/ml/assistant.json';
import mlOperations from './locales/ml/operations.json';

import hiCommon from './locales/hi/common.json';
import hiDashboard from './locales/hi/dashboard.json';
import hiIot from './locales/hi/iot.json';
import hiIrrigation from './locales/hi/irrigation.json';
import hiDisease from './locales/hi/disease.json';
import hiAnalytics from './locales/hi/analytics.json';
import hiAssistant from './locales/hi/assistant.json';
import hiOperations from './locales/hi/operations.json';

import bnCommon from './locales/bn/common.json';
import bnDashboard from './locales/bn/dashboard.json';
import bnIot from './locales/bn/iot.json';
import bnIrrigation from './locales/bn/irrigation.json';
import bnDisease from './locales/bn/disease.json';
import bnAnalytics from './locales/bn/analytics.json';
import bnAssistant from './locales/bn/assistant.json';
import bnOperations from './locales/bn/operations.json';

export const SUPPORTED_LANGUAGES = [
  { code: 'en', label: 'English', nativeLabel: 'English', locale: 'en-IN' },
  { code: 'ta', label: 'Tamil', nativeLabel: 'தமிழ்', locale: 'ta-IN' },
  { code: 'te', label: 'Telugu', nativeLabel: 'తెలుగు', locale: 'te-IN' },
  { code: 'kn', label: 'Kannada', nativeLabel: 'ಕನ್ನಡ', locale: 'kn-IN' },
  { code: 'ml', label: 'Malayalam', nativeLabel: 'മലയാളം', locale: 'ml-IN' },
  { code: 'hi', label: 'Hindi', nativeLabel: 'हिन्दी', locale: 'hi-IN' },
  { code: 'bn', label: 'Bengali', nativeLabel: 'বাংলা', locale: 'bn-IN' },
] as const;

export type SupportedLanguageCode = (typeof SUPPORTED_LANGUAGES)[number]['code'];

const STORAGE_KEY = 'agrisense_lang';

function getInitialLanguage(): SupportedLanguageCode {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && SUPPORTED_LANGUAGES.some((l) => l.code === saved)) {
      return saved as SupportedLanguageCode;
    }
  } catch {
    // Ignore storage read errors in restricted environments
  }
  return 'en';
}

export const resources = {
  en: {
    common: enCommon,
    dashboard: enDashboard,
    iot: enIot,
    irrigation: enIrrigation,
    disease: enDisease,
    analytics: enAnalytics,
    assistant: enAssistant,
    operations: enOperations,
  },
  ta: {
    common: taCommon,
    dashboard: taDashboard,
    iot: taIot,
    irrigation: taIrrigation,
    disease: taDisease,
    analytics: taAnalytics,
    assistant: taAssistant,
    operations: taOperations,
  },
  te: {
    common: teCommon,
    dashboard: teDashboard,
    iot: teIot,
    irrigation: teIrrigation,
    disease: teDisease,
    analytics: teAnalytics,
    assistant: teAssistant,
    operations: teOperations,
  },
  kn: {
    common: knCommon,
    dashboard: knDashboard,
    iot: knIot,
    irrigation: knIrrigation,
    disease: knDisease,
    analytics: knAnalytics,
    assistant: knAssistant,
    operations: knOperations,
  },
  ml: {
    common: mlCommon,
    dashboard: mlDashboard,
    iot: mlIot,
    irrigation: mlIrrigation,
    disease: mlDisease,
    analytics: mlAnalytics,
    assistant: mlAssistant,
    operations: mlOperations,
  },
  hi: {
    common: hiCommon,
    dashboard: hiDashboard,
    iot: hiIot,
    irrigation: hiIrrigation,
    disease: hiDisease,
    analytics: hiAnalytics,
    assistant: hiAssistant,
    operations: hiOperations,
  },
  bn: {
    common: bnCommon,
    dashboard: bnDashboard,
    iot: bnIot,
    irrigation: bnIrrigation,
    disease: bnDisease,
    analytics: bnAnalytics,
    assistant: bnAssistant,
    operations: bnOperations,
  },
} as const;

const initialLang = getInitialLanguage();

i18n.use(initReactI18next).init({
  resources,
  lng: initialLang,
  fallbackLng: 'en',
  defaultNS: 'common',
  ns: [
    'common',
    'dashboard',
    'iot',
    'irrigation',
    'disease',
    'analytics',
    'assistant',
    'operations',
  ],
  interpolation: {
    escapeValue: false,
  },
  saveMissing: true,
  missingKeyHandler: (lngs, ns, key) => {
    if (import.meta.env.DEV) {
      console.warn(`[AgriSense i18n] Missing translation key: ${ns}:${key} for [${lngs.join(', ')}]`);
    }
  },
});

if (typeof document !== 'undefined') {
  document.documentElement.lang = initialLang;
}

export function setAppLanguage(code: SupportedLanguageCode) {
  try {
    localStorage.setItem(STORAGE_KEY, code);
  } catch {
    // Ignore storage write errors
  }
  if (typeof document !== 'undefined') {
    document.documentElement.lang = code;
  }
  return i18n.changeLanguage(code);
}

export default i18n;
