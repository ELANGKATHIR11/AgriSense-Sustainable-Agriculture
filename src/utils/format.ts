import { SUPPORTED_LANGUAGES } from '../i18n';

const LOCALE_MAP: Record<string, string> = Object.fromEntries(
  SUPPORTED_LANGUAGES.map((l) => [l.code, l.locale])
);

const numberFormatterCache = new Map<string, Intl.NumberFormat>();
const dateTimeFormatterCache = new Map<string, Intl.DateTimeFormat>();

export function getLocaleTag(langCode: string): string {
  return LOCALE_MAP[langCode] || 'en-IN';
}

function getNumberFormatter(
  locale: string,
  options?: Intl.NumberFormatOptions
): Intl.NumberFormat {
  const maxFrac = options?.maximumFractionDigits ?? 1;
  const minFrac = options?.minimumFractionDigits ?? 0;
  const key = `${locale}:${minFrac}:${maxFrac}`;
  let fmt = numberFormatterCache.get(key);
  if (!fmt) {
    fmt = new Intl.NumberFormat(locale, {
      maximumFractionDigits: maxFrac,
      minimumFractionDigits: minFrac,
      ...options,
    });
    numberFormatterCache.set(key, fmt);
  }
  return fmt;
}

function getDateTimeFormatter(
  locale: string,
  variant: 'datetime' | 'shortdate'
): Intl.DateTimeFormat {
  const key = `${locale}:${variant}`;
  let fmt = dateTimeFormatterCache.get(key);
  if (!fmt) {
    fmt =
      variant === 'shortdate'
        ? new Intl.DateTimeFormat(locale, {
            month: 'short',
            day: 'numeric',
          })
        : new Intl.DateTimeFormat(locale, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          });
    dateTimeFormatterCache.set(key, fmt);
  }
  return fmt;
}

export function formatNumber(
  value: number,
  langCode: string,
  options?: Intl.NumberFormatOptions
): string {
  if (!Number.isFinite(value)) return '0';
  try {
    return getNumberFormatter(getLocaleTag(langCode), options).format(value);
  } catch {
    return String(value);
  }
}

export function formatDateTime(
  isoString: string,
  langCode: string
): string {
  try {
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) return isoString;
    return getDateTimeFormatter(getLocaleTag(langCode), 'datetime').format(date);
  } catch {
    return isoString;
  }
}

export function formatShortDate(isoString: string, langCode: string): string {
  try {
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) return isoString;
    return getDateTimeFormatter(getLocaleTag(langCode), 'shortdate').format(date);
  } catch {
    return isoString;
  }
}
