import React, { useMemo, useState } from 'react';
import { formatNumber } from '../utils/format';

export interface TelemetryChartPoint {
  label: string;
  primary: number;
  secondary?: number;
  barValue?: number;
}

interface TelemetryChartProps {
  data: TelemetryChartPoint[];
  primaryLabel: string;
  primaryUnit: string;
  primaryColor?: string;
  secondaryLabel?: string;
  secondaryUnit?: string;
  secondaryColor?: string;
  barLabel?: string;
  barUnit?: string;
  barColor?: string;
  lang: string;
  height?: number;
}

export const TelemetryChart: React.FC<TelemetryChartProps> = React.memo(
  ({
    data,
    primaryLabel,
    primaryUnit,
    primaryColor = '#4A8C55',
    secondaryLabel,
    secondaryUnit = '',
    secondaryColor = '#438EBA',
    barLabel,
    barUnit = 'mm',
    barColor = '#DCE8D5',
    lang,
    height = 250,
  }) => {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);

    const width = 760;
    const padLeft = 46;
    const padRight = 36;
    const padTop = 24;
    const padBottom = 36;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const geometry = useMemo(() => {
      if (!data || data.length === 0) return null;

      let minRaw = Infinity;
      let maxRaw = -Infinity;
      let maxBar = 25;

      for (let i = 0; i < data.length; i++) {
        const pt = data[i];
        if (pt.primary < minRaw) minRaw = pt.primary;
        if (pt.primary > maxRaw) maxRaw = pt.primary;
        if (typeof pt.secondary === 'number') {
          if (pt.secondary < minRaw) minRaw = pt.secondary;
          if (pt.secondary > maxRaw) maxRaw = pt.secondary;
        }
        if (typeof pt.barValue === 'number' && pt.barValue > maxBar) {
          maxBar = pt.barValue;
        }
      }

      if (!Number.isFinite(minRaw)) minRaw = 0;
      if (!Number.isFinite(maxRaw)) maxRaw = 100;

      const minVal = Math.max(0, Math.floor(minRaw * 0.85));
      const maxVal = Math.ceil(Math.max(maxRaw, minVal + 10) * 1.08);
      const valSpan = Math.max(1, maxVal - minVal);
      const countSpan = Math.max(1, data.length - 1);

      const getX = (idx: number) => padLeft + (idx / countSpan) * plotWidth;
      const getY = (val: number) =>
        padTop + plotHeight - ((val - minVal) / valSpan) * plotHeight;

      const coords = data.map((d, i) => ({
        x: getX(i),
        y1: getY(d.primary),
        y2: typeof d.secondary === 'number' ? getY(d.secondary) : null,
        label: d.label,
        barValue: d.barValue ?? 0,
      }));

      const primaryPoints = coords.map((c) => `${c.x.toFixed(1)},${c.y1.toFixed(1)}`).join(' ');
      const primaryArea = `${padLeft},${padTop + plotHeight} ${primaryPoints} ${coords[
        coords.length - 1
      ].x.toFixed(1)},${padTop + plotHeight}`;

      const hasSecondary =
        Boolean(secondaryLabel) && coords.every((c) => c.y2 !== null);
      const secondaryPoints = hasSecondary
        ? coords.map((c) => `${c.x.toFixed(1)},${c.y2!.toFixed(1)}`).join(' ')
        : null;

      const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
        const val = Math.round(minVal + t * valSpan);
        return { val, y: getY(val) };
      });

      return {
        coords,
        primaryPoints,
        primaryArea,
        secondaryPoints,
        yTicks,
        maxBar,
      };
    }, [data, plotWidth, plotHeight, secondaryLabel]);

    if (!geometry || data.length === 0) {
      return null;
    }

    const activePoint =
      hoverIndex !== null && data[hoverIndex]
        ? data[hoverIndex]
        : data[data.length - 1];

    const colHitWidth = plotWidth / Math.max(2, data.length);

    return (
      <div className="w-full">
        {/* Legend and active readout bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-3 text-xs text-[var(--agri-muted)]">
          <div className="flex flex-wrap items-center gap-5">
            <span className="inline-flex items-center gap-2 font-medium text-[var(--agri-text)]">
              <span
                className="inline-block w-3 h-0.5"
                style={{ backgroundColor: primaryColor }}
              />
              {primaryLabel}
            </span>
            {secondaryLabel && (
              <span className="inline-flex items-center gap-2 font-medium text-[var(--agri-text)]">
                <span
                  className="inline-block w-3 h-0.5 border-b-2 border-dashed"
                  style={{ borderColor: secondaryColor }}
                />
                {secondaryLabel}
              </span>
            )}
            {barLabel && (
              <span className="inline-flex items-center gap-2 font-medium text-[var(--agri-text)]">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-xs"
                  style={{ backgroundColor: '#438EBA', opacity: 0.35 }}
                />
                {barLabel}
              </span>
            )}
          </div>

          {activePoint && (
            <div className="font-mono-tabular text-xs text-[var(--agri-text)]">
              <span>{activePoint.label}</span>
              <span className="mx-1.5 text-[var(--agri-muted)]">·</span>
              <span>
                {formatNumber(activePoint.primary, lang)}
                {primaryUnit}
              </span>
              {typeof activePoint.secondary === 'number' && (
                <>
                  <span className="mx-1.5 text-[var(--agri-muted)]">·</span>
                  <span>
                    {formatNumber(activePoint.secondary, lang)}
                    {secondaryUnit}
                  </span>
                </>
              )}
              {typeof activePoint.barValue === 'number' && activePoint.barValue > 0 && (
                <>
                  <span className="mx-1.5 text-[var(--agri-muted)]">·</span>
                  <span>
                    {formatNumber(activePoint.barValue, lang)} {barUnit}
                  </span>
                </>
              )}
            </div>
          )}
        </div>

        <div className="w-full overflow-x-auto">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto select-none"
            role="img"
            aria-label={primaryLabel}
            onMouseLeave={() => setHoverIndex(null)}
          >
            <defs>
              <linearGradient id="agriPrimaryFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={primaryColor} stopOpacity="0.20" />
                <stop offset="100%" stopColor={primaryColor} stopOpacity="0.01" />
              </linearGradient>
            </defs>

            {/* Horizontal grid lines & Y axis labels */}
            {geometry.yTicks.map((tick, idx) => (
              <g key={idx}>
                <line
                  x1={padLeft}
                  y1={tick.y}
                  x2={width - padRight}
                  y2={tick.y}
                  stroke="currentColor"
                  className="text-[var(--agri-border)]"
                  strokeDasharray={idx === 0 ? undefined : '3 3'}
                  strokeWidth="1"
                />
                <text
                  x={padLeft - 8}
                  y={tick.y + 4}
                  textAnchor="end"
                  className="fill-[var(--agri-muted)] text-[10px] font-mono-tabular"
                >
                  {formatNumber(tick.val, lang, { maximumFractionDigits: 0 })}
                </text>
              </g>
            ))}

            {/* Optional irrigation bars */}
            {barLabel &&
              geometry.coords.map((c, i) => {
                if (c.barValue <= 0) return null;
                const barH = (c.barValue / geometry.maxBar) * (plotHeight * 0.55);
                const x = c.x - 9;
                const y = padTop + plotHeight - barH;
                return (
                  <rect
                    key={`bar-${i}`}
                    x={x}
                    y={y}
                    width={18}
                    height={barH}
                    rx={2}
                    fill={barColor === '#DCE8D5' ? '#438EBA' : barColor}
                    opacity={0.28}
                  />
                );
              })}

            {/* Area under primary curve */}
            <polygon points={geometry.primaryArea} fill="url(#agriPrimaryFill)" />

            {/* Primary polyline */}
            <polyline
              fill="none"
              stroke={primaryColor}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={geometry.primaryPoints}
            />

            {/* Secondary polyline */}
            {geometry.secondaryPoints && (
              <polyline
                fill="none"
                stroke={secondaryColor}
                strokeWidth="2"
                strokeDasharray="5 4"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={geometry.secondaryPoints}
              />
            )}

            {/* Active hover guide line */}
            {hoverIndex !== null && geometry.coords[hoverIndex] && (
              <line
                x1={geometry.coords[hoverIndex].x}
                y1={padTop}
                x2={geometry.coords[hoverIndex].x}
                y2={padTop + plotHeight}
                stroke={primaryColor}
                strokeWidth="1"
                strokeDasharray="2 2"
              />
            )}

            {/* Data points & X axis labels */}
            {geometry.coords.map((c, i) => {
              const isHovered = hoverIndex === i;
              const showLabel =
                data.length <= 10 ||
                i % Math.ceil(data.length / 8) === 0 ||
                i === data.length - 1;

              return (
                <g
                  key={i}
                  onMouseEnter={() => setHoverIndex(i)}
                  className="cursor-pointer"
                >
                  <circle
                    cx={c.x}
                    cy={c.y1}
                    r={isHovered ? 5 : 3.2}
                    fill={primaryColor}
                    stroke="#FFFFFF"
                    strokeWidth="1.5"
                  />
                  {c.y2 !== null && (
                    <circle
                      cx={c.x}
                      cy={c.y2}
                      r={isHovered ? 4.5 : 2.8}
                      fill={secondaryColor}
                      stroke="#FFFFFF"
                      strokeWidth="1.2"
                    />
                  )}
                  {showLabel && (
                    <text
                      x={c.x}
                      y={height - 10}
                      textAnchor="middle"
                      className="fill-[var(--agri-muted)] text-[10px] font-mono-tabular"
                    >
                      {c.label}
                    </text>
                  )}
                  <rect
                    x={c.x - colHitWidth / 2}
                    y={padTop}
                    width={colHitWidth}
                    height={plotHeight}
                    fill="transparent"
                  />
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    );
  }
);
