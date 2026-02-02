import React from 'react';
import { clsx } from 'clsx';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export interface StatCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    isPositive?: boolean;
  };
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  className?: string;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  unit,
  icon,
  trend,
  variant = 'default',
  className,
}) => {
  const variantStyles = {
    default: {
      bg: 'bg-white dark:bg-gray-800',
      icon: 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300',
      accent: 'text-gray-900 dark:text-white',
    },
    success: {
      bg: 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20',
      icon: 'bg-green-100 dark:bg-green-900/50 text-green-600 dark:text-green-400',
      accent: 'text-green-700 dark:text-green-400',
    },
    warning: {
      bg: 'bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20',
      icon: 'bg-amber-100 dark:bg-amber-900/50 text-amber-600 dark:text-amber-400',
      accent: 'text-amber-700 dark:text-amber-400',
    },
    danger: {
      bg: 'bg-gradient-to-br from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20',
      icon: 'bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400',
      accent: 'text-red-700 dark:text-red-400',
    },
    info: {
      bg: 'bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20',
      icon: 'bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400',
      accent: 'text-blue-700 dark:text-blue-400',
    },
  };

  const styles = variantStyles[variant];

  return (
    <div
      className={clsx(
        'p-5 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 transition-all duration-300 hover:shadow-md',
        styles.bg,
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className={clsx('text-3xl font-bold', styles.accent)}>{value}</span>
            {unit && (
              <span className="text-sm text-gray-500 dark:text-gray-400">{unit}</span>
            )}
          </div>
          
          {trend && (
            <div className="mt-3 flex items-center gap-1">
              {trend.isPositive === undefined ? (
                <Minus className="w-4 h-4 text-gray-400" />
              ) : trend.isPositive ? (
                <TrendingUp className="w-4 h-4 text-green-500" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-500" />
              )}
              <span
                className={clsx(
                  'text-sm font-medium',
                  trend.isPositive === undefined
                    ? 'text-gray-500'
                    : trend.isPositive
                    ? 'text-green-600'
                    : 'text-red-600'
                )}
              >
                {trend.value > 0 ? '+' : ''}{trend.value}%
              </span>
              <span className="text-sm text-gray-400">vs last period</span>
            </div>
          )}
        </div>
        
        {icon && (
          <div className={clsx('p-3 rounded-xl', styles.icon)}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
};

export interface MetricCardProps {
  label: string;
  value: number;
  max?: number;
  unit?: string;
  icon?: React.ReactNode;
  color?: 'green' | 'blue' | 'amber' | 'red';
}

const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  max = 100,
  unit = '%',
  icon,
  color = 'green',
}) => {
  const percentage = (value / max) * 100;
  
  const colors = {
    green: 'bg-agri-500',
    blue: 'bg-water-500',
    amber: 'bg-earth-500',
    red: 'bg-red-500',
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-100 dark:border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-600 dark:text-gray-400">{label}</span>
        {icon && <span className="text-gray-400">{icon}</span>}
      </div>
      <div className="flex items-baseline gap-1 mb-2">
        <span className="text-2xl font-bold text-gray-900 dark:text-white">{value}</span>
        <span className="text-sm text-gray-500">{unit}</span>
      </div>
      <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full transition-all duration-500', colors[color])}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  );
};

export { StatCard, MetricCard };
