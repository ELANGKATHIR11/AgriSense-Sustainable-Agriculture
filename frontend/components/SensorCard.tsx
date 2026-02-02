import React from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';

interface SensorCardProps {
  title: string;
  value: string | number;
  unit: string;
  icon: React.ElementType;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
  colorClass?: string;
}

export const SensorCard: React.FC<SensorCardProps> = ({ 
  title, 
  value, 
  unit, 
  icon: Icon, 
  trend,
  trendValue,
  colorClass = "text-agri-600 bg-agri-50"
}) => {
  return (
    <div className="bg-white p-6 rounded-2xl border border-stone-100 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <div className={`p-3 rounded-xl ${colorClass}`}>
          <Icon size={24} />
        </div>
        {trend && (
          <div className={`flex items-center text-xs font-medium px-2 py-1 rounded-full ${
            trend === 'up' ? 'text-green-700 bg-green-50' : 'text-amber-700 bg-amber-50'
          }`}>
            {trend === 'up' ? <ArrowUp size={12} className="mr-1" /> : <ArrowDown size={12} className="mr-1" />}
            {trendValue}
          </div>
        )}
      </div>
      <div>
        <h3 className="text-stone-500 text-sm font-medium mb-1">{title}</h3>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-bold text-stone-800">{value}</span>
          <span className="text-sm text-stone-400 font-medium">{unit}</span>
        </div>
      </div>
    </div>
  );
};