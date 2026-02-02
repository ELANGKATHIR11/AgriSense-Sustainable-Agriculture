import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Layers,
  Droplets,
  Thermometer,
  Activity,
  BarChart3,
  TrendingUp,
  Info,
  FlaskConical,
  Leaf,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { StatCard, MetricCard } from '../components/ui/StatCard';
import { ProgressBar } from '../components/ui/Loading';
import { fetchLiveSensors } from '../services/api';

interface SoilData {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  ph: number;
  moisture: number;
  temperature: number;
  organicMatter: number;
  salinity: number;
}

const SoilAnalysis = () => {
  const { t } = useTranslation();
  const [soilData, setSoilData] = useState<SoilData>({
    nitrogen: 45,
    phosphorus: 28,
    potassium: 52,
    ph: 6.5,
    moisture: 42,
    temperature: 22,
    organicMatter: 3.2,
    salinity: 0.8,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await fetchLiveSensors();
        setSoilData({
          nitrogen: data.nitrogen || 45,
          phosphorus: data.phosphorus || 28,
          potassium: data.potassium || 52,
          ph: data.phLevel || 6.5,
          moisture: data.soilMoisture || 42,
          temperature: data.soilTemperature || 22,
          organicMatter: 3.2,
          salinity: 0.8,
        });
      } catch (error) {
        console.error('Failed to fetch soil data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getSoilHealthScore = () => {
    // Calculate overall soil health based on parameters
    const npkScore = ((soilData.nitrogen + soilData.phosphorus + soilData.potassium) / 3) / 100;
    const phScore = Math.abs(6.5 - soilData.ph) < 1.5 ? 1 : 0.5;
    const moistureScore = soilData.moisture > 30 && soilData.moisture < 70 ? 1 : 0.6;
    return Math.round((npkScore + phScore + moistureScore) / 3 * 100);
  };

  const getRecommendations = () => {
    const recommendations = [];
    
    if (soilData.nitrogen < 40) {
      recommendations.push({
        type: 'warning',
        title: 'Low Nitrogen',
        description: 'Consider applying urea or ammonium sulfate',
      });
    }
    if (soilData.phosphorus < 25) {
      recommendations.push({
        type: 'warning',
        title: 'Low Phosphorus',
        description: 'Add superphosphate or bone meal',
      });
    }
    if (soilData.ph < 5.5) {
      recommendations.push({
        type: 'danger',
        title: 'Acidic Soil',
        description: 'Apply lime to raise pH level',
      });
    } else if (soilData.ph > 7.5) {
      recommendations.push({
        type: 'danger',
        title: 'Alkaline Soil',
        description: 'Add sulfur or organic matter to lower pH',
      });
    }
    if (soilData.moisture < 30) {
      recommendations.push({
        type: 'warning',
        title: 'Low Moisture',
        description: 'Increase irrigation frequency',
      });
    }

    if (recommendations.length === 0) {
      recommendations.push({
        type: 'success',
        title: 'Optimal Conditions',
        description: 'Soil conditions are suitable for most crops',
      });
    }

    return recommendations;
  };

  const healthScore = getSoilHealthScore();
  const recommendations = getRecommendations();

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Soil Analysis Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-400">Real-time soil health monitoring and recommendations</p>
      </div>

      {/* Health Score Card */}
      <Card className="bg-gradient-to-r from-agri-500 to-earth-500">
        <CardContent className="py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-white">
              <p className="text-lg opacity-80">Overall Soil Health Score</p>
              <div className="flex items-baseline gap-2 mt-2">
                <span className="text-6xl font-bold">{healthScore}</span>
                <span className="text-2xl opacity-80">/100</span>
              </div>
              <Badge 
                variant={healthScore > 70 ? 'success' : healthScore > 50 ? 'warning' : 'danger'}
                className="mt-3"
              >
                {healthScore > 70 ? 'Excellent' : healthScore > 50 ? 'Good' : 'Needs Attention'}
              </Badge>
            </div>
            <div className="w-48 h-48 relative">
              <svg viewBox="0 0 100 100" className="transform -rotate-90">
                <circle
                  cx="50"
                  cy="50"
                  r="45"
                  fill="none"
                  stroke="rgba(255,255,255,0.2)"
                  strokeWidth="8"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="45"
                  fill="none"
                  stroke="white"
                  strokeWidth="8"
                  strokeDasharray={`${healthScore * 2.83} 283`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <Layers className="w-16 h-16 text-white/80" />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* NPK Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          label={t('nitrogen')}
          value={soilData.nitrogen}
          max={100}
          unit="mg/kg"
          icon={<FlaskConical className="w-5 h-5" />}
          color="green"
        />
        <MetricCard
          label={t('phosphorus')}
          value={soilData.phosphorus}
          max={100}
          unit="mg/kg"
          icon={<FlaskConical className="w-5 h-5" />}
          color="blue"
        />
        <MetricCard
          label={t('potassium')}
          value={soilData.potassium}
          max={100}
          unit="mg/kg"
          icon={<FlaskConical className="w-5 h-5" />}
          color="amber"
        />
      </div>

      {/* Other Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title={t('phLevel')}
          value={soilData.ph.toFixed(1)}
          icon={<Activity className="w-6 h-6" />}
          variant={soilData.ph >= 5.5 && soilData.ph <= 7.5 ? 'success' : 'warning'}
        />
        <StatCard
          title={t('soilMoisture')}
          value={`${soilData.moisture}%`}
          icon={<Droplets className="w-6 h-6" />}
          variant={soilData.moisture > 30 && soilData.moisture < 70 ? 'success' : 'warning'}
        />
        <StatCard
          title="Soil Temperature"
          value={`${soilData.temperature}°C`}
          icon={<Thermometer className="w-6 h-6" />}
          variant="info"
        />
        <StatCard
          title="Organic Matter"
          value={`${soilData.organicMatter}%`}
          icon={<Leaf className="w-6 h-6" />}
          variant={soilData.organicMatter > 2 ? 'success' : 'warning'}
        />
      </div>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-agri-600" />
            Soil Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {recommendations.map((rec, index) => (
            <div
              key={index}
              className={`flex items-start gap-3 p-4 rounded-lg ${
                rec.type === 'success' ? 'bg-green-50 dark:bg-green-900/20' :
                rec.type === 'warning' ? 'bg-amber-50 dark:bg-amber-900/20' :
                'bg-red-50 dark:bg-red-900/20'
              }`}
            >
              {rec.type === 'success' ? (
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              ) : rec.type === 'warning' ? (
                <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <p className={`font-medium ${
                  rec.type === 'success' ? 'text-green-800 dark:text-green-400' :
                  rec.type === 'warning' ? 'text-amber-800 dark:text-amber-400' :
                  'text-red-800 dark:text-red-400'
                }`}>
                  {rec.title}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">{rec.description}</p>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Suitable Crops */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Leaf className="w-5 h-5 text-agri-600" />
            Crops Suitable for Current Soil
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {['Rice', 'Wheat', 'Maize', 'Soybean', 'Cotton', 'Sugarcane', 'Groundnut', 'Vegetables'].map((crop) => (
              <div
                key={crop}
                className="flex items-center gap-2 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
              >
                <Leaf className="w-4 h-4 text-agri-600" />
                <span className="text-gray-700 dark:text-gray-300">{crop}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SoilAnalysis;
