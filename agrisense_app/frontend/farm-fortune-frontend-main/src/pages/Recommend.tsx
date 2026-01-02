import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Zap, Droplets, Thermometer, Gauge, Beaker, AlertCircle, CheckCircle, TrendingUp, TrendingDown, Play, Square, Wifi, WifiOff, RefreshCw, Wheat, Loader2, Leaf, CloudRain, Bug, LandPlot } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useTranslation } from "react-i18next";

import { 
  api, 
  type SensorReading, 
  type BackendRecommendation, 
  type PlantListItem, 
  type LiveSensorData, 
  type DeviceStatus,
  type WaterOptimizationRequest,
  type WaterOptimizationResponse,
  type WaterOptimizationInfo,
  type YieldPredictionRequest,
  type YieldPredictionResponse,
  type YieldPredictionInfo
} from "@/lib/api";

interface SensorDataUI {
  temperature: string;
  humidity: string;
  soilMoisture: string;
  ph: string;
  nitrogen: string;
  phosphorus: string;
  potassium: string;
  cropType: string;
  soilType: string;
  areaM2: string;
}

const Recommend = () => {
  const [sensorData, setSensorData] = useState<SensorDataUI>({
    temperature: "",
    humidity: "",
    soilMoisture: "",
    ph: "",
    nitrogen: "",
    phosphorus: "",
    potassium: "",
    cropType: "",
    soilType: "loam",
    areaM2: "100",
  });
  const [recommendation, setRecommendation] = useState<BackendRecommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [plants, setPlants] = useState<PlantListItem[]>([]);
  const [soilTypes, setSoilTypes] = useState<string[]>(["sand","loam","clay"]);
  const [warnings, setWarnings] = useState<Record<string, string>>({});
  
  // Live sensor data state
  const [liveSensorData, setLiveSensorData] = useState<LiveSensorData | null>(null);
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatus[]>([]);
  const [useLiveData, setUseLiveData] = useState(false);
  const [sensorLoading, setSensorLoading] = useState(false);
  const [lastSensorUpdate, setLastSensorUpdate] = useState<string>("");
  
  // Water Optimization ML state
  const [waterModelInfo, setWaterModelInfo] = useState<WaterOptimizationInfo | null>(null);
  const [waterFormData, setWaterFormData] = useState<WaterOptimizationRequest>({
    soil_moisture: 50,
    temperature: 25,
    humidity: 60,
    crop_type: "tomato",
    soil_type: "loam",
    evapotranspiration: 4,
    rainfall_forecast: 0,
    plant_growth_stage: 0.5,
    area_m2: 100,
  });
  const [waterPrediction, setWaterPrediction] = useState<WaterOptimizationResponse | null>(null);
  const [waterPredicting, setWaterPredicting] = useState(false);
  
  // Yield Prediction ML state
  const [yieldModelInfo, setYieldModelInfo] = useState<YieldPredictionInfo | null>(null);
  const [yieldFormData, setYieldFormData] = useState<YieldPredictionRequest>({
    crop_type: "rice",
    soil_type: "loam",
    area_hectares: 5,
    nitrogen: 80,
    phosphorus: 40,
    potassium: 60,
    temperature: 25,
    humidity: 65,
    rainfall: 800,
    irrigation: 200,
    growing_days: 120,
    pest_pressure: 0.2,
  });
  const [yieldPrediction, setYieldPrediction] = useState<YieldPredictionResponse | null>(null);
  const [yieldPredicting, setYieldPredicting] = useState(false);
  
  const { toast } = useToast();
  const { t } = useTranslation();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await api.plants();
        if (!cancelled) setPlants(res.items);
        const soil = await api.soilTypes();
        if (!cancelled && soil.items?.length) setSoilTypes(soil.items);
        
        // Load ML model info
        const [waterInfo, yieldInfo] = await Promise.all([
          api.waterOptimizationInfo().catch(() => null),
          api.yieldPredictionInfo().catch(() => null)
        ]);
        if (!cancelled) {
          if (waterInfo) setWaterModelInfo(waterInfo);
          if (yieldInfo) setYieldModelInfo(yieldInfo);
        }
      } catch {
        // ignore
      }
    })();
    return () => { cancelled = true };
  }, []);
  
  // Water optimization handlers
  const handleWaterChange = useCallback((field: keyof WaterOptimizationRequest, value: number | string) => {
    setWaterFormData(prev => ({ ...prev, [field]: value }));
  }, []);
  
  const handleWaterPredict = useCallback(async () => {
    setWaterPredicting(true);
    try {
      const result = await api.waterOptimization(waterFormData);
      setWaterPrediction(result);
      toast({
        title: "Water Optimization Complete",
        description: `Recommended: ${result.total_irrigation_liters.toFixed(0)}L total irrigation`,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast({ title: "Prediction Failed", description: msg, variant: "destructive" });
    } finally {
      setWaterPredicting(false);
    }
  }, [waterFormData, toast]);
  
  // Yield prediction handlers
  const handleYieldChange = useCallback((field: keyof YieldPredictionRequest, value: number | string) => {
    setYieldFormData(prev => ({ ...prev, [field]: value }));
  }, []);
  
  const handleYieldPredict = useCallback(async () => {
    setYieldPredicting(true);
    try {
      const result = await api.yieldPrediction(yieldFormData);
      setYieldPrediction(result);
      toast({
        title: "Yield Prediction Complete",
        description: `Predicted: ${result.predicted_yield_kg_ha.toFixed(0)} kg/ha (${result.yield_category})`,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast({ title: "Prediction Failed", description: msg, variant: "destructive" });
    } finally {
      setYieldPredicting(false);
    }
  }, [yieldFormData, toast]);
  
  // Utility functions for urgency/yield display
  const getUrgencyColor = (urgency: number) => {
    if (urgency >= 0.7) return "text-red-600 dark:text-red-400";
    if (urgency >= 0.4) return "text-yellow-600 dark:text-yellow-400";
    return "text-green-600 dark:text-green-400";
  };
  
  const getUrgencyBadge = (urgency: number): "destructive" | "secondary" | "default" => {
    if (urgency >= 0.7) return "destructive";
    if (urgency >= 0.4) return "secondary";
    return "default";
  };
  
  const getYieldCategoryColor = (category: string) => {
    switch (category) {
      case "excellent": return "text-green-600 dark:text-green-400";
      case "good": return "text-blue-600 dark:text-blue-400";
      case "average": return "text-yellow-600 dark:text-yellow-400";
      case "below_average": return "text-orange-600 dark:text-orange-400";
      case "poor": return "text-red-600 dark:text-red-400";
      default: return "text-muted-foreground";
    }
  };
  
  const getYieldBadge = (category: string): "default" | "secondary" | "outline" | "destructive" => {
    switch (category) {
      case "excellent": return "default";
      case "good": return "secondary";
      case "average": return "outline";
      case "below_average": return "secondary";
      case "poor": return "destructive";
      default: return "outline";
    }
  };

  // Fetch live sensor data
  const fetchLiveSensorData = async () => {
    setSensorLoading(true);
    try {
      const [sensorRes, deviceRes] = await Promise.all([
        api.sensorsLive().catch(() => null),
        api.sensorsDeviceStatus().catch(() => ({ devices: [] }))
      ]);
      
      if (sensorRes && sensorRes.status === "success") {
        // Handle single device or multiple devices
        let sensorData: LiveSensorData;
        if (sensorRes.data && typeof sensorRes.data === 'object' && 'device_id' in sensorRes.data) {
          sensorData = sensorRes.data as LiveSensorData;
        } else if (sensorRes.data && typeof sensorRes.data === 'object') {
          // Multiple devices - use first available
          const devices = Object.values(sensorRes.data as Record<string, LiveSensorData>);
          sensorData = devices[0];
        } else {
          throw new Error("No sensor data available");
        }
        
        setLiveSensorData(sensorData);
        setLastSensorUpdate(new Date().toLocaleTimeString());
        toast({ title: "Live Data Updated", description: "Sensor data refreshed successfully" });
      }
      
      if (deviceRes && deviceRes.devices) {
        setDeviceStatus(deviceRes.devices);
      }
      
    } catch (error) {
      console.error("Failed to fetch live sensor data:", error);
      toast({ 
        title: "Sensor Data Error", 
        description: "Unable to fetch live sensor data. MQTT bridge may not be running.", 
        variant: "destructive" 
      });
    } finally {
      setSensorLoading(false);
    }
  };

  // Auto-fill sensor data from live sensors
  const useLiveSensorData = () => {
    if (!liveSensorData) {
      toast({ title: "No Live Data", description: "Please fetch live sensor data first", variant: "destructive" });
      return;
    }
    
    setSensorData(prev => ({
      ...prev,
      temperature: liveSensorData.air_temperature.toFixed(1),
      humidity: liveSensorData.humidity.toFixed(1),
      soilMoisture: liveSensorData.soil_moisture_percentage.toFixed(1),
      ph: liveSensorData.ph_level.toFixed(1),
      // Keep existing crop type and soil type
    }));
    
    setUseLiveData(true);
    toast({ title: "Live Data Applied", description: "Sensor values automatically filled" });
  };

  // Get live recommendations directly from sensor data
  const getLiveRecommendations = async () => {
    setLoading(true);
    try {
      const res = await api.sensorsRecommendationsLive();
      setRecommendation(res.recommendations);
      
      // Update sensor data display with live values
      if (res.sensor_data) {
        setSensorData(prev => ({
          ...prev,
          temperature: res.sensor_data.air_temperature.toFixed(1),
          humidity: res.sensor_data.humidity.toFixed(1),
          soilMoisture: res.sensor_data.soil_moisture_percentage.toFixed(1),
          ph: res.sensor_data.ph_level.toFixed(1),
        }));
      }
      
      toast({ 
        title: "Live Recommendations", 
        description: "Recommendations generated from real-time sensor data" 
      });
    } catch (error) {
      console.error("Failed to get live recommendations:", error);
      toast({ 
        title: "Live Recommendations Failed", 
        description: "Unable to get live recommendations. Using manual input instead.", 
        variant: "destructive" 
      });
      // Fall back to manual recommendations
      await generateRecommendations();
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: keyof SensorDataUI, value: string) => {
    setSensorData(prev => ({ ...prev, [field]: value }));
    // Basic inline validation warnings
    const w: Record<string, string> = {};
    const ph = field === 'ph' ? parseFloat(value || 'NaN') : parseFloat(sensorData.ph || 'NaN');
    if (!Number.isNaN(ph) && (ph < 3.5 || ph > 9.5)) w.ph = t('warn_ph_range');
    const moisture = field === 'soilMoisture' ? parseFloat(value || 'NaN') : parseFloat(sensorData.soilMoisture || 'NaN');
    if (!Number.isNaN(moisture) && (moisture < 0 || moisture > 100)) w.soilMoisture = t('warn_moisture_range');
    const temp = field === 'temperature' ? parseFloat(value || 'NaN') : parseFloat(sensorData.temperature || 'NaN');
    if (!Number.isNaN(temp) && (temp < -10 || temp > 60)) w.temperature = t('warn_temp_range');
    const area = field === 'areaM2' ? parseFloat(value || 'NaN') : parseFloat(sensorData.areaM2 || 'NaN');
    if (!Number.isNaN(area) && area <= 0) w.areaM2 = t('warn_area_positive');
    setWarnings(w);
  };

  const generateRecommendations = async () => {
    setLoading(true);
    try {
      const payload: SensorReading = {
        plant: sensorData.cropType || "generic",
        soil_type: (sensorData.soilType || "loam").toLowerCase(),
        area_m2: Math.max(1, parseFloat(sensorData.areaM2 || "100")),
        ph: parseFloat(sensorData.ph || "6.5"),
        moisture_pct: parseFloat(sensorData.soilMoisture || "40"),
        temperature_c: parseFloat(sensorData.temperature || "28"),
        ec_dS_m: 1.0,
        n_ppm: sensorData.nitrogen ? parseFloat(sensorData.nitrogen) : undefined,
        p_ppm: sensorData.phosphorus ? parseFloat(sensorData.phosphorus) : undefined,
        k_ppm: sensorData.potassium ? parseFloat(sensorData.potassium) : undefined,
      };
      const res = await api.recommend(payload);
      setRecommendation(res);
      toast({ title: "Analysis Complete", description: "Smart recommendations generated." });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      toast({ title: "Request failed", description: msg, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const startIrrigation = async () => {
    try {
      // Use suggested runtime if provided; else a simple heuristic based on water_liters and assumed flow
      const seconds = recommendation?.suggested_runtime_min
        ? Math.max(1, Math.round(recommendation.suggested_runtime_min * 60))
        : Math.max(60, Math.round((recommendation?.water_liters || 0) / Math.max(1, recommendation?.assumed_flow_lpm || 20) * 60));
      const r = await api.irrigationStart("Z1", seconds);
      toast({ title: r.ok ? "Irrigation start sent" : "Queued", description: r.status });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      toast({ title: "Failed to start", description: msg, variant: "destructive" });
    }
  };

  const stopIrrigation = async () => {
    try {
      const r = await api.irrigationStop("Z1");
      toast({ title: r.ok ? "Stop sent" : "Stop queued", description: r.status });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      toast({ title: "Failed to stop", description: msg, variant: "destructive" });
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case "high": return <AlertCircle className="w-4 h-4 text-destructive" />;
      case "medium": return <TrendingUp className="w-4 h-4 text-accent-foreground" />;
      default: return <CheckCircle className="w-4 h-4 text-primary" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high": return "border-l-destructive bg-destructive/5";
      case "medium": return "border-l-accent-foreground bg-accent";
      default: return "border-l-primary bg-primary/5";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-secondary">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">{t("smart_recommendations")}</h1>
          <p className="text-muted-foreground">{t("enter_sensor_prompt")}</p>
        </div>

        {/* Live Sensor Data Card */}
        <Card className="mb-8 shadow-medium border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Wifi className="w-5 h-5 text-primary" />
                <span>Live Sensor Data</span>
                <Badge variant={liveSensorData ? "default" : "secondary"}>
                  {liveSensorData ? "Connected" : "No Data"}
                </Badge>
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={fetchLiveSensorData}
                  disabled={sensorLoading}
                >
                  {sensorLoading ? (
                    <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                  ) : (
                    <RefreshCw className="w-4 h-4 mr-2" />
                  )}
                  Refresh
                </Button>
                {liveSensorData && (
                  <Button
                    variant="default"
                    size="sm"
                    onClick={useLiveSensorData}
                  >
                    Use Live Data
                  </Button>
                )}
              </div>
            </CardTitle>
            {lastSensorUpdate && (
              <CardDescription>
                Last updated: {lastSensorUpdate}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent>
            {liveSensorData ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <div className="flex items-center space-x-2">
                  <Thermometer className="w-4 h-4 text-orange-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">Air Temp</p>
                    <p className="text-lg font-semibold">{liveSensorData.air_temperature.toFixed(1)}°C</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Droplets className="w-4 h-4 text-blue-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">Humidity</p>
                    <p className="text-lg font-semibold">{liveSensorData.humidity.toFixed(1)}%</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Gauge className="w-4 h-4 text-green-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">Soil Moisture</p>
                    <p className="text-lg font-semibold">{liveSensorData.soil_moisture_percentage.toFixed(1)}%</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Thermometer className="w-4 h-4 text-amber-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">Soil Temp</p>
                    <p className="text-lg font-semibold">{liveSensorData.soil_temperature.toFixed(1)}°C</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Beaker className="w-4 h-4 text-purple-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">pH Level</p>
                    <p className="text-lg font-semibold">{liveSensorData.ph_level.toFixed(1)}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Zap className="w-4 h-4 text-yellow-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">Light</p>
                    <p className="text-lg font-semibold">{liveSensorData.light_intensity_percentage.toFixed(0)}%</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <WifiOff className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground mb-4">
                  No live sensor data available. Connect ESP32 sensors and start MQTT bridge.
                </p>
                <Button onClick={fetchLiveSensorData} disabled={sensorLoading}>
                  {sensorLoading ? "Connecting..." : "Connect to Sensors"}
                </Button>
              </div>
            )}
            
            {/* Device Status */}
            {deviceStatus.length > 0 && (
              <div className="mt-6 pt-6 border-t">
                <h4 className="text-sm font-semibold mb-3">Device Status</h4>
                <div className="flex flex-wrap gap-2">
                  {deviceStatus.map((device) => (
                    <Badge
                      key={device.device_id}
                      variant={device.is_connected ? "default" : "destructive"}
                      className="flex items-center space-x-1"
                    >
                      {device.is_connected ? (
                        <Wifi className="w-3 h-3" />
                      ) : (
                        <WifiOff className="w-3 h-3" />
                      )}
                      <span>{device.device_id}</span>
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            
            {/* Quick Action Buttons */}
            {liveSensorData && (
              <div className="mt-6 pt-6 border-t flex space-x-3">
                <Button 
                  onClick={getLiveRecommendations}
                  disabled={loading}
                  className="flex-1"
                >
                  {loading ? "Analyzing..." : "Get Live Recommendations"}
                </Button>
                <Button 
                  variant="outline"
                  onClick={() => window.open('/api/sensors/soil-analysis/live', '_blank')}
                  className="flex-1"
                >
                  Live Soil Analysis
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Sensor Input Form */}
          <Card className="shadow-medium">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Gauge className="w-5 h-5 text-primary" />
                <span>{t("sensor_data_input")}</span>
              </CardTitle>
              <CardDescription>
                {t("enter_readings")}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Environmental Sensors */}
              <div className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground">{t("environmental_conditions")}</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="temperature" className="flex items-center space-x-2">
                      <Thermometer className="w-4 h-4" />
                      <span>{t("temperature_c_label")}</span>
                    </Label>
                    <Input
                      id="temperature"
                      placeholder="25.5"
                      value={sensorData.temperature}
                      onChange={(e) => handleInputChange("temperature", e.target.value)}
                    />
                  </div>
                  <div>
                    <Label htmlFor="humidity" className="flex items-center space-x-2">
                      <Droplets className="w-4 h-4" />
                      <span>{t("humidity_pct")}</span>
                    </Label>
                    <Input
                      id="humidity"
                      placeholder="65"
                      value={sensorData.humidity}
                      onChange={(e) => handleInputChange("humidity", e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* Soil Sensors */}
              <div className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground">{t("soil_analysis")}</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="soilMoisture">{t("soil_moisture_pct")}</Label>
                    <Input
                      id="soilMoisture"
                      placeholder="45"
                      value={sensorData.soilMoisture}
                      onChange={(e) => handleInputChange("soilMoisture", e.target.value)}
                    />
                    {warnings.soilMoisture && (<div className="text-xs text-destructive mt-1">{warnings.soilMoisture}</div>)}
                  </div>
                  <div>
                    <Label htmlFor="ph">{t("ph_level")}</Label>
                    <Input
                      id="ph"
                      placeholder="6.5"
                      value={sensorData.ph}
                      onChange={(e) => handleInputChange("ph", e.target.value)}
                    />
                    {warnings.ph && (<div className="text-xs text-destructive mt-1">{warnings.ph}</div>)}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="soilType">{t("soil_type")}</Label>
                    <Select value={sensorData.soilType} onValueChange={(v)=>handleInputChange("soilType", v)}>
                      <SelectTrigger>
                        <SelectValue placeholder={t("select_soil_type")} />
                      </SelectTrigger>
                      <SelectContent>
                        {soilTypes.map(s => (<SelectItem key={s} value={s}>{s}</SelectItem>))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="areaM2">{t("area_m2")}</Label>
                    <Input
                      id="areaM2"
                      placeholder="100"
                      value={sensorData.areaM2}
                      onChange={(e) => handleInputChange("areaM2", e.target.value)}
                    />
                    {warnings.areaM2 && (<div className="text-xs text-destructive mt-1">{warnings.areaM2}</div>)}
                  </div>
                </div>
              </div>

              {/* Nutrient Levels */}
              <div className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground flex items-center space-x-2">
                  <Beaker className="w-4 h-4" />
                  <span>{t("nutrient_levels_ppm")}</span>
                </h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <Label htmlFor="nitrogen">{t("nitrogen_n")}</Label>
                    <Input
                      id="nitrogen"
                      placeholder="120"
                      value={sensorData.nitrogen}
                      onChange={(e) => handleInputChange("nitrogen", e.target.value)}
                    />
                  </div>
                  <div>
                    <Label htmlFor="phosphorus">{t("phosphorus_p")}</Label>
                    <Input
                      id="phosphorus"
                      placeholder="45"
                      value={sensorData.phosphorus}
                      onChange={(e) => handleInputChange("phosphorus", e.target.value)}
                    />
                  </div>
                  <div>
                    <Label htmlFor="potassium">{t("potassium_k")}</Label>
                    <Input
                      id="potassium"
                      placeholder="80"
                      value={sensorData.potassium}
                      onChange={(e) => handleInputChange("potassium", e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* Crop Selection */}
              <div>
                <Label htmlFor="cropType">{t("crop_type")}</Label>
                <Select value={sensorData.cropType} onValueChange={(value) => handleInputChange("cropType", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder={t("select_crop_type")} />
                  </SelectTrigger>
                  <SelectContent>
                    {plants.map((p) => (
                      <SelectItem key={p.value} value={p.value}>{p.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Button 
                onClick={generateRecommendations} 
                className="w-full bg-gradient-primary hover:shadow-glow transition-spring"
                disabled={loading || !sensorData.cropType || !sensorData.soilType || !(parseFloat(sensorData.areaM2||'0')>0)}
              >
                <Zap className="w-4 h-4 mr-2" />
                {loading ? t("analyzing") : t("generate_recommendations")}
              </Button>
            </CardContent>
          </Card>

          {/* Recommendations Display */}
          <Card className="shadow-medium">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                <span>{t("smart_recommendations")}</span>
              </CardTitle>
              <CardDescription>
                {t("ai_insights")}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!recommendation ? (
                <div className="text-center py-12">
                  <Zap className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">{t("enter_sensor_prompt")}</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="border-l-4 p-4 rounded-lg bg-primary/5 border-l-primary">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-semibold text-foreground flex items-center space-x-2">
                        {getPriorityIcon("high")}
                        <span>{t("irrigation_fertilizer_plan")}</span>
                      </h4>
                      <div className="flex gap-2">
                        <Button size="sm" onClick={startIrrigation}>
                          <Play className="w-4 h-4 mr-1" /> {t("start")}
                        </Button>
                        <Button size="sm" variant="destructive" onClick={stopIrrigation}>
                          <Square className="w-4 h-4 mr-1" /> {t("stop")}
                        </Button>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">{t("water_liters_total")}</div>
                        <div className="text-foreground font-medium">{recommendation.water_liters}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">{t("water_source")}</div>
                        <div className="text-foreground font-medium">{recommendation.water_source ?? '-'}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">{t("irrigation_cycles")}</div>
                        <div className="text-foreground font-medium">{recommendation.irrigation_cycles ?? '-'}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">N (g)</div>
                        <div className="text-foreground font-medium">{recommendation.fert_n_g}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">P (g)</div>
                        <div className="text-foreground font-medium">{recommendation.fert_p_g}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">K (g)</div>
                        <div className="text-foreground font-medium">{recommendation.fert_k_g}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">{t("runtime_min")}</div>
                        <div className="text-foreground font-medium">{recommendation.suggested_runtime_min ?? '-'}</div>
                      </div>
                      {recommendation.best_time && (
                        <div>
                          <div className="text-muted-foreground">{t("best_time")}</div>
                          <div className="text-foreground font-medium">{recommendation.best_time}</div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Impact Metrics */}
                  {(recommendation.expected_savings_liters != null || recommendation.expected_cost_saving_rs != null || recommendation.expected_co2e_kg != null) && (
                    <div className="border rounded-lg p-4 bg-card">
                      <div className="text-sm font-semibold mb-2">{t("impact")}</div>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <div className="text-muted-foreground">{t("water_saved_l")}</div>
                          <div className="text-foreground font-medium">{Math.round(recommendation.expected_savings_liters || 0)}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">{t("cost_saved_rs2")}</div>
                          <div className="text-foreground font-medium">{Math.round(recommendation.expected_cost_saving_rs || 0)}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">{t("co2e_kg2")}</div>
                          <div className="text-foreground font-medium">{(recommendation.expected_co2e_kg || 0).toFixed(2)}</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Fertilizer equivalents */}
                  {recommendation.fertilizer_equivalents && (
                    <div className="border rounded-lg p-4 bg-card">
                      <div className="text-sm font-semibold mb-2">{t("fertilizer_equivalents")}</div>
                      <ul className="text-sm grid grid-cols-2 gap-2">
                        {Object.entries(recommendation.fertilizer_equivalents).map(([k, v]) => (
                          <li key={k} className="flex justify-between">
                            <span className="text-muted-foreground">{k}</span>
                            <span className="font-medium">{v}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {/* Detailed Tips */}
                  {recommendation.tips && recommendation.tips.length > 0 && (
                    <div className="border rounded-lg p-4 bg-card">
                      <div className="text-sm font-semibold mb-2">{t("detailed_tips")}</div>
                      <ul className="list-disc ml-5 space-y-1 text-sm">
                        {recommendation.tips.map((tip, i) => (
                          <li key={i}>{tip}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {recommendation.notes && recommendation.notes.length > 0 && (
                    <div className="border-l-4 p-4 rounded-lg bg-accent border-l-accent-foreground">
                      <div className="text-sm text-accent-foreground font-semibold mb-2">{t("notes")}</div>
                      <ul className="list-disc ml-5 space-y-1 text-sm">
                        {recommendation.notes.map((n, i) => (<li key={i}>{n}</li>))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        
        {/* ML Models Section */}
        <div className="mt-8">
          <Tabs defaultValue="water" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="water" className="flex items-center gap-2">
                <Droplets className="w-4 h-4" />
                Water AI Optimization
                {waterModelInfo && (
                  <Badge variant={waterModelInfo.model_available ? "default" : "destructive"} className="ml-2 text-xs">
                    {waterModelInfo.model_available ? `v${waterModelInfo.model_version}` : "Unavailable"}
                  </Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="yield" className="flex items-center gap-2">
                <Wheat className="w-4 h-4" />
                Yield AI Prediction
                {yieldModelInfo && (
                  <Badge variant={yieldModelInfo.model_available ? "default" : "destructive"} className="ml-2 text-xs">
                    {yieldModelInfo.model_available ? `v${yieldModelInfo.model_version}` : "Unavailable"}
                  </Badge>
                )}
              </TabsTrigger>
            </TabsList>
            
            {/* Water Optimization Tab */}
            <TabsContent value="water">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Water Input Form */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Droplets className="h-5 w-5 text-blue-500" />
                      Environmental Conditions
                    </CardTitle>
                    <CardDescription>
                      Enter current field conditions for irrigation recommendation
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Crop & Soil Selection */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Crop Type</Label>
                        <Select value={waterFormData.crop_type} onValueChange={(v) => handleWaterChange("crop_type", v)}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {(waterModelInfo?.supported_crops || ["tomato", "rice", "wheat", "corn", "soybean"]).map((c) => (
                              <SelectItem key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label>Soil Type</Label>
                        <Select value={waterFormData.soil_type} onValueChange={(v) => handleWaterChange("soil_type", v)}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {(waterModelInfo?.supported_soils || ["sandy", "loam", "clay", "silt"]).map((s) => (
                              <SelectItem key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    
                    {/* Sliders */}
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label className="flex items-center gap-1"><Gauge className="w-4 h-4" />Soil Moisture</Label>
                          <span className="text-sm text-muted-foreground">{waterFormData.soil_moisture}%</span>
                        </div>
                        <Slider value={[waterFormData.soil_moisture]} onValueChange={([v]) => handleWaterChange("soil_moisture", v)} min={0} max={100} step={1} />
                        <Progress value={waterFormData.soil_moisture} className="h-2" />
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label className="flex items-center gap-1"><Thermometer className="w-4 h-4" />Temperature</Label>
                          <span className="text-sm text-muted-foreground">{waterFormData.temperature}°C</span>
                        </div>
                        <Slider value={[waterFormData.temperature]} onValueChange={([v]) => handleWaterChange("temperature", v)} min={0} max={50} step={0.5} />
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label className="flex items-center gap-1"><Droplets className="w-4 h-4" />Humidity</Label>
                          <span className="text-sm text-muted-foreground">{waterFormData.humidity}%</span>
                        </div>
                        <Slider value={[waterFormData.humidity]} onValueChange={([v]) => handleWaterChange("humidity", v)} min={0} max={100} step={1} />
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label className="flex items-center gap-1"><CloudRain className="w-4 h-4" />Rainfall Forecast</Label>
                          <span className="text-sm text-muted-foreground">{waterFormData.rainfall_forecast} mm</span>
                        </div>
                        <Slider value={[waterFormData.rainfall_forecast]} onValueChange={([v]) => handleWaterChange("rainfall_forecast", v)} min={0} max={50} step={0.5} />
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label className="flex items-center gap-1"><Leaf className="w-4 h-4" />Growth Stage</Label>
                          <span className="text-sm text-muted-foreground">{(waterFormData.plant_growth_stage * 100).toFixed(0)}%</span>
                        </div>
                        <Slider value={[waterFormData.plant_growth_stage]} onValueChange={([v]) => handleWaterChange("plant_growth_stage", v)} min={0} max={1} step={0.05} />
                      </div>
                    </div>
                    
                    {/* Area Input */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Area (m²)</Label>
                        <Input type="number" value={waterFormData.area_m2} onChange={(e) => handleWaterChange("area_m2", parseFloat(e.target.value) || 0)} min={1} />
                      </div>
                      <div className="space-y-2">
                        <Label>ET (mm/day)</Label>
                        <Input type="number" value={waterFormData.evapotranspiration} onChange={(e) => handleWaterChange("evapotranspiration", parseFloat(e.target.value) || 0)} min={0} step={0.1} />
                      </div>
                    </div>
                    
                    <Button className="w-full" size="lg" onClick={handleWaterPredict} disabled={waterPredicting}>
                      {waterPredicting ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Analyzing...</>) : (<><Droplets className="mr-2 h-4 w-4" />Get Water Recommendation</>)}
                    </Button>
                  </CardContent>
                </Card>
                
                {/* Water Results Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      Irrigation Recommendation
                    </CardTitle>
                    <CardDescription>ML-predicted optimal irrigation schedule</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {!waterPrediction && !waterPredicting && (
                      <div className="text-center py-12 text-muted-foreground">
                        <Droplets className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Enter conditions and click "Get Water Recommendation" to see ML predictions</p>
                      </div>
                    )}
                    {waterPredicting && (
                      <div className="text-center py-12">
                        <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-primary" />
                        <p className="text-muted-foreground">Analyzing with ML model...</p>
                      </div>
                    )}
                    {waterPrediction && !waterPredicting && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                            <div className="text-sm text-muted-foreground">Volume/m²</div>
                            <div className="text-2xl font-bold text-blue-600">{waterPrediction.irrigation_volume_per_m2.toFixed(2)} L</div>
                          </div>
                          <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                            <div className="text-sm text-muted-foreground">Total Irrigation</div>
                            <div className="text-2xl font-bold text-green-600">{waterPrediction.total_irrigation_liters.toFixed(0)} L</div>
                          </div>
                          <div className="p-4 bg-amber-50 dark:bg-amber-950 rounded-lg">
                            <div className="text-sm text-muted-foreground">Urgency</div>
                            <div className={`text-xl font-bold ${getUrgencyColor(waterPrediction.irrigation_urgency)}`}>
                              <Badge variant={getUrgencyBadge(waterPrediction.irrigation_urgency)}>
                                {(waterPrediction.irrigation_urgency * 100).toFixed(0)}%
                              </Badge>
                            </div>
                          </div>
                          <div className="p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                            <div className="text-sm text-muted-foreground">Frequency</div>
                            <div className="text-xl font-bold text-purple-600">Every {waterPrediction.recommended_frequency_days} days</div>
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                          <span className="text-sm">Model Confidence</span>
                          <div className="flex items-center gap-2">
                            <Progress value={waterPrediction.confidence * 100} className="w-24 h-2" />
                            <span className="text-sm font-medium">{(waterPrediction.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        
                        {waterPrediction.recommendations.length > 0 && (
                          <div className="border rounded-lg p-4">
                            <h4 className="font-semibold mb-2">Recommendations</h4>
                            <ul className="space-y-1 text-sm">
                              {waterPrediction.recommendations.map((r, i) => (
                                <li key={i} className="flex items-start gap-2">
                                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                                  <span>{r}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            {/* Yield Prediction Tab */}
            <TabsContent value="yield">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Yield Input Form */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <LandPlot className="h-5 w-5 text-amber-500" />
                      Field & Crop Information
                    </CardTitle>
                    <CardDescription>Enter field conditions and inputs for yield forecast</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Crop & Soil Selection */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Crop Type</Label>
                        <Select value={yieldFormData.crop_type} onValueChange={(v) => handleYieldChange("crop_type", v)}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {(yieldModelInfo?.supported_crops || ["rice", "wheat", "corn", "soybean", "cotton"]).map((c) => (
                              <SelectItem key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label>Soil Type</Label>
                        <Select value={yieldFormData.soil_type} onValueChange={(v) => handleYieldChange("soil_type", v)}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {(yieldModelInfo?.supported_soils || ["sandy", "loam", "clay", "silt"]).map((s) => (
                              <SelectItem key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    
                    {/* Area & Growing Days */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Area (hectares)</Label>
                        <Input type="number" value={yieldFormData.area_hectares} onChange={(e) => handleYieldChange("area_hectares", parseFloat(e.target.value) || 0)} min={0.1} max={1000} step={0.1} />
                      </div>
                      <div className="space-y-2">
                        <Label>Growing Days</Label>
                        <Input type="number" value={yieldFormData.growing_days} onChange={(e) => handleYieldChange("growing_days", parseInt(e.target.value) || 0)} min={30} max={365} />
                      </div>
                    </div>
                    
                    {/* NPK Inputs */}
                    <div className="space-y-3">
                      <Label className="text-base font-medium">Fertilizer (kg/ha)</Label>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs">Nitrogen (N)</Label>
                          <Input type="number" value={yieldFormData.nitrogen} onChange={(e) => handleYieldChange("nitrogen", parseFloat(e.target.value) || 0)} min={0} max={300} />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">Phosphorus (P)</Label>
                          <Input type="number" value={yieldFormData.phosphorus} onChange={(e) => handleYieldChange("phosphorus", parseFloat(e.target.value) || 0)} min={0} max={200} />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">Potassium (K)</Label>
                          <Input type="number" value={yieldFormData.potassium} onChange={(e) => handleYieldChange("potassium", parseFloat(e.target.value) || 0)} min={0} max={200} />
                        </div>
                      </div>
                    </div>
                    
                    {/* Climate */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="flex items-center gap-1"><Thermometer className="w-4 h-4" />Temperature (°C)</Label>
                        <Slider value={[yieldFormData.temperature]} onValueChange={([v]) => handleYieldChange("temperature", v)} min={5} max={45} step={0.5} />
                        <div className="text-right text-sm text-muted-foreground">{yieldFormData.temperature}°C</div>
                      </div>
                      <div className="space-y-2">
                        <Label className="flex items-center gap-1"><Droplets className="w-4 h-4" />Humidity (%)</Label>
                        <Slider value={[yieldFormData.humidity]} onValueChange={([v]) => handleYieldChange("humidity", v)} min={20} max={100} step={1} />
                        <div className="text-right text-sm text-muted-foreground">{yieldFormData.humidity}%</div>
                      </div>
                    </div>
                    
                    {/* Water Inputs */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="flex items-center gap-1"><CloudRain className="w-4 h-4" />Rainfall (mm)</Label>
                        <Input type="number" value={yieldFormData.rainfall} onChange={(e) => handleYieldChange("rainfall", parseFloat(e.target.value) || 0)} min={0} max={3000} />
                      </div>
                      <div className="space-y-2">
                        <Label className="flex items-center gap-1"><Droplets className="w-4 h-4" />Irrigation (mm)</Label>
                        <Input type="number" value={yieldFormData.irrigation} onChange={(e) => handleYieldChange("irrigation", parseFloat(e.target.value) || 0)} min={0} max={1000} />
                      </div>
                    </div>
                    
                    {/* Pest Pressure */}
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label className="flex items-center gap-1"><Bug className="w-4 h-4" />Pest Pressure</Label>
                        <span className="text-sm text-muted-foreground">{(yieldFormData.pest_pressure * 100).toFixed(0)}%</span>
                      </div>
                      <Slider value={[yieldFormData.pest_pressure]} onValueChange={([v]) => handleYieldChange("pest_pressure", v)} min={0} max={1} step={0.05} />
                    </div>
                    
                    <Button className="w-full" size="lg" onClick={handleYieldPredict} disabled={yieldPredicting}>
                      {yieldPredicting ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Predicting...</>) : (<><Wheat className="mr-2 h-4 w-4" />Predict Yield</>)}
                    </Button>
                  </CardContent>
                </Card>
                
                {/* Yield Results Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Wheat className="h-5 w-5 text-amber-500" />
                      Yield Forecast
                    </CardTitle>
                    <CardDescription>ML-predicted crop yield and comparison</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {!yieldPrediction && !yieldPredicting && (
                      <div className="text-center py-12 text-muted-foreground">
                        <Wheat className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Enter field information and click "Predict Yield" to see ML predictions</p>
                      </div>
                    )}
                    {yieldPredicting && (
                      <div className="text-center py-12">
                        <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-primary" />
                        <p className="text-muted-foreground">Running yield prediction model...</p>
                      </div>
                    )}
                    {yieldPrediction && !yieldPredicting && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 bg-amber-50 dark:bg-amber-950 rounded-lg col-span-2">
                            <div className="text-sm text-muted-foreground">Predicted Yield</div>
                            <div className="flex items-center gap-2">
                              <span className="text-3xl font-bold text-amber-600">{yieldPrediction.predicted_yield_kg_ha.toFixed(0)}</span>
                              <span className="text-lg text-muted-foreground">kg/ha</span>
                              <Badge variant={getYieldBadge(yieldPrediction.yield_category)} className="ml-2">
                                {yieldPrediction.yield_category.replace("_", " ")}
                              </Badge>
                            </div>
                          </div>
                          <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                            <div className="text-sm text-muted-foreground">Total Production</div>
                            <div className="text-2xl font-bold text-green-600">{yieldPrediction.total_production_kg.toFixed(0)} kg</div>
                          </div>
                          <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                            <div className="text-sm text-muted-foreground">Regional Average</div>
                            <div className="flex items-center gap-1">
                              <span className="text-2xl font-bold text-blue-600">{yieldPrediction.regional_average_yield.toFixed(0)}</span>
                              <span className="text-sm text-muted-foreground">kg/ha</span>
                              {yieldPrediction.predicted_yield_kg_ha > yieldPrediction.regional_average_yield * 1.1 ? (
                                <TrendingUp className="w-5 h-5 text-green-500" />
                              ) : yieldPrediction.predicted_yield_kg_ha < yieldPrediction.regional_average_yield * 0.9 ? (
                                <TrendingDown className="w-5 h-5 text-red-500" />
                              ) : null}
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                          <span className="text-sm">Model Confidence</span>
                          <div className="flex items-center gap-2">
                            <Progress value={yieldPrediction.confidence * 100} className="w-24 h-2" />
                            <span className="text-sm font-medium">{(yieldPrediction.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        
                        {yieldPrediction.recommendations.length > 0 && (
                          <div className="border rounded-lg p-4">
                            <h4 className="font-semibold mb-2">Yield Improvement Tips</h4>
                            <ul className="space-y-1 text-sm">
                              {yieldPrediction.recommendations.map((r, i) => (
                                <li key={i} className="flex items-start gap-2">
                                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                                  <span>{r}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default Recommend;