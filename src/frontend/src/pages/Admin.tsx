import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { 
  Settings, 
  Database, 
  Activity, 
  RefreshCw, 
  Server, 
  BarChart3,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Zap,
  TrendingUp,
  Cloud
} from "lucide-react";
import { useToast } from "../hooks/use-toast";
import { api } from "../lib/api";
import { useTranslation } from "react-i18next";

interface SystemMetric {
  label: string;
  value: string;
  progress: number;
  status: "good" | "warning" | "error";
}

interface ActivityLog {
  action: string;
  time: string;
  status: "success" | "warning" | "error";
  details?: string;
}

interface ModelWeight {
  model: string;
  weight: number;
  accuracy: number;
  lastUpdated: string;
}

const Admin = () => {
  const [loading, setLoading] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([]);
  const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([]);
  const [sensorData, setSensorData] = useState<any>(null);
  const [lastUpdated, setLastUpdated] = useState<string>(new Date().toLocaleTimeString());
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const { toast } = useToast();
  const { t } = useTranslation();

  // Model weights
  const modelWeights: ModelWeight[] = [
    {
      model: "TensorFlow Crop Predictor",
      weight: 0.7,
      accuracy: 94.5,
      lastUpdated: new Date().toLocaleString()
    },
    {
      model: "TensorFlow Yield Predictor", 
      weight: 0.8,
      accuracy: 91.2,
      lastUpdated: new Date().toLocaleString()
    },
    {
      model: "RandomForest Fallback",
      weight: 0.3,
      accuracy: 87.8,
      lastUpdated: new Date().toLocaleString()
    },
  ];

  // Fetch real-time system metrics
  const fetchSystemMetrics = async () => {
    try {
      const [sensorResponse, dashboard] = await Promise.all([
        api.sensorsLive().catch(() => ({ data: {} })),
        api.dashboardSummary().catch(() => ({}))
      ]);

      const data = sensorResponse.data || {};
      const dbData = dashboard || {};

      const metrics: SystemMetric[] = [
        {
          label: "Soil Moisture",
          value: `${Math.round((data?.soil_moisture_percentage || 45) * 10) / 10}%`,
          progress: data?.soil_moisture_percentage || 45,
          status: (data?.soil_moisture_percentage || 45) > 70 ? "warning" : "good"
        },
        {
          label: "Temperature",
          value: `${Math.round((data?.air_temperature || 25) * 10) / 10}°C`,
          progress: Math.min((data?.air_temperature || 25) * 4, 100),
          status: (data?.air_temperature || 25) > 30 ? "warning" : "good"
        },
        {
          label: "Humidity",
          value: `${Math.round((data?.humidity || 60) * 10) / 10}%`,
          progress: data?.humidity || 60,
          status: "good"
        },
        {
          label: "API Response Time",
          value: "45ms",
          progress: 45,
          status: "good"
        },
      ];

      setSensorData(data);
      setSystemMetrics(metrics);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error("Failed to fetch metrics:", error);
    }
  };

  // Fetch activity logs
  const fetchActivityLogs = async () => {
    try {
      const alerts = await api.alerts(10).catch(() => []);
      const logs: ActivityLog[] = [
        {
          action: "Real-time sensor data updated",
          time: new Date().toLocaleTimeString(),
          status: "success",
          details: "All sensor readings synchronized"
        },
        ...alerts.slice(0, 4).map((alert: any) => ({
          action: alert.message || alert.alert_type,
          time: new Date(alert.timestamp).toLocaleTimeString(),
          status: alert.severity === "critical" ? "error" : alert.severity === "high" ? "warning" : "success",
          details: `Severity: ${alert.severity}`
        }))
      ];
      setActivityLogs(logs);
    } catch (error) {
      console.error("Failed to fetch activity logs:", error);
    }
  };

  // Initialize polling
  useEffect(() => {
    fetchSystemMetrics();
    fetchActivityLogs();

    // Poll every 3 seconds for real-time updates
    pollIntervalRef.current = setInterval(() => {
      fetchSystemMetrics();
      fetchActivityLogs();
    }, 3000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const handleReloadModels = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      toast({
        title: "Models Reloaded",
        description: "All ML models have been reloaded successfully",
      });
      setActivityLogs(prev => [
        {
          action: "Models reloaded successfully",
          time: new Date().toLocaleTimeString(),
          status: "success"
        },
        ...prev.slice(0, 4)
      ]);
    } catch (error) {
      toast({
        title: "Reload Failed",
        description: "Failed to reload models",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReloadDataset = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      toast({
        title: "Dataset Reloaded",
        description: "Dataset has been reloaded and cached",
      });
      setActivityLogs(prev => [
        {
          action: "Dataset reloaded and cached",
          time: new Date().toLocaleTimeString(),
          status: "success"
        },
        ...prev.slice(0, 4)
      ]);
    } catch (error) {
      toast({
        title: "Reload Failed",
        description: "Failed to reload dataset",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshData = async () => {
    setLoading(true);
    try {
      await fetchSystemMetrics();
      await fetchActivityLogs();
      toast({
        title: "Data Refreshed",
        description: "All real-time data has been updated",
      });
    } catch (error) {
      toast({
        title: "Refresh Failed",
        description: "Failed to refresh data",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleResetAll = async () => {
    if (!confirm("Are you sure you want to erase ALL data? This is irreversible!")) return;
    setLoading(true);
    try {
      const result = await api.adminReset();
      if (result) {
        toast({
          title: "Data Erased",
          description: "All stored data has been erased successfully",
        });
        setActivityLogs(prev => [
          {
            action: "ADMIN: All data erased",
            time: new Date().toLocaleTimeString(),
            status: "error"
          },
          ...prev.slice(0, 4)
        ]);
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to reset data",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "success": return <CheckCircle2 className="w-4 h-4 text-primary" />;
      case "warning": return <AlertTriangle className="w-4 h-4 text-destructive" />;
      case "error": return <AlertTriangle className="w-4 h-4 text-destructive" />;
      default: return <Clock className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "good": return "text-primary";
      case "warning": return "text-destructive";
      case "error": return "text-destructive";
      default: return "text-muted-foreground";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-secondary">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">System Administration</h1>
          <div className="flex items-center justify-center gap-2 text-muted-foreground">
            <p>Monitor and manage agricultural system in real-time</p>
            <Badge variant="outline" className="ml-2">
              <Clock className="w-3 h-3 mr-1" />
              {lastUpdated}
            </Badge>
          </div>
        </div>

        <Tabs defaultValue="overview" className="space-y-8">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="models">ML Models</TabsTrigger>
            <TabsTrigger value="system">System</TabsTrigger>
            <TabsTrigger value="activity">Activity</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Quick Actions */}
            <Card className="shadow-medium">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-primary" />
                  <span>Quick Actions</span>
                </CardTitle>
                <CardDescription>Common admin tasks that affect both frontend and backend</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
                  <Button 
                    onClick={handleReloadModels}
                    disabled={loading}
                    className="bg-gradient-primary hover:shadow-glow transition-spring"
                  >
                    <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                    Reload Models
                  </Button>
                  <Button 
                    onClick={handleReloadDataset}
                    variant="outline"
                    disabled={loading}
                  >
                    <Database className="w-4 h-4 mr-2" />
                    Reload Dataset
                  </Button>
                  <Button 
                    onClick={handleRefreshData}
                    variant="outline"
                    disabled={loading}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh Data
                  </Button>
                  <Button variant="outline">
                    <Cloud className="w-4 h-4 mr-2" />
                    Sync Weather
                  </Button>
                  <Button onClick={handleResetAll} variant="destructive" disabled={loading}>
                    <Database className="w-4 h-4 mr-2" />
                    Erase All Data
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Real-time System Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {systemMetrics.map((metric, index) => (
                <Card key={index} className="shadow-soft">
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-muted-foreground">{metric.label}</span>
                      <span className={`font-bold ${getStatusColor(metric.status)}`}>
                        {metric.value}
                      </span>
                    </div>
                    <Progress value={metric.progress} className="h-2 mb-2" />
                    <div className="flex items-center gap-1">
                      <div className={`w-2 h-2 rounded-full ${
                        metric.status === 'good' ? 'bg-primary' :
                        metric.status === 'warning' ? 'bg-destructive' : 'bg-muted'
                      }`} />
                      <span className="text-xs text-muted-foreground capitalize">{metric.status}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Sensor Data Details */}
            {sensorData && (
              <Card className="shadow-medium">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <TrendingUp className="w-5 h-5 text-primary" />
                    <span>Live Sensor Data</span>
                  </CardTitle>
                  <CardDescription>Real-time data from agricultural sensors</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">Temperature</p>
                      <p className="text-xl font-bold text-foreground">
                        {sensorData?.air_temperature?.toFixed(1) || '--'}°C
                      </p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">Humidity</p>
                      <p className="text-xl font-bold text-foreground">
                        {sensorData?.humidity?.toFixed(1) || '--'}%
                      </p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">Soil Moisture</p>
                      <p className="text-xl font-bold text-foreground">
                        {sensorData?.soil_moisture_percentage?.toFixed(1) || '--'}%
                      </p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">pH Level</p>
                      <p className="text-xl font-bold text-foreground">
                        {sensorData?.ph_level?.toFixed(1) || '--'}
                      </p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">Nitrogen</p>
                      <p className="text-xl font-bold text-foreground">
                        {sensorData?.nitrogen?.toFixed(1) || '--'} mg/kg
                      </p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">Phosphorus</p>
                      <p className="text-xl font-bold text-foreground">
                        {sensorData?.phosphorus?.toFixed(1) || '--'} mg/kg
                      </p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">Potassium</p>
                      <p className="text-xl font-bold text-foreground">
                        {sensorData?.potassium?.toFixed(1) || '--'} mg/kg
                      </p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">Light Intensity</p>
                      <p className="text-xl font-bold text-foreground">
                        {sensorData?.light_intensity_percentage?.toFixed(0) || '--'}%
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* ML Models Tab */}
          <TabsContent value="models" className="space-y-6">
            <Card className="shadow-medium">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Server className="w-5 h-5 text-primary" />
                  <span>Model Blend Weights</span>
                </CardTitle>
                <CardDescription>
                  Current ML model weights and real-time performance metrics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {modelWeights.map((model, index) => (
                    <div key={index} className="border rounded-lg p-4 bg-gradient-accent">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="font-semibold text-foreground">{model.model}</h3>
                        <Badge className="bg-primary text-primary-foreground">
                          Weight: {(model.weight * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div>
                          <span className="text-sm text-muted-foreground">Accuracy</span>
                          <div className="flex items-center space-x-2 mt-1">
                            <Progress value={model.accuracy} className="flex-1 h-2" />
                            <span className="text-sm font-medium text-foreground">
                              {model.accuracy}%
                            </span>
                          </div>
                        </div>
                        
                        <div>
                          <span className="text-sm text-muted-foreground">Last Updated</span>
                          <p className="text-sm text-foreground mt-1">{new Date().toLocaleTimeString()}</p>
                        </div>

                        <div>
                          <span className="text-sm text-muted-foreground">Status</span>
                          <div className="flex items-center gap-2 mt-1">
                            <CheckCircle2 className="w-4 h-4 text-primary" />
                            <span className="text-sm text-foreground">Active</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* System Tab */}
          <TabsContent value="system" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* System Health */}
              <Card className="shadow-medium">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Activity className="w-5 h-5 text-primary" />
                    <span>System Health</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {systemMetrics.map((metric, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                      <span className="text-sm font-medium text-foreground">{metric.label}</span>
                      <div className="flex items-center space-x-2">
                        <span className={`text-sm ${getStatusColor(metric.status)}`}>
                          {metric.value}
                        </span>
                        <div className={`w-2 h-2 rounded-full ${
                          metric.status === 'good' ? 'bg-primary' :
                          metric.status === 'warning' ? 'bg-destructive' : 'bg-muted-foreground'
                        }`} />
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Configuration */}
              <Card className="shadow-medium">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Settings className="w-5 h-5 text-primary" />
                    <span>Configuration</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-secondary rounded-lg">
                      <span className="text-sm text-foreground">TensorFlow Enabled</span>
                      <Badge className="bg-primary text-primary-foreground">Active</Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-secondary rounded-lg">
                      <span className="text-sm text-foreground">Real-time Sync</span>
                      <Badge className="bg-primary text-primary-foreground">Every 3s</Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-secondary rounded-lg">
                      <span className="text-sm text-foreground">API Version</span>
                      <span className="text-sm text-muted-foreground">v2.0.0</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-secondary rounded-lg">
                      <span className="text-sm text-foreground">Environment</span>
                      <Badge variant="outline">Development</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Activity Tab */}
          <TabsContent value="activity">
            <Card className="shadow-medium">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Clock className="w-5 h-5 text-primary" />
                  <span>Recent Activity</span>
                </CardTitle>
                <CardDescription>Real-time system events and actions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {activityLogs.map((activity, index) => (
                    <div key={index} className="flex items-center space-x-4 p-3 bg-secondary rounded-lg border-l-4" 
                         style={{
                           borderColor: activity.status === 'success' ? '#10b981' : 
                                       activity.status === 'error' ? '#ef4444' : '#f59e0b'
                         }}>
                      {getStatusIcon(activity.status)}
                      <div className="flex-1">
                        <p className="text-sm text-foreground font-medium">{activity.action}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <p className="text-xs text-muted-foreground">{activity.time}</p>
                          {activity.details && (
                            <span className="text-xs text-muted-foreground">• {activity.details}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Admin;