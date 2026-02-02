import React, { useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Upload,
  ScanLine,
  AlertTriangle,
  CheckCircle,
  Camera,
  X,
  Image,
  Sparkles,
  Leaf,
  Shield,
  Clock,
  ChevronRight,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { toast } from '../components/ui/Toast';
import { MLService } from '../services/api';

interface DiseaseResult {
  disease: string;
  confidence: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  symptoms: string[];
  treatments: { name: string; type: string; description: string }[];
  preventionTips: string[];
}

// Environmental Risk Form Component
const EnvironmentalRiskForm = () => {
  const [formData, setFormData] = useState({
    temperature: 25,
    humidity: 70,
    rainfall: 100,
    ph: 6.5
  });
  const [riskResult, setRiskResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await MLService.predictDiseaseRisk(formData);
      setRiskResult(response);
    } catch (error) {
      toast.error('Failed to predict risk');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Temperature (°C)</label>
          <input
            type="number"
            value={formData.temperature}
            onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
            className="w-full p-2 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Humidity (%)</label>
          <input
            type="number"
            value={formData.humidity}
            onChange={(e) => setFormData({ ...formData, humidity: parseFloat(e.target.value) })}
            className="w-full p-2 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Rainfall (mm)</label>
          <input
            type="number"
            value={formData.rainfall}
            onChange={(e) => setFormData({ ...formData, rainfall: parseFloat(e.target.value) })}
            className="w-full p-2 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Soil pH</label>
          <input
            type="number" step="0.1"
            value={formData.ph}
            onChange={(e) => setFormData({ ...formData, ph: parseFloat(e.target.value) })}
            className="w-full p-2 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700"
          />
        </div>
        <div className="md:col-span-2 lg:col-span-4 flex justify-end">
          <Button type="submit" isLoading={loading} leftIcon={<Sparkles size={16} />}>
            Predict Risk
          </Button>
        </div>
      </form>

      {riskResult && (
        <div className={`p-4 rounded-xl border ${
          riskResult.risk_level === 'High' ? 'bg-red-50 border-red-200 text-red-900' :
          riskResult.risk_level === 'Medium' ? 'bg-yellow-50 border-yellow-200 text-yellow-900' :
          'bg-green-50 border-green-200 text-green-900'
        }`}>
          <div className="flex justify-between items-center">
            <div>
              <h4 className="font-bold text-lg">Risk Level: {riskResult.risk_level}</h4>
              <p className="text-sm opacity-80">Probability: {(riskResult.risk_score * 100).toFixed(1)}%</p>
            </div>
            {riskResult.risk_level === 'High' && <AlertTriangle className="w-8 h-8 text-red-600" />}
            {riskResult.risk_level === 'Low' && <CheckCircle className="w-8 h-8 text-green-600" />}
          </div>
        </div>
      )}
    </div>
  );
}

const DiseaseDetection: React.FC = () => {
  const { t } = useTranslation();
  const [modelType, setModelType] = useState<'general' | 'targeted'>('general');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<DiseaseResult | null>(null);
  const [showCamera, setShowCamera] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    } else {
      toast.error('Please select a valid image file');
    }
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  // Drag and Drop handlers
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  // Camera functions
  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setShowCamera(true);
    } catch (error: any) {
      console.error('Camera error:', error);
      setCameraError('Could not access camera. Please check permissions.');
      toast.error('Camera access denied');
    }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setShowCamera(false);
  }, []);

  const capturePhoto = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            handleFileSelect(file);
            stopCamera();
          }
        }, 'image/jpeg', 0.9);
      }
    }
  }, [handleFileSelect, stopCamera]);

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setAnalyzing(true);

    try {
      let response;
      let analysisResult: any = {};

      if (modelType === 'general') {
        response = await MLService.analyzePlantImage(selectedFile);
        analysisResult = {
          disease: response.disease || 'Unknown condition detected',
          confidence: response.confidence || 0.85,
          severity: response.severity || 'medium',
          description: response.analysis || response.data?.analysis || 'Analysis completed.',
          symptoms: response.symptoms || [],
          treatments: response.treatments || [],
          preventionTips: response.preventionTips || []
        };
      } else {
        // Targeted Model
        response = await MLService.analyzeTargetedDisease(selectedFile);
        const data = response;
        analysisResult = {
          disease: data.prediction,
          confidence: data.confidence / 100, // Convert percentage to 0-1
          severity: data.prediction === 'Healthy' ? 'low' : 'high',
          description: `Analysis by specialized model detected ${data.prediction}.`,
          symptoms: data.prediction === 'Healthy' ? [] : ['Visible fungal growth', 'Leaf discoloration'],
          treatments: data.prediction === 'Healthy' ? [] : [
            { name: 'Targeted Fungicide', type: 'chemical', description: `Apply specific treatment for ${data.prediction}` }
          ],
          preventionTips: data.prediction === 'Healthy' ? ['Continue regular care'] : ['Isolate affected plants']
        };
      }
      
      setResult(analysisResult as DiseaseResult);
      toast.success(`${modelType === 'general' ? 'General' : 'Specialized'} analysis completed!`);
    } catch (error: any) {
      console.error('Disease detection error:', error);
      toast.error(error.message || 'Failed to analyze image');
    } finally {
      setAnalyzing(false);
    }
  };

  const clearImage = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'success';
      case 'medium': return 'warning';
      case 'high': return 'danger';
      case 'critical': return 'danger';
      default: return 'default';
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center justify-center gap-3">
          <ScanLine className="w-8 h-8 text-agri-600" />
          {t('disease')}
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Upload or capture a photo of the affected plant for AI-powered diagnosis
        </p>
      </div>

      {/* Camera Modal */}
      {showCamera && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl overflow-hidden max-w-2xl w-full">
            <div className="p-4 border-b dark:border-gray-700 flex justify-between items-center">
              <h3 className="font-semibold dark:text-white">Capture Plant Photo</h3>
              <button onClick={stopCamera} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="relative bg-black aspect-video">
              <video ref={videoRef} className="w-full h-full object-cover" autoPlay playsInline muted />
            </div>
            <div className="p-4 flex justify-center">
              <Button onClick={capturePhoto} size="lg" leftIcon={<Camera size={20} />}>
                Capture Photo
              </Button>
            </div>
          </div>
          <canvas ref={canvasRef} className="hidden" />
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Environmental Risk Forecast Section */}
        <div className="lg:col-span-2">
          <Card className="bg-gradient-to-r from-teal-50 to-emerald-50 dark:from-teal-900/20 dark:to-emerald-900/20 border-teal-200 dark:border-teal-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-teal-800 dark:text-teal-400">
                <Sparkles className="w-5 h-5" />
                Environmental Disease Risk Forecast
              </CardTitle>
            </CardHeader>
            <CardContent>
              <EnvironmentalRiskForm />
            </CardContent>
          </Card>
        </div>

        {/* Upload Section */}
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle className="flex items-center gap-2">
                <Image className="w-5 h-5 text-agri-600" />
                Easy Scan
              </CardTitle>
              <div className="flex gap-2">
                 <Badge 
                    variant={modelType === 'general' ? 'default' : 'outline'} 
                    className="cursor-pointer"
                    onClick={() => setModelType('general')}
                  >
                    General
                  </Badge>
                  <Badge 
                    variant={modelType === 'targeted' ? 'default' : 'outline'}
                    className="cursor-pointer"
                    onClick={() => setModelType('targeted')}
                  >
                    Specialized (Rust/Powdery)
                  </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Drag and Drop Area */}
            <div
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                isDragging
                  ? 'border-agri-500 bg-agri-50 dark:bg-agri-900/20'
                  : preview
                  ? 'border-agri-400 bg-agri-50/50 dark:bg-gray-700'
                  : 'border-gray-300 dark:border-gray-600 hover:border-agri-400 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleInputChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              
              {preview ? (
                <div className="relative">
                  <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-lg" />
                  <button
                    onClick={(e) => { e.stopPropagation(); clearImage(); }}
                    className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 shadow-lg"
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <div className="py-8">
                  <div className={`w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center ${
                    isDragging ? 'bg-agri-100 text-agri-600' : 'bg-gray-100 dark:bg-gray-700 text-gray-400'
                  }`}>
                    <Upload size={36} />
                  </div>
                  <p className="text-gray-700 dark:text-gray-300 font-medium mb-1">
                    {isDragging ? 'Drop your image here' : 'Drag & drop your image here'}
                  </p>
                  <p className="text-gray-500 dark:text-gray-400 text-sm">or click to browse</p>
                  <p className="text-gray-400 text-xs mt-2">Supports JPG, PNG up to 10MB</p>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
              <Button
                variant="secondary"
                onClick={startCamera}
                leftIcon={<Camera size={18} />}
                className="flex-1"
              >
                Use Camera
              </Button>
              <Button
                onClick={handleAnalyze}
                disabled={!selectedFile || analyzing}
                isLoading={analyzing}
                leftIcon={<Sparkles size={18} />}
                className="flex-1"
              >
                {analyzing ? 'Analyzing...' : `Detect (${modelType === 'general' ? 'General' : 'Specialized'})`}
              </Button>
            </div>

            {cameraError && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg text-sm">
                {cameraError}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Leaf className="w-5 h-5 text-agri-600" />
              Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            {analyzing && (
              <div className="text-center py-12 space-y-4">
                <div className="w-16 h-16 border-4 border-agri-200 border-t-agri-600 rounded-full animate-spin mx-auto" />
                <p className="text-gray-600 dark:text-gray-400 font-medium">Running {modelType} analysis...</p>
                <p className="text-gray-400 text-sm">This may take a few seconds</p>
              </div>
            )}

            {!analyzing && result && (
              <div className="space-y-6 animate-fade-in">
                {/* Disease Detection */}
                <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-xl border border-red-200 dark:border-red-800">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="text-lg font-bold text-red-800 dark:text-red-400">{result.disease}</h3>
                        <Badge variant={getSeverityColor(result.severity) as any}>
                          {result.severity.toUpperCase()}
                        </Badge>
                      </div>
                      <p className="text-red-700 dark:text-red-300 text-sm">{result.description}</p>
                    </div>
                  </div>
                </div>

                {/* Confidence Score */}
                <div className="flex items-center gap-3 p-3 bg-agri-50 dark:bg-agri-900/20 rounded-lg">
                  <CheckCircle className="w-5 h-5 text-agri-600" />
                  <div className="flex-1">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">Confidence Score</span>
                      <span className="font-semibold text-agri-700 dark:text-agri-400">
                        {(result.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-agri-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Treatments */}
                <div>
                  <h4 className="font-semibold text-gray-800 dark:text-white mb-3 flex items-center gap-2">
                    <Shield className="w-4 h-4 text-agri-600" />
                    Treatment Recommendations
                  </h4>
                  <div className="space-y-2">
                    {result.treatments.map((treatment, idx) => (
                      <div key={idx} className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                        <ChevronRight className="w-4 h-4 text-agri-600 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="font-medium text-gray-800 dark:text-gray-200">{treatment.name}</p>
                          <p className="text-sm text-gray-600 dark:text-gray-400">{treatment.description}</p>
                          <Badge variant="info" size="sm" className="mt-1">{treatment.type}</Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Prevention Tips */}
                <div>
                  <h4 className="font-semibold text-gray-800 dark:text-white mb-3 flex items-center gap-2">
                    <Clock className="w-4 h-4 text-agri-600" />
                    Prevention Tips
                  </h4>
                  <ul className="space-y-2">
                    {result.preventionTips.map((tip, idx) => (
                      <li key={idx} className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                        <div className="w-1.5 h-1.5 bg-agri-500 rounded-full" />
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {!analyzing && !result && (
              <div className="text-center py-16 text-gray-400">
                <ScanLine size={80} className="mx-auto mb-4 opacity-20" />
                <p className="font-medium">AI Analysis results will appear here</p>
                <p className="text-sm mt-1">Upload an image to get started</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default DiseaseDetection;