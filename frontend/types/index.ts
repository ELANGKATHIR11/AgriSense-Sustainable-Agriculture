// ============================================
// AgriSense TypeScript Type Definitions
// ============================================

// ===========================================
// API Response Types
// ===========================================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// ===========================================
// Authentication Types
// ===========================================

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'farmer';
  avatar?: string;
  createdAt: string;
  preferences: UserPreferences;
}

export interface UserPreferences {
  language: 'en' | 'hi' | 'ta' | 'te' | 'kn';
  theme: 'light' | 'dark' | 'system';
  units: 'metric' | 'imperial';
  notifications: NotificationPreferences;
}

export interface NotificationPreferences {
  email: boolean;
  push: boolean;
  sms: boolean;
  alerts: boolean;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData extends LoginCredentials {
  name: string;
  confirmPassword: string;
}

export interface AuthResponse {
  user: User;
  token: string;
  refreshToken: string;
  expiresIn: number;
}

// ===========================================
// Sensor & IoT Types
// ===========================================

export interface SensorData {
  temperature: number;
  humidity: number;
  soilMoisture: number;
  soilTemperature: number;
  phLevel: number;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  lightIntensity: number;
  timestamp: string;
}

export interface SensorThreshold {
  min: number;
  max: number;
  unit: string;
  warningMin?: number;
  warningMax?: number;
}

export interface SensorConfig {
  temperature: SensorThreshold;
  humidity: SensorThreshold;
  soilMoisture: SensorThreshold;
  phLevel: SensorThreshold;
  nitrogen: SensorThreshold;
  phosphorus: SensorThreshold;
  potassium: SensorThreshold;
}

export interface IoTDevice {
  id: string;
  name: string;
  type: 'sensor' | 'actuator' | 'controller';
  status: 'online' | 'offline' | 'error';
  lastSeen: string;
  location?: string;
  firmware?: string;
}

// ===========================================
// Crop Types
// ===========================================

export interface CropRecommendationRequest {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
}

export interface CropRecommendationResult {
  crop: string;
  confidence: number;
  details: string;
  alternatives?: CropAlternative[];
}

export interface CropAlternative {
  crop: string;
  confidence: number;
  suitability: 'high' | 'medium' | 'low';
}

export type RecommendationResult = CropRecommendationResult;

export interface AnalysisResult {
  water_requirement: number;
  season: string;
  crop_group: string;
  recommended_crop: string;
  expected_yield: number;
}

export interface CropProfile {
  id: number;
  name: string;
  scientificName: string;
  tempRange: string;
  humidityRange: string;
  phRange: string;
  waterReq: 'High' | 'Medium' | 'Low';
  duration: string;
  season: string;
  type?: string;
  imageUrl?: string;
}

export interface YieldPredictionRequest {
  crop: string;
  area: number;
  season: string;
  state: string;
}

export interface YieldPredictionResult {
  predictedYield: number;
  unit: string;
  confidence: number;
  factors?: YieldFactor[];
}

export interface YieldFactor {
  name: string;
  impact: 'positive' | 'negative' | 'neutral';
  description: string;
}

// ===========================================
// Disease Detection Types
// ===========================================

export interface DiseaseDetectionRequest {
  image: File;
  cropType?: string;
}

export interface DiseaseDetectionResult {
  disease: string;
  confidence: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  symptoms: string[];
  treatments: Treatment[];
  preventionTips: string[];
  imageUrl?: string;
}

export interface Treatment {
  name: string;
  type: 'organic' | 'chemical' | 'biological';
  application: string;
  frequency: string;
  cost?: string;
}

// ===========================================
// Weed Management Types
// ===========================================

export interface WeedDetectionResult {
  weedType: string;
  confidence: number;
  coveragePercent: number;
  severity: 'low' | 'medium' | 'high';
  controlMethods: ControlMethod[];
  estimatedCost: number;
}

export interface ControlMethod {
  name: string;
  type: 'manual' | 'chemical' | 'biological' | 'preventive';
  description: string;
  effectiveness: number;
  timing: string;
}

// ===========================================
// Irrigation Types
// ===========================================

export interface IrrigationSchedule {
  id: string;
  zone: string;
  startTime: string;
  duration: number;
  days: number[];
  enabled: boolean;
}

export interface WaterTankStatus {
  tankId: string;
  level: number;
  capacity: number;
  currentVolume: number;
  lastUpdated: string;
  pumpStatus: 'ON' | 'OFF';
  alerts: Alert[];
}

export interface WaterUsage {
  date: string;
  usage: number;
  unit: 'liters' | 'gallons';
}

// ===========================================
// Chat Types
// ===========================================

export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  image?: string; // Optional image URL/preview
  attachments?: Attachment[];
}

export interface Attachment {
  type: 'image' | 'file';
  url: string;
  name: string;
  size?: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

// ===========================================
// Admin Types
// ===========================================

export interface ActivityLog {
  id: number;
  action: string;
  status: 'success' | 'warning' | 'error';
  timestamp: string;
  details: string;
  userId?: string;
}

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  uptime: string;
  modelStatus: 'loaded' | 'loading' | 'error';
  activeConnections: number;
}

export interface MLModelInfo {
  name: string;
  version: string;
  status: 'active' | 'inactive' | 'error';
  accuracy: number;
  lastTrained: string;
  predictions: number;
}

// ===========================================
// Alert Types
// ===========================================

export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  source: string;
}

// ===========================================
// Form Types
// ===========================================

export interface FormFieldError {
  field: string;
  message: string;
}

export interface FormState<T> {
  values: T;
  errors: FormFieldError[];
  isSubmitting: boolean;
  isValid: boolean;
}

// ===========================================
// Component Prop Types
// ===========================================

export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export type ButtonVariant = 'primary' | 'secondary' | 'success' | 'danger' | 'warning' | 'outline' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends BaseComponentProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

export interface InputProps extends BaseComponentProps {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  required?: boolean;
}

export interface CardProps extends BaseComponentProps {
  variant?: 'default' | 'bordered' | 'elevated' | 'glass';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hoverable?: boolean;
}

export interface ModalProps extends BaseComponentProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  closeOnOverlay?: boolean;
  closeOnEscape?: boolean;
}

export interface BadgeProps extends BaseComponentProps {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  size?: 'sm' | 'md' | 'lg';
  dot?: boolean;
}

export interface StatCardProps extends BaseComponentProps {
  title: string;
  value: string | number;
  icon?: React.ReactNode;
  trend?: number;
  trendLabel?: string;
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
}

export interface DataTableColumn<T> {
  key: keyof T | string;
  header: string;
  sortable?: boolean;
  render?: (value: any, row: T) => React.ReactNode;
  width?: string;
}

export interface DataTableProps<T> extends BaseComponentProps {
  data: T[];
  columns: DataTableColumn<T>[];
  loading?: boolean;
  searchable?: boolean;
  pagination?: boolean;
  pageSize?: number;
  emptyMessage?: string;
  onRowClick?: (row: T) => void;
}

// ===========================================
// Service Types
// ===========================================

export interface ServiceConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
}

export interface RequestOptions {
  headers?: Record<string, string>;
  params?: Record<string, any>;
  signal?: AbortSignal;
}

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

// ===========================================
// Hook Return Types
// ===========================================

export interface UseQueryResult<T> {
  data: T | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

export interface UseMutationResult<T, V> {
  mutate: (variables: V) => void;
  mutateAsync: (variables: V) => Promise<T>;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  data: T | undefined;
}

// ===========================================
// Legacy Types (from types.ts)
// ===========================================

export interface LegacySensorData {
  timestamp: string;
  air_temperature: number;
  humidity: number;
  soil_moisture: number;
  soil_temperature: number;
  ph_level: number;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  light_intensity: number;
}

export interface LegacySystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  uptime_hours: number;
  active_sensors: number;
  ml_models_status: 'Online' | 'Offline' | 'Training';
}

export interface LegacyActivityLog {
  id: number;
  timestamp: string;
  action: string;
  status: 'Success' | 'Warning' | 'Error';
}

export interface LegacyChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

export interface DiseaseResult {
  disease_name: string;
  confidence: number;
  treatment: string;
  image_url?: string;
}

export interface CropInput {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  ph: number;
  rainfall: number;
  temperature: number;
  humidity: number;
}

export interface IrrigationStatus {
  pump_active: boolean;
  water_level: number;
  flow_rate: number;
  last_active: string;
  mode: 'Auto' | 'Manual';
}

export interface CropDetail {
  id: number;
  name: string;
  scientific_name: string;
  category: string;
  season: string;
  soil_type: string;
  description: string;
}

export interface MLDataset {
  id: string;
  name: string;
  type: 'CSV' | 'Image' | 'JSON';
  size: string;
  records: number;
  uploaded_at: string;
  status: 'Ready' | 'Processing';
}

export interface MLModel {
  id: string;
  name: string;
  version: string;
  type: 'Classification' | 'Regression' | 'NLP';
  status: 'Trained' | 'Training' | 'Idle' | 'Failed';
  accuracy: number;
  last_trained: string;
  dataset_id: string;
}
