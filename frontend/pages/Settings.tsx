import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  User,
  Bell,
  Globe,
  Palette,
  Database,
  Download,
  Upload,
  Shield,
  Save,
  Moon,
  Sun,
  Thermometer,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input, Select } from '../components/ui/Input';
import { Badge } from '../components/ui/Badge';
import { toast } from '../components/ui/Toast';

const Settings = () => {
  const { t, i18n } = useTranslation();
  const [activeTab, setActiveTab] = useState('profile');
  const [isDark, setIsDark] = useState(() => document.documentElement.classList.contains('dark'));
  
  const [settings, setSettings] = useState({
    name: 'Farmer User',
    email: 'farmer@agrisense.com',
    language: i18n.language,
    units: 'metric',
    notifications: {
      email: true,
      push: true,
      sms: false,
      alerts: true,
    },
    dataRetention: '30',
  });

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'appearance', label: 'Appearance', icon: Palette },
    { id: 'data', label: 'Data & Export', icon: Database },
  ];

  const languages = [
    { value: 'en', label: 'English' },
    { value: 'hi', label: 'हिंदी (Hindi)' },
    { value: 'ta', label: 'தமிழ் (Tamil)' },
    { value: 'te', label: 'తెలుగు (Telugu)' },
    { value: 'kn', label: 'ಕನ್ನಡ (Kannada)' },
  ];

  const handleSave = () => {
    toast.success('Settings saved successfully!');
  };

  const handleExport = () => {
    toast.info('Exporting data...', 'Your data will be downloaded shortly');
  };

  const toggleDarkMode = () => {
    const newDark = !isDark;
    setIsDark(newDark);
    if (newDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Settings</h1>
        <p className="text-gray-600 dark:text-gray-400">Manage your account and application preferences</p>
      </div>

      <div className="flex flex-col md:flex-row gap-6">
        {/* Sidebar Tabs */}
        <div className="md:w-48 flex-shrink-0">
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${
                  activeTab === tab.id
                    ? 'bg-agri-600 text-white'
                    : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                <tab.icon size={18} />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Content Area */}
        <div className="flex-1">
          {activeTab === 'profile' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="w-5 h-5 text-agri-600" />
                  Profile Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Input
                  label="Full Name"
                  value={settings.name}
                  onChange={(e) => setSettings({ ...settings, name: e.target.value })}
                />
                <Input
                  label="Email Address"
                  type="email"
                  value={settings.email}
                  onChange={(e) => setSettings({ ...settings, email: e.target.value })}
                />
                <Select
                  label="Language"
                  options={languages}
                  value={settings.language}
                  onChange={(e) => {
                    setSettings({ ...settings, language: e.target.value });
                    i18n.changeLanguage(e.target.value);
                  }}
                />
                <Select
                  label="Units"
                  options={[
                    { value: 'metric', label: 'Metric (°C, mm, kg)' },
                    { value: 'imperial', label: 'Imperial (°F, in, lb)' },
                  ]}
                  value={settings.units}
                  onChange={(e) => setSettings({ ...settings, units: e.target.value })}
                />
                <Button onClick={handleSave} leftIcon={<Save size={18} />}>
                  Save Changes
                </Button>
              </CardContent>
            </Card>
          )}

          {activeTab === 'notifications' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="w-5 h-5 text-agri-600" />
                  Notification Preferences
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {[
                  { key: 'email', label: 'Email Notifications', desc: 'Receive updates via email' },
                  { key: 'push', label: 'Push Notifications', desc: 'Browser push notifications' },
                  { key: 'sms', label: 'SMS Alerts', desc: 'Critical alerts via SMS' },
                  { key: 'alerts', label: 'Sensor Alerts', desc: 'Threshold violation alerts' },
                ].map((item) => (
                  <div key={item.key} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">{item.label}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">{item.desc}</p>
                    </div>
                    <button
                      onClick={() => setSettings({
                        ...settings,
                        notifications: {
                          ...settings.notifications,
                          [item.key]: !settings.notifications[item.key as keyof typeof settings.notifications]
                        }
                      })}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        settings.notifications[item.key as keyof typeof settings.notifications]
                          ? 'bg-agri-600'
                          : 'bg-gray-300 dark:bg-gray-600'
                      }`}
                    >
                      <span
                        className={`block w-5 h-5 bg-white rounded-full shadow transition-transform ${
                          settings.notifications[item.key as keyof typeof settings.notifications]
                            ? 'translate-x-6'
                            : 'translate-x-0.5'
                        }`}
                      />
                    </button>
                  </div>
                ))}
                <Button onClick={handleSave} leftIcon={<Save size={18} />}>
                  Save Preferences
                </Button>
              </CardContent>
            </Card>
          )}

          {activeTab === 'appearance' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Palette className="w-5 h-5 text-agri-600" />
                  Appearance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    {isDark ? <Moon className="w-5 h-5 text-blue-400" /> : <Sun className="w-5 h-5 text-amber-500" />}
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">Dark Mode</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Toggle dark/light theme</p>
                    </div>
                  </div>
                  <button
                    onClick={toggleDarkMode}
                    className={`w-12 h-6 rounded-full transition-colors ${
                      isDark ? 'bg-agri-600' : 'bg-gray-300'
                    }`}
                  >
                    <span
                      className={`block w-5 h-5 bg-white rounded-full shadow transition-transform ${
                        isDark ? 'translate-x-6' : 'translate-x-0.5'
                      }`}
                    />
                  </button>
                </div>

                <div className="space-y-3">
                  <p className="font-medium text-gray-900 dark:text-white">Theme Colors</p>
                  <div className="flex gap-3">
                    {['agri', 'water', 'earth', 'soil'].map((color) => (
                      <button
                        key={color}
                        className={`w-10 h-10 rounded-full border-2 border-white shadow-md ring-2 ring-offset-2 ${
                          color === 'agri' ? 'bg-agri-600 ring-agri-600' :
                          color === 'water' ? 'bg-water-600 ring-water-600' :
                          color === 'earth' ? 'bg-earth-600 ring-earth-600' :
                          'bg-soil-600 ring-soil-600'
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {activeTab === 'data' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5 text-agri-600" />
                  Data Management
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Button variant="outline" leftIcon={<Download size={18} />} onClick={handleExport}>
                    Export All Data
                  </Button>
                  <Button variant="outline" leftIcon={<Upload size={18} />}>
                    Import Data
                  </Button>
                </div>

                <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900 dark:text-white">Data Retention</span>
                    <Badge variant="info">{settings.dataRetention} days</Badge>
                  </div>
                  <input
                    type="range"
                    min="7"
                    max="365"
                    value={settings.dataRetention}
                    onChange={(e) => setSettings({ ...settings, dataRetention: e.target.value })}
                    className="w-full accent-agri-600"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>7 days</span>
                    <span>1 year</span>
                  </div>
                </div>

                <div className="p-4 border border-red-200 bg-red-50 dark:bg-red-900/20 dark:border-red-800 rounded-lg">
                  <div className="flex items-center gap-3 mb-2">
                    <Shield className="w-5 h-5 text-red-600" />
                    <span className="font-medium text-red-700 dark:text-red-400">Danger Zone</span>
                  </div>
                  <p className="text-sm text-red-600 dark:text-red-400 mb-3">
                    Permanently delete all your data. This action cannot be undone.
                  </p>
                  <Button variant="danger" size="sm">
                    Delete All Data
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default Settings;
