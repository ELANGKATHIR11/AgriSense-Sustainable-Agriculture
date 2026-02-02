import React, { useState, useEffect } from 'react';
import { Droplets, Power, Activity, AlertTriangle, Settings, Plus, Trash2, RefreshCw } from 'lucide-react';
import { toast } from 'react-toastify';
import { IoTService, WaterTankService } from '../services/api';
import socketService from '../services/socket';

const Irrigation: React.FC = () => {
    const [devices, setDevices] = useState<any[]>([]);
    const [tankStatus, setTankStatus] = useState<any>(null);
    const [usageHistory, setUsageHistory] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [pumpLoading, setPumpLoading] = useState(false);
    const [showAddDevice, setShowAddDevice] = useState(false);
    const [newDevice, setNewDevice] = useState({
        name: '',
        type: 'soil',
        location: '',
        topic: ''
    });

    useEffect(() => {
        loadData();

        // Listen for real-time sensor updates
        socketService.on('sensorUpdate', handleSensorUpdate);
        socketService.on('iotData', handleIoTUpdate);

        return () => {
            socketService.off('sensorUpdate', handleSensorUpdate);
            socketService.off('iotData', handleIoTUpdate);
        };
    }, []);

    const handleSensorUpdate = (data: any) => {
        console.log('Real-time sensor update:', data);
        // Update tank status if sensor data includes tank info
        if (data.tankLevel) {
            setTankStatus((prev: any) => ({
                ...prev,
                level: data.tankLevel,
                lastUpdated: new Date().toISOString()
            }));
        }
    };

    const handleIoTUpdate = (data: any) => {
        console.log('Real-time IoT update:', data);
        loadData();
    };

    const loadData = async () => {
        try {
            setLoading(true);
            const [devicesRes, tankRes, historyRes] = await Promise.all([
                IoTService.getDevices(),
                WaterTankService.getTankStatus(),
                WaterTankService.getUsageHistory()
            ]);

            setDevices(Array.isArray(devicesRes) ? devicesRes : (devicesRes as any).data || []);
            setTankStatus(tankRes);
            setUsageHistory(historyRes);
        } catch (error: any) {
            console.error('Error loading irrigation data:', error);
            toast.error('Failed to load irrigation data');
        } finally {
            setLoading(false);
        }
    };

    const handlePumpControl = async (action: 'ON' | 'OFF') => {
        try {
            setPumpLoading(true);
            await WaterTankService.controlPump(action);
            toast.success(`Pump turned ${action}`);

            // Update tank status
            setTankStatus((prev: any) => ({
                ...prev,
                pumpStatus: action
            }));
        } catch (error: any) {
            toast.error(`Failed to ${action} pump`);
        } finally {
            setPumpLoading(false);
        }
    };

    const handleAddDevice = async () => {
        if (!newDevice.name || !newDevice.location || !newDevice.topic) {
            toast.error('Please fill all fields');
            return;
        }

        try {
            await IoTService.createDevice(newDevice);
            toast.success('Device added successfully');
            setShowAddDevice(false);
            setNewDevice({ name: '', type: 'soil', location: '', topic: '' });
            loadData();
        } catch (error: any) {
            toast.error('Failed to add device');
        }
    };

    const handleDeleteDevice = async (deviceId: string) => {
        if (!confirm('Are you sure you want to delete this device?')) return;

        try {
            await IoTService.deleteDevice(deviceId);
            toast.success('Device deleted');
            loadData();
        } catch (error: any) {
            toast.error('Failed to delete device');
        }
    };

    const getTankColor = (level: number) => {
        if (level >= 70) return 'text-green-600 bg-green-50';
        if (level >= 30) return 'text-yellow-600 bg-yellow-50';
        return 'text-red-600 bg-red-50';
    };

    const getTankBorderColor = (level: number) => {
        if (level >= 70) return 'border-green-600';
        if (level >= 30) return 'border-yellow-600';
        return 'border-red-600';
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="text-center">
                    <RefreshCw className="animate-spin h-12 w-12 text-agri-600 mx-auto mb-4" />
                    <p className="text-stone-500">Loading irrigation system...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold text-agri-900">Smart Irrigation Control</h2>
                    <p className="text-stone-500 mt-1">Monitor and control your irrigation system</p>
                </div>
                <button
                    onClick={loadData}
                    className="flex items-center gap-2 px-4 py-2 bg-stone-100 hover:bg-stone-200 rounded-lg transition-colors"
                >
                    <RefreshCw className="h-4 w-4" />
                    Refresh
                </button>
            </div>

            {/* Water Tank Status */}
            {tankStatus && (
                <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-8 rounded-2xl border border-blue-100">
                    <div className="grid md:grid-cols-3 gap-6">
                        {/* Tank Level Gauge */}
                        <div className="col-span-1">
                            <h3 className="font-semibold text-stone-800 mb-4 flex items-center gap-2">
                                <Droplets className="h-5 w-5 text-blue-600" />
                                Water Tank Level
                            </h3>
                            <div className="relative">
                                <div className="w-full h-64 bg-white rounded-2xl border-4 border-blue-200 relative overflow-hidden">
                                    <div
                                        className={`absolute bottom-0 w-full transition-all duration-1000 ${tankStatus.level >= 70 ? 'bg-green-400' :
                                                tankStatus.level >= 30 ? 'bg-yellow-400' : 'bg-red-400'
                                            } opacity-60`}
                                        // eslint-disable-next-line react/forbid-dom-props
                                        style={{ height: `${tankStatus.level}%` }}
                                    />
                                    <div className="absolute inset-0 flex items-center justify-center flex-col z-10">
                                        <span className="text-5xl font-bold text-stone-800">
                                            {Math.round(tankStatus.level)}%
                                        </span>
                                        <span className="text-sm text-stone-600 mt-2">
                                            {tankStatus.currentVolume?.toLocaleString()} / {tankStatus.capacity?.toLocaleString()} L
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Tank Info & Controls */}
                        <div className="col-span-2 space-y-6">
                            {/* Status Cards */}
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-white p-4 rounded-xl border border-blue-100">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-stone-600">Pump Status</span>
                                        <div className={`px-3 py-1 rounded-full text-xs font-bold ${tankStatus.pumpStatus === 'ON' ? 'bg-green-100 text-green-700' : 'bg-stone-100 text-stone-600'
                                            }`}>
                                            {tankStatus.pumpStatus}
                                        </div>
                                    </div>
                                    <p className="text-2xl font-bold text-stone-800 mt-2">
                                        {tankStatus.pumpStatus === 'ON' ? 'Running' : 'Stopped'}
                                    </p>
                                </div>

                                <div className="bg-white p-4 rounded-xl border border-blue-100">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-stone-600">Capacity</span>
                                        <Activity className="h-4 w-4 text-blue-600" />
                                    </div>
                                    <p className="text-2xl font-bold text-stone-800 mt-2">
                                        {tankStatus.capacity?.toLocaleString()} L
                                    </p>
                                </div>
                            </div>

                            {/* Pump Controls */}
                            <div className="bg-white p-6 rounded-xl border border-blue-100">
                                <h4 className="font-semibold text-stone-800 mb-4">Pump Control</h4>
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => handlePumpControl('ON')}
                                        disabled={pumpLoading || tankStatus.pumpStatus === 'ON'}
                                        className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-green-300 text-white rounded-xl font-medium transition-colors"
                                    >
                                        <Power className="h-5 w-5" />
                                        Turn ON
                                    </button>
                                    <button
                                        onClick={() => handlePumpControl('OFF')}
                                        disabled={pumpLoading || tankStatus.pumpStatus === 'OFF'}
                                        className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-red-300 text-white rounded-xl font-medium transition-colors"
                                    >
                                        <Power className="h-5 w-5" />
                                        Turn OFF
                                    </button>
                                </div>
                            </div>

                            {/* Alerts */}
                            {tankStatus.level < 30 && (
                                <div className="bg-red-50 border border-red-200 p-4 rounded-xl flex items-start gap-3">
                                    <AlertTriangle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                                    <div>
                                        <p className="font-semibold text-red-800">Low Water Level</p>
                                        <p className="text-sm text-red-700">Water tank level is below 30%. Consider refilling soon.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* IoT Devices */}
            <div className="bg-white p-6 rounded-2xl border border-stone-200">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-stone-900">IoT Devices</h3>
                    <button
                        onClick={() => setShowAddDevice(!showAddDevice)}
                        className="flex items-center gap-2 px-4 py-2 bg-agri-600 hover:bg-agri-700 text-white rounded-lg transition-colors"
                    >
                        <Plus className="h-4 w-4" />
                        Add Device
                    </button>
                </div>

                {/* Add Device Form */}
                {showAddDevice && (
                    <div className="bg-stone-50 p-4 rounded-xl mb-6 border border-stone-200">
                        <div className="grid md:grid-cols-2 gap-4">
                            <input
                                type="text"
                                placeholder="Device Name"
                                value={newDevice.name}
                                onChange={(e) => setNewDevice({ ...newDevice, name: e.target.value })}
                                className="px-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-agri-500 focus:border-agri-500"
                            />
                            <select
                                value={newDevice.type}
                                onChange={(e) => setNewDevice({ ...newDevice, type: e.target.value })}
                                className="px-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-agri-500 focus:border-agri-500"
                                aria-label="Device Type"
                            >
                                <option value="soil">Soil Sensor</option>
                                <option value="environmental">Environmental Sensor</option>
                                <option value="pump">Water Pump</option>
                                <option value="valve">Valve Controller</option>
                            </select>
                            <input
                                type="text"
                                placeholder="Location (e.g., Field A)"
                                value={newDevice.location}
                                onChange={(e) => setNewDevice({ ...newDevice, location: e.target.value })}
                                className="px-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-agri-500 focus:border-agri-500"
                            />
                            <input
                                type="text"
                                placeholder="MQTT Topic"
                                value={newDevice.topic}
                                onChange={(e) => setNewDevice({ ...newDevice, topic: e.target.value })}
                                className="px-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-agri-500 focus:border-agri-500"
                            />
                        </div>
                        <div className="flex gap-2 mt-4">
                            <button
                                onClick={handleAddDevice}
                                className="px-6 py-2 bg-agri-600 hover:bg-agri-700 text-white rounded-lg transition-colors"
                            >
                                Add Device
                            </button>
                            <button
                                onClick={() => setShowAddDevice(false)}
                                className="px-6 py-2 bg-stone-200 hover:bg-stone-300 text-stone-700 rounded-lg transition-colors"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                )}

                {/* Devices List */}
                {devices.length === 0 ? (
                    <div className="text-center py-12 text-stone-400">
                        <Settings className="h-16 w-16 mx-auto mb-4 opacity-50" />
                        <p>No devices configured. Add your first IoT device to get started.</p>
                    </div>
                ) : (
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {devices.map((device) => (
                            <div
                                key={device.id}
                                className="bg-stone-50 p-4 rounded-xl border border-stone-200 hover:border-agri-400 transition-colors"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <div>
                                        <h4 className="font-semibold text-stone-900">{device.name}</h4>
                                        <p className="text-sm text-stone-600">{device.location}</p>
                                    </div>
                                    <button
                                        onClick={() => handleDeleteDevice(device.id)}
                                        className="text-red-500 hover:text-red-700 transition-colors"
                                        aria-label={`Delete ${device.name}`}
                                    >
                                        <Trash2 className="h-4 w-4" />
                                    </button>
                                </div>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-stone-600">Type:</span>
                                        <span className="font-medium text-stone-800 capitalize">{device.type}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-stone-600">Status:</span>
                                        <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${device.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                                            }`}>
                                            {device.status}
                                        </span>
                                    </div>
                                    <div className="text-xs text-stone-500 mt-2">
                                        Topic: {device.topic}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Usage History */}
            {usageHistory.length > 0 && (
                <div className="bg-white p-6 rounded-2xl border border-stone-200">
                    <h3 className="text-xl font-bold text-stone-900 mb-6">7-Day Water Usage</h3>
                    <div className="space-y-2">
                        {usageHistory.map((day: any, index: number) => (
                            <div key={index} className="flex items-center gap-4">
                                <span className="text-sm text-stone-600 w-24">{day.date}</span>
                                <div className="flex-1 bg-stone-100 rounded-full h-8 relative overflow-hidden">
                                    <div
                                        className="bg-blue-500 h-full rounded-full transition-all duration-500"
                                        // eslint-disable-next-line react/forbid-dom-props
                                        style={{ width: `${(day.usage / 1000) * 100}%` }}
                                    />
                                    <span className="absolute inset-0 flex items-center justify-center text-sm font-medium text-stone-700">
                                        {day.usage} L
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default Irrigation;
