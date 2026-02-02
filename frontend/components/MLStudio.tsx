import React, { useState, useEffect } from 'react';
import { fetchDatasets, fetchModels, triggerTraining, uploadDataset } from '../services/api';
import { MLDataset, MLModel } from '../types';
import { Database, HardDrive, Cpu, UploadCloud, Play, Trash2, Server, Save, FileText, Image as ImageIcon, Activity } from 'lucide-react';

const MLStudio: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'data' | 'models' | 'database'>('data');

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center">
        <div>
          <h1 className="text-2xl font-bold text-agri-900">ML Studio & Data Center</h1>
          <p className="text-gray-600">Manage datasets, train models, and configure database connections.</p>
        </div>
        <div className="mt-4 md:mt-0 flex space-x-2 bg-white p-1 rounded-lg border border-agri-200 shadow-sm">
          <TabButton active={activeTab === 'data'} onClick={() => setActiveTab('data')} icon={HardDrive} label="Data Hub" />
          <TabButton active={activeTab === 'models'} onClick={() => setActiveTab('models')} icon={Cpu} label="Model Workbench" />
          <TabButton active={activeTab === 'database'} onClick={() => setActiveTab('database')} icon={Database} label="Native DB" />
        </div>
      </div>
      <div className="min-h-[500px]">
        {activeTab === 'data' && <DataHub />}
        {activeTab === 'models' && <ModelWorkbench />}
        {activeTab === 'database' && <DatabaseConfig />}
      </div>
    </div>
  );
};

const TabButton = ({ active, onClick, icon: Icon, label }: any) => (
  <button onClick={onClick} className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-all ${active ? 'bg-agri-600 text-white shadow-md' : 'text-gray-600 hover:bg-agri-50'}`}>
    <Icon className="w-4 h-4 mr-2" />{label}
  </button>
);

const DataHub: React.FC = () => {
  const [datasets, setDatasets] = useState<MLDataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => { fetchDatasets().then(data => { setDatasets(Array.isArray(data) ? data : []); setLoading(false); }); }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setIsUploading(true);
      const newFile = await uploadDataset(e.target.files[0]);
      setDatasets(prev => [...prev, newFile]);
      setIsUploading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-agri-100 overflow-hidden">
      <div className="p-6 border-b border-agri-100 flex justify-between items-center bg-gray-50">
        <div><h2 className="text-lg font-semibold text-gray-800">Local Datasets</h2><p className="text-xs text-gray-500">Stored in backend/ml/datasets</p></div>
        <label className="cursor-pointer bg-agri-600 hover:bg-agri-700 text-white px-4 py-2 rounded-lg flex items-center shadow-sm transition-colors">
          <UploadCloud className="w-4 h-4 mr-2" />{isUploading ? 'Uploading...' : 'Upload Dataset'}
          <input type="file" className="hidden" onChange={handleFileUpload} disabled={isUploading} />
        </label>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="bg-white text-gray-500 border-b border-agri-100">
            <tr>
              <th className="px-6 py-4 font-medium text-xs uppercase tracking-wider">Name</th>
              <th className="px-6 py-4 font-medium text-xs uppercase tracking-wider">Type</th>
              <th className="px-6 py-4 font-medium text-xs uppercase tracking-wider">Size</th>
              <th className="px-6 py-4 font-medium text-xs uppercase tracking-wider">Records</th>
              <th className="px-6 py-4 font-medium text-xs uppercase tracking-wider">Status</th>
              <th className="px-6 py-4 font-medium text-xs uppercase tracking-wider text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {loading ? <tr><td colSpan={6} className="p-8 text-center text-gray-500">Loading datasets...</td></tr> : datasets.map((ds) => (
              <tr key={ds.id} className="hover:bg-agri-50/50 transition-colors">
                <td className="px-6 py-4 text-sm font-medium text-gray-900 flex items-center">
                  {ds.type === 'Image' ? <ImageIcon className="w-4 h-4 mr-2 text-purple-500" /> : <FileText className="w-4 h-4 mr-2 text-blue-500" />}{ds.name}
                </td>
                <td className="px-6 py-4 text-sm text-gray-500">{ds.type}</td>
                <td className="px-6 py-4 text-sm text-gray-500">{ds.size}</td>
                <td className="px-6 py-4 text-sm text-gray-500">{ds.records.toLocaleString()}</td>
                <td className="px-6 py-4">
                  <span className={`px-2 py-1 text-xs rounded-full border ${ds.status === 'Ready' ? 'bg-green-100 text-green-700 border-green-200' : 'bg-yellow-100 text-yellow-700 border-yellow-200'}`}>{ds.status}</span>
                </td>
                <td className="px-6 py-4 text-right"><button className="text-gray-400 hover:text-red-500"><Trash2 className="w-4 h-4" /></button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const ModelWorkbench: React.FC = () => {
  const [models, setModels] = useState<MLModel[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const refreshModels = async () => {
    try {
      const data = await fetchModels();
      setModels(Array.isArray(data) ? data : []);
      setLastUpdated(new Date().toISOString());
    } catch (e) {
      // ignore
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    refreshModels();
    const iv = setInterval(refreshModels, 10000); // poll every 10s for live accuracy
    return () => clearInterval(iv);
  }, []);

  const handleTrain = async (id: string) => {
    await triggerTraining(id);
    refreshModels();
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {models.map((model) => (
        <div key={model.id} className="bg-white p-6 rounded-xl shadow-sm border border-agri-100 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-start mb-4">
              <div><h3 className="text-lg font-bold text-gray-800">{model.name}</h3><p className="text-xs text-gray-500 font-mono mt-1">{model.version}</p></div>
              <span className={`px-3 py-1 rounded-full text-xs font-bold ${model.status === 'Trained' ? 'bg-green-100 text-green-700' : model.status === 'Training' ? 'bg-blue-100 text-blue-700 animate-pulse' : 'bg-gray-100 text-gray-600'}`}>{model.status}</span>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center text-sm"><span className="text-gray-500">Model Type</span><span className="font-medium text-gray-800">{model.type}</span></div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-500">Accuracy</span>
                <div className="flex items-center">
                  <div className="w-36 h-2 bg-gray-100 rounded-full mr-2 overflow-hidden">
                    <div className={`h-full ${((model.accuracy||0) >= 0.9) ? 'bg-green-500' : ((model.accuracy||0) >= 0.75) ? 'bg-yellow-500' : 'bg-red-500'}`} style={{ width: `${(model.accuracy || 0) * 100}%` }} />
                  </div>
                  <div className="text-sm font-semibold text-gray-800 mr-3">{((model.accuracy || 0) * 100).toFixed(2)}%</div>
                  <div className="text-xs text-gray-400">{lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : ''}</div>
                </div>
              </div>
              <div className="flex justify-between items-center text-sm"><span className="text-gray-500">Last Trained</span><span className="text-gray-700">{model.last_trained}</span></div>
            </div>
          </div>
          <div className="mt-6 pt-6 border-t border-gray-100 flex space-x-3">
            <button onClick={() => handleTrain(model.id)} disabled={model.status === 'Training'} className="flex-1 bg-agri-600 hover:bg-agri-700 disabled:opacity-50 text-white py-2 rounded-lg font-medium text-sm flex items-center justify-center">
              {model.status === 'Training' ? 'Training...' : <><Play className="w-4 h-4 mr-2" />Retrain Model</>}
            </button>
            <button className="px-4 py-2 border border-gray-200 rounded-lg hover:bg-gray-50 text-gray-600"><Activity className="w-4 h-4" /></button>
          </div>
        </div>
      ))}
      <div className="border-2 border-dashed border-agri-200 rounded-xl p-6 flex flex-col items-center justify-center text-center hover:bg-agri-50/50 transition-colors cursor-pointer min-h-[280px]">
        <div className="w-12 h-12 bg-agri-100 rounded-full flex items-center justify-center mb-4"><Cpu className="w-6 h-6 text-agri-600" /></div>
        <h3 className="text-lg font-bold text-gray-800">Create New Model</h3>
        <p className="text-sm text-gray-500 mt-2 max-w-xs">Define architecture and train from your datasets.</p>
      </div>
    </div>
  );
};

const DatabaseConfig: React.FC = () => {
  const [dbType, setDbType] = useState('postgresql');
  return (
    <div className="max-w-3xl mx-auto">
      <div className="bg-white rounded-xl shadow-sm border border-agri-100 overflow-hidden">
        <div className="p-6 border-b border-agri-100 bg-slate-50">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-slate-200 rounded-lg"><Server className="w-6 h-6 text-slate-700" /></div>
            <div><h2 className="text-lg font-bold text-gray-900">Database Connection</h2><p className="text-sm text-gray-500">PostgreSQL / SQLite / MongoDB</p></div>
          </div>
        </div>
        <div className="p-8 space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Database Type</label>
            <div className="flex space-x-4">
              <label className={`flex-1 border rounded-lg p-4 cursor-pointer flex items-center justify-center transition-all ${dbType === 'postgresql' ? 'border-agri-500 bg-agri-50 ring-1 ring-agri-500' : 'border-gray-200'}`}>
                <input type="radio" name="db" value={'postgresql'} checked={dbType === 'postgresql'} onChange={() => setDbType('postgresql')} className="hidden" />
                <span className="font-bold text-gray-700">PostgreSQL</span>
              </label>
            </div>
          </div>
          <div className="pt-4 flex items-center justify-between border-t border-gray-100">
            <div className="flex items-center text-sm text-green-600"><span className="w-2 h-2 rounded-full bg-green-500 mr-2"></span>Backend Connected</div>
            <button className="bg-agri-800 hover:bg-agri-900 text-white px-6 py-2 rounded-lg font-bold flex items-center"><Save className="w-4 h-4 mr-2" />Save Configuration</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLStudio;
