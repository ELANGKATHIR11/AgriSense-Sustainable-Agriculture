import React, { useState } from 'react';
import { analyzeCropImage } from '../services/api';
import { Upload, Camera, AlertCircle, CheckCircle, X, Shield, Activity, Sprout } from 'lucide-react';

const DiseaseDetection: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const [result, setResult] = useState<any | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    setLoading(true);
    try {
      const data = await analyzeCropImage(selectedImage);
      setResult(data);
    } catch (err: any) { 
      console.error(err);
      alert(`Failed to analyze image: ${err.message || 'Unknown error'}`); 
    } finally { setLoading(false); }
  };

  const clearImage = () => { setSelectedImage(null); setPreviewUrl(null); setResult(null); };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-agri-900">Plant Disease Detection</h1>
        <p className="text-gray-600 mt-2">Upload a photo of an affected plant leaf to identify diseases and get treatment advice.</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-agri-100">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center"><Camera className="w-5 h-5 mr-2 text-agri-600" />Image Upload</h2>
          <div className="border-2 border-dashed border-agri-200 rounded-lg p-6 flex flex-col items-center justify-center min-h-[300px] relative bg-agri-50/50">
            {previewUrl ? (
              <div className="relative w-full h-full flex flex-col items-center">
                <img src={previewUrl} alt="Preview" className="max-h-64 object-contain rounded-md shadow-sm" />
                <button onClick={clearImage} title="Clear Image" aria-label="Clear Image" className="absolute top-0 right-0 p-1 bg-red-100 text-red-600 rounded-full hover:bg-red-200"><X className="w-4 h-4" /></button>
                <button onClick={handleAnalyze} disabled={loading} className="mt-6 w-full py-3 bg-agri-600 text-white rounded-lg font-medium hover:bg-agri-700 disabled:opacity-50 flex justify-center items-center">
                  {loading ? <><div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>Analyzing...</> : 'Analyze Plant'}
                </button>
              </div>
            ) : (
              <>
                <div className="p-4 bg-agri-100 rounded-full mb-4"><Upload className="w-8 h-8 text-agri-600" /></div>
                <p className="text-gray-600 font-medium">Click to upload</p>
                <p className="text-xs text-gray-400 mt-2">JPG, PNG (Max 5MB)</p>
                <input type="file" accept="image/*" onChange={handleImageChange} title="Upload Plant Image" className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
              </>
            )}
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-agri-100">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center"><CheckCircle className="w-5 h-5 mr-2 text-agri-600" />Analysis Results</h2>
          {result ? (
            <div className="space-y-6">
              {/* Result Cards Logic */}
              {(result.crop_identified === "Non-Crop Image" || result.crop_identified === "System Error") ? (
                  /* ERROR / WARNING CARD */
                  <div className="bg-orange-50 border border-orange-200 rounded-xl p-6 text-center">
                      <div className="flex justify-center mb-4">
                          <AlertCircle className="w-12 h-12 text-orange-500" />
                      </div>
                      <h3 className="text-xl font-bold text-gray-800 mb-2">{result.diagnosis}</h3>
                      <p className="text-gray-600 mb-4">
                          {result.crop_identified === "Non-Crop Image" 
                            ? "We couldn't detect a clear plant leaf in this photo." 
                            : "The system encountered an initialization error."}
                      </p>
                      <div className="text-sm text-left bg-white p-4 rounded-lg border border-orange-100">
                          <p className="font-semibold text-orange-700 mb-2">Suggestions:</p>
                          <ul className="list-disc pl-5 space-y-1 text-gray-600">
                              {result.cure.immediate_actions.map((action: string, i: number) => (
                                  <li key={i}>{action}</li>
                              ))}
                          </ul>
                      </div>
                  </div>
              ) : (
                  /* SUCCESS / DIAGNOSIS CARD */
                  <>
                    {/* Diagnosis Header */}
                    <div className={`p-4 rounded-lg border ${result.confidence > 80 ? 'bg-red-50 border-red-100' : 'bg-yellow-50 border-yellow-100'}`}>
                        <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-500">Diagnostic Report</span>
                        <span className={`text-xs font-bold px-2 py-1 rounded-full ${result.confidence > 80 ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>{Math.round(result.confidence)}% Confidence</span>
                        </div>
                        <div className="flex items-center space-x-2">
                            <Sprout className="w-5 h-5 text-agri-600" />
                            <h3 className="text-2xl font-bold text-gray-900">{result.crop_identified}</h3>
                        </div>
                        <div className="mt-1">
                            <p className="text-lg font-semibold text-agri-800">{result.diagnosis}</p>
                            {result.scientific_name && (
                                <p className="text-xs italic text-gray-500 font-serif">Pathogen: {result.scientific_name}</p>
                            )}
                        </div>
                        {result.severity && <p className="text-sm text-gray-600 mt-1">Severity: <span className="font-medium">{result.severity}</span></p>}
                    </div>

                    {/* Cure Section */}
                    <div className="bg-white rounded-lg border border-agri-100 overflow-hidden">
                        <div className="bg-agri-50 px-4 py-3 border-b border-agri-100 flex items-center">
                            <Activity className="w-4 h-4 mr-2 text-agri-600" />
                            <h4 className="font-semibold text-gray-800">Cure & Treatment</h4>
                        </div>
                        <div className="p-4 space-y-3">
                            <div>
                                <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Immediate Actions</p>
                                <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
                                    {result.cure.immediate_actions.map((action: string, i: number) => (
                                        <li key={i}>{action}</li>
                                    ))}
                                </ul>
                            </div>
                            {result.cure.chemical_treatments.length > 0 && (
                                <div>
                                    <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Chemical</p>
                                    <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
                                        {result.cure.chemical_treatments.map((action: string, i: number) => (
                                        <li key={i}>{action}</li>
                                    ))}
                                </ul>
                            </div>
                            )}
                            {result.cure.biological_treatments && result.cure.biological_treatments.length > 0 && (
                            <div>
                                <p className="text-xs font-bold text-green-600 uppercase tracking-wider mb-1">Biological Control</p>
                                <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
                                    {result.cure.biological_treatments.map((action: string, i: number) => (
                                        <li key={i}>{action}</li>
                                    ))}
                                </ul>
                            </div>
                            )}
                        </div>
                    </div>

                    {/* Prevention Section */}
                    <div className="bg-white rounded-lg border border-blue-100 overflow-hidden">
                        <div className="bg-blue-50 px-4 py-3 border-b border-blue-100 flex items-center">
                            <Shield className="w-4 h-4 mr-2 text-blue-600" />
                            <h4 className="font-semibold text-gray-800">Long-term Prevention</h4>
                        </div>
                        <div className="p-4">
                            <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
                                {result.prevention.long_term_strategy.map((action: string, i: number) => (
                                        <li key={i}>{action}</li>
                                ))}
                            </ul>
                        </div>
                    </div>
                  </>
              )}

            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-400 min-h-[300px]">
              <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mb-4"><Upload className="w-8 h-8 text-gray-300" /></div>
              <p>Upload an image to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DiseaseDetection;
