import React from 'react';
import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import Chatbot from './components/Chatbot';
import DiseaseDetection from './components/DiseaseDetection';
import CropManager from './components/CropManager';
import Irrigation from './components/Irrigation';
import CropLibrary from './components/CropLibrary';
import MLStudio from './components/MLStudio';
import Admin from './components/Admin';

const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat" element={<Chatbot />} />
          <Route path="/disease" element={<DiseaseDetection />} />
          <Route path="/crops" element={<CropManager />} />
          <Route path="/library" element={<CropLibrary />} />
          <Route path="/ml-studio" element={<MLStudio />} />
          <Route path="/irrigation" element={<Irrigation />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
};

export default App;
