import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import Chatbot from './pages/Chatbot'
import DiseaseManagement from './pages/DiseaseManagement'
import WeedManagement from './pages/WeedManagement'
import Crops from './pages/Crops'
import Irrigation from './pages/Irrigation'
import Admin from './pages/Admin'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/chatbot" element={<Chatbot />} />
        <Route path="/disease" element={<DiseaseManagement />} />
        <Route path="/weeds" element={<WeedManagement />} />
        <Route path="/crops" element={<Crops />} />
        <Route path="/irrigation" element={<Irrigation />} />
        <Route path="/admin" element={<Admin />} />
      </Routes>
    </Layout>
  )
}

export default App
