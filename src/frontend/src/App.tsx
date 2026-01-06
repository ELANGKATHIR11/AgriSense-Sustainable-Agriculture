import { Suspense, lazy } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navigation from "./components/Navigation";

// Route-based code splitting (lazy load pages)
const Home = lazy(() => import("./pages/Home"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Recommend = lazy(() => import("./pages/Recommend"));
const SoilAnalysis = lazy(() => import("./pages/SoilAnalysis"));
const Crops = lazy(() => import("./pages/Crops"));
const Admin = lazy(() => import("./pages/Admin"));
const NotFound = lazy(() => import("./pages/NotFound"));
const LiveStats = lazy(() => import("./pages/LiveStats"));
const ImpactGraphs = lazy(() => import("./pages/ImpactGraphs"));
const Irrigation = lazy(() => import("./pages/Irrigation"));
const Harvesting = lazy(() => import("./pages/Harvesting"));
const Chatbot = lazy(() => import("./pages/Chatbot"));
const Tank = lazy(() => import("./pages/Tank"));
const DiseaseManagement = lazy(() => import("./pages/DiseaseManagement"));
const WeedManagement = lazy(() => import("./pages/WeedManagement"));
const Arduino = lazy(() => import("./pages/Arduino"));

// ML RAG Chat Component
const AgriSenseRAGChat = lazy(() => import("./components/AgriSenseRAGChat"));

// Sensible React Query defaults for perf and UX
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000, // 30s fresh data window
      gcTime: 5 * 60_000, // 5 min cache retention
      retry: 2,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

// Router basename - use root path for Firebase Hosting
// Both dev and production use "/" since we unified the vite base path
const routerBasename = "";

const App = () => {
  console.log('App: Rendering...');
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter 
          basename={routerBasename}
          future={{
            v7_startTransition: true,
            v7_relativeSplatPath: true,
          }}
        >
            <div className="min-h-screen bg-gray-50">
              <Navigation />
              <Suspense fallback={<div className="p-6 text-sm text-gray-600 text-center">Loading...</div>}>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/home" element={<Home />} />
                  <Route path="/recommend" element={<Recommend />} />
                  <Route path="/soil-analysis" element={<SoilAnalysis />} />
                  <Route path="/crops" element={<Crops />} />
                  <Route path="/live" element={<LiveStats />} />
                  <Route path="/irrigation" element={<Irrigation />} />
                  <Route path="/tank" element={<Tank />} />
                  <Route path="/harvesting" element={<Harvesting />} />
                  <Route path="/chat" element={<Chatbot />} />
                  <Route path="/ai-chat" element={<AgriSenseRAGChat />} />
                  <Route path="/disease-management" element={<DiseaseManagement />} />
                  <Route path="/weed-management" element={<WeedManagement />} />
                  <Route path="/arduino" element={<Arduino />} />
                  <Route path="/admin" element={<Admin />} />
                  <Route path="/impact" element={<ImpactGraphs />} />
                  {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </Suspense>
            </div>
          </BrowserRouter>
        </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
