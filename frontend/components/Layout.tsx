import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Sprout, 
  Droplets, 
  Bug, 
  MessageSquare, 
  Settings, 
  Menu, 
  X,
  Leaf,
  BookOpen,
  Cpu
} from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Crop Manager', path: '/crops', icon: Sprout },
    { name: 'Crop Library', path: '/library', icon: BookOpen },
    { name: 'ML Studio', path: '/ml-studio', icon: Cpu },
    { name: 'Disease Detect', path: '/disease', icon: Bug },
    { name: 'Irrigation', path: '/irrigation', icon: Droplets },
    { name: 'AI Assistant', path: '/chat', icon: MessageSquare },
    { name: 'Admin', path: '/admin', icon: Settings },
  ];

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div className="flex h-screen bg-agri-50 font-sans">
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 z-20 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      <aside 
        className={`fixed inset-y-0 left-0 z-30 w-64 bg-white border-r border-agri-100 transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-center h-16 border-b border-agri-100 bg-agri-600">
          <div className="flex items-center space-x-2 text-white">
            <Leaf className="w-6 h-6" />
            <span className="text-xl font-bold tracking-wide">AgriSense</span>
          </div>
        </div>

        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setIsSidebarOpen(false)}
                className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors duration-150 ${
                  isActive 
                    ? 'bg-agri-100 text-agri-900' 
                    : 'text-gray-600 hover:bg-agri-50 hover:text-agri-700'
                }`}
              >
                <Icon className={`w-5 h-5 mr-3 ${isActive ? 'text-agri-600' : 'text-gray-400'}`} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        <div className="absolute bottom-0 w-full p-4 border-t border-agri-100 bg-agri-50">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 rounded-full bg-agri-200 flex items-center justify-center text-agri-700 font-bold">
                JD
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">John Doe</p>
              <p className="text-xs text-green-600 flex items-center">
                <span className="w-2 h-2 rounded-full bg-green-500 mr-1 animate-pulse"></span>
                System Online
              </p>
            </div>
          </div>
        </div>
      </aside>

      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="flex items-center justify-between px-6 py-4 bg-white border-b border-agri-100 lg:hidden">
          <div className="flex items-center space-x-2 text-agri-800">
            <Leaf className="w-6 h-6 text-agri-600" />
            <span className="font-bold text-lg">AgriSense</span>
          </div>
          <button 
            onClick={toggleSidebar} 
            className="text-gray-500 hover:text-agri-600 focus:outline-none"
          >
            {isSidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </header>

        <main className="flex-1 overflow-x-hidden overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
