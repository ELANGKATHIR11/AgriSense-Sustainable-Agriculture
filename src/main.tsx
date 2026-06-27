import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import { QueryProvider } from './providers/QueryProvider.tsx';
import { LanguageProvider } from './providers/LanguageProvider.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryProvider>
      <LanguageProvider>
        <App />
      </LanguageProvider>
    </QueryProvider>
  </StrictMode>,
);

