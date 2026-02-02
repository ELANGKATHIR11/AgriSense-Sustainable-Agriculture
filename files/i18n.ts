import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

const resources = {
  en: {
    translation: {
      welcome: "Welcome to AgriSense",
      dashboard: "Dashboard",
      chatbot: "AI Assistant",
      crops: "Crops",
      disease: "Disease Detection",
      weeds: "Weed Management",
      irrigation: "Smart Irrigation",
    },
  },
  hi: {
    translation: {
      welcome: "एग्रीसेंस में आपका स्वागत है",
      dashboard: "डैशबोर्ड",
      chatbot: "एआई सहायक",
      crops: "फसलें",
      disease: "रोग पहचान",
      weeds: "खरपतवार प्रबंधन",
      irrigation: "स्मार्ट सिंचाई",
    },
  },
}

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: 'en',
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false,
    },
  })

export default i18n
