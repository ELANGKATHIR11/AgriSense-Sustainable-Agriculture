/**
 * License: GNU Affero General Public License v3.0 (AGPL-3.0)
 * This file is part of AgriSense.
 * 
 * TERMS OF USE:
 * This project is licensed under the AGPL-3.0. Private modifications or private use
 * without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
 * AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
 * Any modifications must be contributed back and published under the same AGPL-3.0 license.
 */

import React, { createContext, useState, useEffect, useContext } from "react";

export type Language = "en" | "ta" | "te" | "ml" | "hi";

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

// High-fidelity dictionaries for 100% translatable UI covering 5 languages
const TRANSLATIONS: Record<Language, Record<string, string>> = {
  en: {
    "nav.overview": "Overview",
    "nav.dashboard": "Dashboard",
    "nav.twin": "Digital Twin",
    "nav.aivision": "AI Vision",
    "nav.disease": "Disease Vision",
    "nav.fieldintel": "Field Intelligence",
    "nav.suitability": "Crop Suitability",
    "nav.catalog": "Crop Catalog",
    "nav.irrigation": "Irrigation",
    "nav.sensors": "IoT Sensors",
    "nav.weather": "Weather Intel",
    "nav.yield": "Yield Forecast",
    "nav.commerce": "Commerce & Infrastructure",
    "nav.marketplace": "Agri Marketplace",
    "nav.market_intel": "Market Intelligence",
    "nav.aihub": "Local AI Hub",
    "nav.chat": "AgriGPT Chat",
    "nav.swarm": "ASO Swarm",
    "nav.mlops": "MLOps Control",
    "nav.agriops": "AgriOps Hub",
    "nav.settings": "Settings",
    "settings.title": "System Preferences",
    "settings.subtitle": "Configure farm parameters, alert thresholds, and review edge AI system status.",
    "settings.farm_config": "Farmland Configuration",
    "settings.farm_name": "Registered Farm Sector",
    "settings.moisture_label": "Moisture Alert Threshold",
    "settings.save": "Save Configuration",
    "settings.saving": "Saving...",
    "settings.saved": "Settings Saved!",
    "settings.status": "Edge System Status",
    "settings.edge_title": "Edge-Only Architecture",
    "settings.edge_desc": "All AI inference runs locally on your hardware via Ollama. No cloud API keys required — zero external data transmission.",
    "settings.large_fonts": "Large Fonts",
    "settings.high_contrast": "High Contrast",
    "settings.reduced_motion": "Reduced Motion",
    "settings.accessibility": "Accessibility Settings",
    "dashboard.header": "Farm Control Center",
    "dashboard.live": "Live Telemetry",
    "dashboard.alerts": "Active Alerts",
    "dashboard.actions": "Quick Actions",
    "agrigpt.title": "AgriGPT Assistant",
    "agrigpt.placeholder": "Ask anything about crop disease, yield, or weather..."
  },
  ta: {
    "nav.overview": "கண்ணோட்டம்",
    "nav.dashboard": "டாஷ்போர்டு",
    "nav.twin": "டிஜிட்டல் இரட்டை",
    "nav.aivision": "ஏஐ பார்வை",
    "nav.disease": "நோய் கண்டறிதல்",
    "nav.fieldintel": "புல நுண்ணறிவு",
    "nav.suitability": "பயிர் பொருத்தம்",
    "nav.catalog": "பயிர் பட்டியல்",
    "nav.irrigation": "நீர்ப்பாசனம்",
    "nav.sensors": "ஐஓடி சென்சார்கள்",
    "nav.weather": "வானிலை தகவல்",
    "nav.yield": "மகசூல் முன்னறிவிப்பு",
    "nav.commerce": "வர்த்தகம் மற்றும் உள்கட்டமைப்பு",
    "nav.marketplace": "விவசாய சந்தை",
    "nav.market_intel": "சந்தை நுண்ணறிவு",
    "nav.aihub": "உள்ளூர் ஏஐ ஹப்",
    "nav.chat": "அக்ரிஜிபிடி அரட்டை",
    "nav.swarm": "ஏஎஸ்ஓ ஸ்வார்ம்",
    "nav.mlops": "எம்எல்ஓப்ஸ் கட்டுப்பாடு",
    "nav.agriops": "அக்ரிஆப்ஸ் மையம்",
    "nav.settings": "அமைப்புகள்",
    "settings.title": "அமைப்பு விருப்பத்தேர்வுகள்",
    "settings.subtitle": "பண்ணை அளவுருக்கள், எச்சரிக்கை வரம்புகளை உள்ளமைக்கவும் மற்றும் எட்ஜ் ஏஐ கணினி நிலையை மதிப்பாய்வு செய்யவும்.",
    "settings.farm_config": "விவசாய நில கட்டமைப்பு",
    "settings.farm_name": "பதிவு செய்யப்பட்ட பண்ணை பிரிவு",
    "settings.moisture_label": "ஈரப்பதம் எச்சரிக்கை வரம்பு",
    "settings.save": "கட்டமைப்பை சேமிக்கவும்",
    "settings.saving": "சேமிக்கிறது...",
    "settings.saved": "அமைப்புகள் சேமிக்கப்பட்டன!",
    "settings.status": "எட்ஜ் கணினி நிலை",
    "settings.edge_title": "உள்ளூர் கட்டமைப்பு மட்டுமே",
    "settings.edge_desc": "அனைத்து ஏஐ அனுமானங்களும் உங்கள் வன்பொருளில் ஒல்லாமா வழியாக உள்நாட்டில் இயங்குகின்றன. கிளவுட் ஏபிஐ விசைகள் தேவையில்லை — பூஜ்ஜிய வெளிப்புற தரவு பரிமாற்றம்.",
    "settings.large_fonts": "பெரிய எழுத்துருக்கள்",
    "settings.high_contrast": "அதிக மாறுபாடு",
    "settings.reduced_motion": "குறைக்கப்பட்ட இயக்கம்",
    "settings.accessibility": "அணுகல்தன்மை அமைப்புகள்",
    "dashboard.header": "பண்ணை கட்டுப்பாட்டு மையம்",
    "dashboard.live": "நேரடி அளவீடு",
    "dashboard.alerts": "செயலில் உள்ள எச்சரிக்கைகள்",
    "dashboard.actions": "விரைவான செயல்கள்",
    "agrigpt.title": "அக்ரிஜிபிடி உதவியாளர்",
    "agrigpt.placeholder": "பயிர் நோய், மகசூல் அல்லது வானிலை பற்றி எதையும் கேளுங்கள்..."
  },
  te: {
    "nav.overview": "అవలోకనం",
    "nav.dashboard": "డాష్‌బోర్డ్",
    "nav.twin": "డిజిటల్ ట్విన్",
    "nav.aivision": "AI విజన్",
    "nav.disease": "వ్యాధి గుర్తింపు",
    "nav.fieldintel": "ఫీల్డ్ ఇంటెలిజెన్స్",
    "nav.suitability": "పంట అనుకూలత",
    "nav.catalog": "పంటల జాబితా",
    "nav.irrigation": "నీటి పారుదల",
    "nav.sensors": "IoT సెన్సార్లు",
    "nav.weather": "వాతావరణ సమాచారం",
    "nav.yield": "దిగుబడి సూచన",
    "nav.commerce": "వాణిజ్యం & మౌలిక సదుపాయాలు",
    "nav.marketplace": "వ్యవసాయ మార్కెట్",
    "nav.market_intel": "మార్కెట్ ఇంటెలిజెన్స్",
    "nav.aihub": "స్థానిక AI హబ్",
    "nav.chat": "AgriGPT చాట్",
    "nav.swarm": "ASO స్వార్మ్",
    "nav.mlops": "MLOps నియంత్రణ",
    "nav.agriops": "AgriOps హబ్",
    "nav.settings": "సెట్టింగులు",
    "settings.title": "సిస్టమ్ ప్రాధాన్యతలు",
    "settings.subtitle": "పంట పారామితులు, హెచ్చరిక పరిమితులను కాన్ఫిగర్ చేయండి మరియు ఎడ్జ్ AI సిస్టమ్ స్థితిని సమీక్షించండి.",
    "settings.farm_config": "వ్యవసాయ క్షేత్రం కాన్ఫిగరేషన్",
    "settings.farm_name": "నమోదిత వ్యవసాయ రంగం",
    "settings.moisture_label": "తేమ హెచ్చరిక పరిమితి",
    "settings.save": "కాన్ఫిగరేషన్‌ను సేవ్ చేయి",
    "settings.saving": "సేవ్ అవుతోంది...",
    "settings.saved": "సెట్టింగులు సేవ్ చేయబడ్డాయి!",
    "settings.status": "ఎడ్జ్ సిస్టమ్ స్థితి",
    "settings.edge_title": "స్థానిక ఆర్కిటెక్చర్ మాత్రమే",
    "settings.edge_desc": "అన్ని AI అనుమితులు Ollama ద్వారా మీ హార్డ్‌వేర్‌లో స్థానికంగా నడుస్తాయి. క్లౌడ్ API కీలు అవసరం లేదు — జీరో బాహ్య డేటా బదిలీ.",
    "settings.large_fonts": "పెద్ద ఫాంట్లు",
    "settings.high_contrast": "అధిక కాంట్రాస్ట్",
    "settings.reduced_motion": "తగ్గించబడిన చలనం",
    "settings.accessibility": "యాక్సెస్బిలిటీ సెట్టింగులు",
    "dashboard.header": "వ్యవసాయ నియంత్రణ కేంద్రం",
    "dashboard.live": "ప్రత్యక్ష టెలిమెట్రీ",
    "dashboard.alerts": "సక్రియ హెచ్చరికలు",
    "dashboard.actions": "త్వరిత చర్యలు",
    "agrigpt.title": "AgriGPT సహాయకుడు",
    "agrigpt.placeholder": "పంట వ్యాధి, దిగుబడి లేదా వాతావరణం గురించి ఏదైనా అడగండి..."
  },
  ml: {
    "nav.overview": "അവലോകനം",
    "nav.dashboard": "ഡാഷ്‌ബോർഡ്",
    "nav.twin": "ഡിജിറ്റൽ ട്വിൻ",
    "nav.aivision": "AI വിഷൻ",
    "nav.disease": "രോഗ നിർണ്ണയം",
    "nav.fieldintel": "ഫീൽഡ് ഇന്റലിജൻസ്",
    "nav.suitability": "വിള അനുയോജ്യത",
    "nav.catalog": "വിള വിവരങ്ങൾ",
    "nav.irrigation": "ജലസേചനം",
    "nav.sensors": "IoT സെൻസറുകൾ",
    "nav.weather": "കാലാവസ്ഥാ വിവരങ്ങൾ",
    "nav.yield": "വിളവ് പ്രവചനം",
    "nav.commerce": "കൊമേഴ്‌സ് & ഇൻഫ്രാസ്ട്രക്ചർ",
    "nav.marketplace": "അഗ്രി മാർക്കറ്റ്",
    "nav.market_intel": "മാർക്കറ്റ് ഇന്റലിജൻസ്",
    "nav.aihub": "ലോക്കൽ AI ഹബ്",
    "nav.chat": "AgriGPT ചാറ്റ്",
    "nav.swarm": "ASO സ്വാം",
    "nav.mlops": "MLOps നിയന്ത്രണം",
    "nav.agriops": "AgriOps ഹബ്",
    "nav.settings": "ക്രമീകരണങ്ങൾ",
    "settings.title": "സിസ്റ്റം മുൻഗണനകൾ",
    "settings.subtitle": "ഫാം പാരാമീറ്ററുകൾ, അലേർട്ട് പരിധികൾ കോൺഫിഗർ ചെയ്യുക, എഡ്ജ് AI സിസ്റ്റം നില അവലോകനം ചെയ്യുക.",
    "settings.farm_config": "കൃഷിഭൂമി ക്രമീകരണം",
    "settings.farm_name": "രജിസ്റ്റർ ചെയ്ത ഫാം മേഖല",
    "settings.moisture_label": "ഈർപ്പ അലേർട്ട് പരിധി",
    "settings.save": "ക്രമീകരണം സംരക്ഷിക്കുക",
    "settings.saving": "സംരക്ഷിക്കുന്നു...",
    "settings.saved": "ക്രമീകരണങ്ങൾ സംരക്ഷിച്ചു!",
    "settings.status": "എഡ്ജ് സിസ്റ്റം നില",
    "settings.edge_title": "ലോക്കൽ ആർക്കിടെക്ചർ മാത്രം",
    "settings.edge_desc": "എല്ലാ AI അനുമാനങ്ങളും ഒല്ലാമ വഴി നിങ്ങളുടെ ഹാർഡ്‌വെയറിൽ പ്രാദേശികമായി പ്രവർത്തിക്കുന്നു. ക്ലൗഡ് API കീകൾ ആവശ്യമില്ല — പൂജ്യം ബാഹ്യ ഡാറ്റ കൈമാറ്റം.",
    "settings.large_fonts": "വലിയ ഫോണ്ടുകൾ",
    "settings.high_contrast": "ഉയർന്ന ദൃശ്യതീവ്രത",
    "settings.reduced_motion": "കുറഞ്ഞ ചലനം",
    "settings.accessibility": "പ്രവേശനക്ഷമത ക്രമീകരണങ്ങൾ",
    "dashboard.header": "ഫാം നിയന്ത്രണ കേന്ദ്രം",
    "dashboard.live": "തത്സമയ ടെലിമെട്രി",
    "dashboard.alerts": "സജീവ അലേർട്ടുകൾ",
    "dashboard.actions": "ദ്രുത നടപടികൾ",
    "agrigpt.title": "AgriGPT സഹായി",
    "agrigpt.placeholder": "വിള രോഗങ്ങൾ, വിളവ് അല്ലെങ്കിൽ കാലാവസ്ഥ എന്നിവയെക്കുറിച്ച് എന്തും ചോദിക്കുക..."
  },
  hi: {
    "nav.overview": "अवलोकन",
    "nav.dashboard": "डैशबोर्ड",
    "nav.twin": "डिजिटल ट्विन",
    "nav.aivision": "एआई विजन",
    "nav.disease": "रोग पहचान",
    "nav.fieldintel": "कृषि बुद्धिमत्ता",
    "nav.suitability": "फसल उपयुक्तता",
    "nav.catalog": "फसल सूची",
    "nav.irrigation": "सिंचाई प्रबंधन",
    "nav.sensors": "आईओटी सेंसर",
    "nav.weather": "मौसम सलाह",
    "nav.yield": "उपज पूर्वानुमान",
    "nav.commerce": "वाणिज्य और बुनियादी ढांचा",
    "nav.marketplace": "कृषि बाजार",
    "nav.market_intel": "बाजार बुद्धिमत्ता",
    "nav.aihub": "स्थानीय एआई हब",
    "nav.chat": "AgriGPT चैट",
    "nav.swarm": "ASO झुंड",
    "nav.mlops": "MLOps नियंत्रण",
    "nav.agriops": "AgriOps हब",
    "nav.settings": "सेटिंग्स",
    "settings.title": "सिस्टम प्राथमिकताएं",
    "settings.subtitle": "कृषि मापदंडों, अलर्ट थ्रेशोल्ड को कॉन्फ़िगर करें और एज एआई सिस्टम स्थिति की समीक्षा करें।",
    "settings.farm_config": "कृषि भूमि विन्यास",
    "settings.farm_name": "पंजीकृत फार्म क्षेत्र",
    "settings.moisture_label": "नमी चेतावनी थ्रेशोल्ड",
    "settings.save": "विन्यास सहेजें",
    "settings.saving": "सहेज रहा है...",
    "settings.saved": "सेटिंग्स सहेजी गईं!",
    "settings.status": "एज सिस्टम स्थिति",
    "settings.edge_title": "केवल स्थानीय आर्किटेक्चर",
    "settings.edge_desc": "सभी एआई निष्कर्ष स्थानीय रूप से आपके हार्डवेयर पर ओल्लामा के माध्यम से चलते हैं। क्लाउड एपीआई कुंजी की कोई आवश्यकता नहीं — शून्य बाहरी डेटा ट्रांसमिशन।",
    "settings.large_fonts": "बड़े फ़ॉन्ट",
    "settings.high_contrast": "उच्च कंट्रास्ट",
    "settings.reduced_motion": "कम गतिशीलता",
    "settings.accessibility": "अभिगम्यता सेटिंग्स",
    "dashboard.header": "फार्म नियंत्रण केंद्र",
    "dashboard.live": "लाइव टेलीमेट्री",
    "dashboard.alerts": "सक्रिय अलर्ट",
    "dashboard.actions": "त्वरित क्रियाएं",
    "agrigpt.title": "AgriGPT सहायक",
    "agrigpt.placeholder": "फसल रोग, उपज या मौसम के बारे में कुछ भी पूछें..."
  }
};

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<Language>(() => {
    return (localStorage.getItem("agrisense_language") as Language) || "en";
  });

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem("agrisense_language", lang);
    
    // Attempt syncing preferred language back to DB dynamically
    const profileStr = localStorage.getItem("agrisense_profile");
    if (profileStr) {
      try {
        const profile = JSON.parse(profileStr);
        fetch("/api/auth/language", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: profile.email, preferred_language: lang })
        }).catch(err => console.debug("Language sync failed", err));
      } catch (e) {
        console.debug(e);
      }
    }
  };

  const t = (key: string): string => {
    return TRANSLATIONS[language][key] || TRANSLATIONS["en"][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useTranslation = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useTranslation must be used within a LanguageProvider");
  }
  return context;
};
