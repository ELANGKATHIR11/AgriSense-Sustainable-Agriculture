import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Translation resources for 5 languages
const resources = {
  en: {
    translation: {
      // Navigation
      welcome: "Welcome to AgriSense",
      dashboard: "Dashboard",
      chatbot: "AI Assistant",
      crops: "Crop Guide",
      disease: "Disease Detection",
      weeds: "Weed Management",
      irrigation: "Smart Irrigation",
      admin: "Admin Panel",
      settings: "Settings",
      
      // Dashboard
      liveMonitoring: "Live Monitoring",
      sensorData: "Sensor Data",
      temperature: "Temperature",
      humidity: "Humidity",
      soilMoisture: "Soil Moisture",
      phLevel: "pH Level",
      nitrogen: "Nitrogen",
      phosphorus: "Phosphorus",
      potassium: "Potassium",
      lightIntensity: "Light Intensity",
      
      // Actions
      analyze: "Analyze",
      predict: "Predict",
      recommend: "Recommend",
      upload: "Upload",
      submit: "Submit",
      cancel: "Cancel",
      save: "Save",
      delete: "Delete",
      loading: "Loading...",
      
      // Crop Recommendation
      cropRecommendation: "Crop Recommendation",
      yieldPrediction: "Yield Prediction",
      waterRequirement: "Water Requirement",
      seasonClassification: "Season Classification",
      predictedCrop: "Predicted Crop",
      confidence: "Confidence",
      
      // Disease Detection
      uploadImage: "Upload Plant Image",
      dragDrop: "Drag and drop or click to upload",
      detectDisease: "Detect Disease",
      healthStatus: "Health Status",
      treatment: "Treatment Recommendations",
      
      // Chatbot
      askQuestion: "Ask a question about farming...",
      sendMessage: "Send",
      thinking: "Thinking...",
      
      // Admin
      systemStatus: "System Status",
      mlModels: "ML Models",
      activities: "Activities",
      
      // Common
      noData: "No data available",
      error: "An error occurred",
      success: "Success",
      warning: "Warning",
    },
  },
  hi: {
    translation: {
      // Navigation
      welcome: "एग्रीसेंस में आपका स्वागत है",
      dashboard: "डैशबोर्ड",
      chatbot: "एआई सहायक",
      crops: "फसल मार्गदर्शिका",
      disease: "रोग पहचान",
      weeds: "खरपतवार प्रबंधन",
      irrigation: "स्मार्ट सिंचाई",
      admin: "व्यवस्थापक पैनल",
      settings: "सेटिंग्स",
      
      // Dashboard
      liveMonitoring: "लाइव निगरानी",
      sensorData: "सेंसर डेटा",
      temperature: "तापमान",
      humidity: "आर्द्रता",
      soilMoisture: "मिट्टी की नमी",
      phLevel: "पीएच स्तर",
      nitrogen: "नाइट्रोजन",
      phosphorus: "फास्फोरस",
      potassium: "पोटेशियम",
      lightIntensity: "प्रकाश तीव्रता",
      
      // Actions
      analyze: "विश्लेषण करें",
      predict: "भविष्यवाणी करें",
      recommend: "सुझाव दें",
      upload: "अपलोड करें",
      submit: "जमा करें",
      cancel: "रद्द करें",
      save: "सहेजें",
      delete: "हटाएं",
      loading: "लोड हो रहा है...",
      
      // Crop Recommendation
      cropRecommendation: "फसल सिफारिश",
      yieldPrediction: "उपज भविष्यवाणी",
      waterRequirement: "पानी की आवश्यकता",
      seasonClassification: "मौसम वर्गीकरण",
      predictedCrop: "अनुशंसित फसल",
      confidence: "विश्वास स्तर",
      
      // Disease Detection
      uploadImage: "पौधे की छवि अपलोड करें",
      dragDrop: "खींचें और छोड़ें या अपलोड करने के लिए क्लिक करें",
      detectDisease: "रोग का पता लगाएं",
      healthStatus: "स्वास्थ्य स्थिति",
      treatment: "उपचार सिफारिशें",
      
      // Chatbot
      askQuestion: "खेती के बारे में कोई सवाल पूछें...",
      sendMessage: "भेजें",
      thinking: "सोच रहा है...",
      
      // Admin
      systemStatus: "सिस्टम स्थिति",
      mlModels: "एमएल मॉडल",
      activities: "गतिविधियां",
      
      // Common
      noData: "कोई डेटा उपलब्ध नहीं",
      error: "एक त्रुटि हुई",
      success: "सफलता",
      warning: "चेतावनी",
    },
  },
  ta: {
    translation: {
      // Navigation
      welcome: "அக்ரிசென்ஸுக்கு வரவேற்கிறோம்",
      dashboard: "டாஷ்போர்டு",
      chatbot: "AI உதவியாளர்",
      crops: "பயிர் வழிகாட்டி",
      disease: "நோய் கண்டறிதல்",
      weeds: "களை மேலாண்மை",
      irrigation: "ஸ்மார்ட் நீர்ப்பாசனம்",
      admin: "நிர்வாகி பேனல்",
      settings: "அமைப்புகள்",
      
      // Dashboard
      liveMonitoring: "நேரடி கண்காணிப்பு",
      sensorData: "சென்சார் தரவு",
      temperature: "வெப்பநிலை",
      humidity: "ஈரப்பதம்",
      soilMoisture: "மண் ஈரப்பதம்",
      phLevel: "pH நிலை",
      nitrogen: "நைட்ரஜன்",
      phosphorus: "பாஸ்பரஸ்",
      potassium: "பொட்டாசியம்",
      lightIntensity: "ஒளி தீவிரம்",
      
      // Actions
      analyze: "பகுப்பாய்வு",
      predict: "கணிக்க",
      recommend: "பரிந்துரைக்க",
      upload: "பதிவேற்றம்",
      submit: "சமர்ப்பி",
      cancel: "ரத்து செய்",
      save: "சேமி",
      delete: "நீக்கு",
      loading: "ஏற்றுகிறது...",
      
      // Crop Recommendation
      cropRecommendation: "பயிர் பரிந்துரை",
      yieldPrediction: "விளைச்சல் கணிப்பு",
      waterRequirement: "நீர் தேவை",
      seasonClassification: "பருவ வகைப்பாடு",
      predictedCrop: "பரிந்துரைக்கப்பட்ட பயிர்",
      confidence: "நம்பகத்தன்மை",
      
      // Disease Detection
      uploadImage: "தாவர படத்தை பதிவேற்றவும்",
      dragDrop: "இழுத்து விடவும் அல்லது கிளிக் செய்யவும்",
      detectDisease: "நோயை கண்டறியவும்",
      healthStatus: "ஆரோக்கிய நிலை",
      treatment: "சிகிச்சை பரிந்துரைகள்",
      
      // Chatbot
      askQuestion: "விவசாயம் பற்றி கேள்வி கேளுங்கள்...",
      sendMessage: "அனுப்பு",
      thinking: "யோசிக்கிறேன்...",
      
      // Admin
      systemStatus: "கணினி நிலை",
      mlModels: "ML மாடல்கள்",
      activities: "செயல்பாடுகள்",
      
      // Common
      noData: "தரவு இல்லை",
      error: "பிழை ஏற்பட்டது",
      success: "வெற்றி",
      warning: "எச்சரிக்கை",
    },
  },
  te: {
    translation: {
      // Navigation
      welcome: "అగ్రిసెన్స్‌కు స్వాగతం",
      dashboard: "డాష్‌బోర్డ్",
      chatbot: "AI సహాయకుడు",
      crops: "పంట గైడ్",
      disease: "వ్యాధి గుర్తింపు",
      weeds: "కలుపు నిర్వహణ",
      irrigation: "స్మార్ట్ నీటిపారుదల",
      admin: "అడ్మిన్ ప్యానల్",
      settings: "సెట్టింగ్‌లు",
      
      // Dashboard
      liveMonitoring: "లైవ్ మానిటరింగ్",
      sensorData: "సెన్సార్ డేటా",
      temperature: "ఉష్ణోగ్రత",
      humidity: "తేమ",
      soilMoisture: "నేల తేమ",
      phLevel: "pH స్థాయి",
      nitrogen: "నైట్రోజన్",
      phosphorus: "ఫాస్ఫరస్",
      potassium: "పొటాషియం",
      lightIntensity: "కాంతి తీవ్రత",
      
      // Actions
      analyze: "విశ్లేషించు",
      predict: "అంచనా వేయండి",
      recommend: "సిఫారసు చేయండి",
      upload: "అప్‌లోడ్",
      submit: "సమర్పించు",
      cancel: "రద్దు చేయి",
      save: "సేవ్ చేయి",
      delete: "తొలగించు",
      loading: "లోడ్ అవుతోంది...",
      
      // Crop Recommendation
      cropRecommendation: "పంట సిఫారసు",
      yieldPrediction: "దిగుబడి అంచనా",
      waterRequirement: "నీటి అవసరం",
      seasonClassification: "సీజన్ వర్గీకరణ",
      predictedCrop: "సిఫారసు చేసిన పంట",
      confidence: "నమ్మకం",
      
      // Disease Detection
      uploadImage: "మొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి",
      dragDrop: "డ్రాగ్ అండ్ డ్రాప్ లేదా క్లిక్ చేయండి",
      detectDisease: "వ్యాధిని గుర్తించండి",
      healthStatus: "ఆరోగ్య స్థితి",
      treatment: "చికిత్స సిఫారసులు",
      
      // Chatbot
      askQuestion: "వ్యవసాయం గురించి ప్రశ్న అడగండి...",
      sendMessage: "పంపు",
      thinking: "ఆలోచిస్తున్నాను...",
      
      // Admin
      systemStatus: "సిస్టమ్ స్థితి",
      mlModels: "ML మోడల్స్",
      activities: "కార్యకలాపాలు",
      
      // Common
      noData: "డేటా అందుబాటులో లేదు",
      error: "లోపం సంభవించింది",
      success: "విజయం",
      warning: "హెచ్చరిక",
    },
  },
  kn: {
    translation: {
      // Navigation
      welcome: "ಅಗ್ರಿಸೆನ್ಸ್‌ಗೆ ಸ್ವಾಗತ",
      dashboard: "ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
      chatbot: "AI ಸಹಾಯಕ",
      crops: "ಬೆಳೆ ಮಾರ್ಗದರ್ಶಿ",
      disease: "ರೋಗ ಪತ್ತೆ",
      weeds: "ಕಳೆ ನಿರ್ವಹಣೆ",
      irrigation: "ಸ್ಮಾರ್ಟ್ ನೀರಾವರಿ",
      admin: "ಅಡ್ಮಿನ್ ಪ್ಯಾನಲ್",
      settings: "ಸೆಟ್ಟಿಂಗ್ಸ್",
      
      // Dashboard
      liveMonitoring: "ಲೈವ್ ಮಾನಿಟರಿಂಗ್",
      sensorData: "ಸೆನ್ಸಾರ್ ಡೇಟಾ",
      temperature: "ತಾಪಮಾನ",
      humidity: "ಆರ್ದ್ರತೆ",
      soilMoisture: "ಮಣ್ಣಿನ ತೇವಾಂಶ",
      phLevel: "pH ಮಟ್ಟ",
      nitrogen: "ಸಾರಜನಕ",
      phosphorus: "ರಂಜಕ",
      potassium: "ಪೊಟ್ಯಾಸಿಯಂ",
      lightIntensity: "ಬೆಳಕಿನ ತೀವ್ರತೆ",
      
      // Actions
      analyze: "ವಿಶ್ಲೇಷಿಸಿ",
      predict: "ಊಹಿಸಿ",
      recommend: "ಶಿಫಾರಸು ಮಾಡಿ",
      upload: "ಅಪ್‌ಲೋಡ್",
      submit: "ಸಲ್ಲಿಸಿ",
      cancel: "ರದ್ದುಮಾಡಿ",
      save: "ಉಳಿಸಿ",
      delete: "ಅಳಿಸಿ",
      loading: "ಲೋಡ್ ಆಗುತ್ತಿದೆ...",
      
      // Crop Recommendation
      cropRecommendation: "ಬೆಳೆ ಶಿಫಾರಸು",
      yieldPrediction: "ಇಳುವರಿ ಊಹೆ",
      waterRequirement: "ನೀರಿನ ಅವಶ್ಯಕತೆ",
      seasonClassification: "ಋತು ವರ್ಗೀಕರಣ",
      predictedCrop: "ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆ",
      confidence: "ವಿಶ್ವಾಸ",
      
      // Disease Detection
      uploadImage: "ಗಿಡದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
      dragDrop: "ಡ್ರ್ಯಾಗ್ ಮತ್ತು ಡ್ರಾಪ್ ಅಥವಾ ಕ್ಲಿಕ್ ಮಾಡಿ",
      detectDisease: "ರೋಗವನ್ನು ಪತ್ತೆ ಮಾಡಿ",
      healthStatus: "ಆರೋಗ್ಯ ಸ್ಥಿತಿ",
      treatment: "ಚಿಕಿತ್ಸೆ ಶಿಫಾರಸುಗಳು",
      
      // Chatbot
      askQuestion: "ಕೃಷಿ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ...",
      sendMessage: "ಕಳುಹಿಸು",
      thinking: "ಯೋಚಿಸುತ್ತಿದ್ದೇನೆ...",
      
      // Admin
      systemStatus: "ಸಿಸ್ಟಮ್ ಸ್ಥಿತಿ",
      mlModels: "ML ಮಾಡೆಲ್ಸ್",
      activities: "ಚಟುವಟಿಕೆಗಳು",
      
      // Common
      noData: "ಡೇಟಾ ಲಭ್ಯವಿಲ್ಲ",
      error: "ದೋಷ ಸಂಭವಿಸಿದೆ",
      success: "ಯಶಸ್ಸು",
      warning: "ಎಚ್ಚರಿಕೆ",
    },
  },
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    lng: 'en',
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
    },
  });

export default i18n;
