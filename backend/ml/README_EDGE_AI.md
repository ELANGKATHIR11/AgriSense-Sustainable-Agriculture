# Edge AI Agricultural Assistant

## Overview

This edge AI system provides **offline, local AI capabilities** for agricultural assistance without requiring external API calls. It includes:

1. **Edge AI Chatbot** - Neural Network-based chatbot for farmer questions
2. **Edge AI Vision** - Plant disease detection and analysis
3. **Cultivation Guides** - Comprehensive guides for all 96 crops
4. **Disease Knowledge Base** - Complete disease information for all crops

## Features

### Edge AI Chatbot
- ✅ Works completely offline (no internet required)
- ✅ Neural Network-based Q&A system
- ✅ Provides cultivation guides for all 96 crops
- ✅ Answers questions about:
  - Water efficiency (reduce water usage by 30-50%)
  - Fertilizer optimization (reduce usage by 20-40%)
  - Yield enhancement (increase yield by 15-30%)
  - Pest and disease management
  - Soil management
  - Irrigation scheduling

### Edge AI Vision
- ✅ Offline plant disease detection
- ✅ Analyzes plant images locally
- ✅ Provides:
  - Disease identification
  - Treatment recommendations (chemical & organic)
  - Prevention measures
  - Cure instructions
- ✅ Supports all 96 crops

## Setup

### 1. Generate Knowledge Bases

```bash
cd backend/ml
python setup_edge_ai.py
```

This will generate:
- `knowledge_base/cultivation_guides.json` - Guides for all 96 crops
- `knowledge_base/disease_knowledge.json` - Disease database

### 2. Install Dependencies

```bash
pip install flask flask-cors scikit-learn pandas numpy pillow opencv-python
```

### 3. Start Edge AI Service

```bash
cd backend/ml
python edge_ai_service.py
```

Service runs on port 5002 by default.

## Usage

### Chatbot API

**Query Endpoint:**
```bash
POST http://localhost:5002/chatbot/query
Content-Type: application/json

{
  "query": "how to reduce water usage for rice",
  "crop_name": "Rice",
  "context": {}
}
```

**Cultivation Guide:**
```bash
GET http://localhost:5002/chatbot/cultivation-guide/Rice
```

### Vision API

**Analyze Image:**
```bash
POST http://localhost:5002/vision/analyze
Content-Type: multipart/form-data

image: <file>
crop_name: Rice
```

**Analyze from Path:**
```bash
POST http://localhost:5002/vision/analyze-path
Content-Type: application/json

{
  "image_path": "/path/to/image.jpg",
  "crop_name": "Rice"
}
```

**Get Crop Diseases:**
```bash
GET http://localhost:5002/diseases/Rice
```

## Integration

The edge AI system is automatically integrated into:
- `backend/services/llmService.js` - Uses edge AI chatbot
- `backend/services/vlmService.js` - Uses edge AI vision

Set environment variable:
```env
USE_EDGE_AI=true
```

## Architecture

```
Edge AI System
├── edge_ai_chatbot.py      # Chatbot with Neural Network
├── edge_ai_vision.py       # Vision model for disease detection
├── edge_ai_service.py       # Flask API service
├── generate_cultivation_guides.py  # Generate crop guides
├── generate_disease_knowledge.py   # Generate disease DB
└── knowledge_base/
    ├── cultivation_guides.json    # 96 crop guides
    └── disease_knowledge.json      # Disease database
```

## Supported Crops (96 Total)

Almond, Apple, Arecanut, Arhar, Bajra, Banana, Barley, Barnyard_Millet, Beetroot, Bitter_Gourd, Black_Pepper, Bottle_Gourd, Brinjal, Buckwheat, Cabbage, Cardamom, Carrot, Cashew, Castor, Cauliflower, Chickpea, Chilli, Cluster_Bean, Coconut, Coffee, Coriander, Cotton, Cucumber, Cumin, Custard_Apple, Dragon_Fruit, Fenugreek, Field_Pea, Foxtail_Millet, French_Bean, Garlic, Ginger, Grapes, Green_Pea, Groundnut, Guava, Horse_Gram, Jackfruit, Jowar, Jute, Kidney_Bean, Kodo_Millet, Lentil, Lettuce, Linseed, Litchi, Little_Millet, Maize, Mango, Masoor, Moong, Moth_Bean, Muskmelon, Mustard, Niger, Oats, Okra, Onion, Orange, Papaya, Pearl_Millet, Pigeon_Pea, Pineapple, Pomegranate, Potato, Proso_Millet, Pumpkin, Radish, Ragi, Rice, Ridge_Gourd, Rubber, Safflower, Sapota, Sesame, Sorghum, Soybean, Spinach, Strawberry, Sugarcane, Sunflower, Sweet_Potato, Tea, Tobacco, Tomato, Turmeric, Turnip, Urad, Walnut, Watermelon, Wheat

## Benefits

1. **Privacy** - All processing happens locally
2. **Cost** - No API costs
3. **Speed** - Fast local processing
4. **Reliability** - Works offline
5. **Customization** - Can be trained on your data

## Future Enhancements

- Train neural networks on actual agricultural Q&A datasets
- Add CNN model for better image-based disease detection
- Expand knowledge base with more detailed information
- Add multi-language support
- Integrate with IoT sensors for real-time advice

---

**Edge AI - Empowering farmers with local, intelligent assistance! 🌾**
