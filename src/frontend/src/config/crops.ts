// Comprehensive list of 104 supported crops matching Backend VLM database
export const ALL_CROPS = [
    // Cereals & Millets
    "rice", "wheat", "maize", "bajra", "jowar", "ragi", "barley", "oats", 
    "pearl_millet", "foxtail_millet", "kodo_millet", "little_millet", 
    "proso_millet", "barnyard_millet", "sorghum", "buck_wheat", "buckwheat",

    // Pulses & Legumes
    "chickpea", "pigeon_pea", "moong", "urad", "masoor", "arhar", 
    "kidney_bean", "horse_gram", "moth_bean", "field_pea", "lentil", 
    "green_pea", "french_bean", "cluster_bean", "beans", "peas",

    // Vegetables
    "potato", "tomato", "onion", "cabbage", "cauliflower", "brinjal", 
    "chilli", "okra", "carrot", "radish", "pumpkin", "bottle_gourd", 
    "bitter_gourd", "ridge_gourd", "cucumber", "spinach", "beetroot", 
    "turnip", "lettuce", "sweet_potato", "broccoli",

    // Fruits
    "mango", "banana", "papaya", "guava", "apple", "grapes", "orange", 
    "pomegranate", "sapota", "pineapple", "litchi", "jackfruit", 
    "watermelon", "muskmelon", "strawberry", "custard_apple", 
    "dragon_fruit", "mandarin_orange", "passion_fruit", "raspberry",

    // Spices
    "ginger", "turmeric", "garlic", "coriander", "cumin", "fenugreek", 
    "black_pepper", "cardamom", "large_cardamom",

    // Oilseeds
    "groundnut", "soybean", "mustard", "sunflower", "safflower", 
    "sesame", "linseed", "niger", "castor",

    // Cash Crops & Plantation
    "sugarcane", "cotton", "tobacco", "jute", "tea", "coffee", 
    "rubber", "coconut", "arecanut",

    // Nuts
    "cashew", "almond", "walnut"
];

// Helper to check if a crop is supported
export const isSupportedCrop = (crop: string) => ALL_CROPS.includes(crop.toLowerCase());
