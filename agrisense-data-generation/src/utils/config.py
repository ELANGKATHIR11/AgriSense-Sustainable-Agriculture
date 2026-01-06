import os

class Config:
    """Configuration settings for the data generation process."""
    
    # Base directory for data generation
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths for synthetic data output
    SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')
    AUGMENTED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'augmented')
    
    # Configuration for data generation
    GENERATION_CONFIG = {
        'sensor_data': {
            'num_samples': 1000,
            'fields': ['temperature', 'humidity', 'moisture'],
        },
        'crop_data': {
            'num_samples': 500,
            'crops': ['wheat', 'corn', 'soybean'],
        },
        'weather_data': {
            'num_samples': 365,
            'parameters': ['temperature', 'precipitation', 'humidity'],
        },
        'soil_data': {
            'num_samples': 300,
            'parameters': ['pH', 'nitrogen', 'phosphorus', 'potassium'],
        },
    }