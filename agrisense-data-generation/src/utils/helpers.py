def generate_random_crop_data(num_samples):
    import random
    crops = ['Wheat', 'Corn', 'Rice', 'Soybean', 'Barley']
    data = []
    
    for _ in range(num_samples):
        crop = random.choice(crops)
        yield {
            'crop_type': crop,
            'yield': round(random.uniform(1.0, 10.0), 2),  # Yield in tons per hectare
            'area': round(random.uniform(0.5, 5.0), 2),   # Area in hectares
            'fertilizer_used': round(random.uniform(50, 200), 2)  # Fertilizer in kg
        }

def generate_random_weather_data(num_samples):
    import random
    data = []
    
    for _ in range(num_samples):
        yield {
            'temperature': round(random.uniform(15.0, 35.0), 2),  # Temperature in Celsius
            'humidity': round(random.uniform(30.0, 90.0), 2),     # Humidity in percentage
            'precipitation': round(random.uniform(0.0, 100.0), 2)  # Precipitation in mm
        }

def generate_random_soil_data(num_samples):
    import random
    soil_types = ['Clay', 'Sandy', 'Loamy', 'Silty']
    data = []
    
    for _ in range(num_samples):
        yield {
            'soil_type': random.choice(soil_types),
            'ph_level': round(random.uniform(5.0, 8.0), 2),  # pH level
            'organic_matter': round(random.uniform(1.0, 10.0), 2)  # Organic matter in percentage
        }

def generate_random_sensor_data(num_samples):
    import random
    data = []
    
    for _ in range(num_samples):
        yield {
            'sensor_id': f'SENSOR_{random.randint(1000, 9999)}',
            'temperature': round(random.uniform(10.0, 40.0), 2),  # Temperature in Celsius
            'humidity': round(random.uniform(20.0, 100.0), 2),     # Humidity in percentage
            'moisture': round(random.uniform(0.0, 100.0), 2)       # Soil moisture in percentage
        }

def save_to_csv(data, filename):
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def load_config(config_file):
    import yaml
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)