def generate_soil_data(num_samples=100):
    import random
    import pandas as pd

    # Define soil properties and their ranges
    soil_types = ['Clay', 'Sandy', 'Loamy', 'Silty']
    ph_range = (4.0, 8.0)  # pH range
    moisture_range = (5.0, 30.0)  # Moisture percentage
    nutrient_levels = {
        'Nitrogen': (0.1, 1.5),  # in percentage
        'Phosphorus': (0.1, 1.0),  # in percentage
        'Potassium': (0.1, 2.0)  # in percentage
    }

    # Generate synthetic soil data
    data = []
    for _ in range(num_samples):
        soil_type = random.choice(soil_types)
        ph = round(random.uniform(*ph_range), 2)
        moisture = round(random.uniform(*moisture_range), 2)
        nutrients = {nutrient: round(random.uniform(*level_range), 2) for nutrient, level_range in nutrient_levels.items()}
        
        sample = {
            'Soil Type': soil_type,
            'pH': ph,
            'Moisture (%)': moisture,
            **nutrients
        }
        data.append(sample)

    # Create a DataFrame
    soil_data_df = pd.DataFrame(data)
    
    return soil_data_df

if __name__ == "__main__":
    # Generate and print synthetic soil data
    synthetic_soil_data = generate_soil_data(100)
    print(synthetic_soil_data)