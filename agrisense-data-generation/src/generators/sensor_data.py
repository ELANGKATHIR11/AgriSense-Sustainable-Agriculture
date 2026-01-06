def generate_sensor_data(num_samples=100):
    import random
    import pandas as pd
    from datetime import datetime, timedelta

    # Generate synthetic sensor data
    data = {
        "timestamp": [],
        "device_id": [],
        "temperature": [],
        "humidity": [],
        "soil_moisture": [],
        "light_intensity": []
    }

    start_time = datetime.now()

    for i in range(num_samples):
        data["timestamp"].append(start_time + timedelta(minutes=i))
        data["device_id"].append(f"device_{random.randint(1, 10)}")
        data["temperature"].append(round(random.uniform(15.0, 35.0), 2))  # Temperature in Celsius
        data["humidity"].append(round(random.uniform(30.0, 90.0), 2))      # Humidity in percentage
        data["soil_moisture"].append(round(random.uniform(0.0, 100.0), 2)) # Soil moisture in percentage
        data["light_intensity"].append(round(random.uniform(0.0, 1000.0), 2)) # Light intensity in lux

    # Create a DataFrame
    df = pd.DataFrame(data)

    return df

def save_sensor_data_to_csv(df, file_path):
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    num_samples = 100  # Number of samples to generate
    sensor_data = generate_sensor_data(num_samples)
    save_sensor_data_to_csv(sensor_data, 'data/synthetic/sensor_data.csv')