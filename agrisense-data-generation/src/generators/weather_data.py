import random
import pandas as pd
from datetime import datetime, timedelta

def generate_weather_data(num_records: int, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate synthetic weather data for agricultural conditions.

    Parameters:
    - num_records (int): Number of records to generate.
    - start_date (str): Start date for the data generation in 'YYYY-MM-DD' format.
    - end_date (str): End date for the data generation in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: A DataFrame containing synthetic weather data.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    weather_data = []

    for _ in range(num_records):
        date = random.choice(date_range)
        temperature = round(random.uniform(-5, 40), 2)  # Temperature in Celsius
        humidity = round(random.uniform(0, 100), 2)      # Humidity in percentage
        precipitation = round(random.uniform(0, 50), 2)   # Precipitation in mm
        wind_speed = round(random.uniform(0, 15), 2)      # Wind speed in m/s

        weather_data.append({
            'date': date,
            'temperature': temperature,
            'humidity': humidity,
            'precipitation': precipitation,
            'wind_speed': wind_speed
        })

    return pd.DataFrame(weather_data)

def save_weather_data_to_csv(data: pd.DataFrame, file_path: str):
    """
    Save the generated weather data to a CSV file.

    Parameters:
    - data (pd.DataFrame): The weather data to save.
    - file_path (str): The path where the CSV file will be saved.
    """
    data.to_csv(file_path, index=False)