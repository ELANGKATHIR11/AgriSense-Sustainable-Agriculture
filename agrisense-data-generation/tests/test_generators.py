import pytest
from src.generators.sensor_data import generate_sensor_data
from src.generators.crop_data import generate_agri_data
from src.generators.weather_data import generate_weather_data
from src.generators.soil_data import generate_soil_data

def test_generate_sensor_data():
    data = generate_sensor_data(num_samples=100)
    assert len(data) == 100
    assert all('temperature' in sample for sample in data)
    assert all('humidity' in sample for sample in data)

def test_generate_agri_data():
    crop_data = generate_agri_data(num_samples=50)
    assert len(crop_data) == 50
    assert all('crop_type' in sample for sample in crop_data)
    assert all('yield' in sample for sample in crop_data)

def test_generate_weather_data():
    weather_data = generate_weather_data(num_samples=30)
    assert len(weather_data) == 30
    assert all('temperature' in sample for sample in weather_data)
    assert all('precipitation' in sample for sample in weather_data)

def test_generate_soil_data():
    soil_data = generate_soil_data(num_samples=20)
    assert len(soil_data) == 20
    assert all('ph' in sample for sample in soil_data)
    assert all('moisture' in sample for sample in soil_data)