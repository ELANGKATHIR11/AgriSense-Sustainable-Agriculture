def main():
    print("Starting the AgriSense Data Generation and Augmentation Process...")
    
    # Import necessary modules
    from generators.sensor_data import generate_sensor_data
    from generators.crop_data import generate_agri_data
    from generators.weather_data import generate_weather_data
    from generators.soil_data import generate_soil_data
    from augmentation.image_augmentor import augment_vision_data
    
    # Generate synthetic datasets
    generate_sensor_data()
    generate_agri_data()
    generate_weather_data()
    generate_soil_data()
    
    # Perform image augmentation
    augment_vision_data()
    
    print("Data generation and augmentation completed successfully.")

if __name__ == "__main__":
    main()