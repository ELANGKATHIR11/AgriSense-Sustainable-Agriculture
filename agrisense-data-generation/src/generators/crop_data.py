from faker import Faker
import pandas as pd
import random

fake = Faker()

def generate_crop_data(num_samples=100):
    data = {
        "crop_id": [],
        "crop_name": [],
        "growth_stage": [],
        "yield_per_hectare": [],
        "planting_date": [],
        "harvest_date": [],
        "location": [],
        "soil_type": [],
    }

    crop_names = ["Wheat", "Corn", "Rice", "Soybean", "Barley"]
    growth_stages = ["Seedling", "Vegetative", "Flowering", "Mature", "Harvest"]

    for _ in range(num_samples):
        crop_id = fake.uuid4()
        crop_name = random.choice(crop_names)
        growth_stage = random.choice(growth_stages)
        yield_per_hectare = round(random.uniform(1.0, 10.0), 2)  # Yield in tons
        planting_date = fake.date_between(start_date='-2y', end_date='today')
        harvest_date = fake.date_between(start_date=planting_date, end_date='+6m')
        location = fake.city()
        soil_type = random.choice(["Clay", "Sandy", "Loamy", "Silty"])

        data["crop_id"].append(crop_id)
        data["crop_name"].append(crop_name)
        data["growth_stage"].append(growth_stage)
        data["yield_per_hectare"].append(yield_per_hectare)
        data["planting_date"].append(planting_date)
        data["harvest_date"].append(harvest_date)
        data["location"].append(location)
        data["soil_type"].append(soil_type)

    return pd.DataFrame(data)

if __name__ == "__main__":
    crop_data_df = generate_crop_data(1000)
    crop_data_df.to_csv("data/synthetic/crop_data.csv", index=False)