# AgriSense Data Generation

This project is designed to generate synthetic agricultural datasets and augment vision data for the AgriSense platform. The goal is to provide a comprehensive set of tools for simulating agricultural data, which can be used for machine learning and data analysis purposes.

## Project Structure

The project is organized into the following main directories:

- **src/**: Contains the source code for data generation and augmentation.
  - **generators/**: Scripts for generating synthetic data related to sensors, crops, weather, and soil.
  - **augmentation/**: Scripts for augmenting vision datasets, including image transformations.
  - **schemas/**: Data models and schemas for the generated datasets.
  - **utils/**: Utility functions and configuration settings.

- **tests/**: Contains unit tests for the data generation and augmentation scripts.

- **data/**: Directory for storing generated datasets.
  - **raw/**: For raw data files.
  - **synthetic/**: For synthetic data files.
  - **augmented/**: For augmented image files.

- **configs/**: Configuration files for data generation settings.

- **pyproject.toml**: Project metadata and dependencies.

- **requirements.txt**: List of required Python packages.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ELANGKATHIR11/AgriSense-Sustainable-Agriculture.git
cd agrisense-data-generation
pip install -r requirements.txt
```

## Usage

### Data Generation

To generate synthetic agricultural datasets, run the main script:

```bash
python src/main.py
```

This will invoke the data generation processes defined in the `src/generators` module.

### Image Augmentation

To perform image augmentation on vision datasets, use the image augmentor script:

```bash
python src/augmentation/image_augmentor.py
```

This script will apply various transformations to the images in the specified dataset.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.