# AgriDoctor AI 🌱

## Bilingual Multi-Crop Plant Disease Detection and Smart Farming Advisory System

AgriDoctor AI identifies potential plant diseases from leaf images and provides agricultural guidance in English and Tamil.

## Supported Crop Groups

| Crop group | Classes |
|---|---:|
| PlantVillage (Pepper, Potato, Tomato) | 15 |
| Rice | 4 |
| Sugarcane | 11 |
| Total | 30 |

## Key Features
- Multi-crop disease classification
- Three MobileNetV2 deep-learning models
- Image quality checks
- Prediction uncertainty checks
- English and Tamil support
- Symptoms, causes, prevention, medicine guidance and recovery information
- Interactive Streamlit application

## Technology Stack
- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit

## Installation
1. Clone this repository.
2. Install dependencies using: pip install -r requirements.txt
3. Run: streamlit run app.py

## Model Evaluation
- PlantVillage accuracy: 92.73%
- Rice accuracy: 99.10%
- Sugarcane accuracy: 91.12%

These are held-out dataset results, not guaranteed real-world field accuracy.

## Important Limitations
- Predictions are AI-assisted candidates, not confirmed diagnoses.
- Field photographs may produce different results.
- Consult local agricultural experts before treatment decisions.
- Fertilizer and pesticide requirements depend on local conditions.

## Languages
- English
- Tamil

## Project Status
Core model development and Streamlit interface completed. Testing and public deployment are in progress.

## Author
Developed as an AI and Machine Learning portfolio project.
