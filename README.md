# Open-Domain Image Classification App

This Streamlit app uses BioCLIP-2 for open-domain image classification of plants and animals. It integrates USDA Invasive Species data and Wikipedia summaries for selected species.

## Features
- Local model loading (BioCLIP-2)
- Automatic image classification (top-5 predictions)
- USDA Invasive Species API integration
- Data grouping and summary by FS Unit Name
- Interactive map visualization (pydeck)
- Wikipedia info tab for selected species
- Robust error handling and user-friendly UI

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download BioCLIP-2 model and embeddings, place in `models/` and `embeddings/` folders.
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Upload an image and explore results in the tabs.

## Requirements
See `requirements.txt` for Python package dependencies.

## File Structure
- `app.py`: Main Streamlit app
- `models/`: BioCLIP-2 model files
- `embeddings/`: Text embeddings files
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Notes
- USDA API and Wikipedia require internet access.
- For Apple Silicon, MPS device is used if available.
- All processing is local except for API queries.
