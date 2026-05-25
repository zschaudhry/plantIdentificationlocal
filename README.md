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

### 1. Set up a virtual environment (Recommended)
To prevent package conflicts with other global tools:
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup models
Download the BioCLIP-2 model and embeddings, and place them in the `models/` and `embeddings/` folders.

### 4. Run the app
```bash
streamlit run app.py
```

## Requirements
See `requirements.txt` for Python package dependencies.

## File Structure
- `app.py`: Main Streamlit app
- `models/`: BioCLIP-2 model files
- `embeddings/`: Text embeddings files
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Notes
- **API Caching:** USDA Invasive Species and Wikipedia API results are cached using `@st.cache_data` to ensure fluid tab navigation and prevent redundant network requests.
- **Taxonomic Rank State:** Interactive dropdown selection state is managed via Streamlit session state binding to avoid selection resets when switching views.
- **Hardware Acceleration:** For Apple Silicon (M1/M2/M3), the MPS device is automatically selected if available.
- **API Access:** USDA API queries and Wikipedia summaries require active internet access.

