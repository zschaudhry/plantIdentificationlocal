
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from open_clip import create_model, get_tokenizer
from torchvision import transforms

from PIL import Image
import streamlit as st


# --- Streamlit page config must be first ---
st.set_page_config(page_title="Open-Domain Image Classification", layout="wide")

# Set up logging (set to WARNING for performance; change to INFO for debugging)
log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=log_format)
logger = logging.getLogger()


######################################################################
# NOTE: To avoid unnecessary Streamlit cache invalidation:
# - Do NOT change the code or global variables used by the cached functions below
#   unless you want to force a reload of the model or embeddings.
# - Keep device selection and file paths outside the cached functions (as done here).
# - Avoid passing changing arguments to cached functions.
# - Avoid unnecessary code changes above the cached functions.
# - Restarting the app or changing dependencies will also clear the cache.
######################################################################
# Define constants for local files
MODEL_DIR = "models"
EMBEDDINGS_DIR = "embeddings"
MODEL_BIN = os.path.join(MODEL_DIR, "open_clip_pytorch_model.bin")
MODEL_CONFIG = os.path.join(MODEL_DIR, "open_clip_config.json")
TOKENIZER_JSON = os.path.join(MODEL_DIR, "tokenizer.json")
TOKENIZER_CONFIG = os.path.join(MODEL_DIR, "tokenizer_config.json")
VOCAB_JSON = os.path.join(MODEL_DIR, "vocab.json")
SPECIAL_TOKENS_MAP = os.path.join(MODEL_DIR, "special_tokens_map.json")
MERGES_TXT = os.path.join(MODEL_DIR, "merges.txt")
EMBEDDINGS_NPY = os.path.join(EMBEDDINGS_DIR, "txt_emb_species.npy")
EMBEDDINGS_JSON = os.path.join(EMBEDDINGS_DIR, "txt_emb_species.json")

# Device selection: Prefer MPS (Apple Silicon), then CUDA, then CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    logger.info("Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA device (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    logger.info("Using CPU device")

min_prob = 1e-9
k = 1


# Load model and tokenizer from local files


# Robust model and tokenizer loading with error handling
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = create_model(
            "ViT-L-14",  # architecture
            pretrained=MODEL_BIN,
            output_dict=True
        )
        model = model.to(device)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise
    try:
        # Use the default tokenizer for the architecture
        tokenizer = get_tokenizer("ViT-L-14")
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        raise
    return model, tokenizer

# Always show spinner while loading model/tokenizer
with st.spinner("Loading model and tokenizer... :hourglass:"):
    model, tokenizer = load_model_and_tokenizer()

# Show a message when the app is ready for use
st.success("Model and embeddings loaded! Ready for image upload and classification. :white_check_mark:")

# Define preprocess function
preprocess_img = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


# Robust embeddings loading with error handling
@st.cache_data
def load_text_embeddings():
    try:
        txt_emb = torch.from_numpy(np.load(EMBEDDINGS_NPY)).to(device)
    except Exception as e:
        st.error(f"Error loading embeddings .npy: {e}")
        raise
    try:
        with open(EMBEDDINGS_JSON, 'r') as fd:
            txt_names = json.load(fd)
    except Exception as e:
        st.error(f"Error loading embeddings .json: {e}")
        raise
    return txt_emb, txt_names

# Always show spinner while loading embeddings
with st.spinner("Loading text embeddings... :hourglass:"):
    txt_emb, txt_names = load_text_embeddings()

# Define ranks and format name function
ranks = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
def format_name(taxon):
    return " ".join(taxon)

# Define open-domain classification function
def open_domain_classification(img):
    logger.info(f"Starting open domain classification")
    img = preprocess_img(img).to(device)
    img_features = model.encode_image(img.unsqueeze(0))
    img_features = F.normalize(img_features, dim=-1)

    logits = (model.logit_scale.exp() * img_features @ txt_emb).squeeze()
    probs = F.softmax(logits, dim=0)

    topk = probs.topk(k)
    prediction = txt_names[topk.indices[0].item()][0]

    # Safely handle the Species field in case prediction has fewer than 7 elements
    species = " ".join(prediction[5:]) if len(prediction) > 6 else prediction[5]
    classification_result = {
        "Kingdom": prediction[0],
        "Phylum": prediction[1],
        "Class": prediction[2],
        "Order": prediction[3],
        "Family": prediction[4],
        "Genus": prediction[5],
        "Species": species
    }

    return classification_result





st.markdown("""
<style>
.main .block-container {padding-top: 2rem;}
.stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)


st.title(":seedling: Open-Domain Image Classification :owl:")
st.write("Upload an image of a plant or animal to classify it into taxonomic ranks using BioCLIP-2. :leaves: :tiger: :owl: :herb:")


# Sidebar for upload and info
with st.sidebar:
    st.header("Upload Image :frame_with_picture:")
    uploaded_image = st.file_uploader("Choose an image :camera:", type=["jpg", "jpeg", "png"])
    st.markdown("""
    **Instructions :memo::**
    - Upload a clear image of a plant or animal. :deciduous_tree: :dog: :bird:
    - Click **Classify** :mag: to see results.
    """)

# Main content area
if uploaded_image is not None:
    @st.cache_data(show_spinner=False)
    def load_image(image_file):
        img = Image.open(image_file)
        img = img.convert("RGB")
        img = img.resize((224, 224), resample=Image.BICUBIC)
        return img
    image = load_image(uploaded_image)
    st.image(image, caption='Uploaded Image (224x224) :framed_picture:', use_container_width=False, width=128)
    # Automatically classify and show top 5 results in tabs below the image
    with st.spinner("Classifying... :hourglass_flowing_sand:"):
        def open_domain_classification(img, k=5):
            img = preprocess_img(img).to(device)
            img_features = model.encode_image(img.unsqueeze(0))
            img_features = F.normalize(img_features, dim=-1)
            logits = (model.logit_scale.exp() * img_features @ txt_emb).squeeze()
            probs = F.softmax(logits, dim=0)
            topk = probs.topk(k)
            results = []
            for idx, (prob, idx_val) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
                prediction = txt_names[idx_val][0]
                genus = prediction[5] if len(prediction) > 5 else ""
                species_epithet = " ".join(prediction[6:]) if len(prediction) > 6 else ""
                scientific_name = f"{genus} {species_epithet}".strip()
                result = {
                    "Rank": idx + 1,
                    "Kingdom": prediction[0],
                    "Phylum": prediction[1],
                    "Class": prediction[2],
                    "Order": prediction[3],
                    "Family": prediction[4],
                    "Genus": genus,
                    "Scientific Name": scientific_name,
                    "Confidence (%)": f"{prob * 100:.2f}"
                }
                results.append(result)
            return results
        classification_results = open_domain_classification(image, k=5)
        import pandas as pd
        df = pd.DataFrame(classification_results)
        # Create selectbox above tabs with first scientific name selected
        scientific_names = df["Scientific Name"].tolist()
        selected_scientific_name = st.selectbox("Select Scientific Name:", options=scientific_names, index=0)

        # USDA query function
        import requests
        def query_invasive_species_database(scientific_name):
            url = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_InvasiveSpecies_01/MapServer/0/query"
            out_fields = [
                "NRCS_PLANT_CODE", "SCIENTIFIC_NAME", "COMMON_NAME", "PROJECT_CODE", "PLANT_STATUS",
                "FS_UNIT_NAME", "EXAMINERS", "LAST_UPDATE"
            ]
            params = {
                'where': f"SCIENTIFIC_NAME='{scientific_name}'",
                'outFields': ",".join(out_fields),
                'returnGeometry': 'true',
                'f': 'json'
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            return None

        # Query USDA for selected scientific name
        usda_data = query_invasive_species_database(selected_scientific_name)
        usda_df = None
        if usda_data and 'features' in usda_data:
            features = usda_data['features']
            grid_data = [f['attributes'] for f in features] if features else []
            import pandas as pd
            usda_df = pd.DataFrame(grid_data) if grid_data else pd.DataFrame()

        # Create five tabs in the main container, full width
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Top Predictions",
            "USDA Data",
            "Summary",
            "Map",
            "Wikipedia Info"
        ])
        with tab1:
            st.dataframe(df, use_container_width=True, hide_index=True)
        with tab2:
            if usda_df is not None and not usda_df.empty:
                invasive_df = usda_df.copy()
                import pandas as pd
                for col in invasive_df.columns:
                    col_dtype = invasive_df[col].dtype
                    if pd.api.types.is_object_dtype(col_dtype):
                        sample = invasive_df[col].dropna().astype(str).head(10)
                        if sample.str.match(r"^\d{4}-\d{2}-\d{2}T").any() or sample.str.match(r"^\d{4}-\d{2}-\d{2}$").any():
                            invasive_df[col] = pd.to_datetime(invasive_df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                        elif sample.str.match(r"^\d{12,}").any() or sample.str.match(r"^\d{10,}").any():
                            dt = pd.to_datetime(invasive_df[col], errors='coerce', unit='ms')
                            if dt.isna().all():
                                dt = pd.to_datetime(invasive_df[col], errors='coerce', unit='s')
                            invasive_df[col] = dt.dt.strftime('%Y-%m-%d')
                    elif pd.api.types.is_integer_dtype(col_dtype) or pd.api.types.is_float_dtype(col_dtype):
                        sample = invasive_df[col].dropna().astype(str).head(10)
                        if sample.str.match(r"^\d{12,}").any() or sample.str.match(r"^\d{10,}").any():
                            dt = pd.to_datetime(invasive_df[col], errors='coerce', unit='ms')
                            if dt.isna().all():
                                dt = pd.to_datetime(invasive_df[col], errors='coerce', unit='s')
                            invasive_df[col] = dt.dt.strftime('%Y-%m-%d')
                if 'LAST_UPDATE' in invasive_df.columns:
                    try:
                        invasive_df['LAST_UPDATE_sort'] = pd.to_datetime(invasive_df['LAST_UPDATE'], errors='coerce')
                        invasive_df = invasive_df.sort_values('LAST_UPDATE_sort', ascending=False).drop(columns=['LAST_UPDATE_sort'])
                    except Exception:
                        pass
                st.dataframe(invasive_df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No USDA Invasive Species data found for: {selected_scientific_name}")
        with tab3:
            if usda_df is not None and not usda_df.empty and 'FS_UNIT_NAME' in usda_df.columns:
                summary_df = usda_df.groupby('FS_UNIT_NAME').size().reset_index(name='Record Count')
                summary_df = summary_df.rename(columns={'FS_UNIT_NAME': 'FS Unit Name'})
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.info("No USDA data available to summarize by FS Unit Name.")
        with tab4:
            import pydeck as pdk
            def show_invasive_map(invasive_map_df, width=800, height=600):
                """
                Display an interactive heatmap of invasive species points using pydeck (Deck.gl).
                """
                if invasive_map_df.empty or not {'lat', 'lon'}.issubset(invasive_map_df.columns):
                    st.info("No map data available.")
                    return
                df = invasive_map_df.dropna(subset=['lat', 'lon']).copy()
                if df.empty:
                    st.info("No valid map coordinates available.")
                    return
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=df,
                    opacity=0.9,
                    radius_scale=6,
                    radius_min_pixels=1,
                    radius_max_pixels=100,
                    get_position='[lon, lat]'
                )
                view_state = pdk.ViewState(
                    latitude=df['lat'].mean(),
                    longitude=df['lon'].mean(),
                    zoom=5,
                    pitch=0,
                )
                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style='road'
                )
                st.pydeck_chart(deck, use_container_width=True)

            # Extract coordinates from USDA response
            coords = []
            if usda_data and 'features' in usda_data:
                for f in usda_data['features']:
                    geom = f.get('geometry', {})
                    # ESRI geometry can be point, multipoint, polyline, polygon, etc.
                    if 'x' in geom and 'y' in geom:
                        coords.append({'lat': geom['y'], 'lon': geom['x']})
                    elif 'points' in geom and isinstance(geom['points'], list):
                        for pt in geom['points']:
                            if len(pt) == 2:
                                coords.append({'lat': pt[1], 'lon': pt[0]})
                    elif 'paths' in geom and isinstance(geom['paths'], list):
                        for path in geom['paths']:
                            for pt in path:
                                if len(pt) == 2:
                                    coords.append({'lat': pt[1], 'lon': pt[0]})
                    elif 'rings' in geom and isinstance(geom['rings'], list):
                        for ring in geom['rings']:
                            for pt in ring:
                                if len(pt) == 2:
                                    coords.append({'lat': pt[1], 'lon': pt[0]})
            import pandas as pd
            invasive_map_df = pd.DataFrame(coords)
            show_invasive_map(invasive_map_df)
        with tab5:
            import requests
            import re
            from typing import Optional

            def _clean_wikipedia_html(html: str) -> str:
                html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
                html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
                html = re.sub(r'<sup[^>]*>.*?</sup>', '', html, flags=re.DOTALL)
                html = re.sub(r'<a [^>]+>(.*?)</a>', r'\1', html, flags=re.DOTALL)
                clean_text = re.sub('<[^<]+?>', '', html)
                clean_text = re.sub(r'\[[^\]]*\]', '', clean_text)
                clean_text = '\n'.join([line for line in clean_text.splitlines() if not line.strip().startswith('^')])
                clean_text = re.sub(r'/\*.*?\*/', '', clean_text, flags=re.DOTALL)
                clean_text = re.sub(r'\{[^\}]*\}', '', clean_text, flags=re.DOTALL)
                clean_text = '\n'.join([line for line in clean_text.splitlines() if line.strip()])
                return clean_text.strip()

            def get_wikipedia_summary(scientific_name: str) -> Optional[dict]:
                wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{scientific_name.replace(' ', '_')}"
                try:
                    resp = requests.get(wiki_url)
                    if resp.status_code == 200:
                        return resp.json()
                except Exception as e:
                    st.error(f"Wikipedia request failed: {e}")
                return None

            wiki_data = get_wikipedia_summary(selected_scientific_name)
            if wiki_data:
                title = wiki_data.get('title', selected_scientific_name)
                extract = wiki_data.get('extract', '')
                st.subheader(title)
                st.write(extract)
                if 'thumbnail' in wiki_data and 'source' in wiki_data['thumbnail']:
                    st.image(wiki_data['thumbnail']['source'], caption=title)
            else:
                st.warning(f"No Wikipedia summary found for: {selected_scientific_name}")
else:
    st.info("Please upload an image to begin classification. :arrow_up:")