

import os
import torch
import streamlit as st
import pandas as pd

import pandas as pd
# Import refactored logic modules
from utils.model import load_model_and_tokenizer, load_text_embeddings, open_domain_classification
from utils.image import load_image
from utils.usda import query_invasive_species_database, format_usda_dates, extract_usda_coordinates
from utils.wikipedia import get_wikipedia_summary


# --- Streamlit page config must be first ---
st.set_page_config(page_title="Open-Domain Image Classification", layout="wide")

# Set up logging (set to WARNING for performance; change to INFO for debugging)
 

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
VOCAB_JSON = os.path.join(MODEL_DIR, "vocab.json")
SPECIAL_TOKENS_MAP = os.path.join(MODEL_DIR, "special_tokens_map.json")
EMBEDDINGS_NPY = os.path.join(EMBEDDINGS_DIR, "txt_emb_species.npy")
EMBEDDINGS_JSON = os.path.join(EMBEDDINGS_DIR, "txt_emb_species.json")
# Device selection: Prefer MPS (Apple Silicon), then CUDA, then CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

min_prob = 1e-9
k = 1


# Load model and tokenizer from local files



# Load model/tokenizer and embeddings using refactored logic
@st.cache_resource
def get_model_and_tokenizer():
    return load_model_and_tokenizer(MODEL_BIN, device)

with st.spinner("Loading model and tokenizer... :hourglass:"):
    model, tokenizer = get_model_and_tokenizer()

@st.cache_data
def get_text_embeddings():
    return load_text_embeddings(EMBEDDINGS_NPY, EMBEDDINGS_JSON, device)

with st.spinner("Loading text embeddings... :hourglass:"):
    txt_emb, txt_names = get_text_embeddings()

st.success("Model and embeddings loaded! Ready for image upload and classification. :white_check_mark:")

# Define ranks and format name function
ranks = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
def format_name(taxon):
    return " ".join(taxon)

# Define open-domain classification function






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
    def cached_load_image(image_file):
        return load_image(image_file)
    image = cached_load_image(uploaded_image)
    st.image(image, caption='Uploaded Image (224x224) :framed_picture:', use_container_width=False, width=128)
    # Automatically classify and show top 5 results in tabs below the image
    with st.spinner("Classifying... :hourglass_flowing_sand:"):
        classification_results = open_domain_classification(
            image, model, txt_emb, txt_names, device, k=5
        )
        df = pd.DataFrame(classification_results)
        # Create selectbox above tabs with first scientific name selected
        scientific_names = df["Scientific Name"].tolist()
        selected_scientific_name = st.selectbox("Select Scientific Name:", options=scientific_names, index=0)

        # Query USDA for selected scientific name
        usda_data = query_invasive_species_database(selected_scientific_name)
        usda_df = None
        if usda_data and 'features' in usda_data:
            features = usda_data['features']
            grid_data = [f['attributes'] for f in features] if features else []
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
                invasive_df = format_usda_dates(usda_df)
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
            invasive_map_df = extract_usda_coordinates(usda_data)
            def show_invasive_map(invasive_map_df, width=800, height=600):
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
            show_invasive_map(invasive_map_df)
        with tab5:
            wiki_data = get_wikipedia_summary(selected_scientific_name)
            if wiki_data:
                title = wiki_data.get('title', selected_scientific_name)
                extract = wiki_data.get('extract', '')
                st.subheader(title)
                st.write(extract)
                if 'thumbnail' in wiki_data and 'source' in wiki_data['thumbnail']:
                    st.image(wiki_data['thumbnail']['source'], caption=title)
                if 'page_url' in wiki_data and wiki_data['page_url']:
                    st.markdown(f"[View on Wikipedia]({wiki_data['page_url']})")
            else:
                st.warning(f"No Wikipedia summary found for: {selected_scientific_name}")
else:
    st.info("Please upload an image to begin classification. :arrow_up:")