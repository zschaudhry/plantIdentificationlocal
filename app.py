
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
col1, col2 = st.columns([1, 2])


with col1:
    if uploaded_image is not None:
        @st.cache_data(show_spinner=False)
        def load_image(image_file):
            img = Image.open(image_file)
            # Resize to model input size (224x224) immediately for both display and classification
            img = img.convert("RGB")
            img = img.resize((224, 224), resample=Image.BICUBIC)
            return img
        image = load_image(uploaded_image)
        st.image(image, caption='Uploaded Image (224x224) :framed_picture:', use_container_width=False, width=128)


with col2:
    if uploaded_image is not None:
        classify_btn = st.button(":mag: Classify :rocket:", use_container_width=True)
        if classify_btn:
            with st.spinner("Classifying... :hourglass_flowing_sand:"):
                classification_result = open_domain_classification(image)
            st.success("Classification Results :trophy: :sparkles:")
            st.table({rank + " :label:": classification for rank, classification in classification_result.items()})
    else:
        st.info("Please upload an image to begin classification. :arrow_up:")