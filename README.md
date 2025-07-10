# Open-Domain Image Classification with BioCLIP-2

This Streamlit app performs open-domain image classification (plants and animals) using the BioCLIP-2 model and local embeddings.

## Features
- Upload an image of a plant or animal and classify it into taxonomic ranks.
- Uses local model weights and embeddings for fast, offline inference.
- Modern, responsive UI with thumbnail preview and error handling.

## Folder Structure
```
plantIdentificationlocal/
├── app.py
├── requirements.txt
├── .gitignore
├── models/
│   ├── open_clip_pytorch_model.bin
│   ├── open_clip_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── special_tokens_map.json
│   └── merges.txt
├── embeddings/
│   ├── txt_emb_species.npy
│   └── txt_emb_species.json
```

## Setup
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Download model and embedding files:**
   - Place all model files in `models/` and embedding files in `embeddings/` as shown above.
3. **Run the app:**
   ```sh
   streamlit run app.py
   ```
   If you see a watcher error, run:
   ```sh
   STREAMLIT_WATCHER_TYPE=none streamlit run app.py
   ```

## Notes
- Requires Python 3.10+ (3.12+ may have Streamlit/PyTorch watcher issues).
- All inference is local; no files are uploaded to the cloud.
- For best performance, use a machine with a GPU (CUDA).

## Credits
- Model: [BioCLIP-2](https://huggingface.co/imageomics/bioclip-2)
- Embeddings: [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
