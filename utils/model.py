import torch
import torch.nn.functional as F
from open_clip import create_model, get_tokenizer
from torchvision import transforms
import numpy as np
import json
import os
from PIL import Image

def load_model_and_tokenizer(model_bin, device):
    model = create_model(
        "ViT-L-14",
        pretrained=model_bin,
        output_dict=True
    )
    model = model.to(device)
    tokenizer = get_tokenizer("ViT-L-14")
    return model, tokenizer

def load_text_embeddings(embeddings_npy, embeddings_json, device):
    txt_emb = torch.from_numpy(np.load(embeddings_npy)).to(device)
    with open(embeddings_json, 'r') as fd:
        txt_names = json.load(fd)
    return txt_emb, txt_names

def preprocess_image():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

def open_domain_classification(img, model, txt_emb, txt_names, device, k=5):
    preprocess_img = preprocess_image()
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
