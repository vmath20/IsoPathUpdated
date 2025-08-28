from PIL import Image
import os
import numpy as np
import torch
import pandas as pd
from math import ceil
from tqdm import tqdm
from huggingface_hub import login
from transformers import CLIPProcessor, CLIPModel

#Constants
num_slides = 250
num_patches_per_slide = 250
patch_size = 224

preprocessed_patches_dir_brca = "/lotterlab/users/vmishra/RSA_updated100/preprocessed_patches_BRCA"
preprocessed_patches_dir_luad = "/lotterlab/users/vmishra/RSA_updated100/preprocessed_patches_LUAD"
preprocessed_patches_dir_lusc = "/lotterlab/users/vmishra/RSA_updated100/preprocessed_patches_LUSC"
preprocessed_patches_dir_coad = "/lotterlab/users/vmishra/RSA_updated100/preprocessed_patches_COAD"

login(token = "YOUR_HF_TOKEN")

def embed(patches, model, processor, device, batch_size=64, verbose=True):
    num_batches = ceil(len(patches) / batch_size)
    opt_embs = []

    for batch_idx in tqdm(range(num_batches), disable=not verbose):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(patches))
        batch_np = patches[start:end]

        # Convert numpy arrays to PIL Images for processing
        batch_pil = [Image.fromarray(patch.astype('uint8')) for patch in batch_np]

        batch_transformed = torch.cat([processor(images=img, return_tensors="pt")["pixel_values"] for img in batch_pil])
        
        # Move batch to device
        batch = batch_transformed.to(device)

        # Call model
        model = model.to(device)

        with torch.no_grad():
            batch_emb = model.get_image_features(batch)

        # Copy to host and append
        opt_embs.append(batch_emb.cpu())

    # Stack to contiguous array
    opt_embs = torch.cat(opt_embs, dim=0)

    return opt_embs.numpy()

model = CLIPModel.from_pretrained("vinid/plip")
preprocess = CLIPProcessor.from_pretrained("vinid/plip")
model.eval()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def load_patches_from_individual_files(patches_dir, normalized=False):
    patches_list = []
    
    if not os.path.exists(patches_dir):
        print(f"Directory not found: {patches_dir}")
        return np.array([])
    
    if normalized:
        pattern = "_patches-normalized.npy"
    else:
        pattern = "_patches.npy"
    
    filenames = [f for f in os.listdir(patches_dir) if f.endswith(pattern)]
    
    if not filenames:
        print(f"No files found matching pattern '{pattern}' in {patches_dir}")
        return np.array([])
    
    print(f"Found {len(filenames)} patch files in {patches_dir}")
    
    for filename in tqdm(filenames, desc=f"Loading {'normalized' if normalized else 'original'} patches"):
        try:
            patches = np.load(os.path.join(patches_dir, filename))
            patches_list.append(patches)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    if patches_list:
        all_patches = np.concatenate(patches_list, axis=0)
        print(f"Total patches loaded: {len(all_patches)}")
        return all_patches
    else:
        print("No patches could be loaded")
        return np.array([])

def embed_patches(patches, model, transform, device):
    if len(patches) == 0:
        return np.array([])
    model = model.to(device)
    return embed(patches, model, transform, device)

brca_patches = load_patches_from_individual_files(preprocessed_patches_dir_brca, normalized=False)
brca_patches_norm = load_patches_from_individual_files(preprocessed_patches_dir_brca, normalized=True)

luad_patches = load_patches_from_individual_files(preprocessed_patches_dir_luad, normalized=False)
luad_patches_norm = load_patches_from_individual_files(preprocessed_patches_dir_luad, normalized=True)

lusc_patches = load_patches_from_individual_files(preprocessed_patches_dir_lusc, normalized=False)
lusc_patches_norm = load_patches_from_individual_files(preprocessed_patches_dir_lusc, normalized=True)

coad_patches = load_patches_from_individual_files(preprocessed_patches_dir_coad, normalized=False)
coad_patches_norm = load_patches_from_individual_files(preprocessed_patches_dir_coad, normalized=True)

brca_embeddings = embed_patches(brca_patches, model, preprocess, device)
luad_embeddings = embed_patches(luad_patches, model, preprocess, device)
lusc_embeddings = embed_patches(lusc_patches, model, preprocess, device)
coad_embeddings = embed_patches(coad_patches, model, preprocess, device)

brca_embeddings_norm = embed_patches(brca_patches_norm, model, preprocess, device)
luad_embeddings_norm = embed_patches(luad_patches_norm, model, preprocess, device)
lusc_embeddings_norm = embed_patches(lusc_patches_norm, model, preprocess, device)
coad_embeddings_norm = embed_patches(coad_patches_norm, model, preprocess, device)

num_brca = len(brca_embeddings)
num_luad = len(luad_embeddings)
num_lusc = len(lusc_embeddings)
num_coad = len(coad_embeddings)

num_brca_norm = len(brca_embeddings_norm)
num_luad_norm = len(luad_embeddings_norm)
num_lusc_norm = len(lusc_embeddings_norm)
num_coad_norm = len(coad_embeddings_norm)

brca_labels = [f"BRCA_{i+1}" for i in range(num_brca)]
luad_labels = [f"LUAD_{i+1}" for i in range(num_luad)]
lusc_labels = [f"LUSC_{i+1}" for i in range(num_lusc)]
coad_labels = [f"COAD_{i+1}" for i in range(num_coad)]

brca_labels_norm = [f"BRCA_norm_{i+1}" for i in range(num_brca_norm)]
luad_labels_norm = [f"LUAD_norm_{i+1}" for i in range(num_luad_norm)]
lusc_labels_norm = [f"LUSC_norm_{i+1}" for i in range(num_lusc_norm)]
coad_labels_norm = [f"COAD_norm_{i+1}" for i in range(num_coad_norm)]

np.save("/lotterlab/users/vmishra/RSA_updated100/brca_embeddings_plip_updated.npy", brca_embeddings)
np.save("/lotterlab/users/vmishra/RSA_updated100/luad_embeddings_plip_updated.npy", luad_embeddings)
np.save("/lotterlab/users/vmishra/RSA_updated100/lusc_embeddings_plip_updated.npy", lusc_embeddings)
np.save("/lotterlab/users/vmishra/RSA_updated100/coad_embeddings_plip_updated.npy", coad_embeddings)

np.save("/lotterlab/users/vmishra/RSA_updated100/brca_embeddings_plip_normalized_updated.npy", brca_embeddings_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/luad_embeddings_plip_normalized_updated.npy", luad_embeddings_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/lusc_embeddings_plip_normalized_updated.npy", lusc_embeddings_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/coad_embeddings_plip_normalized_updated.npy", coad_embeddings_norm)

np.save("/lotterlab/users/vmishra/RSA_updated100/brca_labels_plip_updated.npy", brca_labels)
np.save("/lotterlab/users/vmishra/RSA_updated100/luad_labels_plip_updated.npy", luad_labels)
np.save("/lotterlab/users/vmishra/RSA_updated100/lusc_labels_plip_updated.npy", lusc_labels)
np.save("/lotterlab/users/vmishra/RSA_updated100/coad_labels_plip_updated.npy", coad_labels)

np.save("/lotterlab/users/vmishra/RSA_updated100/brca_labels_plip_norm_updated.npy", brca_labels_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/luad_labels_plip_norm_updated.npy", luad_labels_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/lusc_labels_plip_norm_updated.npy", lusc_labels_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/coad_labels_plip_norm_updated.npy", coad_labels_norm)
