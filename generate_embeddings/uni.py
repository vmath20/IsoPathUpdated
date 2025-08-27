from PIL import Image
import os
import numpy as np
import torch
import pandas as pd
from math import ceil
from tqdm import tqdm
from huggingface_hub import login
import timm
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

#Constants
num_slides = 250
num_patches_per_slide = 250
patch_size = 224

preprocessed_patches_dir_brca = "/lotterlab/users/vmishra/RSA_updated100/preprocessed_patches_BRCA"
preprocessed_patches_dir_luad = "/lotterlab/users/vmishra/RSA_updated100/preprocessed_patches_LUAD"
preprocessed_patches_dir_lusc = "/lotterlab/users/vmishra/RSA_updated100/preprocessed_patches_LUSC"
preprocessed_patches_dir_coad = "/lotterlab/users/vmishra/RSA_updated100/preprocessed_patches_COAD"

login(token = "YORU_HF_TOKEN")

def embed(
    patches,
    model,
    transform,
    device,
    batch_size=64,
    verbose=True,
):
    num_batches = ceil(len(patches) / batch_size)
    opt_embs = []

    for batch_idx in tqdm(range(num_batches), disable=not verbose):
        # Slice batch
        start = batch_idx * batch_size
        end = min(start + batch_size, len(patches))
        batch_np = patches[start:end]

        # Convert numpy arrays to PIL Images for transform
        batch_pil = [Image.fromarray(patch.astype('uint8')) for patch in batch_np]
        
        # Apply transform to each image
        batch_transformed = [transform(img) for img in batch_pil]
        
        # Stack transformed images
        batch = torch.stack(batch_transformed).to(device)

        # Call model
        with torch.no_grad():
            batch_emb = model(batch)

        # Copy to host and append
        opt_embs.append(batch_emb.cpu())

    # Stack to contiguous array
    opt_embs = torch.cat(opt_embs, dim=0)

    return opt_embs


# Initialize model
timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
model.to(device)
model.eval()
preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

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

def embed_patches(patches, model, transform, device, description=""):
    if len(patches) == 0:
        print(f"No patches found for {description}")
        return np.array([])
    
    print(f"Embedding {len(patches)} patches for {description}...")
    return embed(patches, model, transform, device).numpy()

brca_patches = load_patches_from_individual_files(preprocessed_patches_dir_brca, normalized=False)
brca_patches_norm = load_patches_from_individual_files(preprocessed_patches_dir_brca, normalized=True)

luad_patches = load_patches_from_individual_files(preprocessed_patches_dir_luad, normalized=False)
luad_patches_norm = load_patches_from_individual_files(preprocessed_patches_dir_luad, normalized=True)

lusc_patches = load_patches_from_individual_files(preprocessed_patches_dir_lusc, normalized=False)
lusc_patches_norm = load_patches_from_individual_files(preprocessed_patches_dir_lusc, normalized=True)

coad_patches = load_patches_from_individual_files(preprocessed_patches_dir_coad, normalized=False)
coad_patches_norm = load_patches_from_individual_files(preprocessed_patches_dir_coad, normalized=True)

brca_embeddings = embed_patches(brca_patches, model, preprocess, device, "BRCA original")
luad_embeddings = embed_patches(luad_patches, model, preprocess, device, "LUAD original")
lusc_embeddings = embed_patches(lusc_patches, model, preprocess, device, "LUSC original")
coad_embeddings = embed_patches(coad_patches, model, preprocess, device, "COAD original")

brca_embeddings_norm = embed_patches(brca_patches_norm, model, preprocess, device, "BRCA normalized")
luad_embeddings_norm = embed_patches(luad_patches_norm, model, preprocess, device, "LUAD normalized")
lusc_embeddings_norm = embed_patches(lusc_patches_norm, model, preprocess, device, "LUSC normalized")
coad_embeddings_norm = embed_patches(coad_patches_norm, model, preprocess, device, "COAD normalized")

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

np.save("/lotterlab/users/vmishra/RSA_updated100/brca_embeddings_uni_updated.npy", brca_embeddings)
np.save("/lotterlab/users/vmishra/RSA_updated100/luad_embeddings_uni_updated.npy", luad_embeddings)
np.save("/lotterlab/users/vmishra/RSA_updated100/lusc_embeddings_uni_updated.npy", lusc_embeddings)
np.save("/lotterlab/users/vmishra/RSA_updated100/coad_embeddings_uni_updated.npy", coad_embeddings)

np.save("/lotterlab/users/vmishra/RSA_updated100/brca_embeddings_uni_normalized_updated.npy", brca_embeddings_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/luad_embeddings_uni_normalized_updated.npy", luad_embeddings_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/lusc_embeddings_uni_normalized_updated.npy", lusc_embeddings_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/coad_embeddings_uni_normalized_updated.npy", coad_embeddings_norm)

np.save("/lotterlab/users/vmishra/RSA_updated100/brca_labels_uni_updated.npy", brca_labels)
np.save("/lotterlab/users/vmishra/RSA_updated100/luad_labels_uni_updated.npy", luad_labels)
np.save("/lotterlab/users/vmishra/RSA_updated100/lusc_labels_uni_updated.npy", lusc_labels)
np.save("/lotterlab/users/vmishra/RSA_updated100/coad_labels_uni_updated.npy", coad_labels)

np.save("/lotterlab/users/vmishra/RSA_updated100/brca_labels_uni_norm_updated.npy", brca_labels_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/luad_labels_uni_norm_updated.npy", luad_labels_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/lusc_labels_uni_norm_updated.npy", lusc_labels_norm)
np.save("/lotterlab/users/vmishra/RSA_updated100/coad_labels_uni_norm_updated.npy", coad_labels_norm)
