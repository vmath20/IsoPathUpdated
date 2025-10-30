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
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoModel, CLIPProcessor, CLIPModel
import sys
from torchvision import transforms
import torch
from conch.open_clip_custom import create_model_from_pretrained

from constants import PROJECT_SAVE_DIR


def embed(
        patches,
        model,
        model_name,
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

        if model_name == 'virchow2':
            # Call model with Virchow2-specific processing
            with torch.no_grad():
                output = model(batch)
                cls_token = output[:, 0]
                patch_tokens = output[:, 5:]
                pooled = torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)

            # Copy to host and append
            opt_embs.append(pooled.cpu())
        elif model_name == 'musk':
            with torch.inference_mode():
                batch_emb = model(
                    image=batch,
                    with_head=False,
                    out_norm=False,
                    ms_aug=True,
                    return_global=True
                )[0]
            opt_embs.append(batch_emb.cpu())

        elif model_name in ['conch', 'keep']:
            with torch.no_grad():
                batch_emb = model.encode_image(batch)

            # Move to CPU and append
            opt_embs.append(batch_emb.cpu())

        elif model_name in ['uni2', 'prov', 'dinov2']:
            with torch.no_grad():
                batch_emb = model(batch)

            # Copy to host and append
            opt_embs.append(batch_emb.cpu())

        elif model_name == 'plip':
            with torch.no_grad():
                inputs = transform(images=batch_pil, return_tensors="pt").to(device)
                batch_emb = model.get_image_features(batch)            

            opt_embs.append(batch_emb.cpu())


    # Stack to contiguous array
    opt_embs = torch.cat(opt_embs, dim=0)

    return opt_embs


def load_model(model_name, gpu_num):
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

    if model_name == 'virchow2':
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )

        preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    elif model_name == 'uni2':
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)

        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    elif model_name == 'prov':
        model = create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == 'keep':
        model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)

        # Define transforms for KEEP
        preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    elif model_name == 'conch':
        model, preprocess = create_model_from_pretrained(
            'conch_ViT-B-16',
            "hf_hub:MahmoodLab/conch",
            hf_auth_token="YOUR_HF_TOKEN"
        )

    elif model_name == 'plip':
        model = CLIPModel.from_pretrained("vinid/plip")
        preprocess = CLIPProcessor.from_pretrained("vinid/plip")

    elif model_name == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    else:
        raise ValueError()

    model = model.eval()
    model.to(device)

    return model, device, preprocess


if __name__ == '__main__':
    model_name = sys.argv[1] # model name read in as first argument
    gpu_num = sys.argv[2] # gpu num read in as second

    num_slides = 250
    num_patches_per_slide = 250
    patch_size = 224

    login(token="YOUR_HF_TOKEN")
    model, device, preprocess = load_model(model_name, gpu_num)
    os.makedirs(f"{PROJECT_SAVE_DIR}/embeddings", exist_ok=True)
    
    for cancer_type in ['COAD', 'BRCA', 'LUSC', 'LUAD']:
        for norm_tag in ['', '-normalized']:
            print(f'Embedding {cancer_type}{norm_tag} with model {model_name} on gpu{gpu_num}')

            patch_tag = f'_patches{norm_tag}.npy'
            patch_dir = f"{PROJECT_SAVE_DIR}/preprocessed_patches_{cancer_type}/"
            patch_files = [f for f in os.listdir(patch_dir) if f.endswith(patch_tag)]

            assert len(patch_files) == num_slides

            patch_files = np.sort(patch_files) # force consistent order

            all_patches = []
            for f in patch_files:
                file_patches = np.load(os.path.join(patch_dir, f))
                assert file_patches.shape[0] == num_patches_per_slide
                all_patches.append(file_patches)

            all_patches = np.concatenate(all_patches, axis=0)

            embeddings = embed(all_patches, model, model_name, preprocess, device).numpy()

            embeddings_path = f"{PROJECT_SAVE_DIR}/embeddings/embeddings_{cancer_type}{norm_tag}-{model_name}.npy"
            np.save(embeddings_path, embeddings)

            embeddings_order_path = f"{PROJECT_SAVE_DIR}/embeddings/embeddings_{cancer_type}{norm_tag}-file_order-{model_name}.npy"
            np.save(embeddings_order_path, patch_files)
