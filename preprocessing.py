import cv2
from PIL import Image
import os
import numpy as np
import pandas as pd
from skimage.transform import downscale_local_mean
from skimage import filters, color
import openslide
from tiatoolbox.tools.stainnorm import MacenkoNormalizer

# Macenko normalizer
def initialize_normalizer(target_image):
    norm = MacenkoNormalizer()
    norm.fit(target_image)
    return norm

def crop(im, patch_size):
    height, width, _ = im.shape
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size
    height_crop = patch_size * n_patches_h
    width_crop = patch_size * n_patches_w
    im = im[:height_crop, :width_crop, :]
    return im, n_patches_h, n_patches_w

# Segment function
def segment(thumb):
    im_gray = color.rgb2gray(thumb)
    thres = filters.threshold_otsu(im_gray)
    mask = im_gray < thres
    return mask

# Patchify function
def patchify(im, mask, patch_size, n_patches_h, n_patches_w):
    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            if not mask[i, j]:
                continue
            start_i = i * patch_size
            end_i = start_i + patch_size
            start_j = j * patch_size
            end_j = start_j + patch_size
            patch = im[start_i:end_i, start_j:end_j, :]
            patches.append(patch)
    return np.stack(patches)


def preprocess_and_save_patches(slides_df, patch_size, num_patches, base_save_dir, normalizer, target_slides=250):
    successfully_processed_slides = 0
    total_processed = 0
    
    for slide_idx, (_, slide_info) in enumerate(slides_df.iterrows()):
        if successfully_processed_slides >= target_slides:
            print(f"Target of {target_slides} slides successfully processed. Stopping.")
            break
            
        print(f"Processing slide {slide_idx + 1}/{len(slides_df)} (Successfully processed: {successfully_processed_slides}/{target_slides})")
        slide_path = slide_info['Full Path']
        slide_filename = os.path.basename(slide_path).replace('.svs', f'_patches.npy')
        slide_filename_norm = os.path.basename(slide_path).replace('.svs', f'_patches-normalized.npy')
        slide_filename_indices = os.path.basename(slide_path).replace('.svs', f'_patches-indices.npy')
        
        slide_save_path = os.path.join(base_save_dir, slide_filename)
        slide_save_path_norm = os.path.join(base_save_dir, slide_filename_norm)
        slide_save_path_indices = os.path.join(base_save_dir, slide_filename_indices)

        if os.path.exists(slide_save_path) and os.path.exists(slide_save_path_norm):
            print(f"Skipping {slide_filename}, already processed.")
            successfully_processed_slides += 1
            total_processed += 1
            continue
        
        if not os.path.exists(slide_path):
            print(f"Warning: File not found: {slide_path}, skipping slide.")
            continue

        try:
            slide = openslide.OpenSlide(slide_path)
            region = slide.read_region((0, 0), 0, slide.level_dimensions[0])
            im = np.array(region.convert('RGB'))
            slide.close()

            im, n_patches_h, n_patches_w = crop(im, patch_size)
            thumb = downscale_local_mean(im, (patch_size, patch_size, 1))
            mask = segment(thumb)
            patches = patchify(im, mask, patch_size, n_patches_h, n_patches_w)

            patch_buffer = min(100, len(patches) - num_patches)
            patches_to_sample = min(num_patches + patch_buffer, len(patches))
            
            # Check if slide has insufficient patches (even with buffer)
            if len(patches) < num_patches:
                print(f"Warning: Only {len(patches)} patches available for slide {slide_idx + 1} (need {num_patches}), skipping slide.")
                continue

            # Randomly sample patches (more than needed)
            available_indices = np.random.choice(len(patches), patches_to_sample, replace=False)
            sampled_patches = patches[available_indices]
            
            # Try to normalize patches and collect successful ones
            successful_patches = []
            successful_indices = []
            successful_normalized_patches = []
            
            for i, (patch, original_idx) in enumerate(zip(sampled_patches, available_indices)):
                if len(successful_patches) >= num_patches:
                    break  # We have enough successful patches
                    
                try:
                    normalized_patch = normalizer.transform(patch)
                    successful_patches.append(patch)
                    successful_indices.append(original_idx)
                    successful_normalized_patches.append(normalized_patch)
                except Exception as norm_error:
                    print(f"Warning: Failed to normalize patch {i}, skipping patch: {str(norm_error)}")
                    continue
            
            # Check if we have enough successful patches
            if len(successful_patches) < num_patches:
                print(f"Warning: Only {len(successful_patches)} patches successfully normalized for slide {slide_idx + 1} (need {num_patches}), skipping slide.")
                continue

            # Take exactly the number of patches needed
            final_patches = np.stack(successful_patches[:num_patches])
            final_indices = np.array(successful_indices[:num_patches])
            final_normalized_patches = np.stack(successful_normalized_patches[:num_patches])

            # Save all data
            np.save(slide_save_path, final_patches)
            np.save(slide_save_path_indices, final_indices)
            np.save(slide_save_path_norm, final_normalized_patches)
            
            successfully_processed_slides += 1
            total_processed += 1
            print(f"Successfully processed slide {slide_idx + 1} with {num_patches} patches (Success count: {successfully_processed_slides})")

        except Exception as e:
            print(f"Error processing slide {slide_path}: {str(e)}, skipping slide.")
            continue
    
    print(f"Processing complete: {successfully_processed_slides} slides successfully processed, {total_processed - successfully_processed_slides} slides had errors")
    return successfully_processed_slides, total_processed


num_slides = 250
num_patches_per_slide = 250
patch_size = 224

metadata_path = "/tcga/open-access/gdc_data_portal/biospecimen/tcga_Biospecimen_SAMPLE_METADATA/2025-05-09/gdc_sample_sheet.2025-05-14.tsv"
metadata_df = pd.read_csv(metadata_path, sep='\t')
slides_df = metadata_df[metadata_df['Data Type'] == 'Slide Image']
slides_df = slides_df.sort_values(by='Project ID').reset_index(drop=True)
base_dir = '/tcga/open-access/gdc_data_portal/biospecimen/tcga_Biospecimen_FILES/'
slides_df['Full Path'] = slides_df.apply(lambda row: os.path.join(base_dir, row['File ID'], row['File Name']), axis=1)

target_image = np.array(Image.open('normalization_template.jpg'))
normalizer = initialize_normalizer(target_image)

for cancer_type in ['COAD', 'BRCA', 'LUSC', 'LUAD']:
    print(f"\nProcessing {cancer_type} slides...")
    all_slides = slides_df[slides_df['Project ID'] == 'TCGA-' + cancer_type]
    
    sample_size = min(num_slides + 100, len(all_slides))
    sampled_slides = all_slides.sample(n=sample_size, random_state=42)

    out_dir = f"/lotterlab/users/vmishra/RSA_updated101/preprocessed_patches_{cancer_type}/"
    os.makedirs(out_dir, exist_ok=True)
    sampled_slides.to_csv(out_dir + f'sampled_{cancer_type}_slides.csv', index=False)

    successfully_processed, total_processed = preprocess_and_save_patches(
        sampled_slides, 
        patch_size=patch_size, 
        num_patches=num_patches_per_slide, 
        base_save_dir=out_dir, 
        normalizer=normalizer,
        target_slides=num_slides
    )
    
    print(f"{cancer_type}: {successfully_processed} slides successfully processed out of {total_processed} attempted")
    
    if successfully_processed < num_slides:
        print(f"Warning: Only {successfully_processed} slides successfully processed for {cancer_type} (target was {num_slides})")
