# Representational Similarity Analysis of Computational Pathology Models
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/vmath20/IsoPathUpdated)

This repository contains the code and analysis pipeline for comparing the feature representations of various pretrained computational pathology models. The primary method used is Representational Similarity Analysis (RSA), applied to whole-slide image (WSI) patches from The Cancer Genome Atlas (TCGA). The analysis covers four cancer types: Breast Cancer (BRCA), Colon Adenocarcinoma (COAD), Lung Adenocarcinoma (LUAD), and Lung Squamous Cell Carcinoma (LUSC).

## Project Overview

The project follows a three-stage pipeline:

1.  **Preprocessing**: WSIs from the TCGA database are processed. This involves tissue segmentation, generation of 224x224 pixel patches, and Macenko stain normalization.
2.  **Embedding Generation**: A suite of seven different pretrained models is used to generate feature embeddings for the preprocessed image patches.
3.  **Analysis**: The generated embeddings are analyzed to understand and compare their representational geometries. This includes:
    *   Calculating Representational Dissimilarity Matrices (RDMs).
    *   Measuring the similarity between model RDMs using Spearman correlation and Cosine similarity.
    *   Evaluating slide-level and disease-level specificity using Cliff's Delta.
    *   Performing spectral analysis (SVD) to assess the effective dimensionality of the embeddings.

## Models Analyzed

The following seven pretrained models are evaluated in this study:

*   **UNI2**: [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)
*   **Virchow2**: [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)
*   **Prov-Gigapath**: [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath)
*   **CONCH**: [MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch)
*   **PLIP**: [vinid/plip](https://huggingface.co/vinid/plip)
*   **KEEP**: [Astaxanthin/KEEP](https://huggingface.co/Astaxanthin/KEEP)
*   **ViT-DinoV2**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

## Repository Structure

*   `preprocessing.py`: Script to process raw TCGA WSIs, perform stain normalization, and extract patches.
*   `run_analysis.ipynb`: A Jupyter Notebook for conducting the full analysis pipeline on the generated embeddings. It calculates RDMs, specificity scores, and generates all figures.
*   `generate_embeddings/`: Contains scripts to generate feature embeddings from image patches using the different models.
    *   `generate_embeddings.py`: The main script to generate embeddings for all models. It accepts the model name and GPU index as command-line arguments.
    *   Individual model scripts (e.g., `uni.py`, `conch.ipynb`): Development scripts for generating embeddings for specific models.
*   `constants.py`: Defines the base directory for saving project data. **Note:** All paths are hardcoded and must be modified for your local environment.

## Getting Started

### Prerequisites

You will need Python 3.x and the following libraries. You can install them using `pip`:

```bash
pip install torch torchvision pandas numpy scikit-image openslide-python tiatoolbox rsatoolbox seaborn timm transformers huggingface_hub cliffs_delta
```

You will also need to have access to the TCGA slide images (`.svs` files).

### Instructions

#### 1. Configure Paths

First, update the hardcoded paths in `constants.py` and other scripts to match your local environment. Specifically, modify `PROJECT_SAVE_DIR` in `constants.py` to your desired output directory.

```python
# In constants.py
PROJECT_SAVE_DIR = '/path/to/your/project/directory/'
```

You will also need to update data paths within `preprocessing.py` and `run_analysis.ipynb`.

#### 2. Preprocess Data

The `preprocessing.py` script handles the extraction and normalization of patches from WSIs.

1.  Place a reference image for stain normalization named `normalization_template.jpg` in the root directory.
2.  Update the path to the TCGA metadata file (`metadata_path`) and the base directory for slide images (`base_dir`) in `preprocessing.py`.
3.  Run the script:

```bash
python preprocessing.py
```

This will create subdirectories (e.g., `preprocessed_patches_BRCA/`) containing the extracted patches as `.npy` files.

#### 3. Generate Embeddings

Embeddings are generated using the `generate_embeddings/generate_embeddings.py` script. This script requires a model name and a GPU index as command-line arguments. You will also need to add your Hugging Face access token in the script.

```bash
# Example for generating embeddings with UNI2 on GPU 0
python generate_embeddings/generate_embeddings.py uni2 0

# Example for generating embeddings with Virchow2 on GPU 1
python generate_embeddings/generate_embeddings.py virchow2 1
```

Repeat this command for all models listed in the `models` array in `run_analysis.ipynb`. The script will save the embeddings in the `embeddings/` directory specified in your configuration.

#### 4. Run Analysis

Once all embeddings are generated, open and run the `run_analysis.ipynb` notebook. This notebook will:

1.  Load the embeddings and split them into batches.
2.  Calculate RDMs for each model and batch.
3.  Generate heatmaps comparing model RDMs (Spearman correlation and Cosine similarity).
4.  Create hierarchical clustering dendrograms.
5.  Calculate and save slide and disease specificity scores.
6.  Perform and plot the spectral analysis.

All plots and results will be saved in the `plots/` and `rdms/` directories.

#### Customizing Distance Functions and Batch Sizes

To test other distance functions change the following snippet code:

```python
rdm = calc_rdm(dataset, method='euclidean')
```

For instance, using Pearson correlation instead of euclidean distance would be changing that line to:

```python
rdm = calc_rdm(dataset, method='correlation')
```

To change number of slides and patches per slide, change the following code snippet:

```python
num_slides_per_batch = total_slides // n_batches
num_patches_per_batch = total_patches // n_batches
```

to:

```python
num_slides_per_batch = 25
num_patches_per_batch = 100 # using 25 WSIs and 100 patches per WSI for instance.
```
