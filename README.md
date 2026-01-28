# Lightweight Alignment of Vision-Language Representations via Embedding Denoising

This project explores a diffusion-based approach for aligning vision-language embeddings in low-data regimes by denoising CLIP image representations using text-conditioned latent diffusion.

## Overview

Vision-Language Models (VLMs) such as CLIP achieve strong image–text alignment through large-scale contrastive training, but their performance degrades significantly when training data is limited. This project proposes a **lightweight diffusion-inspired alignment mechanism** that operates directly in CLIP’s latent embedding space.

A compact **1D U-Net denoiser (3.67M parameters)** is trained on frozen CLIP ViT-B/32 embeddings. Noised image embeddings are reconstructed using corresponding text embeddings as cross-modal conditioning. A hybrid objective combining **MSE reconstruction loss** and **contrastive alignment loss** ensures both denoising fidelity and semantic consistency.

## Key Features

* Diffusion-based denoising in embedding space (no pixel-level diffusion)
* Frozen CLIP ViT-B/32 backbone
* Lightweight 1D U-Net architecture
* Hybrid reconstruction + contrastive loss
* Designed for low-data training scenarios

## Setup

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open the provided notebook to:

* Extract CLIP image and text embeddings
* Train the diffusion denoiser
* Evaluate retrieval performance

## Evaluation Metrics

The project evaluates image-to-text retrieval using:

* **Recall@1**: Correct caption ranked first
* **Recall@3**: Correct caption in the top 3
* **Recall@5**: Correct caption in the top 5
* **Cosine Similarity**: Mean similarity between aligned image–text embeddings

## Results Summary

Experiments conducted with **1,000 training samples** from a **3,000-sample dataset** show:

* **Recall@1** improves from **39.5% (raw CLIP)** and **60.3% (contrastive fine-tuning)** to **87.4%** with diffusion-denoised embeddings
* Mean cosine similarity increases from **0.2856** to **0.6993**
* Cluster analysis demonstrates improved semantic organization and modality alignment

These results indicate that diffusion-based denoising produces smoother, more robust multimodal representations and consistently outperforms contrastive fine-tuning in low-data settings.

## Notes

* All CLIP parameters remain frozen during training
* The diffusion model operates entirely in latent space, making it computationally efficient
* Designed for research and educational experimentation
