# CLIP Image-Text Matching Evaluation

This project evaluates the CLIP (Contrastive Language-Image Pre-training) model for image-text matching tasks across multiple datasets.

## Overview

The project uses OpenAI's CLIP-ViT-Base model to perform zero-shot image classification and image-text matching. It evaluates the model's performance using Recall@K metrics (Recall@1, Recall@3, Recall@5) on various image captioning datasets.

## Model

- **Model**: `openai/clip-vit-base-patch32`
- **Parameters**: ~151.28M
- **Task**: Zero-shot image classification and image-text matching


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

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

## Evaluation Metrics

The project measures:
- **Recall@1**: Percentage of cases where the correct caption is ranked first
- **Recall@3**: Percentage of cases where the correct caption is in the top 3
- **Recall@5**: Percentage of cases where the correct caption is in the top 5

## Results

The notebook includes evaluation results and visualization plots showing Recall@K performance across different datasets.

