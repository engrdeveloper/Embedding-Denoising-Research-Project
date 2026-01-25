# Visual Guide: Diffusion Alignment Code

Quick visual reference for understanding the code flow.

---

## 1. Cross-Entropy Loss (Visual Example)

```
Batch Size = 4

Similarity Matrix (logits_per_image):
         Text 0  Text 1  Text 2  Text 3
Image 0   0.9    0.1    0.2    0.1   ← Should pick Text 0
Image 1   0.2    0.8    0.1    0.1   ← Should pick Text 1
Image 2   0.1    0.2    0.85   0.1   ← Should pick Text 2
Image 3   0.1    0.1    0.2    0.9   ← Should pick Text 3

Labels = [0, 1, 2, 3]  ← Correct positions

Cross-Entropy checks:
- Row 0: Is position 0 the highest? → Yes → Low loss
- Row 1: Is position 1 the highest? → Yes → Low loss
- Row 2: Is position 2 the highest? → Yes → Low loss
- Row 3: Is position 3 the highest? → Yes → Low loss

If wrong position is highest → High loss!
```

---

## 2. Diffusion Alignment (Conceptual Flow)

```
┌─────────────────────────────────────────────────────────────┐
│ Forward Process (Training):                                  │
└─────────────────────────────────────────────────────────────┘

Clean Image Embedding           Clean Text Embedding
    [0.5, 0.3, 0.8, ...]            [0.48, 0.32, 0.79, ...]
         │                                │
         │  ADD NOISE                     │
         ↓                                ↓
Noisy Image Embedding            Noisy Text Embedding
    [0.52, 0.29, 0.83, ...]          [0.50, 0.30, 0.81, ...]
         │                                │
         │  CROSS-MODAL DENOISING         │
         │  (Text guides image)           │
         ↓                                ↓
Denoised Image Embedding         Denoised Text Embedding
    [0.501, 0.3001, 0.7999, ...]     [0.4801, 0.3199, 0.7901, ...]
         │                                │
         └────────────┬───────────────────┘
                      │
                      ↓
              Now more aligned!

┌─────────────────────────────────────────────────────────────┐
│ Inference Process:                                           │
└─────────────────────────────────────────────────────────────┘

Clean Image Embedding           Clean Text Embedding
    [0.5, 0.3, 0.8, ...]            [0.48, 0.32, 0.79, ...]
         │                                │
         │  APPLY TRAINED DENOISER        │
         │  (No noise added!)             │
         ↓                                ↓
Aligned Image Embedding          Aligned Text Embedding
    [0.499, 0.301, 0.799, ...]       [0.481, 0.319, 0.791, ...]
         │                                │
         └────────────┬───────────────────┘
                      │
                      ↓
              Better alignment!
```

---

## 3. U-Net Architecture (Data Flow)

```
INPUT: Noisy Embedding [batch_size, 512]
       + Condition [batch_size, 512]
       = Concatenated [batch_size, 1024]
                           │
                           ↓
┌──────────────────────────────────────────────┐
│ ENCODER (Downsampling/Feature Extraction)    │
├──────────────────────────────────────────────┤
│                                               │
│  Linear(1024 → 1024) + ReLU                  │
│         ↓                                     │
│    e1: [batch_size, 1024]                    │
│         │ (SAVED FOR SKIP CONNECTION)        │
│         ↓                                     │
│  Linear(1024 → 1024) + ReLU                  │
│         ↓                                     │
│    e2: [batch_size, 1024]                    │
│                                               │
└──────────────────────────────────────────────┘
                           │
                           ↓
┌──────────────────────────────────────────────┐
│ DECODER (Upsampling/Reconstruction)          │
├──────────────────────────────────────────────┤
│                                               │
│  Linear(1024 → 1024) + ReLU                  │
│         ↓                                     │
│    (ADD e1 - SKIP CONNECTION!)               │
│         ↓                                     │
│    d1: [batch_size, 1024]                    │
│         ↓                                     │
│  Linear(1024 → 512)                          │
│         ↓                                     │
│    d2: [batch_size, 512] (Predicted Noise)   │
│                                               │
└──────────────────────────────────────────────┘
                           │
                           ↓
┌──────────────────────────────────────────────┐
│ OUTPUT: Residual Subtraction                 │
├──────────────────────────────────────────────┤
│                                               │
│  x_denoised = x_noisy - d2                   │
│                                               │
│  Output: [batch_size, 512]                   │
│                                               │
└──────────────────────────────────────────────┘
```

---

## 4. Training Process (One Batch)

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Get Clean CLIP Embeddings                           │
└─────────────────────────────────────────────────────────────┘

32 Images                   32 Captions
    │                            │
    ├─> CLIP Image Encoder       ├─> CLIP Text Encoder
    │                            │
    └─> img_embeds                └─> txt_embeds
        [32, 512]                     [32, 512]

┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Add Noise                                           │
└─────────────────────────────────────────────────────────────┘

Sample t ∈ [0, 1]  (e.g., t = 0.5)
    │
    ├─> Noise = Normal(0, t * max_std)
    │
    └─> noisy_img = img_embeds + noise
        noisy_txt = txt_embeds + noise
        [32, 512] each

┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Denoise with Cross-Modal Conditioning              │
└─────────────────────────────────────────────────────────────┘

noisy_img [32, 512]          txt_embeds [32, 512]
    │                            │
    ├─> residual = noisy - clean │
    │    [32, 512]               │
    │                            │
    ├─> Concatenate: [residual, txt_embeds]
    │    [32, 1024]
    │                            │
    ├─> U-Net Forward Pass       │
    │    [32, 1024] → [32, 512]  │
    │                            │
    └─> denoised_img             │
        [32, 512]                │
                                  │
        (Same for txt with img as condition)

┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Compute Loss                                        │
└─────────────────────────────────────────────────────────────┘

denoised_img [32, 512]       denoised_txt [32, 512]
img_embeds [32, 512]         txt_embeds [32, 512]
    │                            │
    ├─> MSE Loss                 │
    │    ||denoised - original||²│
    │                            │
    ├─> Contrastive Loss         │
    │    sim_matrix = denoised_img @ denoised_txt.T
    │    [32, 32] similarity matrix
    │    cross_entropy(sim_matrix, labels)
    │                            │
    └─> Total Loss = λ₁*MSE_img + λ₂*MSE_txt + Contrastive

┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Backpropagate and Update                            │
└─────────────────────────────────────────────────────────────┘

    Loss
     │
     ↓
Backward Pass
(Gradients computed)
     │
     ↓
Update U-Net Weights
(Adam optimizer)
```

---

## 5. Inference Process (Evaluation)

```
┌─────────────────────────────────────────────────────────────┐
│ TEST BATCH: 32 image-text pairs                             │
└─────────────────────────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Get CLIP Embeddings                                 │
└─────────────────────────────────────────────────────────────┘

32 Test Images           32 Test Captions
    │                        │
    └─> CLIP Encoders ──────┘
                │
                ↓
        img_embeds [32, 512]
        txt_embeds [32, 512]
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Apply Trained Denoiser (NO NOISE!)                  │
└─────────────────────────────────────────────────────────────┘

img_embeds [32, 512]     txt_embeds [32, 512]
    │                        │
    ├─> denoiser(img_embeds, cond=txt_embeds)
    │    └─> denoised_img [32, 512]
    │
    └─> denoiser(txt_embeds, cond=img_embeds)
        └─> denoised_txt [32, 512]
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Compute Similarity Matrix                           │
└─────────────────────────────────────────────────────────────┘

sim_matrix = denoised_img @ denoised_txt.T
[32, 32] matrix

         Text 0  Text 1  ...  Text 31
Image 0   0.95   0.20   ...   0.15
Image 1   0.18   0.92   ...   0.22
...
Image 31  0.21   0.19   ...   0.93
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Evaluate                                             │
└─────────────────────────────────────────────────────────────┘

For each image i:
    scores = sim_matrix[i]  # Row i: similarities to all texts
    top1 = argmax(scores)   # Best matching text index
    
    if top1 == i:           # Correct match?
        recall_at_1 += 1

Compute metrics:
- Recall@1: % of images with correct text in top-1
- Recall@3: % of images with correct text in top-3
- Recall@5: % of images with correct text in top-5
- Cosine Similarity: Average pair similarity
```

---

## 6. Key Differences: Training vs Inference

```
┌─────────────────────────────────────────────────────────────┐
│ TRAINING                                                     │
└─────────────────────────────────────────────────────────────┘

Input: Clean embeddings
    │
    ├─> ADD NOISE (random timestep t)
    │
    ├─> Denoise with U-Net
    │
    ├─> Compute loss (MSE + Contrastive)
    │
    ├─> Backpropagate
    │
    └─> Update weights

Goal: Learn to denoise and align

┌─────────────────────────────────────────────────────────────┐
│ INFERENCE                                                    │
└─────────────────────────────────────────────────────────────┘

Input: Clean embeddings
    │
    ├─> NO NOISE ADDED
    │
    ├─> Apply trained denoiser directly
    │
    ├─> Compute similarity matrix
    │
    └─> Evaluate (Recall@K, Cosine Similarity)

Goal: Use learned alignment for better matching
```

---

## 7. Data Shapes Throughout Pipeline

```
INPUT DATA:
- Images: [32] (list of PIL Images)
- Captions: [32] (list of strings)

AFTER CLIP ENCODING:
- img_embeds: [32, 512]
- txt_embeds: [32, 512]

AFTER NOISE ADDITION:
- noisy_img: [32, 512]
- noisy_txt: [32, 512]

RESIDUAL COMPUTATION:
- residual_img: [32, 512] = noisy_img - img_embeds
- residual_txt: [32, 512] = noisy_txt - txt_embeds

CONCATENATION (for U-Net input):
- x_img: [32, 1024] = [residual_img, txt_embeds]
- x_txt: [32, 1024] = [residual_txt, img_embeds]

U-NET INTERMEDIATE:
- e1: [32, 1024]
- e2: [32, 1024]
- d1: [32, 1024]
- d2: [32, 512] (predicted noise)

U-NET OUTPUT:
- denoised_img: [32, 512] = noisy_img - d2
- denoised_txt: [32, 512] = noisy_txt - d2

SIMILARITY MATRIX:
- sim_matrix: [32, 32] = denoised_img @ denoised_txt.T
```

---

## 8. Loss Function Components

```
Total Loss = MSE_Loss + Contrastive_Loss

┌─────────────────────────────────────────────────────────────┐
│ MSE Loss (Reconstruction)                                   │
└─────────────────────────────────────────────────────────────┘

MSE_Loss = λ₁ * ||denoised_img - img_embeds||²
         + λ₂ * ||denoised_txt - txt_embeds||²

Purpose: Ensure denoised embeddings are close to originals

Example:
- Original: [0.5, 0.3, 0.8]
- Denoised: [0.52, 0.31, 0.79]
- MSE: 0.0002 (should be small)

┌─────────────────────────────────────────────────────────────┐
│ Contrastive Loss (Alignment)                                │
└─────────────────────────────────────────────────────────────┘

Contrastive_Loss = CrossEntropy(sim_matrix, labels)

sim_matrix = denoised_img @ denoised_txt.T / temperature
[32, 32] matrix

labels = [0, 1, 2, ..., 31]  (diagonal elements)

Purpose: Ensure correct pairs have high similarity

Example:
sim_matrix = [
    [0.9, 0.1, 0.2, ...],  ← Image 0 should match Text 0 (0.9)
    [0.2, 0.8, 0.1, ...],  ← Image 1 should match Text 1 (0.8)
    ...
]

Cross-entropy maximizes diagonal elements!
```

---

## 9. Cross-Modal Conditioning (Key Innovation)

```
┌─────────────────────────────────────────────────────────────┐
│ Image Denoising (Text as Condition)                         │
└─────────────────────────────────────────────────────────────┘

noisy_img: [32, 512]          txt_embeds: [32, 512]
    │                            │
    ├─> residual_img              │
    │    [32, 512]                │
    │                            │
    ├───────────────┐            │
                    │            │
                    ↓            ↓
            [residual_img, txt_embeds]
                    [32, 1024]
                    │
                    ↓
                U-Net
                    │
                    ↓
            denoised_img [32, 512]

Key: Text embedding guides image denoising!

┌─────────────────────────────────────────────────────────────┐
│ Text Denoising (Image as Condition)                         │
└─────────────────────────────────────────────────────────────┘

noisy_txt: [32, 512]          img_embeds: [32, 512]
    │                            │
    ├─> residual_txt              │
    │    [32, 512]                │
    │                            │
    ├───────────────┐            │
                    │            │
                    ↓            ↓
            [residual_txt, img_embeds]
                    [32, 1024]
                    │
                    ↓
                U-Net
                    │
                    ↓
            denoised_txt [32, 512]

Key: Image embedding guides text denoising!

Result: Both modalities help each other align!
```

---

## 10. Quick Reference: Code Flow Summary

```
TRAINING ONE BATCH:
───────────────────
1. Get CLIP embeddings (frozen)
   → img_embeds [32, 512], txt_embeds [32, 512]

2. Sample t ∈ [0, 1], add noise
   → noisy_img [32, 512], noisy_txt [32, 512]

3. Compute residual
   → residual_img = noisy_img - img_embeds
   → residual_txt = noisy_txt - txt_embeds

4. Concatenate with condition
   → x_img = [residual_img, txt_embeds] [32, 1024]
   → x_txt = [residual_txt, img_embeds] [32, 1024]

5. U-Net forward pass
   → denoised_img [32, 512]
   → denoised_txt [32, 512]

6. Compute loss
   → MSE + Contrastive

7. Backpropagate and update
   → Update U-Net weights

───────────────────────────────────────────────────────

INFERENCE ONE BATCH:
────────────────────
1. Get CLIP embeddings (frozen)
   → img_embeds [32, 512], txt_embeds [32, 512]

2. Apply trained denoiser (NO NOISE!)
   → denoised_img = denoiser(img_embeds, cond=txt_embeds)
   → denoised_txt = denoiser(txt_embeds, cond=img_embeds)

3. Compute similarity matrix
   → sim_matrix = denoised_img @ denoised_txt.T [32, 32]

4. Evaluate
   → Recall@K, Cosine Similarity
```
