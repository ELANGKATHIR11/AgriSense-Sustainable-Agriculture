# Patent Novelty Claim #02: SCOLD + Triplet Stress Manifold for Crop Disease Embedding

## Title
System and Method for Dual-Loss Projection Creating Stress-Aware Feature Spaces for Agricultural Disease Classification

## Mechanism

The system combines two complementary loss functions applied to frozen DeiT-Small vision transformer embeddings:

### SCOLD (Soft-target Contrastive Learning on Distilled embeddings)
```
L_SCOLD = -Σ p_soft * log(sim(z_i, z_j) / τ)
```
Where `p_soft` are soft label distributions derived from the DeiT distillation token, `sim` is cosine similarity in the projected space, and `τ` is a temperature parameter. A 2-layer MLP (384→256→128) projects frozen DeiT CLS tokens into a contrastive space where similar diseases cluster and dissimilar ones separate.

### Triplet Stress Network
```
L_triplet = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```
With semi-hard negative mining within each batch: negatives are selected where `d(a,p) < d(a,n) < d(a,p) + margin`, creating a "stress manifold" where the network learns nuanced boundaries between visually similar diseases.

### Dual-Loss Combination
The SCOLD projection creates globally separable clusters; the triplet network refines local boundaries. The resulting 128-dimensional manifold captures both macro-level disease categories and micro-level visual distinctions essential for accurate agricultural diagnosis.

## Why Non-Obvious

1. **Single-loss limitations**: Standard contrastive learning (SimCLR, MoCo) uses hard labels, losing the rich soft-target information available from distillation. Standard triplet networks learn pairwise relationships but miss global cluster structure. The combination is non-obvious because each loss alone appears sufficient.

2. **Stress manifold concept**: The semi-hard negative mining in the triplet network creates a learned "stress zone" at decision boundaries — the manifold encodes not just similarity but the difficulty of discrimination. This property enables the downstream VRAG system to assess reliability of matches.

3. **Frozen backbone efficiency**: Both projection networks train on frozen DeiT features, requiring <2 MB of trainable parameters total. This is non-obvious because the dominant approach in agricultural AI is to fine-tune entire vision models (hundreds of MB), which is impractical on edge devices.

4. **Domain-specific soft targets**: Using DeiT's distillation token (trained on ImageNet teacher) as soft targets for agricultural diseases creates a transfer learning signal — the ImageNet visual features inform disease similarity in ways that crop-specific hard labels cannot.

## System Claim

A computer-implemented system for crop disease embedding comprising:
- A frozen DeiT-Small vision transformer extracting 384-dimensional CLS token embeddings from crop images
- A SCOLD projection module applying soft-target contrastive loss using distillation-derived soft label distributions to train a 2-layer MLP (384→256→128)
- A Triplet stress network applying triplet margin loss with semi-hard negative mining to learn discriminative boundaries
- A combined 128-dimensional manifold encoding both global disease clusters and local stress boundaries
- A FAISS-backed retrieval index operating on the manifold for similarity search

## Method Claim

A method for creating a stress-aware disease embedding manifold comprising:
1. Extracting 384-dimensional CLS token embeddings from crop images using a frozen DeiT-Small vision transformer with distillation
2. Training a first projection head using soft-target contrastive loss where soft targets are derived from distillation token distributions
3. Training a second projection head using triplet margin loss with semi-hard negative mining, selecting negatives within the margin boundary
4. Combining the projected embeddings to form a 128-dimensional manifold
5. Building a FAISS inner-product index on the normalized manifold for nearest-neighbor retrieval
6. Using retrieval results as evidence for downstream decision-making

## Dependent Claims

1. The system of the main claim wherein the SCOLD temperature parameter τ is set to 0.07 and the triplet margin is set to 0.3, calibrated for inter-disease visual similarity in agricultural images.
2. The method of the main claim wherein training uses mixed precision (FP16) on the SCOLD projection head, enabling training under 4 GB VRAM on edge devices.
3. The system of the main claim wherein the stress manifold coordinates are used to compute a reliability score for VRAG retrieval, where matches near stress boundaries receive lower reliability.
4. The method of the main claim wherein batch formation ensures at least 2 positive pairs per disease class for effective contrastive learning.
