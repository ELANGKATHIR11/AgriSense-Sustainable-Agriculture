# Phenology-Adaptive Temporal Attention (PATA)

## System Overview
The **PATA Network** solves the "variable sowing date" problem in crop yield forecasting. Instead of relying on farmer-reported sowing dates or trying to force-fit calendar-based models, PATA implements a **Differential Temporal Alignment** mechanism.

## Components
1.  **`phenology_anchor.py`**:
    *   **Class**: `PhenologyAnchorDetector`
    *   **Function**: A 1D-CNN that scans the input time-series to identify the "Biological Zero" (Sowing/Green-up point).
    *   **Output**: A scalar `shift` parameter [-1, 1].

2.  **`temporal_warping.py`**:
    *   **Class**: `TemporalWarpingLayer`
    *   **Function**: A differentiable sampling layer (inspired by Spatial Transformer Networks) that re-samples the input time-series onto a standardized "Biological Timeline".
    *   **Patent Novelty**: "Deep learning architecture comprising a dynamic temporal resampling layer controlled by a latent phenological regressor."

## Usage Concept
```python
import torch
from temporal_warping import PATANetwork

# Input: Batch of unaligned NDVI series (e.g., sequences starting Jan 1st)
# Some farms sowed in Jan, some in Feb.
unaligned_batch = torch.randn(32, 1, 52) 

model = PATANetwork()
yield_pred, detected_shift, aligned_series = model(unaligned_batch)

# 'aligned_series' is now normalized: Index 0 = Sowing Date for ALL samples.
```
