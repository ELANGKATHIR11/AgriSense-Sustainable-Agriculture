# Patent Novelty Claim #05: Edge-Autonomous Agricultural OS Under 4 GB VRAM

## Title
System Architecture for an Edge-Autonomous Agricultural Operating System Running a Full AI Decision Pipeline Under 4 GB VRAM

## Mechanism

AGRI-OS runs a complete agricultural AI pipeline on edge hardware (Raspberry Pi 5 + 8 GB, Jetson Nano, or similar) within a strict VRAM/memory budget:

### Memory Architecture
| Component | Memory | Purpose |
|-----------|--------|---------|
| DeiT-Small (distilled) | 850 MB | 384-dim CLS token extraction |
| Phi-3-mini Q4 (GGUF via Ollama) | 2,500 MB | Evidence-grounded explanation |
| FAISS index (FlatIP) | ~50 MB | VRAG retrieval |
| SCOLD projection head | ~1 MB | 128-dim contrastive projection |
| Triplet network | ~1 MB | 128-dim manifold projection |
| Isolation Forest | ~5 MB | Anomaly gate |
| **Total** | **~3,407 MB** | **Under 4 GB budget** |

### Storage Architecture
| Component | Size | Purpose |
|-----------|------|---------|
| Existing codebase | 3,000 MB | FastAPI + React + ML models |
| DeiT-Small weights | 85 MB | Vision transformer |
| Phi-3-mini Q4 GGUF | 2,300 MB | Language model |
| FAISS + embeddings | 600 MB | Visual retrieval |
| **Total** | **~5,985 MB** | **Under 20 GB budget** |

### Pipeline Architecture
```
Image → DeiT (850MB) → Anomaly Gate (5MB) → VRAG (50MB) → Governor (CPU) → Phi-3 (2.5GB) → Response
                                    ↕
                        Sensor Data (from LoRa/BLE IoT)
```

Key design decisions enabling edge deployment:
1. **DeiT-Small over ViT-Base**: Saves ~1 GB VRAM, 384-dim embeddings sufficient for VRAG
2. **Q4 quantization for Phi-3**: 4-bit quantization reduces 7.6 GB FP16 → 2.3 GB GGUF
3. **Singleton model loading**: Each model loaded exactly once, shared across requests
4. **FAISS on CPU**: Inner-product search runs on CPU, avoiding GPU memory competition
5. **Lazy loading**: Models loaded only when first needed, reducing startup memory

## Why Non-Obvious

1. **Full pipeline, not single model**: Most edge AI deploys a single purpose model (e.g., disease classifier). AGRI-OS runs an entire pipeline: vision extraction → embedding search → anomaly detection → multi-signal decision → LLM explanation. Fitting this pipeline under 4 GB requires non-obvious architectural choices.

2. **Co-located LLM and vision**: Running both a vision transformer (DeiT) and a language model (Phi-3) simultaneously on edge hardware is non-obvious because these are typically deployed on separate infrastructure or in the cloud.

3. **Graceful degradation**: Each pipeline stage fails independently — if DeiT is unavailable, the system falls back to existing scikit-learn models; if Phi-3 is unavailable, template-based explanations are used; if FAISS index is empty, the Governor decides on sensor data alone. This fault-tolerant design is non-obvious for edge AI.

4. **Wrap, don't rewrite**: The system wraps 5312 lines of existing backend code with new layers rather than replacing them. The existing disease detection, weed management, and chatbot systems continue to operate; AGRI-OS adds post-processing hooks. This composition pattern is non-obvious for agricultural AI systems which typically require monolithic deployments.

## System Claim

An edge-autonomous agricultural operating system comprising:
- A DeiT-Small vision transformer (850 MB) for crop image embedding extraction
- A Phi-3-mini language model in Q4 quantization (2.5 GB) for evidence-grounded explanations
- A FAISS inner-product index for visual retrieval-augmented generation
- A Decision Governor implementing minimax regret-based action gating
- An Isolation Forest anomaly gate preventing action on out-of-distribution inputs
- All components co-located on a single edge device within a 4 GB VRAM budget
- Graceful degradation at each pipeline stage, maintaining functionality when individual components are unavailable

## Method Claim

A method for deploying a full agricultural AI decision pipeline on edge hardware comprising:
1. Loading a frozen DeiT-Small vision transformer using singleton pattern with lazy initialization
2. Loading a Q4-quantized Phi-3-mini language model via Ollama with strict memory limits
3. Maintaining a FAISS inner-product index in CPU memory for embedding retrieval
4. Processing agricultural inputs through a sequential pipeline: image embedding → anomaly gating → visual retrieval → regret-based decision → evidence-grounded explanation
5. Falling back to alternative methods at each stage when primary components are unavailable
6. Operating the entire pipeline within a 4 GB VRAM budget on commodity edge hardware

## Dependent Claims

1. The system of the main claim wherein the total storage footprint is under 6 GB including all model weights, indices, and application code.
2. The method of the main claim wherein a precomputed demo pipeline provides instant results without any model loading, enabling <2 second response times for demonstrations.
3. The system of the main claim wherein sensor data is received from LoRa-connected IoT devices and fused with vision results in the Decision Governor.
4. The method of the main claim wherein the system operates offline without internet connectivity after initial model download.
