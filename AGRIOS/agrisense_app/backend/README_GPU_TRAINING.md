GPU Training Quickstart
======================

Overview
--------
This document explains how to run GPU-backed training for LLM, VLM, and hybrid models using TensorFlow 2.20.0 in a Docker container.

Prerequisites
-------------
- A GPU-enabled Linux host with NVIDIA drivers and the NVIDIA Container Toolkit (nvidia-docker) installed.
- Docker 20.10+ and nvidia-docker configured (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- Sufficient disk space for model checkpoints and datasets.

Build the GPU image
-------------------
From the repository root run:

```bash
docker build -f agrisense_app/backend/Dockerfile.gpu -t agrisense/agriml:gpu .
```

Run training (examples)
-----------------------
Start a container and run the built-in runner which will call your project scripts if present, otherwise run a tiny TF smoke training to validate GPU:

```bash
# Run LLM training (example)
docker run --gpus all --rm -it \
  -v $(pwd):/app \
  --workdir /app/agrisense_app/backend \
  agrisense/agriml:gpu \
  "/bin/bash" -lc "./scripts/run_training.sh llm 3 16"

# Replace 'llm' with 'vlm' or 'hybrid' as needed. Adjust epochs and batch size.
```

Notes
-----
- The image is based on the official `tensorflow/tensorflow:2.20.0-gpu` base which includes the correct CUDA and cuDNN versions. No host CUDA installation is required if you use the official GPU image and NVIDIA container runtime.
- To run large-scale LLM/VLM training you should mount a data volume for datasets and a volume for checkpoints to persist between runs.
- For distributed training across multiple GPUs or nodes, I can add Horovod / TF MirroredStrategy examples — tell me if you want that.

Next steps I can take
---------------------
- Build and push the image to a registry you provide (or to Docker Hub) so you can run it on your GPU host.
- Add job scripts for Slurm / Kubernetes (K8s) if you have a cluster.
- Add example training configs that map to your existing model code (if you point me to training entrypoints).
