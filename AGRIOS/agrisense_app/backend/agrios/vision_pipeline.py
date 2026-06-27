"""
AGRI-OS Vision Pipeline
========================
DeiT Embedding Extraction, SCOLD Training, and Triplet Network.

Components:
- DeiTEmbeddingExtractor: Frozen DeiT-Small for 384-dim CLS token extraction
- SCOLDLoss / SCOLDTrainer: Soft-target contrastive learning on distilled embeddings
- TripletProjection: Semi-hard triplet network for manifold learning
- extract_dataset_embeddings: Batch embedding pipeline for image directories
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agrios.vision")

# ---------------------------------------------------------------------------
# Lazy imports to avoid hard failure when torch/timm not installed
# ---------------------------------------------------------------------------
_torch = None
_nn = None
_timm = None
_cv2 = None


def _lazy_imports():
    """Import heavy dependencies lazily."""
    global _torch, _nn, _timm, _cv2
    if _torch is None:
        import torch as _torch_mod
        import torch.nn as _nn_mod
        import timm as _timm_mod

        _torch = _torch_mod
        _nn = _nn_mod
        _timm = _timm_mod
    if _cv2 is None:
        import cv2 as _cv2_mod

        _cv2 = _cv2_mod


# ============================= DeiT Extractor ==============================


class DeiTEmbeddingExtractor:
    """
    Singleton DeiT-Small extractor producing 384-dim CLS token embeddings.

    Model: facebook/deit-small-distilled-patch16-224 via timm
    All parameters frozen — inference only.
    """

    _instance: Optional["DeiTEmbeddingExtractor"] = None

    def __new__(cls, device: str = "cpu") -> "DeiTEmbeddingExtractor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: str = "cpu") -> None:
        if self._initialized:
            return
        _lazy_imports()
        self.device = device
        logger.info("Loading DeiT-Small (distilled) on %s …", device)
        self.model = _timm.create_model("deit_small_distilled_patch16_224", pretrained=True, num_classes=0)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.to(self.device)

        # timm data config for normalization
        data_cfg = _timm.data.resolve_model_data_config(self.model)
        self.transform = _timm.data.create_transform(**data_cfg, is_training=False)
        self._initialized = True
        logger.info("DeiT-Small loaded — embedding dim=384")

    # -- public API -----------------------------------------------------------

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 384-dim CLS embedding from an OpenCV BGR image.

        Parameters
        ----------
        image : np.ndarray
            OpenCV image (BGR, any size).

        Returns
        -------
        np.ndarray
            Shape (384,) float32 embedding, L2-normalized.
        """
        _lazy_imports()
        from PIL import Image as PILImage

        # BGR → RGB → PIL
        rgb = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with _torch.no_grad():
            features = self.model(tensor)  # (1, 384)
        emb = features.squeeze(0).cpu().numpy().astype(np.float32)
        # L2-normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_file(self, image_path: str | Path) -> np.ndarray:
        """Load image from disk and extract embedding."""
        _lazy_imports()
        img = _cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.embed_image(img)

    def embed_base64(self, b64_data: str) -> np.ndarray:
        """Decode base64 image string and extract embedding."""
        import base64

        _lazy_imports()
        raw = base64.b64decode(b64_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 image")
        return self.embed_image(img)


# ============================== SCOLD Loss =================================


class SCOLDLoss:
    """
    Soft-target Contrastive Learning on Distilled embeddings.

    Loss: L = -Σ p_soft * log(sim(z_i, z_j) / τ)

    Uses soft label distributions from DeiT distillation token to create
    richer training signal than hard labels alone.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        _lazy_imports()
        self.temperature = temperature

    def __call__(
        self,
        projections: Any,  # torch.Tensor (B, D)
        soft_targets: Any,  # torch.Tensor (B, C) soft class distributions
    ) -> Any:
        """Compute SCOLD loss over a batch of projected embeddings."""
        _lazy_imports()
        # Normalize projections
        z = _torch.nn.functional.normalize(projections, dim=1)
        # Cosine similarity matrix
        sim_matrix = _torch.mm(z, z.t()) / self.temperature  # (B, B)
        # Soft target similarity (B, B) — how similar labels are
        if soft_targets.dim() == 1:
            # Hard labels → one-hot
            soft_targets = _torch.nn.functional.one_hot(
                soft_targets.long(), num_classes=soft_targets.max().item() + 1
            ).float()
        p_soft = _torch.mm(soft_targets, soft_targets.t())  # (B, B)
        p_soft = p_soft / p_soft.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Log-softmax over similarity
        log_prob = _torch.nn.functional.log_softmax(sim_matrix, dim=1)
        loss = -(p_soft * log_prob).sum(dim=1).mean()
        return loss


# ============================ SCOLD Trainer ================================


class SCOLDTrainer:
    """
    Train a 2-layer MLP projection head (384 → 256 → 128) on frozen DeiT
    embeddings using SCOLD loss.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.07,
        lr: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        _lazy_imports()
        self.device = device
        self.proj_head = _nn.Sequential(
            _nn.Linear(embed_dim, hidden_dim),
            _nn.BatchNorm1d(hidden_dim),
            _nn.ReLU(inplace=True),
            _nn.Linear(hidden_dim, proj_dim),
        ).to(device)
        self.criterion = SCOLDLoss(temperature=temperature)
        self.optimizer = _torch.optim.Adam(self.proj_head.parameters(), lr=lr)
        self.scaler = _torch.amp.GradScaler("cuda") if device != "cpu" else None

    def train(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 16,
        epochs: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the SCOLD projection head.

        Parameters
        ----------
        embeddings : np.ndarray  (N, 384)
        labels : np.ndarray       (N,)  integer class labels
        batch_size : int
        epochs : int
        save_path : str, optional  Path to save scold_projection.pt

        Returns
        -------
        dict with training metrics
        """
        _lazy_imports()
        dataset = _torch.utils.data.TensorDataset(
            _torch.tensor(embeddings, dtype=_torch.float32),
            _torch.tensor(labels, dtype=_torch.long),
        )
        loader = _torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.proj_head.train()
        history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for emb_batch, lbl_batch in loader:
                emb_batch = emb_batch.to(self.device)
                lbl_batch = lbl_batch.to(self.device)

                self.optimizer.zero_grad()

                if self.scaler is not None:
                    with _torch.amp.autocast("cuda"):
                        proj = self.proj_head(emb_batch)
                        loss = self.criterion(proj, lbl_batch)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    proj = self.proj_head(emb_batch)
                    loss = self.criterion(proj, lbl_batch)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append(avg_loss)
            logger.info("SCOLD Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            _torch.save(self.proj_head.state_dict(), save_path)
            logger.info("SCOLD projection head saved to %s", save_path)

        return {
            "epochs": epochs,
            "final_loss": history[-1] if history else 0.0,
            "history": history,
        }

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project a 384-dim embedding to 128-dim SCOLD space."""
        _lazy_imports()
        self.proj_head.eval()
        with _torch.no_grad():
            t = _torch.tensor(embedding, dtype=_torch.float32).unsqueeze(0).to(self.device)
            proj = self.proj_head(t).squeeze(0).cpu().numpy()
        norm = np.linalg.norm(proj)
        if norm > 0:
            proj = proj / norm
        return proj


# ============================ Triplet Network ==============================


class TripletProjection:
    """
    Triplet network projecting 384-dim DeiT embeddings → 128-dim manifold.
    Uses semi-hard negative mining and triplet margin loss.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        proj_dim: int = 128,
        margin: float = 0.3,
        lr: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        _lazy_imports()
        self.device = device
        self.margin = margin
        self.net = _nn.Sequential(
            _nn.Linear(embed_dim, 256),
            _nn.ReLU(inplace=True),
            _nn.Linear(256, proj_dim),
        ).to(device)
        self.optimizer = _torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = _nn.TripletMarginLoss(margin=margin)

    def _mine_semi_hard(self, embeddings: Any, labels: Any) -> Tuple[Any, Any, Any]:
        """Mine semi-hard triplets from a batch."""
        _lazy_imports()
        with _torch.no_grad():
            dists = _torch.cdist(embeddings, embeddings, p=2)

        anchors, positives, negatives = [], [], []
        for i in range(len(labels)):
            pos_mask = labels == labels[i]
            neg_mask = labels != labels[i]
            pos_mask[i] = False  # exclude self

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # Random positive
            pos_indices = _torch.where(pos_mask)[0]
            p_idx = pos_indices[_torch.randint(len(pos_indices), (1,))].item()
            ap_dist = dists[i, p_idx]

            # Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
            neg_indices = _torch.where(neg_mask)[0]
            neg_dists = dists[i, neg_indices]
            semi_hard = (neg_dists > ap_dist) & (neg_dists < ap_dist + self.margin)

            if semi_hard.sum() > 0:
                sh_indices = neg_indices[semi_hard]
                n_idx = sh_indices[_torch.randint(len(sh_indices), (1,))].item()
            else:
                # Fallback: hardest negative
                n_idx = neg_indices[neg_dists.argmin()].item()

            anchors.append(i)
            positives.append(p_idx)
            negatives.append(n_idx)

        if not anchors:
            return None, None, None
        return (
            _torch.tensor(anchors, dtype=_torch.long),
            _torch.tensor(positives, dtype=_torch.long),
            _torch.tensor(negatives, dtype=_torch.long),
        )

    def train(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 16,
        epochs: int = 5,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the triplet projection network."""
        _lazy_imports()
        dataset = _torch.utils.data.TensorDataset(
            _torch.tensor(embeddings, dtype=_torch.float32),
            _torch.tensor(labels, dtype=_torch.long),
        )
        loader = _torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.net.train()
        history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for emb_batch, lbl_batch in loader:
                emb_batch = emb_batch.to(self.device)
                lbl_batch = lbl_batch.to(self.device)

                projected = self.net(emb_batch)
                a_idx, p_idx, n_idx = self._mine_semi_hard(projected, lbl_batch)
                if a_idx is None:
                    continue

                loss = self.loss_fn(projected[a_idx], projected[p_idx], projected[n_idx])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append(avg_loss)
            logger.info("Triplet Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            _torch.save(self.net.state_dict(), save_path)
            logger.info("Triplet network saved to %s", save_path)

        return {
            "epochs": epochs,
            "final_loss": history[-1] if history else 0.0,
            "history": history,
        }


# ======================== Batch Embedding Pipeline =========================


def extract_dataset_embeddings(
    image_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Batch-extract DeiT embeddings from an image directory.

    Expects structure: image_dir/class_name/image.jpg
    Saves .npy embeddings and metadata JSON.

    Parameters
    ----------
    image_dir : path to image folder (PlantVillage-style)
    output_dir : path to save .npy embeddings + metadata
    batch_size : images per lazy-load batch
    device : 'cpu' or 'cuda'

    Returns
    -------
    dict with extraction stats
    """
    _lazy_imports()
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = DeiTEmbeddingExtractor(device=device)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    all_metadata: List[Dict[str, str]] = []
    all_embeddings: List[np.ndarray] = []
    file_list: List[Path] = []

    # Collect files
    for class_dir in sorted(image_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                file_list.append(img_path)

    logger.info("Found %d images in %s", len(file_list), image_dir)
    processed = 0
    failed = 0

    # Process in batches
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i : i + batch_size]
        for img_path in batch_files:
            try:
                emb = extractor.embed_file(img_path)
                all_embeddings.append(emb)

                # Parse class info from path: image_dir/CropName___DiseaseName/img.jpg
                class_name = img_path.parent.name
                parts = class_name.split("___")
                crop = parts[0] if parts else class_name
                disease = parts[1] if len(parts) > 1 else "unknown"

                emb_filename = f"{img_path.stem}.npy"
                emb_path = output_dir / emb_filename
                np.save(str(emb_path), emb)

                all_metadata.append(
                    {
                        "filename": img_path.name,
                        "crop": crop,
                        "disease": disease,
                        "region": "general",
                        "embedding_path": str(emb_path),
                        "source_image": str(img_path),
                    }
                )
                processed += 1
            except Exception as e:
                logger.warning("Failed to process %s: %s", img_path, e)
                failed += 1

        logger.info("Processed %d / %d images", processed, len(file_list))

    # Save metadata
    meta_path = output_dir / "embedding_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Save concatenated embeddings
    if all_embeddings:
        concat = np.stack(all_embeddings)
        np.save(str(output_dir / "all_embeddings.npy"), concat)
        labels = [m["disease"] for m in all_metadata]
        unique_labels = sorted(set(labels))
        label_map = {name: idx for idx, name in enumerate(unique_labels)}
        label_array = np.array(
            [label_map[lbl] for lbl in labels],
            dtype=np.int64,
        )
        np.save(str(output_dir / "all_labels.npy"), label_array)
        with open(output_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

    stats = {
        "total_images": len(file_list),
        "processed": processed,
        "failed": failed,
        "output_dir": str(output_dir),
        "metadata_path": str(meta_path),
    }
    logger.info("Embedding extraction complete: %s", stats)
    return stats
