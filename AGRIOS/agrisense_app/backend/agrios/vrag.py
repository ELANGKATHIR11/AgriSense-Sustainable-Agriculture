"""
AGRI-OS VRAG — Visual Retrieval-Augmented Generation
=====================================================
FAISS-backed index for vision-embedding retrieval.

Components:
- VRAGIndexBuilder: Build and save FAISS indices from .npy embeddings
- VRAGEngine: Query the index with a DeiT embedding → List[VRAGResult]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .schemas import VRAGResult

logger = logging.getLogger("agrios.vrag")

# Lazy FAISS import
_faiss = None


def _lazy_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_mod

        _faiss = _faiss_mod
    return _faiss


# ========================= FAISS Index Builder =============================


class VRAGIndexBuilder:
    """
    Build a FAISS index from precomputed .npy embeddings.

    Supports:
    - IndexFlatIP (inner product on L2-normalized vectors) for small datasets
    - IndexIVFFlat with nprobe tuning for larger datasets (>10k embeddings)

    Saves: vrag_index.faiss + vrag_metadata.json
    """

    def __init__(self, embed_dim: int = 384) -> None:
        self.embed_dim = embed_dim
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_from_directory(self, emb_dir: str | Path) -> int:
        """
        Load embeddings + metadata from a directory produced by
        extract_dataset_embeddings().

        Returns number of embeddings loaded.
        """
        emb_dir = Path(emb_dir)

        # Try concatenated file first
        concat_path = emb_dir / "all_embeddings.npy"
        meta_path = emb_dir / "embedding_metadata.json"

        if concat_path.exists() and meta_path.exists():
            embeddings = np.load(str(concat_path))
            with open(meta_path) as f:
                metadata = json.load(f)

            for i, meta in enumerate(metadata):
                self.embeddings.append(embeddings[i])
                self.metadata.append(
                    {
                        "id": f"emb_{len(self.metadata)}",
                        "crop": meta.get("crop", "unknown"),
                        "disease": meta.get("disease", "unknown"),
                        "region": meta.get("region", "general"),
                        "severity": meta.get("severity", "unknown"),
                        "outcome": meta.get("outcome", ""),
                        "source_image": meta.get("source_image", ""),
                        "embedding_path": meta.get("embedding_path", ""),
                    }
                )
            logger.info("Loaded %d embeddings from %s", len(metadata), concat_path)
            return len(metadata)

        # Fallback: load individual .npy files
        count = 0
        for npy_file in sorted(emb_dir.glob("*.npy")):
            if npy_file.name in ("all_embeddings.npy", "all_labels.npy"):
                continue
            emb = np.load(str(npy_file))
            if emb.shape != (self.embed_dim,):
                continue
            self.embeddings.append(emb)
            self.metadata.append(
                {
                    "id": f"emb_{len(self.metadata)}",
                    "crop": "unknown",
                    "disease": npy_file.stem,
                    "region": "general",
                    "severity": "unknown",
                    "outcome": "",
                    "source_image": "",
                    "embedding_path": str(npy_file),
                }
            )
            count += 1
        logger.info("Loaded %d individual embedding files from %s", count, emb_dir)
        return count

    def add_embedding(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add a single embedding with metadata."""
        self.embeddings.append(embedding.astype(np.float32))
        metadata.setdefault("id", f"emb_{len(self.metadata)}")
        self.metadata.append(metadata)

    def build_and_save(
        self,
        output_dir: str | Path,
        use_ivf: bool = False,
        nlist: int = 100,
    ) -> Dict[str, Any]:
        """
        Build FAISS index and save to disk.

        Parameters
        ----------
        output_dir : directory for .faiss and .json files
        use_ivf : use IVFFlat instead of FlatIP (for >10k entries)
        nlist : number of IVF clusters

        Returns
        -------
        dict with build stats
        """
        faiss = _lazy_faiss()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.embeddings:
            raise ValueError("No embeddings to index — add embeddings first")

        matrix = np.stack(self.embeddings).astype(np.float32)
        # Ensure L2 normalization for IP-based retrieval
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        n, d = matrix.shape
        logger.info("Building FAISS index: %d vectors × %d dims", n, d)

        if use_ivf and n > nlist * 10:
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, min(nlist, n // 10))
            index.train(matrix)
            index.add(matrix)
            index.nprobe = 10
        else:
            index = faiss.IndexFlatIP(d)
            index.add(matrix)

        index_path = output_dir / "vrag_index.faiss"
        meta_path = output_dir / "vrag_metadata.json"

        faiss.write_index(index, str(index_path))
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        stats = {
            "index_path": str(index_path),
            "metadata_path": str(meta_path),
            "num_vectors": n,
            "dimension": d,
            "index_type": "IVFFlat" if use_ivf else "FlatIP",
        }
        logger.info("FAISS index saved: %s", stats)
        return stats


# ============================ VRAG Engine ==================================


class VRAGEngine:
    """
    Query engine for the FAISS-backed VRAG index.

    Loads index and metadata from disk, provides similarity search
    with optional crop/region filtering.
    """

    def __init__(
        self,
        index_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
    ) -> None:
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._loaded = False

        if index_path and metadata_path:
            self.load(index_path, metadata_path)

    def load(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Load FAISS index and metadata from disk."""
        faiss = _lazy_faiss()
        index_path = str(index_path)
        metadata_path = str(metadata_path)

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self._loaded = True
        logger.info(
            "VRAG engine loaded: %d vectors, %d metadata entries",
            self.index.ntotal,
            len(self.metadata),
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.index is not None

    @property
    def index_size(self) -> int:
        return self.index.ntotal if self.index else 0

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        crop_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
    ) -> List[VRAGResult]:
        """
        Query the VRAG index with an embedding.

        Parameters
        ----------
        embedding : np.ndarray (384,) — L2-normalized query embedding
        top_k : number of results
        crop_filter : optional crop type filter (post-retrieval)
        region_filter : optional region filter (post-retrieval)

        Returns
        -------
        List[VRAGResult] — sorted by similarity descending
        """
        if not self.is_loaded:
            logger.warning("VRAG engine not loaded — returning empty results")
            return []

        # Prepare query vector
        q = embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Search with expanded k for post-filtering
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(q, search_k)

        results: List[VRAGResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]

            # Post-retrieval filters
            if crop_filter and meta.get("crop", "").lower() != crop_filter.lower():
                continue
            if region_filter and meta.get("region", "").lower() != region_filter.lower():
                continue

            results.append(
                VRAGResult(
                    doc_id=meta.get("id", f"emb_{idx}"),
                    similarity_score=float(score),
                    metadata=meta,
                    evidence_text=self._generate_evidence(meta, float(score)),
                    source_image=meta.get("source_image"),
                )
            )
            if len(results) >= top_k:
                break

        return results

    def _generate_evidence(self, meta: Dict[str, Any], score: float) -> str:
        """Generate human-readable evidence text from metadata."""
        crop = meta.get("crop", "unknown crop")
        disease = meta.get("disease", "unknown condition")
        severity = meta.get("severity", "unknown")
        outcome = meta.get("outcome", "")

        text = f"Similar case: {crop} with {disease} " f"(severity: {severity}, similarity: {score:.3f})"
        if outcome:
            text += f". Outcome: {outcome}"
        return text

    # ---------- Feedback helpers ----------

    def add_embedding(
        self,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        """Add a single embedding to the live index (feedback loop).

        The embedding is L2-normalised before insertion.
        """
        if self.index is None:
            faiss = _lazy_faiss()
            d = len(embedding)
            self.index = faiss.IndexFlatIP(d)
            self._loaded = True

        vec = embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.index.add(vec)

        meta = dict(metadata)
        meta.setdefault("id", f"fb_{self.index.ntotal}")
        self.metadata.append(meta)
        logger.info(
            "Feedback embedding added — index now has %d vectors",
            self.index.ntotal,
        )

    def get_all_embeddings(self) -> Optional[np.ndarray]:
        """Return all indexed embeddings as an ndarray.

        Useful for retraining the Isolation Forest.
        Returns None if index is empty.
        """
        if not self.is_loaded or self.index is None:
            return None
        n = self.index.ntotal
        if n == 0:
            return None
        d = self.index.d
        return np.array(
            [self.index.reconstruct(i) for i in range(n)],
            dtype=np.float32,
        ).reshape(n, d)
