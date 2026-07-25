# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps Multimodal Retrieval-Augmented Generation (MRAG) Engine
Upgraded to use Qdrant DB for zero-dependency local Edge AI operations.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.rag.embedding_service import get_embedding

_KNOWLEDGE_BASE = [
    {
        "text": "Tomato Leaf Mold is caused by Passalora fulva. Symptoms include yellow spots on upper leaf surfaces and olive-green velvet mold underneath. Improve greenhouse ventilation and avoid overhead watering.",
        "disease": "Tomato Leaf Mold",
    },
    {
        "text": "Late Blight on Squash is caused by Phytophthora. Dark water-soaked lesions appear on leaves. Apply copper-based biological fungicide immediately.",
        "disease": "Late Blight",
    },
    {
        "text": "Nitrogen deficiency causes uniform yellowing of older leaves starting at the tips. Supplement with organic blood meal, urea, or legume compost.",
        "nutrient": "Nitrogen",
    },
    {
        "text": "Potassium deficiency leads to leaf margin curling and brown necrotic edges. Apply wood ash or kelp meal for correction.",
        "nutrient": "Potassium",
    },
    {
        "text": "Phosphorus deficiency shows as purple/red discolouration on undersides of leaves. Apply bone meal or rock phosphate.",
        "nutrient": "Phosphorus",
    },
    {
        "text": "Powdery Mildew shows white talcum-like powdery spots on leaves. Apply neem oil extract or potassium bicarbonate. Ensure full sunlight and good spacing.",
        "disease": "Powdery Mildew",
    },
    {
        "text": "Corn Common Rust shows reddish-brown powdery pustules on both leaf surfaces. Apply strobilurin or triazole fungicides. Plant rust-resistant hybrids.",
        "disease": "Corn Rust",
    },
    {
        "text": "Weeds compete for soil nitrogen and moisture. Use mulching or selective organic pre-emergents to control weed growth.",
        "weed": "Weed competition",
    },
    {
        "text": "Sandy soil has low water retention and nutrients. Add organic matter (compost), use drip irrigation, and apply slow-release NPK fertilizers.",
        "soil": "Sandy Soil",
    },
    {
        "text": "For rice cultivation, optimal soil pH is 5.5-6.5. Maintain 80-90% soil moisture. Use nitrogen fertilizers in split doses.",
        "crop": "Rice",
    },
    {
        "text": "For tomato cultivation, optimal soil pH is 6.0-6.8. Water deeply but infrequently. Apply calcium to prevent blossom end rot.",
        "crop": "Tomato",
    },
    {
        "text": "Irrigation optimization: water requirements depend on crop type, soil moisture, temperature and evapotranspiration. Use drip irrigation for 40% water savings.",
        "topic": "Irrigation",
    },
]

logger = logging.getLogger("AgriOps.MRAG")

QDRANT_DB_DIR = "ml/models/qdrant_data"
os.makedirs(QDRANT_DB_DIR, exist_ok=True)


class MRAGOrchestrator:
    def __init__(self):
        try:
            self.db = QdrantClient(path=QDRANT_DB_DIR)
        except Exception as e:
            logger.warning(f"File storage lock for Qdrant directory active, falling back to in-memory Qdrant instance: {e}")
            self.db = QdrantClient(":memory:")
        self.dimension = 1024  # Dimension for BGE-M3 text embeddings
        self._init_collections()
        self._migrate_legacy_data()


    def get_table_names(self) -> List[str]:
        try:
            cols = self.db.get_collections().collections
            return [c.name for c in cols]
        except Exception:
            return []

    def _init_collections(self):
        """
        Creates Qdrant collections with matching dimensions.
        """
        collections_to_init = [
            "documents",
            "images",
            "diseases",
            "crops",
            "weeds",
            "weather",
            "satellite",
            "sensors",
            "agent_memory",
            "conversations",
            "research",
            "government_docs",
            "market_prices",
        ]

        for col in collections_to_init:
            try:
                if col not in self.get_table_names():
                    dim = 512 if col == "images" else self.dimension
                    self.db.create_collection(
                        collection_name=col,
                        vectors_config=qmodels.VectorParams(
                            size=dim,
                            distance=qmodels.Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant collection: {col}")
            except Exception as e:
                logger.error(f"Error creating collection {col}: {e}")

    def _migrate_legacy_data(self):
        """
        Migrates legacy knowledge base into Qdrant 'documents' collection.
        """
        try:
            info = self.db.get_collection("documents")
            if info.points_count == 0:
                logger.info(
                    "Migrating legacy FAISS/NumPy knowledge base to Qdrant 'documents'..."
                )
                points = []
                for idx, doc in enumerate(_KNOWLEDGE_BASE):
                    vec = get_embedding(doc["text"]).tolist()
                    points.append(
                        qmodels.PointStruct(
                            id=idx,
                            vector=vec,
                            payload={
                                "id": f"leg-{idx}",
                                "text": doc["text"],
                                "metadata": {k: v for k, v in doc.items() if k not in ("text")},
                                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                            }
                        )
                    )
                self.db.upsert(collection_name="documents", points=points)
                logger.info(
                    f"Successfully migrated {len(points)} legacy entries to Qdrant."
                )
        except Exception as e:
            logger.error(f"Legacy data migration failed: {e}")

    def search_collection(
        self,
        collection_name: str,
        query: str,
        k: int = 3,
        metadata_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs vector similarity search on a Qdrant collection.
        """
        if collection_name not in self.get_table_names():
            logger.warning(f"Collection {collection_name} does not exist.")
            return []

        try:
            query_vector = get_embedding(query).tolist()

            # Parse simple metadata filter (e.g. "agent_id = 'agent-999'")
            qfilter = None
            if metadata_filter and "=" in metadata_filter:
                try:
                    parts = metadata_filter.split("=")
                    filter_key = parts[0].strip()
                    filter_val = parts[1].strip().strip("'").strip('"')
                    # Standard payload key path
                    qfilter = qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key=f"metadata.{filter_key}",
                                match=qmodels.MatchValue(value=filter_val)
                            )
                        ]
                    )
                except Exception:
                    pass

            results = self.db.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=k,
                query_filter=qfilter
            ).points

            formatted = []
            for item in results:
                payload = item.payload or {}
                meta_str = payload.get("metadata", "{}")
                if isinstance(meta_str, str):
                    meta = json.loads(meta_str)
                else:
                    meta = meta_str

                # Map cosine similarity to matching score contract [0, 1]
                formatted.append(
                    {
                        "id": payload.get("id", str(item.id)),
                        "text": payload.get("text", ""),
                        "score": float(item.score),
                        "metadata": meta,
                        "timestamp": payload.get("timestamp"),
                    }
                )

            return formatted
        except Exception as e:
            logger.error(f"Search failed on collection {collection_name}: {e}")
            return []

    def index_document(
        self, collection_name: str, doc_id: str, text: str, metadata: Dict[str, Any]
    ):
        """
        Dynamically adds a text document to the vector collection.
        """
        if collection_name not in self.get_table_names():
            self._init_collections()

        try:
            vec = get_embedding(text).tolist()
            import random
            qdrant_id = random.randint(100000, 999999)
            self.db.upsert(
                collection_name=collection_name,
                points=[
                    qmodels.PointStruct(
                        id=qdrant_id,
                        vector=vec,
                        payload={
                            "id": doc_id,
                            "text": text,
                            "metadata": metadata,
                            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                        }
                    )
                ]
            )
            logger.info(f"Indexed document {doc_id} into collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to index document in {collection_name}: {e}")

    def get_orchestrated_mrag_context(
        self, query: str, sensor_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fuses RAG results from Qdrant collections.
        """
        docs_res = self.search_collection("documents", query, k=2)
        diseases_res = self.search_collection("diseases", query, k=2)
        crops_res = self.search_collection("crops", query, k=2)
        weather_res = self.search_collection("weather", query, k=1)

        all_results = docs_res + diseases_res + crops_res + weather_res
        all_results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "sensor_context": sensor_context or {},
            "retrieved_context": all_results[:4],
            "highest_score": all_results[0]["score"] if all_results else 0.0,
            "sources_count": len(all_results),
        }


# Global singleton orchestrator
mrag_orchestrator = MRAGOrchestrator()
