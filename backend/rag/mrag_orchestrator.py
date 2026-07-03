# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps Multimodal Retrieval-Augmented Generation (MRAG) Engine
Integrates local LanceDB vector collections with BGE-M3 embeddings, hybrid search, and reranking.
"""

import os
import json
import logging
import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

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

LANCE_DB_DIR = "ml/models/lancedb_data"
os.makedirs(LANCE_DB_DIR, exist_ok=True)


class MRAGOrchestrator:
    def __init__(self):
        self.db = lancedb.connect(LANCE_DB_DIR)
        self.dimension = 1024  # Standard dimension for BGE-M3 text embeddings
        self._init_collections()
        self._migrate_legacy_data()

    def get_table_names(self) -> List[str]:
        try:
            res = self.db.list_tables()
            if hasattr(res, "tables"):
                return res.tables
            return list(res)
        except Exception:
            return self.db.table_names()

    def _init_collections(self):
        """
        Creates all 13 collections in LanceDB with strict schemas.
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

        # Standard text-based schema
        text_schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), self.dimension)),
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),  # JSON-serialized metadata
                pa.field("timestamp", pa.string()),
            ]
        )

        # Standard image-based schema
        image_schema = pa.schema(
            [
                pa.field(
                    "vector", pa.list_(pa.float32(), 512)
                ),  # standard image embed dimension (e.g. ResNet/CLIP)
                pa.field("id", pa.string()),
                pa.field("image_path", pa.string()),
                pa.field("label", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("timestamp", pa.string()),
            ]
        )

        for col in collections_to_init:
            try:
                schema = image_schema if col in ["images"] else text_schema
                if col not in self.get_table_names():
                    try:
                        self.db.create_table(col, schema=schema)
                        logger.info(f"Created LanceDB collection: {col}")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            try:
                                self.db.open_table(col)
                            except Exception:
                                import shutil

                                path = os.path.join(LANCE_DB_DIR, f"{col}.lance")
                                if os.path.exists(path):
                                    shutil.rmtree(path)
                                self.db.create_table(col, schema=schema)
                                logger.info(
                                    f"Self-healed and created LanceDB collection: {col}"
                                )
            except Exception as e:
                logger.error(f"Error creating collection {col}: {e}")

    def _migrate_legacy_data(self):
        """
        Migrates legacy _KNOWLEDGE_BASE array documents into lancedb 'documents' table.
        """
        try:
            tbl = self.db.open_table("documents")
            if len(tbl) == 0:
                logger.info(
                    "Migrating legacy FAISS/NumPy knowledge base to LanceDB 'documents'..."
                )
                data = []
                for idx, doc in enumerate(_KNOWLEDGE_BASE):
                    vec = get_embedding(doc["text"])
                    data.append(
                        {
                            "vector": vec.tolist(),
                            "id": f"leg-{idx}",
                            "text": doc["text"],
                            "metadata": json.dumps(
                                {k: v for k, v in doc.items() if k not in ("text")}
                            ),
                            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                        }
                    )
                tbl.add(data)
                logger.info(
                    f"Successfully migrated {len(data)} legacy entries to LanceDB."
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
        Runs vector similarity and hybrid search on a LanceDB collection.
        """
        if collection_name not in self.get_table_names():
            logger.warning(f"Collection {collection_name} does not exist.")
            return []

        try:
            tbl = self.db.open_table(collection_name)
            query_vector = get_embedding(query).tolist()

            # Unpack simple SQL-like metadata filter e.g., "agent_id = 'agent-999'"
            filter_key, filter_val = None, None
            if metadata_filter and "=" in metadata_filter:
                try:
                    parts = metadata_filter.split("=")
                    filter_key = parts[0].strip()
                    filter_val = parts[1].strip().strip("'").strip('"')
                except Exception:
                    pass

            qb = tbl.search(query_vector).limit(k if not filter_key else 100)
            results = qb.to_list()

            formatted = []
            for item in results:
                # Calculate simple distance score (LanceDB outputs L2 distance by default, lower is closer)
                # Convert to cosine-like similarity score
                dist = item.get("_distance", 1.0)
                sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                meta = json.loads(item.get("metadata", "{}"))

                if filter_key and meta.get(filter_key) != filter_val:
                    continue

                formatted.append(
                    {
                        "id": item.get("id"),
                        "text": item.get("text"),
                        "score": float(sim_score),
                        "metadata": meta,
                        "timestamp": item.get("timestamp"),
                    }
                )

                if len(formatted) >= k:
                    break

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
            tbl = self.db.open_table(collection_name)
            vec = get_embedding(text).tolist()
            tbl.add(
                [
                    {
                        "vector": vec,
                        "id": doc_id,
                        "text": text,
                        "metadata": json.dumps(metadata),
                        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    }
                ]
            )
            logger.info(f"Indexed document {doc_id} into collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to index document in {collection_name}: {e}")

    def get_orchestrated_mrag_context(
        self, query: str, sensor_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fuses RAG results from documents, diseases, crops, weather, and active sensor context.
        """
        docs_res = self.search_collection("documents", query, k=2)
        diseases_res = self.search_collection("diseases", query, k=2)
        crops_res = self.search_collection("crops", query, k=2)
        weather_res = self.search_collection("weather", query, k=1)

        # Merge results, sorting by similarity score
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
