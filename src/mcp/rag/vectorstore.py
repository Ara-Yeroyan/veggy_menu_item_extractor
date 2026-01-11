import logging
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from configs import get_settings
from .embeddings import get_embedding_service

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class VectorStore:
    """
    ChromaDB-based vector store for ingredient and dish knowledge base.
    
    Stores vegetarian/non-vegetarian ingredient information for RAG retrieval.
    """

    _instance = None
    _client = None
    _collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and load knowledge base."""
        logger.info("Initializing vector store")
        
        self._client = chromadb.Client(ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        self._collection = self._client.get_or_create_collection(
            name="ingredient_knowledge",
            metadata={"description": "Vegetarian ingredient classification"}
        )
        
        if self._collection.count() == 0:
            self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load ingredient knowledge base from JSON file."""
        kb_path = DATA_DIR / "knowledge_base.json"
        
        if not kb_path.exists():
            logger.warning("Knowledge base not found, creating default")
            self._create_default_knowledge_base()
            return
        
        with open(kb_path, "r") as f:
            knowledge = json.load(f)
        
        self._index_knowledge(knowledge)

    def _create_default_knowledge_base(self):
        """Create and index default knowledge base."""
        from .data.knowledge_base import KNOWLEDGE_BASE
        
        kb_path = DATA_DIR / "knowledge_base.json"
        kb_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(kb_path, "w") as f:
            json.dump(KNOWLEDGE_BASE, f, indent=2)
        
        self._index_knowledge(KNOWLEDGE_BASE)

    def _index_knowledge(self, knowledge: dict):
        """
        Index knowledge base into vector store.

        Parameters
        ----------
        knowledge : dict
            Knowledge base with ingredients and dishes
        """
        embedding_service = get_embedding_service()
        
        documents = []
        metadatas = []
        ids = []
        
        for item in knowledge.get("ingredients", []):
            doc = f"{item['name']}: {item.get('description', '')}"
            documents.append(doc)
            metadatas.append({
                "name": item["name"],
                "is_vegetarian": item["is_vegetarian"],
                "category": item.get("category", "unknown"),
                "type": "ingredient",
                "notes": item.get("notes", "")
            })
            ids.append(f"ing_{item['name'].lower().replace(' ', '_')}")
        
        for item in knowledge.get("dishes", []):
            doc = f"{item['name']}: {item.get('description', '')}"
            documents.append(doc)
            metadatas.append({
                "name": item["name"],
                "is_vegetarian": item["is_vegetarian"],
                "category": item.get("category", "unknown"),
                "type": "dish",
                "notes": item.get("notes", "")
            })
            ids.append(f"dish_{item['name'].lower().replace(' ', '_')}")
        
        if documents:
            embeddings = embedding_service.embed_batch(documents)
            self._collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(documents)} items into vector store")

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Search for relevant ingredients/dishes.

        Parameters
        ----------
        query : str
            Search query (dish name or description)
        top_k : int, optional
            Number of results to return

        Returns
        -------
        list[dict]
            List of relevant items with metadata and scores
        """
        settings = get_settings()
        k = top_k or settings.rag_top_k
        
        embedding_service = get_embedding_service()
        query_embedding = embedding_service.embed(query)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance_score": 1 - results["distances"][0][i]
            })
        
        return items

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "total_items": self._collection.count(),
            "collection_name": self._collection.name
        }


def get_vectorstore() -> VectorStore:
    """Get vector store instance."""
    return VectorStore()
