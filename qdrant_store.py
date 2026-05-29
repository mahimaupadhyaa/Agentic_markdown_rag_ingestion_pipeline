import os
from typing import List, Optional

from config import config

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from dotenv import load_dotenv
load_dotenv()


# -----------------------
# Utility Functions
# -----------------------

def _get_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def drop_collection(collection_name: str, url: str = "http://localhost:6333") -> bool:
    try:
        client = _get_client(url)
        if not client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return False
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' dropped successfully.")
        return True
    except Exception as e:
        print("Error dropping collection:", e)
        return False


def drop_all_collections(url: str = "http://localhost:6333", confirm: bool = False) -> bool:
    if not confirm:
        print("WARNING: This will delete ALL collections. Set confirm=True to proceed.")
        return False
    try:
        client = _get_client(url)
        collections = [c.name for c in client.get_collections().collections]
        print(f"Found {len(collections)} collections.")
        for name in collections:
            print(f"Dropping {name}")
            client.delete_collection(name)
        return True
    except Exception as e:
        print("Error dropping all collections:", e)
        return False


# -----------------------
# QdrantStore Class
# -----------------------

class QdrantStore:
    """
    Drop-in replacement for MilvusStore using Qdrant.

    Same public interface:
      - add_documents()
      - as_retriever()
      - similarity_search()
      - similarity_search_with_score()
      - drop_collection()
      - drop_all_collections()

    Hybrid search: dense (OpenAI) + sparse (BM25 via FastEmbed).
    Namespace filtering via payload field.
    """

    SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        url: str = None,
        collection_name: str = None,
        embed_model: str = None,
        api_key: str = None,
        drop_old: bool = None,
        namespace: str = None,
        # Accept db_name for interface compatibility — not used in Qdrant
        db_name: str = None,
    ):
        print("\nInitializing QdrantStore")

        self.url = url or config.get("database", "uri", default="http://localhost:6333")
        self.collection_name = collection_name or config.get(
            "database", "collection_name", default="collection_demo"
        )
        self.embed_model = embed_model or config.get(
            "model", "embeddings", default="text-embedding-3-small"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.namespace = namespace or config.get("database", "namespace", default="default_namespace")

        print(f"URL: {self.url}")
        print(f"Collection Name: {self.collection_name}")
        print(f"Embed Model: {self.embed_model}")
        print(f"API Key loaded")
        print(f"Namespace: {self.namespace}")

        self.client = QdrantClient(url=self.url)

        print(f"Loading Embedding model: {self.embed_model}")
        self.embeddings_model = OpenAIEmbeddings(
            model=self.embed_model,
            api_key=self.api_key
        )
        print("Embedding model loaded successfully.")

        # Sparse embeddings via FastEmbed (BM25)
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self._initialize_collection(drop_old)

        self.vector_store = self._create_vector_store()

    def _initialize_collection(self, drop_old: bool):
        """
        Create the Qdrant collection with dense + sparse vector config.
        Drops existing collection first if drop_old=True.
        """
        print("\nInitializing collection")

        exists = self.client.collection_exists(self.collection_name)

        if exists and drop_old:
            print(f"Dropping old collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            print(f"Creating collection: {self.collection_name}")

            # Get dense vector size from a test embedding
            test_vec = self.embeddings_model.embed_query("test")
            dense_size = len(test_vec)
            print(f"Dense vector size: {dense_size}")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            print("Collection created successfully.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def _create_vector_store(self) -> QdrantVectorStore:
        """
        Create the LangChain QdrantVectorStore in hybrid retrieval mode.
        """
        print("\nCreating vector store")

        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings_model,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name=self.SPARSE_VECTOR_NAME,
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        Namespace is stored as a payload field for filtering.
        """
        print(f"\nAdding {len(documents)} documents to Qdrant...")
        uids = self.vector_store.add_documents(documents=documents)
        print(f"Inserted {len(uids)} documents.")
        return uids

    def as_retriever(
        self,
        k: int = 4,
        namespace: str = None,
        # ranker_type and ranker_weights kept for interface compatibility
        ranker_type: str = "weighted",
        ranker_weights=None,
    ) -> BaseRetriever:

        namespace = namespace or self.namespace

        print("\nCreating retriever")
        print("k =", k)
        print("namespace =", namespace)

        search_kwargs = {"k": k}

        if namespace:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            search_kwargs["filter"] = Filter(
                must=[
                    FieldCondition(
                        key="metadata.namespace",
                        match=MatchValue(value=namespace),
                    )
                ]
            )

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        namespace: str = None,
    ) -> List[Document]:
        namespace = namespace or self.namespace
        print("\nRunning similarity search")

        kwargs = {"k": k}
        if namespace:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            kwargs["filter"] = Filter(
                must=[FieldCondition(key="metadata.namespace", match=MatchValue(value=namespace))]
            )

        return self.vector_store.similarity_search(query, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        namespace: str = None,
    ):
        namespace = namespace or self.namespace
        print("\nRunning similarity search with score")

        kwargs = {"k": k}
        if namespace:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            kwargs["filter"] = Filter(
                must=[FieldCondition(key="metadata.namespace", match=MatchValue(value=namespace))]
            )

        return self.vector_store.similarity_search_with_score(query, **kwargs)

    def drop_collection(self):
        return drop_collection(self.collection_name, self.url)

    def drop_all_collections(self, confirm=False):
        return drop_all_collections(self.url, confirm)

    # Qdrant has no concept of databases — this is a no-op for compatibility
    def drop_database(self, confirm=False):
        print("Qdrant does not use databases. Use drop_all_collections() instead.")
        return False