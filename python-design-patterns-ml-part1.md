# Python Design Patterns for AI/ML Production Systems - Part 1

**10 Most Common Design Patterns for Production-Grade ML Applications**

*For experienced engineers with .NET background transitioning to Python AI/ML projects*

---

## Table of Contents
1. [Repository Pattern](#1-repository-pattern)
2. [Factory Pattern](#2-factory-pattern)
3. [Strategy Pattern](#3-strategy-pattern)
4. [Singleton Pattern](#4-singleton-pattern)
5. [Observer Pattern](#5-observer-pattern)
6. [Decorator Pattern](#6-decorator-pattern)
7. [Adapter Pattern](#7-adapter-pattern)
8. [Pipeline Pattern](#8-pipeline-pattern)
9. [Chain of Responsibility](#9-chain-of-responsibility)
10. [Builder Pattern](#10-builder-pattern)

---

## 1. Repository Pattern

**Purpose:** Abstract data access logic from business logic. Critical for ML systems that interact with multiple data sources (vector databases, SQL, document stores, model registries).

**When to Use:**
- Working with multiple data sources (vector DB, SQL, blob storage)
- Need to swap data sources (dev vs. prod)
- Want testable code (mock repositories in tests)
- Building RAG systems with multiple storage backends

**Python Implementation for ML/AI:**

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime

# Abstract base repository
class VectorRepository(ABC):
    """Abstract repository for vector storage operations"""
    
    @abstractmethod
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add vector to storage"""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def get_by_id(self, id: str) -> Optional[Dict]:
        """Get vector by ID"""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete vector by ID"""
        pass
    
    @abstractmethod
    def batch_add(self, vectors: List[Dict]) -> None:
        """Batch add vectors for performance"""
        pass


# Concrete implementation for Pinecone
class PineconeVectorRepository(VectorRepository):
    """Pinecone implementation of vector repository"""
    
    def __init__(self, index_name: str, api_key: str, environment: str):
        import pinecone
        from pinecone import Pinecone
        
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
    
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add single vector to Pinecone"""
        self.index.upsert(
            vectors=[{
                'id': id,
                'values': vector.tolist(),
                'metadata': metadata
            }]
        )
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search Pinecone index"""
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            }
            for match in results['matches']
        ]
    
    def get_by_id(self, id: str) -> Optional[Dict]:
        """Fetch vector by ID"""
        results = self.index.fetch(ids=[id])
        
        if id in results['vectors']:
            vec = results['vectors'][id]
            return {
                'id': id,
                'values': vec['values'],
                'metadata': vec.get('metadata', {})
            }
        return None
    
    def delete(self, id: str) -> bool:
        """Delete vector from Pinecone"""
        try:
            self.index.delete(ids=[id])
            return True
        except Exception:
            return False
    
    def batch_add(self, vectors: List[Dict]) -> None:
        """Batch upsert for efficiency"""
        # Pinecone recommends batches of 100
        batch_size = 100
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            formatted_vectors = [
                {
                    'id': vec['id'],
                    'values': vec['vector'].tolist(),
                    'metadata': vec.get('metadata', {})
                }
                for vec in batch
            ]
            
            self.index.upsert(vectors=formatted_vectors)


# Concrete implementation for Qdrant
class QdrantVectorRepository(VectorRepository):
    """Qdrant implementation of vector repository"""
    
    def __init__(self, collection_name: str, url: str = "http://localhost:6333"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        
        # Create collection if doesn't exist
        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
    
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add single vector to Qdrant"""
        from qdrant_client.models import PointStruct
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=vector.tolist(),
                    payload=metadata
                )
            ]
        )
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search Qdrant collection"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'metadata': hit.payload
            }
            for hit in results
        ]
    
    def get_by_id(self, id: str) -> Optional[Dict]:
        """Retrieve point by ID"""
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id]
        )
        
        if results:
            point = results[0]
            return {
                'id': point.id,
                'values': point.vector,
                'metadata': point.payload
            }
        return None
    
    def delete(self, id: str) -> bool:
        """Delete point from Qdrant"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[id]
            )
            return True
        except Exception:
            return False
    
    def batch_add(self, vectors: List[Dict]) -> None:
        """Batch add for performance"""
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=vec['id'],
                vector=vec['vector'].tolist(),
                payload=vec.get('metadata', {})
            )
            for vec in vectors
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )


# In-memory implementation for testing
class InMemoryVectorRepository(VectorRepository):
    """In-memory repository for testing"""
    
    def __init__(self):
        self.storage: Dict[str, Dict] = {}
    
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        self.storage[id] = {
            'vector': vector,
            'metadata': metadata
        }
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Simple cosine similarity search"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not self.storage:
            return []
        
        # Calculate similarities
        results = []
        for id, data in self.storage.items():
            similarity = cosine_similarity(
                query_vector.reshape(1, -1),
                data['vector'].reshape(1, -1)
            )[0][0]
            
            results.append({
                'id': id,
                'score': float(similarity),
                'metadata': data['metadata']
            })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_by_id(self, id: str) -> Optional[Dict]:
        if id in self.storage:
            return {
                'id': id,
                'values': self.storage[id]['vector'],
                'metadata': self.storage[id]['metadata']
            }
        return None
    
    def delete(self, id: str) -> bool:
        if id in self.storage:
            del self.storage[id]
            return True
        return False
    
    def batch_add(self, vectors: List[Dict]) -> None:
        for vec in vectors:
            self.add(vec['id'], vec['vector'], vec.get('metadata', {}))


# Service that uses repository (Dependency Injection)
class DocumentService:
    """Service layer using repository pattern"""
    
    def __init__(self, vector_repo: VectorRepository):
        """
        Inject repository dependency
        
        This allows switching between Pinecone, Qdrant, or in-memory
        without changing service code
        """
        self.vector_repo = vector_repo
    
    def index_document(self, doc_id: str, content: str, 
                      embedder: Any, metadata: Dict) -> None:
        """Index document using injected repository"""
        # Generate embedding
        vector = embedder.embed(content)
        
        # Add to repository (works with any implementation)
        self.vector_repo.add(
            id=doc_id,
            vector=vector,
            metadata={
                'content': content,
                'indexed_at': datetime.utcnow().isoformat(),
                **metadata
            }
        )
    
    def search_documents(self, query: str, embedder: Any, 
                        top_k: int = 5) -> List[Dict]:
        """Search documents using injected repository"""
        query_vector = embedder.embed(query)
        return self.vector_repo.search(query_vector, top_k)
    
    def batch_index_documents(self, documents: List[Dict], 
                             embedder: Any) -> None:
        """Batch index for performance"""
        vectors = []
        
        for doc in documents:
            vector = embedder.embed(doc['content'])
            vectors.append({
                'id': doc['id'],
                'vector': vector,
                'metadata': {
                    'content': doc['content'],
                    **doc.get('metadata', {})
                }
            })
        
        self.vector_repo.batch_add(vectors)


# Usage example
if __name__ == "__main__":
    # Development: Use in-memory
    dev_repo = InMemoryVectorRepository()
    dev_service = DocumentService(dev_repo)
    
    # Production: Use Pinecone
    prod_repo = PineconeVectorRepository(
        index_name="prod-index",
        api_key="your-api-key",
        environment="us-west1-gcp"
    )
    prod_service = DocumentService(prod_repo)
    
    # Testing: Use in-memory (fast, no external dependencies)
    test_repo = InMemoryVectorRepository()
    test_service = DocumentService(test_repo)
    
    # All services have identical interface!
    # Easy to swap implementations
```

**Benefits for ML/AI Projects:**
- ✅ Swap vector databases without changing business logic
- ✅ Easy unit testing with in-memory implementation
- ✅ Support multiple environments (dev/staging/prod)
- ✅ Isolate vendor-specific code
- ✅ Enable A/B testing of different vector stores

---

## 2. Factory Pattern

**Purpose:** Create objects without specifying exact class. Essential for ML systems that need to instantiate different models, embedders, or data loaders based on configuration.

**When to Use:**
- Multiple model types (GPT-4, Claude, Llama)
- Different embedding models (OpenAI, Cohere, local)
- Various data sources (PDF, DOCX, HTML)
- Environment-based configuration

**Python Implementation for ML/AI:**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
import numpy as np

# Abstract product
class Embedder(ABC):
    """Abstract base class for all embedders"""
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding for efficiency"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension"""
        pass


# Concrete products
class OpenAIEmbedder(Embedder):
    """OpenAI embeddings implementation"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if "small" in model else 3072
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed (up to 2048 texts per request)"""
        batch_size = 2048
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class CohereEmbedder(Embedder):
    """Cohere embeddings implementation"""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        import cohere
        
        self.client = cohere.Client(api_key)
        self.model = model
        self._dimension = 1024
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using Cohere API"""
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document"
        )
        return np.array(response.embeddings[0])
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed (up to 96 texts per request)"""
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document"
            )
            
            all_embeddings.extend(response.embeddings)
        
        return np.array(all_embeddings)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedder(Embedder):
    """Local embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding locally"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed locally"""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=True
        )
    
    @property
    def dimension(self) -> int:
        return self._dimension


# Enum for embedder types
class EmbedderType(Enum):
    OPENAI_SMALL = "openai-small"
    OPENAI_LARGE = "openai-large"
    COHERE = "cohere"
    LOCAL = "local"


# Factory
class EmbedderFactory:
    """Factory for creating embedders based on configuration"""
    
    @staticmethod
    def create(embedder_type: EmbedderType, config: Dict[str, Any]) -> Embedder:
        """
        Create embedder instance based on type
        
        Args:
            embedder_type: Type of embedder to create
            config: Configuration dictionary
            
        Returns:
            Embedder instance
            
        Raises:
            ValueError: If embedder type not supported
        """
        if embedder_type == EmbedderType.OPENAI_SMALL:
            return OpenAIEmbedder(
                api_key=config['api_key'],
                model="text-embedding-3-small"
            )
        
        elif embedder_type == EmbedderType.OPENAI_LARGE:
            return OpenAIEmbedder(
                api_key=config['api_key'],
                model="text-embedding-3-large"
            )
        
        elif embedder_type == EmbedderType.COHERE:
            return CohereEmbedder(
                api_key=config['api_key'],
                model=config.get('model', 'embed-english-v3.0')
            )
        
        elif embedder_type == EmbedderType.LOCAL:
            return LocalEmbedder(
                model_name=config.get('model_name', 'all-MiniLM-L6-v2')
            )
        
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")
    
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> Embedder:
        """
        Create embedder from configuration dictionary
        
        Config format:
        {
            "type": "openai-small",
            "api_key": "...",
            "model": "..."  # optional
        }
        """
        embedder_type = EmbedderType(config_dict['type'])
        return EmbedderFactory.create(embedder_type, config_dict)


# Abstract factory for creating families of related objects
class LLMProvider(ABC):
    """Abstract LLM provider"""
    
    @abstractmethod
    def create_embedder(self) -> Embedder:
        """Create embedder for this provider"""
        pass
    
    @abstractmethod
    def create_chat_model(self) -> Any:
        """Create chat model for this provider"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider creates OpenAI-specific products"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def create_embedder(self) -> Embedder:
        return OpenAIEmbedder(self.api_key, "text-embedding-3-small")
    
    def create_chat_model(self) -> Any:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=self.api_key, model="gpt-4")


class CohereProvider(LLMProvider):
    """Cohere provider creates Cohere-specific products"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def create_embedder(self) -> Embedder:
        return CohereEmbedder(self.api_key)
    
    def create_chat_model(self) -> Any:
        from langchain_cohere import ChatCohere
        return ChatCohere(cohere_api_key=self.api_key)


class LocalProvider(LLMProvider):
    """Local provider creates local models"""
    
    def create_embedder(self) -> Embedder:
        return LocalEmbedder("all-MiniLM-L6-v2")
    
    def create_chat_model(self) -> Any:
        from langchain_community.llms import Ollama
        return Ollama(model="llama2")


# Configuration-driven factory
class ConfigurableEmbedderFactory:
    """Factory that reads from configuration files"""
    
    def __init__(self, config_path: str):
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def create_embedder_for_env(self, environment: str = "production") -> Embedder:
        """
        Create embedder based on environment
        
        Config YAML format:
        environments:
          development:
            embedder:
              type: local
              model_name: all-MiniLM-L6-v2
          
          production:
            embedder:
              type: openai-small
              api_key: ${OPENAI_API_KEY}
        """
        env_config = self.config['environments'][environment]
        embedder_config = env_config['embedder']
        
        # Replace environment variables
        for key, value in embedder_config.items():
            if isinstance(value, str) and value.startswith('${'):
                env_var = value[2:-1]  # Extract var name
                embedder_config[key] = os.environ.get(env_var)
        
        return EmbedderFactory.create_from_config(embedder_config)


# Usage examples
if __name__ == "__main__":
    # Simple factory usage
    config = {
        'type': 'openai-small',
        'api_key': 'your-api-key'
    }
    embedder = EmbedderFactory.create_from_config(config)
    
    # Embed text
    vector = embedder.embed("Hello, world!")
    print(f"Embedding dimension: {embedder.dimension}")
    
    # Abstract factory usage (create family of products)
    provider = OpenAIProvider(api_key="your-api-key")
    embedder = provider.create_embedder()
    chat_model = provider.create_chat_model()
    
    # Easy to swap providers
    # provider = CohereProvider(api_key="cohere-key")
    # provider = LocalProvider()  # No API key needed!
    
    # Configuration-driven factory
    factory = ConfigurableEmbedderFactory("config.yml")
    
    # Development uses local embedder (free, fast)
    dev_embedder = factory.create_embedder_for_env("development")
    
    # Production uses OpenAI (high quality)
    prod_embedder = factory.create_embedder_for_env("production")
```

**Configuration File (config.yml):**
```yaml
environments:
  development:
    embedder:
      type: local
      model_name: all-MiniLM-L6-v2
  
  staging:
    embedder:
      type: openai-small
      api_key: ${OPENAI_API_KEY}
  
  production:
    embedder:
      type: openai-large
      api_key: ${OPENAI_API_KEY}
```

**Benefits for ML/AI Projects:**
- ✅ Easy model swapping (dev/staging/prod)
- ✅ Configuration-driven model selection
- ✅ Support A/B testing different models
- ✅ Isolate vendor-specific code
- ✅ Consistent interface across all models

---

## 3. Strategy Pattern

**Purpose:** Define a family of algorithms, encapsulate each one, and make them interchangeable. Critical for ML systems with multiple approaches to same problem (different retrieval strategies, reranking methods, chunking algorithms).

**When to Use:**
- Multiple retrieval strategies (vector, keyword, hybrid)
- Different chunking methods (fixed, semantic, recursive)
- Various reranking algorithms
- Multiple prompt templates

**Python Implementation for ML/AI:**

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

# Strategy interface
class RetrievalStrategy(ABC):
    """Abstract retrieval strategy"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents for query"""
        pass


# Concrete strategies
class VectorRetrievalStrategy(RetrievalStrategy):
    """Pure vector similarity retrieval"""
    
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve using vector similarity"""
        # Embed query
        query_vector = self.embedder.embed(query)
        
        # Search vector store
        results = self.vector_store.search(query_vector, top_k)
        
        return results


class KeywordRetrievalStrategy(RetrievalStrategy):
    """BM25 keyword retrieval"""
    
    def __init__(self, documents: List[str]):
        from rank_bm25 import BM25Okapi
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve using BM25"""
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Format results
        results = [
            {
                'id': str(idx),
                'content': self.documents[idx],
                'score': float(scores[idx])
            }
            for idx in top_indices
        ]
        
        return results


class HybridRetrievalStrategy(RetrievalStrategy):
    """Hybrid retrieval combining vector and keyword search"""
    
    def __init__(self, vector_strategy: VectorRetrievalStrategy, 
                 keyword_strategy: KeywordRetrievalStrategy,
                 vector_weight: float = 0.7):
        """
        Args:
            vector_strategy: Vector retrieval strategy
            keyword_strategy: Keyword retrieval strategy
            vector_weight: Weight for vector scores (0-1)
        """
        self.vector_strategy = vector_strategy
        self.keyword_strategy = keyword_strategy
        self.vector_weight = vector_weight
        self.keyword_weight = 1 - vector_weight
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve using hybrid approach"""
        # Get results from both strategies
        vector_results = self.vector_strategy.retrieve(query, top_k * 2)
        keyword_results = self.keyword_strategy.retrieve(query, top_k * 2)
        
        # Combine scores using Reciprocal Rank Fusion
        combined_scores = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = result['id']
            # RRF score
            score = self.vector_weight / (60 + rank)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score
        
        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result['id']
            score = self.keyword_weight / (60 + rank)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score
        
        # Sort by combined score
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Format results
        doc_map = {r['id']: r for r in vector_results + keyword_results}
        
        results = [
            {
                **doc_map[doc_id],
                'combined_score': score
            }
            for doc_id, score in sorted_docs
        ]
        
        return results


class SemanticRerankerStrategy(RetrievalStrategy):
    """Retrieval with semantic reranking"""
    
    def __init__(self, base_strategy: RetrievalStrategy, reranker_model):
        self.base_strategy = base_strategy
        self.reranker = reranker_model
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve and rerank"""
        # Get more candidates than needed
        candidates = self.base_strategy.retrieve(query, top_k * 3)
        
        # Rerank using cross-encoder
        query_doc_pairs = [
            (query, candidate['content'])
            for candidate in candidates
        ]
        
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # Combine with original scores
        for candidate, rerank_score in zip(candidates, rerank_scores):
            candidate['rerank_score'] = float(rerank_score)
            candidate['final_score'] = (
                0.3 * candidate.get('score', 0) + 
                0.7 * rerank_score
            )
        
        # Sort by final score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return candidates[:top_k]


# Context class that uses strategies
class RAGRetriever:
    """RAG retriever that can switch strategies"""
    
    def __init__(self, strategy: RetrievalStrategy):
        """
        Initialize with a retrieval strategy
        
        Strategy can be changed at runtime
        """
        self._strategy = strategy
    
    @property
    def strategy(self) -> RetrievalStrategy:
        """Get current strategy"""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: RetrievalStrategy) -> None:
        """
        Set retrieval strategy
        
        Allows runtime strategy switching
        """
        self._strategy = strategy
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve documents using current strategy
        
        Delegates to strategy implementation
        """
        return self._strategy.retrieve(query, top_k)


# Strategy for chunking
class ChunkingStrategy(ABC):
    """Abstract chunking strategy"""
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks"""
        pass


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """Fixed-size chunking"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """Split into fixed-size chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks


class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic chunking using embeddings"""
    
    def __init__(self, embedder, threshold: float = 0.5):
        self.embedder = embedder
        self.threshold = threshold
    
    def chunk(self, text: str) -> List[str]:
        """Split at semantic boundaries"""
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', text)
        
        # Embed all sentences
        embeddings = self.embedder.embed_batch(sentences)
        
        # Find semantic boundaries
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            if similarity < self.threshold:
                # Semantic boundary - start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                # Continue current chunk
                current_chunk.append(sentences[i])
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class RecursiveChunkingStrategy(ChunkingStrategy):
    """Recursive chunking respecting document structure"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str) -> List[str]:
        """Recursively split using hierarchical separators"""
        return self._split_text(text, self.separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting logic"""
        if not separators or len(text) <= self.chunk_size:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = text.split(separator)
        
        # Merge small splits and recursively split large ones
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if len(split) > self.chunk_size:
                    # Recursively split with next separator
                    chunks.extend(
                        self._split_text(split, remaining_separators)
                    )
                    current_chunk = ""
                else:
                    current_chunk = split + separator
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# Document processor using chunking strategy
class DocumentProcessor:
    """Process documents with configurable chunking"""
    
    def __init__(self, chunking_strategy: ChunkingStrategy):
        self.chunking_strategy = chunking_strategy
    
    def process(self, text: str) -> List[str]:
        """Process document using current chunking strategy"""
        return self.chunking_strategy.chunk(text)
    
    def set_strategy(self, strategy: ChunkingStrategy):
        """Change chunking strategy"""
        self.chunking_strategy = strategy


# Usage examples
if __name__ == "__main__":
    # Retrieval strategies
    vector_strategy = VectorRetrievalStrategy(vector_store, embedder)
    keyword_strategy = KeywordRetrievalStrategy(documents)
    
    # Start with vector retrieval
    retriever = RAGRetriever(strategy=vector_strategy)
    results = retriever.retrieve("What is RAG?", top_k=5)
    
    # Switch to keyword retrieval
    retriever.strategy = keyword_strategy
    results = retriever.retrieve("What is RAG?", top_k=5)
    
    # Switch to hybrid (best of both)
    hybrid_strategy = HybridRetrievalStrategy(
        vector_strategy,
        keyword_strategy,
        vector_weight=0.7
    )
    retriever.strategy = hybrid_strategy
    results = retriever.retrieve("What is RAG?", top_k=5)
    
    # Chunking strategies
    processor = DocumentProcessor(
        chunking_strategy=FixedSizeChunkingStrategy(1000, 200)
    )
    chunks = processor.process(long_document)
    
    # Switch to semantic chunking for better quality
    processor.set_strategy(
        SemanticChunkingStrategy(embedder, threshold=0.5)
    )
    semantic_chunks = processor.process(long_document)
    
    # A/B test different strategies
    strategies = {
        'vector': VectorRetrievalStrategy(vector_store, embedder),
        'keyword': KeywordRetrievalStrategy(documents),
        'hybrid': HybridRetrievalStrategy(vector_strategy, keyword_strategy)
    }
    
    for name, strategy in strategies.items():
        retriever.strategy = strategy
        results = retriever.retrieve(test_query)
        print(f"{name}: {evaluate_results(results)}")
```

**Benefits for ML/AI Projects:**
- ✅ Easy A/B testing of different approaches
- ✅ Runtime strategy switching
- ✅ Compare performance of different methods
- ✅ Clean separation of algorithm implementations
- ✅ Easy to add new strategies without changing existing code

---

## 4. Singleton Pattern

**Purpose:** Ensure a class has only one instance and provide global access. Critical for expensive resources like model loaders, database connections, and configuration managers.

**When to Use:**
- Model loading (heavy models should load once)
- Database connection pools
- Configuration management
- Logging systems
- Cache managers

**Python Implementation for ML/AI:**

```python
from typing import Optional, Dict, Any
import threading
from functools import wraps

# Thread-safe singleton metaclass
class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass
    
    Usage:
        class MyClass(metaclass=SingletonMeta):
            pass
    """
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """
        Thread-safe singleton instance creation
        
        Double-checked locking pattern for performance
        """
        # First check (without lock for performance)
        if cls not in cls._instances:
            # Acquire lock for thread safety
            with cls._lock:
                # Second check (with lock)
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        
        return cls._instances[cls]


# Singleton decorator (alternative approach)
def singleton(cls):
    """
    Singleton decorator
    
    Usage:
        @singleton
        class MyClass:
            pass
    """
    instances = {}
    lock = threading.Lock()
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


# Example 1: Model Manager Singleton
class ModelManager(metaclass=SingletonMeta):
    """
    Singleton for managing ML models
    
    Ensures models are loaded only once and shared across application
    """
    
    def __init__(self):
        """
        Initialize model manager
        
        Only called once due to Singleton pattern
        """
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()
        print("ModelManager initialized (happens only once)")
    
    def load_model(self, model_name: str, model_path: str) -> Any:
        """
        Load model or return cached instance
        
        Args:
            model_name: Identifier for the model
            model_path: Path to model file
            
        Returns:
            Model instance
        """
        if model_name not in self._models:
            with self._lock:
                if model_name not in self._models:
                    print(f"Loading model: {model_name}")
                    
                    # Simulate expensive model loading
                    import time
                    time.sleep(2)  # In reality, this could be 10-60 seconds
                    
                    # Load model (example with sentence-transformers)
                    if "embedding" in model_name:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer(model_path)
                    else:
                        # Load other model types
                        model = self._load_custom_model(model_path)
                    
                    self._models[model_name] = model
                    print(f"Model {model_name} loaded and cached")
        
        return self._models[model_name]
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get cached model if exists"""
        return self._models.get(model_name)
    
    def unload_model(self, model_name: str) -> bool:
        """Unload model to free memory"""
        if model_name in self._models:
            with self._lock:
                if model_name in self._models:
                    del self._models[model_name]
                    print(f"Model {model_name} unloaded")
                    return True
        return False
    
    def list_loaded_models(self) -> list:
        """List all currently loaded models"""
        return list(self._models.keys())
    
    def _load_custom_model(self, model_path: str) -> Any:
        """Load custom model (placeholder)"""
        # Implementation for loading custom models
        pass


# Example 2: Configuration Manager Singleton
class ConfigManager(metaclass=SingletonMeta):
    """
    Singleton for application configuration
    
    Loads config once and provides global access
    """
    
    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()
        print(f"Configuration loaded from {config_path}")
    
    def _load_config(self):
        """Load configuration from file"""
        import yaml
        
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found, using defaults")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'model': {
                'embedder': 'openai-small',
                'llm': 'gpt-4'
            },
            'retrieval': {
                'top_k': 5,
                'strategy': 'hybrid'
            },
            'database': {
                'vector_store': 'pinecone',
                'cache': 'redis'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        
        Example:
            config.get('model.embedder')  # Returns 'openai-small'
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        print("Configuration reloaded")


# Example 3: Database Connection Pool Singleton
class DatabasePool(metaclass=SingletonMeta):
    """
    Singleton for database connection pooling
    
    Manages connection pool for application
    """
    
    def __init__(self, database_url: str, pool_size: int = 10):
        """
        Initialize connection pool
        
        Args:
            database_url: Database connection string
            pool_size: Number of connections in pool
        """
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool
        
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600
        )
        
        print(f"Database pool created with {pool_size} connections")
    
    def get_connection(self):
        """Get connection from pool"""
        return self.engine.connect()
    
    def execute(self, query: str):
        """Execute query using pool connection"""
        with self.engine.connect() as conn:
            return conn.execute(query)


# Example 4: Logger Singleton
@singleton
class Logger:
    """
    Singleton logger using decorator approach
    
    Provides centralized logging for application
    """
    
    def __init__(self, log_file: str = "app.log"):
        """
        Initialize logger
        
        Args:
            log_file: Path to log file
        """
        import logging
        
        self.logger = logging.getLogger("RAGApplication")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        print(f"Logger initialized with log file: {log_file}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)


# Example 5: Cache Manager Singleton
class CacheManager(metaclass=SingletonMeta):
    """
    Singleton for managing application cache
    
    Provides centralized caching for embeddings, queries, etc.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize cache manager
        
        Args:
            redis_url: Redis connection URL
        """
        import redis
        
        self.redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            max_connections=50
        )
        
        # Test connection
        self.redis_client.ping()
        print("Cache manager initialized")
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        return self.redis_client.get(key)
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache with TTL"""
        self.redis_client.setex(key, ttl, value)
    
    def delete(self, key: str):
        """Delete key from cache"""
        self.redis_client.delete(key)
    
    def clear_all(self):
        """Clear entire cache"""
        self.redis_client.flushdb()


# Usage examples
if __name__ == "__main__":
    # Example 1: Model Manager
    # First call - creates instance and loads model
    model_manager1 = ModelManager()
    embedder1 = model_manager1.load_model(
        "embedding-model",
        "all-MiniLM-L6-v2"
    )
    
    # Second call - returns same instance, model already loaded!
    model_manager2 = ModelManager()
    embedder2 = model_manager2.load_model(
        "embedding-model",
        "all-MiniLM-L6-v2"
    )
    
    # Verify it's the same instance
    assert model_manager1 is model_manager2  # True
    assert embedder1 is embedder2  # True
    
    print(f"Loaded models: {model_manager1.list_loaded_models()}")
    
    # Example 2: Configuration Manager
    config1 = ConfigManager("config.yml")
    config2 = ConfigManager()  # Same instance, config already loaded
    
    assert config1 is config2  # True
    
    # Get configuration
    embedder_type = config1.get('model.embedder')
    top_k = config1.get('retrieval.top_k', default=5)
    
    # Example 3: Logger (decorator-based singleton)
    logger1 = Logger("app.log")
    logger2 = Logger()  # Same instance
    
    assert logger1 is logger2  # True
    
    logger1.info("Application started")
    logger2.info("This goes to the same logger!")
    
    # Example 4: Thread-safe access
    import concurrent.futures
    
    def worker(worker_id):
        """Worker function accessing singleton"""
        model_manager = ModelManager()
        logger = Logger()
        config = ConfigManager()
        
        logger.info(f"Worker {worker_id} using ModelManager")
        return model_manager is ModelManager()
    
    # Run multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        results = [f.result() for f in futures]
    
    # All workers got the same singleton instance
    assert all(results)  # All True
    
    print("All workers accessed the same singleton instance!")
```

**Comparison with .NET Singleton:**

```csharp
// .NET Singleton (for reference)
public sealed class ModelManager
{
    private static readonly Lazy<ModelManager> instance = 
        new Lazy<ModelManager>(() => new ModelManager());
    
    private ModelManager() { }
    
    public static ModelManager Instance => instance.Value;
}

// Usage
var manager = ModelManager.Instance;
```

**Python equivalent:**
```python
# Python Singleton (much simpler!)
class ModelManager(metaclass=SingletonMeta):
    def __init__(self):
        pass

# Usage
manager = ModelManager()  # Automatically singleton!
```

**Benefits for ML/AI Projects:**
- ✅ Load expensive models once
- ✅ Share connection pools across application
- ✅ Centralized configuration management
- ✅ Thread-safe resource access
- ✅ Reduced memory footprint

**Cautions:**
- ⚠️ Can make testing harder (use dependency injection for testability)
- ⚠️ Global state can lead to tight coupling
- ⚠️ Consider alternatives like dependency injection containers

---

*Continue with remaining 6 patterns...*

Would you like me to continue with patterns 5-10 in this file, or would you prefer I create a second part?
