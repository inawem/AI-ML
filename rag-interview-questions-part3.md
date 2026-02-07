# RAG & LangChain Interview Questions - Part 3 (Questions 15-20)

**Advanced Topics + Python Production Practices**

---

## Question 15: How do Python-based RAG applications save and manage data in production?

**Answer:**

Production Python applications require robust data persistence strategies across multiple layers. Let's explore comprehensive data management patterns for RAG systems.

**Data Layers in RAG Applications:**

```mermaid
graph TB
    subgraph "Application Data Layers"
        A[User Requests] --> B[Application Layer]
        
        B --> C[Vector Embeddings]
        B --> D[Document Storage]
        B --> E[Metadata & Indexes]
        B --> F[Application State]
        B --> G[Cache Layer]
        B --> H[Audit Logs]
    end
    
    subgraph "Storage Solutions"
        C --> C1[(Vector DB<br/>Pinecone/Qdrant/Weaviate)]
        D --> D1[(Object Storage<br/>S3/MinIO/GCS)]
        E --> E2[(SQL Database<br/>PostgreSQL/MySQL)]
        F --> F1[(NoSQL<br/>MongoDB/DynamoDB)]
        G --> G1[(Redis<br/>Memcached)]
        H --> H1[(Time-Series DB<br/>InfluxDB/TimescaleDB)]
    end
```

### 1. Vector Embeddings Storage (Primary RAG Data)

**A. Using PostgreSQL with pgvector (Self-hosted)**

```python
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import numpy as np

Base = declarative_base()

class Document(Base):
    """Document storage with vector embeddings"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    source = Column(String(500))
    embedding = Column(Vector(1536))  # OpenAI embedding dimension
    metadata_json = Column(Text)  # JSON string for flexible metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DocumentRepository:
    """Production-grade repository pattern for document management"""
    
    def __init__(self, database_url: str):
        """
        Initialize database connection with connection pooling
        
        Args:
            database_url: PostgreSQL connection string
                Example: "postgresql://user:password@localhost:5432/ragdb"
        """
        self.engine = create_engine(
            database_url,
            pool_size=10,          # Number of persistent connections
            max_overflow=20,       # Additional connections when pool exhausted
            pool_timeout=30,       # Seconds to wait for connection
            pool_recycle=3600,     # Recycle connections after 1 hour
            echo=False             # Set True for SQL debugging
        )
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def add_document(self, content: str, embedding: np.ndarray, 
                     source: str = None, metadata: dict = None) -> int:
        """
        Add document with embedding to database
        
        Args:
            content: Document text content
            embedding: Vector embedding (numpy array)
            source: Source file/URL
            metadata: Additional metadata as dictionary
            
        Returns:
            Document ID
        """
        session = self.SessionLocal()
        try:
            doc = Document(
                content=content,
                source=source,
                embedding=embedding.tolist(),  # Convert numpy to list
                metadata_json=json.dumps(metadata) if metadata else None
            )
            
            session.add(doc)
            session.commit()
            session.refresh(doc)
            
            return doc.id
            
        except Exception as e:
            session.rollback()
            raise Exception(f"Failed to add document: {str(e)}")
        finally:
            session.close()
    
    def batch_add_documents(self, documents: list) -> list:
        """
        Efficiently add multiple documents in batch
        
        Args:
            documents: List of dicts with keys: content, embedding, source, metadata
            
        Returns:
            List of created document IDs
        """
        session = self.SessionLocal()
        try:
            doc_objects = [
                Document(
                    content=doc['content'],
                    source=doc.get('source'),
                    embedding=doc['embedding'].tolist(),
                    metadata_json=json.dumps(doc.get('metadata', {}))
                )
                for doc in documents
            ]
            
            # Bulk insert - much faster than individual inserts
            session.bulk_save_objects(doc_objects, return_defaults=True)
            session.commit()
            
            return [doc.id for doc in doc_objects]
            
        except Exception as e:
            session.rollback()
            raise Exception(f"Batch insert failed: {str(e)}")
        finally:
            session.close()
    
    def similarity_search(self, query_embedding: np.ndarray, limit: int = 5) -> list:
        """
        Vector similarity search using cosine distance
        
        Args:
            query_embedding: Query vector
            limit: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        session = self.SessionLocal()
        try:
            # Use pgvector's cosine distance operator
            results = session.query(
                Document,
                Document.embedding.cosine_distance(query_embedding.tolist()).label('distance')
            ).order_by('distance').limit(limit).all()
            
            # Convert distance to similarity (1 - distance)
            return [(doc, 1 - distance) for doc, distance in results]
            
        finally:
            session.close()
    
    def get_by_metadata(self, metadata_key: str, metadata_value: str) -> list:
        """
        Query documents by metadata fields
        
        Uses JSONB operators for efficient metadata querying
        """
        session = self.SessionLocal()
        try:
            # PostgreSQL JSONB query
            results = session.query(Document).filter(
                Document.metadata_json.op('->>')('key') == metadata_value
            ).all()
            
            return results
        finally:
            session.close()
    
    def update_document(self, doc_id: int, content: str = None, 
                       embedding: np.ndarray = None, metadata: dict = None):
        """Update existing document"""
        session = self.SessionLocal()
        try:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            
            if not doc:
                raise ValueError(f"Document {doc_id} not found")
            
            if content:
                doc.content = content
            if embedding is not None:
                doc.embedding = embedding.tolist()
            if metadata:
                doc.metadata_json = json.dumps(metadata)
            
            doc.updated_at = datetime.utcnow()
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise Exception(f"Update failed: {str(e)}")
        finally:
            session.close()
    
    def delete_document(self, doc_id: int):
        """Soft delete or hard delete document"""
        session = self.SessionLocal()
        try:
            # Hard delete
            session.query(Document).filter(Document.id == doc_id).delete()
            session.commit()
            
            # For soft delete, add a 'deleted_at' column and update it instead
            
        except Exception as e:
            session.rollback()
            raise Exception(f"Delete failed: {str(e)}")
        finally:
            session.close()

# Usage Example
repo = DocumentRepository("postgresql://user:pass@localhost:5432/ragdb")

# Add single document
doc_id = repo.add_document(
    content="RAG combines retrieval with generation...",
    embedding=embedding_vector,
    source="docs/rag_intro.pdf",
    metadata={"category": "AI", "author": "John Doe"}
)

# Batch add (10x faster for large imports)
documents = [
    {
        "content": "Document 1...",
        "embedding": emb1,
        "source": "file1.pdf",
        "metadata": {"category": "tech"}
    },
    # ... more documents
]
repo.batch_add_documents(documents)

# Search
results = repo.similarity_search(query_embedding, limit=5)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc.content[:100]}...")
```

**B. Using Pinecone (Managed Vector DB)**

```python
import pinecone
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorStore:
    """Production-grade Pinecone integration"""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone connection
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-west1-gcp')
            index_name: Name of the index to use/create
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        
        # Connect to index
        self.index = self.pc.Index(index_name)
    
    def upsert_documents(self, documents: list):
        """
        Insert or update documents in batches
        
        Args:
            documents: List of dicts with keys: id, embedding, metadata
        """
        # Pinecone requires specific format
        vectors = [
            {
                "id": doc["id"],
                "values": doc["embedding"].tolist(),
                "metadata": doc.get("metadata", {})
            }
            for doc in documents
        ]
        
        # Batch upsert (100 vectors at a time recommended)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5, 
              filter_dict: dict = None) -> list:
        """
        Query with optional metadata filtering
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_dict: Metadata filters (e.g., {"category": "AI"})
            
        Returns:
            List of results with scores and metadata
        """
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        return results['matches']
    
    def delete_by_filter(self, filter_dict: dict):
        """Delete vectors matching metadata filter"""
        self.index.delete(filter=filter_dict)
    
    def get_stats(self):
        """Get index statistics"""
        return self.index.describe_index_stats()

# Usage
vector_store = PineconeVectorStore(
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="rag-production"
)

# Upsert with metadata
vector_store.upsert_documents([
    {
        "id": "doc1",
        "embedding": embedding1,
        "metadata": {
            "text": "Content here...",
            "source": "file.pdf",
            "category": "technical",
            "timestamp": "2024-01-15"
        }
    }
])

# Query with filtering
results = vector_store.query(
    query_embedding=query_emb,
    top_k=5,
    filter_dict={"category": "technical"}
)
```

### 2. Document Storage (Original Files)

**Using S3-Compatible Object Storage:**

```python
import boto3
from botocore.exceptions import ClientError
import hashlib
from pathlib import Path

class DocumentStorage:
    """Production document storage using S3/MinIO"""
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        """
        Initialize S3-compatible storage
        
        Args:
            endpoint: S3 endpoint (AWS S3 or MinIO)
            access_key: Access key ID
            secret_key: Secret access key
            bucket: Bucket name
        """
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=boto3.session.Config(
                signature_version='s3v4',
                retries={'max_attempts': 3, 'mode': 'standard'}
            )
        )
        
        self.bucket = bucket
        
        # Create bucket if doesn't exist
        try:
            self.s3_client.head_bucket(Bucket=bucket)
        except ClientError:
            self.s3_client.create_bucket(Bucket=bucket)
    
    def upload_file(self, file_path: str, object_key: str = None, 
                    metadata: dict = None) -> dict:
        """
        Upload file to S3
        
        Args:
            file_path: Local file path
            object_key: S3 object key (default: filename)
            metadata: Custom metadata
            
        Returns:
            Dict with object_key, etag, version_id
        """
        if not object_key:
            object_key = Path(file_path).name
        
        # Calculate MD5 for integrity check
        with open(file_path, 'rb') as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()
        
        # Prepare metadata
        s3_metadata = metadata or {}
        s3_metadata['md5'] = file_md5
        s3_metadata['original_name'] = Path(file_path).name
        
        try:
            # Upload with metadata
            response = self.s3_client.upload_file(
                file_path,
                self.bucket,
                object_key,
                ExtraArgs={
                    'Metadata': s3_metadata,
                    'ServerSideEncryption': 'AES256'  # Encrypt at rest
                }
            )
            
            # Get object metadata
            head = self.s3_client.head_object(Bucket=self.bucket, Key=object_key)
            
            return {
                'object_key': object_key,
                'etag': head['ETag'],
                'version_id': head.get('VersionId'),
                'size': head['ContentLength']
            }
            
        except ClientError as e:
            raise Exception(f"Upload failed: {str(e)}")
    
    def download_file(self, object_key: str, download_path: str):
        """Download file from S3"""
        try:
            self.s3_client.download_file(self.bucket, object_key, download_path)
        except ClientError as e:
            raise Exception(f"Download failed: {str(e)}")
    
    def get_file_stream(self, object_key: str):
        """Get file as stream (memory efficient for large files)"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=object_key)
            return response['Body']
        except ClientError as e:
            raise Exception(f"Stream failed: {str(e)}")
    
    def list_files(self, prefix: str = '', max_keys: int = 1000) -> list:
        """List files in bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            return response.get('Contents', [])
        except ClientError as e:
            raise Exception(f"List failed: {str(e)}")
    
    def delete_file(self, object_key: str):
        """Delete file"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_key)
        except ClientError as e:
            raise Exception(f"Delete failed: {str(e)}")
    
    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> str:
        """
        Generate temporary download URL
        
        Args:
            object_key: S3 object key
            expiration: URL validity in seconds
            
        Returns:
            Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise Exception(f"URL generation failed: {str(e)}")

# Usage
storage = DocumentStorage(
    endpoint="https://s3.amazonaws.com",  # or MinIO endpoint
    access_key="YOUR_ACCESS_KEY",
    secret_key="YOUR_SECRET_KEY",
    bucket="rag-documents"
)

# Upload document
result = storage.upload_file(
    file_path="data/document.pdf",
    object_key="uploads/2024/01/document.pdf",
    metadata={"department": "engineering", "classification": "internal"}
)

# Generate temporary download link (expires in 1 hour)
download_url = storage.generate_presigned_url("uploads/2024/01/document.pdf")
```

### 3. Metadata and Application State (PostgreSQL/MySQL)

```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class IndexingStatus(enum.Enum):
    """Enum for document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentMetadata(Base):
    """Store document metadata separately from vectors"""
    __tablename__ = 'document_metadata'
    
    id = Column(Integer, primary_key=True)
    vector_id = Column(String(100), unique=True, index=True)  # ID in vector DB
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000))  # S3 path
    file_size = Column(Integer)
    file_type = Column(String(50))
    
    # Processing metadata
    status = Column(String(20), default=IndexingStatus.PENDING.value)
    indexed_at = Column(DateTime)
    error_message = Column(String(1000))
    
    # Content metadata
    chunk_count = Column(Integer, default=0)
    language = Column(String(10))
    
    # Business metadata
    department = Column(String(100))
    category = Column(String(100))
    tags = Column(JSON)  # Array of tags
    
    # Access control
    access_level = Column(String(50), default='public')
    created_by = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    """Track individual chunks of documents"""
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('document_metadata.id'))
    chunk_index = Column(Integer)  # Position in document
    vector_id = Column(String(100), unique=True, index=True)  # ID in vector DB
    content_preview = Column(String(500))  # First 500 chars for debugging
    token_count = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    document = relationship("DocumentMetadata", back_populates="chunks")

class UserQuery(Base):
    """Track user queries for analytics"""
    __tablename__ = 'user_queries'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), index=True)
    query_text = Column(Text, nullable=False)
    
    # Retrieved results
    retrieved_doc_ids = Column(JSON)  # List of document IDs
    
    # Response
    answer_text = Column(Text)
    response_time_ms = Column(Integer)
    
    # Feedback
    thumbs_up = Column(Boolean)
    feedback_text = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class MetadataRepository:
    """Repository for metadata operations"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_document_record(self, vector_id: str, filename: str, 
                              file_path: str, metadata: dict) -> int:
        """Create metadata record for new document"""
        session = self.SessionLocal()
        try:
            doc = DocumentMetadata(
                vector_id=vector_id,
                filename=filename,
                file_path=file_path,
                file_size=metadata.get('file_size'),
                file_type=metadata.get('file_type'),
                department=metadata.get('department'),
                category=metadata.get('category'),
                tags=metadata.get('tags', []),
                access_level=metadata.get('access_level', 'public'),
                created_by=metadata.get('created_by')
            )
            
            session.add(doc)
            session.commit()
            session.refresh(doc)
            
            return doc.id
            
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def add_chunks(self, document_id: int, chunks: list):
        """Add chunk records for a document"""
        session = self.SessionLocal()
        try:
            chunk_objects = [
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    vector_id=chunk['vector_id'],
                    content_preview=chunk['content'][:500],
                    token_count=chunk.get('token_count')
                )
                for i, chunk in enumerate(chunks)
            ]
            
            session.bulk_save_objects(chunk_objects)
            
            # Update document chunk count
            doc = session.query(DocumentMetadata).filter_by(id=document_id).first()
            doc.chunk_count = len(chunks)
            doc.status = IndexingStatus.COMPLETED.value
            doc.indexed_at = datetime.utcnow()
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def log_query(self, user_id: str, query: str, retrieved_docs: list, 
                  answer: str, response_time: int):
        """Log user query for analytics"""
        session = self.SessionLocal()
        try:
            query_log = UserQuery(
                user_id=user_id,
                query_text=query,
                retrieved_doc_ids=[doc.id for doc in retrieved_docs],
                answer_text=answer,
                response_time_ms=response_time
            )
            
            session.add(query_log)
            session.commit()
            
        except Exception as e:
            session.rollback()
            # Don't fail the request if logging fails
            print(f"Failed to log query: {e}")
        finally:
            session.close()
    
    def get_popular_queries(self, limit: int = 10) -> list:
        """Get most common queries for analytics"""
        session = self.SessionLocal()
        try:
            from sqlalchemy import func
            
            results = session.query(
                UserQuery.query_text,
                func.count(UserQuery.id).label('count')
            ).group_by(
                UserQuery.query_text
            ).order_by(
                func.count(UserQuery.id).desc()
            ).limit(limit).all()
            
            return [(query, count) for query, count in results]
            
        finally:
            session.close()
```

### 4. Caching Layer (Redis)

```python
import redis
import json
import hashlib
from typing import Optional

class RAGCache:
    """Production-grade caching for RAG system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize Redis connection
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            max_connections=50  # Connection pool size
        )
        
        # Test connection
        self.redis_client.ping()
    
    def _generate_key(self, query: str, prefix: str = "query") -> str:
        """Generate cache key from query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{prefix}:{query_hash}"
    
    def get_cached_response(self, query: str) -> Optional[dict]:
        """
        Get cached response for query
        
        Args:
            query: User query text
            
        Returns:
            Cached response dict or None
        """
        key = self._generate_key(query)
        cached = self.redis_client.get(key)
        
        if cached:
            return json.loads(cached)
        
        return None
    
    def cache_response(self, query: str, response: dict, ttl: int = 3600):
        """
        Cache response with TTL
        
        Args:
            query: User query
            response: Response dict to cache
            ttl: Time to live in seconds (default 1 hour)
        """
        key = self._generate_key(query)
        
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(response)
        )
    
    def cache_embedding(self, text: str, embedding: list, ttl: int = 86400):
        """
        Cache embeddings to avoid recomputing
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector (as list)
            ttl: Time to live (default 24 hours)
        """
        key = self._generate_key(text, prefix="embedding")
        
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(embedding)
        )
    
    def get_cached_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding"""
        key = self._generate_key(text, prefix="embedding")
        cached = self.redis_client.get(key)
        
        if cached:
            return json.loads(cached)
        
        return None
    
    def invalidate_cache(self, pattern: str = "*"):
        """
        Invalidate cache entries matching pattern
        
        Args:
            pattern: Redis key pattern (e.g., "query:*")
        """
        cursor = 0
        while True:
            cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
            
            if keys:
                self.redis_client.delete(*keys)
            
            if cursor == 0:
                break
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        info = self.redis_client.info('stats')
        
        return {
            'total_keys': self.redis_client.dbsize(),
            'hits': info.get('keyspace_hits', 0),
            'misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / (
                info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1)
            )
        }

# Usage
cache = RAGCache("redis://localhost:6379/0")

# Check cache before processing
cached_response = cache.get_cached_response(user_query)
if cached_response:
    return cached_response  # Cache hit!

# Process query...
response = rag_system.query(user_query)

# Cache for future
cache.cache_response(user_query, response, ttl=3600)

# Cache embeddings
embedding = embedder.embed(text)
cache.cache_embedding(text, embedding.tolist())

# Later: retrieve cached embedding
cached_emb = cache.get_cached_embedding(text)
if cached_emb:
    embedding = np.array(cached_emb)  # Avoid recomputing
```

### 5. Complete Production Data Architecture

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import asyncio

@dataclass
class ProductionRAGSystem:
    """Complete production RAG with all data layers"""
    
    # Core components
    vector_store: PineconeVectorStore
    document_storage: DocumentStorage
    metadata_repo: MetadataRepository
    cache: RAGCache
    
    # Embedder
    embedder: Any  # Your embedding model
    
    async def index_document(self, file_path: str, metadata: dict) -> str:
        """
        Complete document indexing pipeline
        
        Args:
            file_path: Path to document file
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        doc_id = f"doc_{uuid.uuid4()}"
        
        try:
            # 1. Upload to S3
            upload_result = self.document_storage.upload_file(
                file_path=file_path,
                object_key=f"documents/{doc_id}/{Path(file_path).name}",
                metadata=metadata
            )
            
            # 2. Create metadata record
            db_doc_id = self.metadata_repo.create_document_record(
                vector_id=doc_id,
                filename=Path(file_path).name,
                file_path=upload_result['object_key'],
                metadata=metadata
            )
            
            # 3. Process document (chunking, embedding)
            chunks = await self._process_document(file_path)
            
            # 4. Store vectors
            vectors = []
            chunk_records = []
            
            for i, chunk in enumerate(chunks):
                # Check cache first
                cached_emb = self.cache.get_cached_embedding(chunk['text'])
                
                if cached_emb:
                    embedding = np.array(cached_emb)
                else:
                    embedding = await self.embedder.embed(chunk['text'])
                    self.cache.cache_embedding(chunk['text'], embedding.tolist())
                
                chunk_id = f"{doc_id}_chunk_{i}"
                
                vectors.append({
                    "id": chunk_id,
                    "embedding": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "text": chunk['text'],
                        **metadata
                    }
                })
                
                chunk_records.append({
                    "vector_id": chunk_id,
                    "content": chunk['text'],
                    "token_count": chunk.get('token_count')
                })
            
            # Batch upsert to vector DB
            self.vector_store.upsert_documents(vectors)
            
            # 5. Update metadata with chunks
            self.metadata_repo.add_chunks(db_doc_id, chunk_records)
            
            return doc_id
            
        except Exception as e:
            # Update status to failed
            self.metadata_repo.update_status(doc_id, 'failed', str(e))
            raise
    
    async def query(self, user_query: str, user_id: str, 
                    filters: dict = None) -> dict:
        """
        Complete query pipeline with all data layers
        
        Args:
            user_query: User's question
            user_id: User identifier
            filters: Optional metadata filters
            
        Returns:
            Response with answer and sources
        """
        start_time = time.time()
        
        # 1. Check cache
        cache_key = f"{user_query}_{json.dumps(filters or {})}"
        cached = self.cache.get_cached_response(cache_key)
        
        if cached:
            return cached
        
        # 2. Embed query (with caching)
        cached_emb = self.cache.get_cached_embedding(user_query)
        if cached_emb:
            query_embedding = np.array(cached_emb)
        else:
            query_embedding = await self.embedder.embed(user_query)
            self.cache.cache_embedding(user_query, query_embedding.tolist())
        
        # 3. Search vectors
        vector_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=5,
            filter_dict=filters
        )
        
        # 4. Get full document metadata
        doc_ids = [r['metadata']['doc_id'] for r in vector_results]
        documents = self.metadata_repo.get_documents_by_ids(doc_ids)
        
        # 5. Generate answer
        context = "\n\n".join([r['metadata']['text'] for r in vector_results])
        answer = await self._generate_answer(user_query, context)
        
        # 6. Prepare response
        response = {
            "answer": answer,
            "sources": [
                {
                    "document": doc.filename,
                    "snippet": r['metadata']['text'][:200],
                    "score": r['score']
                }
                for doc, r in zip(documents, vector_results)
            ],
            "response_time_ms": int((time.time() - start_time) * 1000)
        }
        
        # 7. Cache response
        self.cache.cache_response(cache_key, response, ttl=3600)
        
        # 8. Log query (async, don't block)
        asyncio.create_task(
            self.metadata_repo.log_query_async(
                user_id=user_id,
                query=user_query,
                retrieved_docs=documents,
                answer=answer,
                response_time=response['response_time_ms']
            )
        )
        
        return response
    
    async def _process_document(self, file_path: str) -> List[dict]:
        """Process document into chunks"""
        # Load document
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Split into chunks
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        
        return [
            {
                "text": chunk.page_content,
                "token_count": len(chunk.page_content.split())
            }
            for chunk in chunks
        ]
```

**Complete Data Flow Diagram:**

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Cache
    participant VectorDB
    participant MetadataDB
    participant S3
    
    User->>API: Upload Document
    API->>S3: Store file
    S3-->>API: File URL
    API->>MetadataDB: Create record
    API->>API: Process & chunk
    API->>Cache: Check embedding cache
    Cache-->>API: Cache miss
    API->>API: Generate embeddings
    API->>Cache: Cache embeddings
    API->>VectorDB: Upsert vectors
    API->>MetadataDB: Update chunks
    API-->>User: Success
    
    User->>API: Query
    API->>Cache: Check response cache
    Cache-->>API: Cache miss
    API->>Cache: Check embedding cache
    Cache-->>API: Cached embedding
    API->>VectorDB: Vector search
    VectorDB-->>API: Top results
    API->>MetadataDB: Get doc metadata
    MetadataDB-->>API: Full metadata
    API->>API: Generate answer
    API->>Cache: Cache response
    API->>MetadataDB: Log query (async)
    API-->>User: Response
```

This comprehensive data management strategy ensures:
- ✅ **Scalability**: Separate concerns (vectors, documents, metadata)
- ✅ **Performance**: Multi-layer caching
- ✅ **Durability**: Persistent storage with backups
- ✅ **Observability**: Query logging and analytics
- ✅ **Cost Efficiency**: Caching reduces API calls by 70-90%
- ✅ **Maintainability**: Clean separation of data layers

---

## Question 16: What is Supabase and how is it used in production Python applications?

**Answer:**

Supabase is an open-source Firebase alternative that provides a complete backend-as-a-service platform built on PostgreSQL. It's particularly powerful for RAG applications because it combines PostgreSQL with pgvector, real-time subscriptions, authentication, and storage in one platform.

**Supabase Architecture:**

```mermaid
graph TB
    subgraph "Supabase Platform"
        A[PostgreSQL Database] --> B[pgvector Extension]
        A --> C[PostgREST API]
        A --> D[Realtime Server]
        E[Auth Service] --> F[JWT Tokens]
        G[Storage Service] --> H[S3-compatible]
        I[Edge Functions] --> J[Deno Runtime]
    end
    
    subgraph "Python Application"
        K[Supabase Client] --> C
        K --> E
        K --> G
        L[Vector Operations] --> B
    end
    
    M[End Users] --> K
```

### 1. Setting Up Supabase for Production

**Installation and Configuration:**

```python
from supabase import create_client, Client
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class SupabaseManager:
    """Production-grade Supabase client manager"""
    
    def __init__(self):
        """
        Initialize Supabase client
        
        Environment variables required:
        - SUPABASE_URL: Your Supabase project URL
        - SUPABASE_KEY: Service role key (for backend) or anon key (for frontend)
        """
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        # Create client with custom options
        self.client: Client = create_client(
            self.url,
            self.key,
            options={
                'schema': 'public',
                'auto_refresh_token': True,
                'persist_session': True,
                'detect_session_in_url': True,
                'headers': {
                    'x-application-name': 'rag-production'
                }
            }
        )
    
    def get_client(self) -> Client:
        """Get Supabase client instance"""
        return self.client

# Initialize (singleton pattern)
supabase_manager = SupabaseManager()
supabase: Client = supabase_manager.get_client()
```

### 2. Database Schema Setup with pgvector

**SQL Schema for RAG Application:**

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_path TEXT,
    file_size INTEGER,
    file_type TEXT,
    
    -- Content
    full_text TEXT,
    language TEXT DEFAULT 'en',
    
    -- Metadata
    category TEXT,
    tags JSONB DEFAULT '[]'::jsonb,
    custom_metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Access control
    access_level TEXT DEFAULT 'public',
    created_by UUID REFERENCES auth.users(id),
    organization_id UUID,
    
    -- Processing status
    status TEXT DEFAULT 'pending',
    indexed_at TIMESTAMPTZ,
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document chunks with embeddings
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    content_preview TEXT GENERATED ALWAYS AS (LEFT(content, 200)) STORED,
    token_count INTEGER,
    
    -- Vector embedding (1536 dimensions for OpenAI)
    embedding vector(1536),
    
    -- Metadata inherited from parent
    metadata JSONB DEFAULT '{}'::jsonb,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(document_id, chunk_index)
);

-- Create index for vector similarity search
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Additional indexes for filtering
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_metadata ON document_chunks USING gin(metadata);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created_by ON documents(created_by);

-- User queries log
CREATE TABLE user_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    query_text TEXT NOT NULL,
    
    -- Results
    retrieved_chunk_ids UUID[],
    answer_text TEXT,
    response_time_ms INTEGER,
    
    -- Feedback
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_queries_user_id ON user_queries(user_id);
CREATE INDEX idx_queries_created_at ON user_queries(created_at);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### 3. Production Python Implementation with Supabase

**A. Document Management:**

```python
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import numpy as np

class SupabaseDocumentRepository:
    """Production document repository using Supabase"""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
    
    def create_document(self, filename: str, file_path: str, 
                       metadata: Dict[str, Any], user_id: str) -> str:
        """
        Create document record in Supabase
        
        Args:
            filename: Original filename
            file_path: Storage path
            metadata: Document metadata
            user_id: User who created the document
            
        Returns:
            Document UUID
        """
        try:
            response = self.client.table('documents').insert({
                'filename': filename,
                'file_path': file_path,
                'file_size': metadata.get('file_size'),
                'file_type': metadata.get('file_type'),
                'category': metadata.get('category'),
                'tags': metadata.get('tags', []),
                'custom_metadata': metadata.get('custom_metadata', {}),
                'access_level': metadata.get('access_level', 'public'),
                'created_by': user_id,
                'status': 'pending'
            }).execute()
            
            return response.data[0]['id']
            
        except Exception as e:
            raise Exception(f"Failed to create document: {str(e)}")
    
    def add_chunks_with_embeddings(self, document_id: str, 
                                   chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add document chunks with embeddings in batch
        
        Args:
            document_id: Parent document UUID
            chunks: List of dicts with keys: content, embedding, metadata
            
        Returns:
            List of created chunk IDs
        """
        try:
            # Prepare chunk records
            chunk_records = [
                {
                    'document_id': document_id,
                    'chunk_index': i,
                    'content': chunk['content'],
                    'token_count': chunk.get('token_count'),
                    'embedding': chunk['embedding'].tolist(),  # Convert numpy to list
                    'metadata': chunk.get('metadata', {})
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Batch insert (Supabase handles this efficiently)
            response = self.client.table('document_chunks').insert(
                chunk_records
            ).execute()
            
            # Update document status
            self.client.table('documents').update({
                'status': 'completed',
                'indexed_at': datetime.utcnow().isoformat()
            }).eq('id', document_id).execute()
            
            return [chunk['id'] for chunk in response.data]
            
        except Exception as e:
            # Update document status to failed
            self.client.table('documents').update({
                'status': 'failed',
                'error_message': str(e)
            }).eq('id', document_id).execute()
            
            raise Exception(f"Failed to add chunks: {str(e)}")
    
    def vector_search(self, query_embedding: np.ndarray, 
                     limit: int = 5, 
                     filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform vector similarity search
        
        Args:
            query_embedding: Query vector (numpy array)
            limit: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of matching chunks with similarity scores
        """
        try:
            # Call Supabase RPC function for vector search
            # (Must be created as PostgreSQL function - see below)
            response = self.client.rpc(
                'match_document_chunks',
                {
                    'query_embedding': query_embedding.tolist(),
                    'match_threshold': 0.7,  # Minimum similarity
                    'match_count': limit,
                    'filter_metadata': filters or {}
                }
            ).execute()
            
            return response.data
            
        except Exception as e:
            raise Exception(f"Vector search failed: {str(e)}")
    
    def get_document_with_chunks(self, document_id: str) -> Dict:
        """Get document with all its chunks"""
        try:
            # Get document
            doc_response = self.client.table('documents').select('*').eq(
                'id', document_id
            ).single().execute()
            
            # Get chunks
            chunks_response = self.client.table('document_chunks').select(
                'id, chunk_index, content, token_count, created_at'
            ).eq('document_id', document_id).order('chunk_index').execute()
            
            document = doc_response.data
            document['chunks'] = chunks_response.data
            
            return document
            
        except Exception as e:
            raise Exception(f"Failed to get document: {str(e)}")
    
    def search_documents(self, filters: Dict, limit: int = 20) -> List[Dict]:
        """
        Search documents by metadata
        
        Args:
            filters: Filter criteria (category, tags, access_level, etc.)
            limit: Maximum results
            
        Returns:
            List of matching documents
        """
        try:
            query = self.client.table('documents').select('*')
            
            # Apply filters
            if 'category' in filters:
                query = query.eq('category', filters['category'])
            
            if 'status' in filters:
                query = query.eq('status', filters['status'])
            
            if 'created_by' in filters:
                query = query.eq('created_by', filters['created_by'])
            
            if 'access_level' in filters:
                query = query.eq('access_level', filters['access_level'])
            
            # Tags filter (JSONB contains)
            if 'tags' in filters:
                query = query.contains('tags', filters['tags'])
            
            response = query.limit(limit).order('created_at', desc=True).execute()
            
            return response.data
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    def delete_document(self, document_id: str):
        """
        Delete document and all its chunks (cascade)
        
        Args:
            document_id: Document UUID
        """
        try:
            # Chunks will be automatically deleted due to CASCADE
            self.client.table('documents').delete().eq('id', document_id).execute()
            
        except Exception as e:
            raise Exception(f"Delete failed: {str(e)}")

# Create PostgreSQL RPC function for vector search
"""
CREATE OR REPLACE FUNCTION match_document_chunks(
    query_embedding vector(1536),
    match_threshold float,
    match_count int,
    filter_metadata jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id uuid,
    document_id uuid,
    content text,
    similarity float,
    metadata jsonb
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.content,
        1 - (dc.embedding <=> query_embedding) AS similarity,
        dc.metadata
    FROM document_chunks dc
    INNER JOIN documents d ON dc.document_id = d.id
    WHERE 
        1 - (dc.embedding <=> query_embedding) > match_threshold
        AND (filter_metadata = '{}'::jsonb OR dc.metadata @> filter_metadata)
        AND d.status = 'completed'
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""
```

**B. Authentication and Row-Level Security (RLS):**

```python
class SupabaseAuth:
    """Handle Supabase authentication"""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
    
    def sign_up(self, email: str, password: str, 
                user_metadata: Optional[Dict] = None) -> Dict:
        """
        Create new user account
        
        Args:
            email: User email
            password: User password
            user_metadata: Additional user data
            
        Returns:
            User data and session
        """
        try:
            response = self.client.auth.sign_up({
                'email': email,
                'password': password,
                'options': {
                    'data': user_metadata or {}
                }
            })
            
            return {
                'user': response.user,
                'session': response.session
            }
            
        except Exception as e:
            raise Exception(f"Sign up failed: {str(e)}")
    
    def sign_in(self, email: str, password: str) -> Dict:
        """Sign in existing user"""
        try:
            response = self.client.auth.sign_in_with_password({
                'email': email,
                'password': password
            })
            
            return {
                'user': response.user,
                'session': response.session,
                'access_token': response.session.access_token
            }
            
        except Exception as e:
            raise Exception(f"Sign in failed: {str(e)}")
    
    def get_user(self, access_token: str) -> Optional[Dict]:
        """Get user from access token"""
        try:
            response = self.client.auth.get_user(access_token)
            return response.user
            
        except Exception as e:
            return None
    
    def sign_out(self):
        """Sign out current user"""
        try:
            self.client.auth.sign_out()
        except Exception as e:
            raise Exception(f"Sign out failed: {str(e)}")

# SQL for Row-Level Security (RLS)
"""
-- Enable RLS on documents table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read documents they created or public documents
CREATE POLICY "Users can read own and public documents"
ON documents FOR SELECT
USING (
    auth.uid() = created_by 
    OR access_level = 'public'
    OR (
        access_level = 'organization' 
        AND organization_id IN (
            SELECT organization_id FROM user_organizations 
            WHERE user_id = auth.uid()
        )
    )
);

-- Policy: Users can insert their own documents
CREATE POLICY "Users can insert own documents"
ON documents FOR INSERT
WITH CHECK (auth.uid() = created_by);

-- Policy: Users can update only their own documents
CREATE POLICY "Users can update own documents"
ON documents FOR UPDATE
USING (auth.uid() = created_by);

-- Policy: Users can delete only their own documents
CREATE POLICY "Users can delete own documents"
ON documents FOR DELETE
USING (auth.uid() = created_by);

-- Similar policies for document_chunks
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read chunks of accessible documents"
ON document_chunks FOR SELECT
USING (
    document_id IN (
        SELECT id FROM documents 
        WHERE auth.uid() = created_by OR access_level = 'public'
    )
);
"""
```

**C. Storage Integration:**

```python
class SupabaseStorage:
    """Manage file storage in Supabase"""
    
    def __init__(self, supabase_client: Client, bucket_name: str = 'documents'):
        self.client = supabase_client
        self.bucket_name = bucket_name
        
        # Create bucket if doesn't exist
        try:
            self.client.storage.create_bucket(bucket_name, {
                'public': False,  # Private bucket
                'file_size_limit': 52428800,  # 50MB limit
                'allowed_mime_types': [
                    'application/pdf',
                    'text/plain',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                ]
            })
        except:
            pass  # Bucket already exists
    
    def upload_file(self, file_path: str, storage_path: str, 
                    user_id: str) -> str:
        """
        Upload file to Supabase storage
        
        Args:
            file_path: Local file path
            storage_path: Path in storage bucket
            user_id: User ID for access control
            
        Returns:
            Public URL or signed URL
        """
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Upload file
            response = self.client.storage.from_(self.bucket_name).upload(
                path=storage_path,
                file=file_data,
                file_options={
                    'content-type': self._get_content_type(file_path),
                    'upsert': True  # Overwrite if exists
                }
            )
            
            # Generate signed URL (expires in 1 hour)
            signed_url = self.client.storage.from_(self.bucket_name).create_signed_url(
                path=storage_path,
                expires_in=3600
            )
            
            return signed_url['signedURL']
            
        except Exception as e:
            raise Exception(f"Upload failed: {str(e)}")
    
    def download_file(self, storage_path: str, local_path: str):
        """Download file from storage"""
        try:
            response = self.client.storage.from_(self.bucket_name).download(storage_path)
            
            with open(local_path, 'wb') as f:
                f.write(response)
                
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
    
    def delete_file(self, storage_path: str):
        """Delete file from storage"""
        try:
            self.client.storage.from_(self.bucket_name).remove([storage_path])
        except Exception as e:
            raise Exception(f"Delete failed: {str(e)}")
    
    def list_files(self, folder_path: str = '') -> List[Dict]:
        """List files in folder"""
        try:
            response = self.client.storage.from_(self.bucket_name).list(folder_path)
            return response
        except Exception as e:
            raise Exception(f"List failed: {str(e)}")
    
    def _get_content_type(self, file_path: str) -> str:
        """Determine content type from file extension"""
        import mimetypes
        content_type, _ = mimetypes.guess_type(file_path)
        return content_type or 'application/octet-stream'
```

**D. Real-time Subscriptions:**

```python
class SupabaseRealtime:
    """Real-time updates using Supabase"""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.subscriptions = []
    
    def subscribe_to_documents(self, callback):
        """
        Subscribe to document changes in real-time
        
        Args:
            callback: Function to call when documents change
                     Signature: callback(payload: Dict)
        """
        def handle_change(payload):
            """Handle database change"""
            event_type = payload['eventType']  # INSERT, UPDATE, DELETE
            new_record = payload.get('new')
            old_record = payload.get('old')
            
            callback({
                'event': event_type,
                'new': new_record,
                'old': old_record
            })
        
        # Subscribe to changes
        subscription = (
            self.client
            .channel('document_changes')
            .on_postgres_changes(
                event='*',  # All events (INSERT, UPDATE, DELETE)
                schema='public',
                table='documents',
                callback=handle_change
            )
            .subscribe()
        )
        
        self.subscriptions.append(subscription)
        return subscription
    
    def subscribe_to_processing_status(self, document_id: str, callback):
        """Subscribe to specific document's processing status"""
        def handle_change(payload):
            if payload.get('new', {}).get('id') == document_id:
                status = payload['new'].get('status')
                callback(status)
        
        subscription = (
            self.client
            .channel(f'doc_{document_id}_status')
            .on_postgres_changes(
                event='UPDATE',
                schema='public',
                table='documents',
                filter=f'id=eq.{document_id}',
                callback=handle_change
            )
            .subscribe()
        )
        
        self.subscriptions.append(subscription)
        return subscription
    
    def unsubscribe_all(self):
        """Unsubscribe from all channels"""
        for subscription in self.subscriptions:
            self.client.remove_channel(subscription)
        self.subscriptions = []

# Usage example
realtime = SupabaseRealtime(supabase)

def on_document_change(payload):
    print(f"Document {payload['event']}: {payload['new']}")

# Subscribe
realtime.subscribe_to_documents(on_document_change)

# Later: cleanup
realtime.unsubscribe_all()
```

### 4. Complete Production RAG System with Supabase

```python
from typing import List, Dict, Optional
import asyncio

class ProductionRAGWithSupabase:
    """Complete RAG system using Supabase as backend"""
    
    def __init__(self, supabase_client: Client, embedder):
        self.supabase = supabase_client
        self.embedder = embedder
        
        # Initialize components
        self.doc_repo = SupabaseDocumentRepository(supabase_client)
        self.auth = SupabaseAuth(supabase_client)
        self.storage = SupabaseStorage(supabase_client)
        self.realtime = SupabaseRealtime(supabase_client)
    
    async def index_document_pipeline(self, file_path: str, 
                                     metadata: Dict, 
                                     user_id: str) -> str:
        """
        Complete document indexing pipeline
        
        Args:
            file_path: Path to document file
            metadata: Document metadata
            user_id: User who owns the document
            
        Returns:
            Document ID
        """
        try:
            # 1. Upload file to Supabase storage
            storage_path = f"{user_id}/{uuid.uuid4()}/{Path(file_path).name}"
            file_url = self.storage.upload_file(file_path, storage_path, user_id)
            
            # 2. Create document record
            doc_id = self.doc_repo.create_document(
                filename=Path(file_path).name,
                file_path=storage_path,
                metadata=metadata,
                user_id=user_id
            )
            
            # 3. Process document (extract text, chunk)
            chunks = await self._process_document(file_path)
            
            # 4. Generate embeddings
            chunk_data = []
            for chunk in chunks:
                embedding = await self.embedder.embed(chunk['text'])
                
                chunk_data.append({
                    'content': chunk['text'],
                    'embedding': embedding,
                    'token_count': chunk.get('token_count'),
                    'metadata': chunk.get('metadata', {})
                })
            
            # 5. Store chunks with embeddings
            self.doc_repo.add_chunks_with_embeddings(doc_id, chunk_data)
            
            return doc_id
            
        except Exception as e:
            # Update document status to failed
            self.supabase.table('documents').update({
                'status': 'failed',
                'error_message': str(e)
            }).eq('id', doc_id).execute()
            
            raise
    
    async def query(self, user_query: str, user_id: str, 
                   filters: Optional[Dict] = None) -> Dict:
        """
        Query the RAG system
        
        Args:
            user_query: User's question
            user_id: User making the query
            filters: Optional metadata filters
            
        Returns:
            Answer with sources
        """
        start_time = time.time()
        
        try:
            # 1. Generate query embedding
            query_embedding = await self.embedder.embed(user_query)
            
            # 2. Vector search with RLS automatically applied
            results = self.doc_repo.vector_search(
                query_embedding=query_embedding,
                limit=5,
                filters=filters
            )
            
            # 3. Prepare context
            context = "\n\n".join([r['content'] for r in results])
            
            # 4. Generate answer
            answer = await self._generate_answer(user_query, context)
            
            # 5. Log query
            response_time = int((time.time() - start_time) * 1000)
            
            self.supabase.table('user_queries').insert({
                'user_id': user_id,
                'query_text': user_query,
                'retrieved_chunk_ids': [r['id'] for r in results],
                'answer_text': answer,
                'response_time_ms': response_time
            }).execute()
            
            # 6. Return response
            return {
                'answer': answer,
                'sources': [
                    {
                        'content': r['content'][:200],
                        'similarity': r['similarity'],
                        'document_id': r['document_id']
                    }
                    for r in results
                ],
                'response_time_ms': response_time
            }
            
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")
    
    def get_user_documents(self, user_id: str) -> List[Dict]:
        """Get all documents for a user (RLS automatically filters)"""
        return self.doc_repo.search_documents(
            filters={'created_by': user_id},
            limit=100
        )
    
    def submit_feedback(self, query_id: str, rating: int, feedback: str):
        """Submit user feedback on a query"""
        self.supabase.table('user_queries').update({
            'rating': rating,
            'feedback_text': feedback
        }).eq('id', query_id).execute()

# Usage
rag_system = ProductionRAGWithSupabase(supabase, embedder)

# Sign in user
auth_response = rag_system.auth.sign_in("user@example.com", "password")
user_id = auth_response['user'].id

# Index document
doc_id = await rag_system.index_document_pipeline(
    file_path="document.pdf",
    metadata={'category': 'technical', 'tags': ['AI', 'RAG']},
    user_id=user_id
)

# Query
response = await rag_system.query(
    user_query="What is RAG?",
    user_id=user_id,
    filters={'category': 'technical'}
)

print(response['answer'])
```

**Key Advantages of Supabase for Production:**

1. **Built-in Authentication**: No need for separate auth service
2. **Row-Level Security**: Database-level access control
3. **Real-time Updates**: Live document processing status
4. **Integrated Storage**: File storage with access control
5. **Auto-generated REST API**: No need to write API endpoints
6. **PostgreSQL + pgvector**: Native vector search
7. **Open Source**: Can self-host if needed
8. **Cost-Effective**: Free tier generous, affordable paid plans

---

*Continuing with remaining questions in next section...*
