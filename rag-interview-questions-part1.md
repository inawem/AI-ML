# RAG & LangChain Interview Questions - Part 1 (Questions 1-10)

**For experienced engineers with 20+ years in software development**

---

## Question 1: Explain the core problem that RAG solves and how it differs from fine-tuning an LLM.

**Answer:**

RAG addresses fundamental limitations of standalone LLMs:

1. **Knowledge Cutoff**: LLMs frozen at training date; RAG provides current information
2. **Hallucinations**: RAG grounds responses in retrieved facts
3. **Domain Knowledge**: RAG accesses private/specialized knowledge bases
4. **Attribution**: RAG enables source tracking
5. **Dynamic Updates**: RAG knowledge updates instantly vs. costly retraining

**RAG vs Fine-Tuning:**

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Cost** | Low (retrieval + inference) | High (GPU hours) |
| **Update Speed** | Instant (add documents) | Slow (retrain) |
| **Knowledge Type** | Factual, dynamic | Behavior, style |
| **Transparency** | High (see sources) | Low (black box) |
| **Use Case** | Q&A, current info | Custom behavior |

**When to Use:**
- **RAG**: Documentation search, current events, citations needed
- **Fine-Tuning**: Custom writing style, domain-specific tasks
- **Both**: Customer support (fine-tune for tone, RAG for product docs)

---

## Question 2: What are embeddings and why are they critical for RAG?

**Answer:**

Embeddings are dense vector representations capturing semantic meaning. Similar concepts have similar vectors.

**Why Critical:**
1. **Semantic Search**: Find conceptually similar documents, not just keywords
2. **Language Independence**: "dog" and "canine" have similar embeddings  
3. **Context Understanding**: Handles synonyms, paraphrasing
4. **Efficiency**: Fast vector similarity computation

**Choosing Embedding Models:**

| Model | Dimensions | Best For | Cost (1M tokens) |
|-------|------------|----------|------------------|
| **text-embedding-3-large** | 3072 | Production, high-stakes | $0.13 |
| **text-embedding-3-small** | 1536 | General use | $0.02 |
| **Cohere embed-v3** | 1024 | Multilingual | $0.10 |
| **all-MiniLM-L6-v2** | 384 | Local, real-time | Free |

**Key Considerations:**
- **Dimension Size**: Higher = more nuanced but slower
- **Domain Alignment**: General vs. specialized (code, medical, legal)
- **Consistency**: Same model for indexing and retrieval

---

## Question 3: Walk through the complete RAG pipeline from ingestion to generation.

**Answer:**

**Phase 1: Ingestion (Offline)**
1. Load documents (PDF, HTML, code, DB)
2. Parse and clean text
3. Chunk into manageable pieces
4. Generate embeddings
5. Store in vector database

**Phase 2: Query Processing (Online)**
1. Receive user query
2. Understand intent
3. Transform/rewrite query if needed
4. Generate query embedding

**Phase 3: Retrieval**
1. Vector similarity search
2. Optional: Hybrid search (vector + keyword)
3. Rerank results
4. Filter by metadata
5. Compress context

**Phase 4: Generation**
1. Construct prompt with context
2. Invoke LLM
3. Generate answer
4. Post-process and verify

**Critical Decision Points:**

**1. Chunking Strategy**
- Smaller chunks (256-512): Better precision, less context
- Larger chunks (1024-2048): More context, diluted relevance
- Recommended: 1000 chars with 200 overlap

**2. Retrieval Approach**
- Single-stage: Fast, simple
- Multi-stage: Better quality, slower (retrieve 100 → rerank to 5)

**3. Context Management**
- How many docs to send LLM?
- Use similarity threshold vs. fixed top-K
- Consider context compression

---

## Question 4: What's the difference between naive RAG and advanced RAG?

**Answer:**

**Naive RAG:**
- Single query → one search → one answer
- No query optimization
- Direct vector similarity only
- No verification
- Fast but limited quality

**Problems:**
- Poor retrieval for complex queries
- Sensitive to query phrasing
- No handling of ambiguity
- Can hallucinate on edge cases

**Advanced RAG Techniques:**

**A. Query Transformation**
- **Multi-Query**: Generate multiple query perspectives
- **HyDE**: Generate hypothetical answer, embed it
- **Step-Back**: Ask broader question first
- **Decomposition**: Break complex query into sub-queries

**B. Retrieval Enhancement**
- **Hybrid Search**: Vector + keyword (BM25)
- **Reranking**: Cross-encoder models for better relevance
- **Context Compression**: Extract only relevant parts

**C. Self-Correction**
- **Verification**: Check if answer is grounded in context
- **Iterative Refinement**: Refine search based on initial results
- **Hallucination Detection**: Validate claims against sources

**When to Use Each:**

| Factor | Naive RAG | Advanced RAG |
|--------|-----------|--------------|
| **Development Time** | Days | Weeks |
| **Query Complexity** | Simple | Complex, multi-part |
| **Accuracy Target** | 70-80% | 90%+ |
| **Latency Budget** | <1s | 1-5s OK |
| **User Base** | Internal/small | External/large |

**Progressive Path:**
1. Start with naive RAG (validate concept)
2. Add reranking (better relevance)
3. Add query enhancement (multi-query/HyDE)
4. Add self-correction (verification loops)
5. Move to agentic RAG (tool use & planning)

---

## Question 5: How do you handle different document types in a unified RAG system?

**Answer:**

**Multi-Format Processing Strategy:**

**1. PDFs**
```python
from langchain_community.document_loaders import (
    PyPDFLoader,          # Fast, basic
    PDFPlumberLoader,     # Better for tables
    UnstructuredPDFLoader # Layout-aware
)

# Extract tables separately
import pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        # Convert to markdown for LLM
```

**Challenges**: Multi-column layouts, tables split across pages, headers/footers

**2. HTML/Web**
```python
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

# Remove noise
for tag in soup(['script', 'style', 'nav', 'footer']):
    tag.decompose()

# Extract structured data
metadata = {
    "url": url,
    "title": soup.title.string,
    "headings": [h.text for h in soup.find_all(['h1', 'h2'])]
}
```

**3. Code Repositories**
```python
import ast

# Parse code structure
tree = ast.parse(code)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        # Extract function with metadata
        metadata = {
            "type": "function",
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "line_number": node.lineno
        }
```

**Challenges**: Don't split mid-function, preserve syntax

**4. Structured Data (Databases)**
```python
# Index schema + sample data
schema = db.get_table_info([table])
sample = db.run(f"SELECT * FROM {table} LIMIT 5")

# For queries, convert NL to SQL
chain = create_sql_query_chain(llm, db)
sql = chain.invoke({"question": natural_language_query})
results = db.run(sql)
```

**Unified Metadata Schema:**
```python
metadata = {
    # Universal
    "doc_id": "",
    "doc_type": "pdf|html|code|database",
    "source": "",
    
    # Type-specific (optional)
    "page": None,        # PDFs
    "url": None,         # Web
    "language": None,    # Code
    "function_name": None # Code
}
```

**Format-Aware Retrieval:**
- Classify query type (code query vs. documentation)
- Apply type-specific filters
- Use specialized tools (SQL for databases, AST search for code)
- Render results appropriately (syntax highlighting for code, tables for data)

---

## Question 6: Explain chunking strategies. How do chunk size and overlap affect performance?

**Answer:**

**Chunking Strategies:**

**1. Fixed-Size**
```python
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```
- ✅ Simple, predictable
- ❌ May split mid-sentence

**2. Recursive (Recommended)**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical
)
```
- ✅ Respects document structure
- ✅ Production standard

**3. Semantic**
```python
from langchain_experimental.text_splitter import SemanticChunker
splitter = SemanticChunker(embeddings)
```
- ✅ Meaning-based boundaries
- ❌ Variable sizes, expensive

**4. Document Structure-Aware**
```python
# Code
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000
)

# Markdown
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")]
)
```

**Chunk Size Impact:**

| Size | Retrieval Precision | Context Quality | Use Case |
|------|-------------------|-----------------|----------|
| **256-512** | High | Low | FAQ, definitions |
| **512-1024** | Medium | Medium | **General docs (recommended)** |
| **1024-2048** | Low | High | Long-form content |
| **2048+** | Very Low | Very High | Code, reports |

**Overlap Analysis:**

| Overlap % | Storage | Retrieval Quality | Recommended For |
|-----------|---------|-------------------|-----------------|
| **0%** | 1x | Poor (context breaks) | Separate entities |
| **10-20%** | 1.1-1.2x | Good | **Standard production** |
| **20-30%** | 1.2-1.3x | Better | High-stakes applications |
| **>50%** | 1.5x+ | Redundant | Rarely justified |

**Advanced: Parent-Child Chunking**
```python
from langchain.retrievers import ParentDocumentRetriever

# Small chunks for retrieval (400 tokens)
# Large chunks for context (2000 tokens)
# Best of both worlds: precise retrieval + full context
```

**Rule of Thumb:**
1. Start with **1000 chars, 200 overlap** (works for 80% of cases)
2. Measure retrieval quality
3. Adjust based on your specific use case:
   - Low precision? Decrease chunk size
   - Insufficient context? Increase chunk size
   - Context breaks? Increase overlap

---

## Question 7: What is the best tech stack for production RAG?

**Answer:**

**Stack by Stage:**

**A. Startup/MVP (Speed to Market)**
```python
stack = {
    "llm": "Azure OpenAI (GPT-4)",
    "embeddings": "text-embedding-3-small",
    "vector_db": "Chroma → Pinecone when scaling",
    "framework": "LangChain",
    "deployment": "Vercel/Railway",
    "monitoring": "LangSmith",
    "cost": "$100-500/month for <10K queries"
}
```

**B. Production (Balanced)**
```python
stack = {
    "llm": "Azure OpenAI + smaller model for simple queries",
    "embeddings": "text-embedding-3-large",
    "vector_db": "Qdrant (self-hosted on K8s)",
    "framework": "LangChain + custom components",
    "deployment": "AWS EKS / GCP GKE",
    "caching": "Redis (semantic cache)",
    "monitoring": "LangSmith + Datadog",
    "cost": "$1K-5K/month for 100K+ queries"
}
```

**C. Enterprise (Scale & Security)**
```python
stack = {
    "llm": "Azure OpenAI (private) + self-hosted fallback",
    "embeddings": "Custom fine-tuned",
    "vector_db": "Milvus (distributed cluster)",
    "framework": "Custom orchestration",
    "deployment": "Multi-region Kubernetes",
    "caching": "Redis cluster",
    "monitoring": "Full observability (Prometheus, Grafana, Jaeger)",
    "cost": "$10K-50K+/month for millions of queries"
}
```

**Vector Database Comparison:**

| Database | Scale | Best For | Pros | Cons |
|----------|-------|----------|------|------|
| **FAISS** | <1M docs | Development | Fast, free | Not persistent |
| **Chroma** | <100K | Small production | Easy setup | Single node |
| **Pinecone** | Unlimited | Fast deployment | Managed, scales | Cost, lock-in |
| **Qdrant** | <10M | Production | Open-source, performant | Self-manage |
| **Milvus** | Unlimited | Enterprise | Distributed, mature | Complex ops |

**LLM Provider Comparison:**

| Provider | Best For | Cost (1M tokens) | Strengths |
|----------|----------|------------------|-----------|
| **Azure OpenAI** | Enterprise | $2-30 | SLA, compliance |
| **OpenAI** | Startups | $0.50-30 | Latest models |
| **Anthropic Claude** | Long docs | $3-15 | 200K context |
| **Cohere** | Multilingual | $0.50-15 | Great embeddings |
| **Self-hosted Llama** | Privacy | $0.10-1 | Full control |

**Cost Optimization:**
```python
# Semantic caching: 70-90% cost reduction
semantic_cache.get(query, threshold=0.95)  # Cache hit saves LLM call

# Smaller embedding model: 5x cheaper
"text-embedding-3-small" vs "text-embedding-3-large"

# Mix of models: Use GPT-3.5 for simple, GPT-4 for complex
if is_simple_query(query):
    llm = "gpt-3.5-turbo"  # 10x cheaper
```

**My Recommendation:**
1. **Phase 1**: Start with managed services (Azure OpenAI + Pinecone)
2. **Phase 2**: Self-host vector DB (Qdrant), keep LLM managed
3. **Phase 3**: Add self-hosted LLM fallback for cost/redundancy

---

## Question 8: How do you evaluate RAG system performance?

**Answer:**

**Evaluation Framework:**

**1. Retrieval Metrics**

```python
# Precision@K: What fraction of retrieved docs are relevant?
def precision_at_k(retrieved, relevant, k=5):
    return len(set(retrieved[:k]) & relevant) / k

# Recall@K: What fraction of relevant docs were retrieved?
def recall_at_k(retrieved, relevant, k=5):
    return len(set(retrieved[:k]) & relevant) / len(relevant)

# MRR: How highly ranked is first relevant doc?
def mrr(retrieved, relevant):
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1 / i
    return 0.0
```

**2. Generation Metrics (RAGAS Framework)**

```python
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,    # Does answer address question?
    faithfulness,        # Is answer grounded in context?
    context_recall,      # Was all needed info retrieved?
    context_precision    # Are retrieved docs relevant?
)

results = evaluate(dataset, metrics=[
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
])
```

**Metric Interpretation:**

| Metric | Target | Critical | Issue if Low |
|--------|--------|----------|--------------|
| **Answer Relevancy** | >0.85 | >0.70 | Poor query understanding |
| **Faithfulness** | >0.90 | >0.80 | Hallucination problem |
| **Context Precision** | >0.70 | >0.60 | Retrieval noise |
| **Context Recall** | >0.80 | >0.70 | Missing information |

**3. System Metrics**

```python
metrics = {
    "p50_latency": 1.2,      # 50% under 1.2s
    "p95_latency": 3.5,      # 95% under 3.5s
    "avg_cost": 0.015,       # $0.015 per query
    "cache_hit_rate": 0.65   # 65% from cache
}
```

**Production Targets:**

| Metric | Target | Critical |
|--------|--------|----------|
| P95 Latency | <3s | <5s |
| Cost per Query | <$0.02 | <$0.10 |
| Cache Hit Rate | >60% | >40% |

**4. Continuous Evaluation**

```python
class ProductionMonitoring:
    def __init__(self, sample_rate=0.1):  # Evaluate 10% of queries
        self.sample_rate = sample_rate
    
    async def evaluate_query(self, question, answer, contexts):
        if random.random() > self.sample_rate:
            return
        
        # Async evaluation (doesn't block user)
        asyncio.create_task(
            self.async_llm_judge(question, answer, contexts)
        )
```

**5. A/B Testing**

```python
# Compare two configurations
test = ABTest(
    config_a={"chunk_size": 500},
    config_b={"chunk_size": 1000},
    test_queries=evaluation_queries
)

results = test.run_test()
# Output: Config B improved by 15.3% (p=0.003, significant)
```

**Key Takeaways:**
1. **Multi-level**: Component (retrieval, generation) + end-to-end + user feedback
2. **Continuous**: Don't just evaluate once - monitor in production  
3. **Right Metrics**: Question-answering ≠ search (different metrics needed)
4. **Invest in Test Data**: 50-100 high-quality test cases is critical

---

## Question 9: Explain conversational RAG with memory.

**Answer:**

**Key Challenges:**
1. **Coreference Resolution**: "What about it?" → What does "it" refer to?
2. **Context Accumulation**: Maintain history without exceeding token limits
3. **Query Reformulation**: Convert contextual follow-ups to standalone queries

**Implementation:**

**1. Query Reformulation**

```python
class ConversationalRAG:
    def _create_standalone_question(self, follow_up):
        """Convert contextual question to standalone"""
        if not self.conversation_history:
            return follow_up  # First turn, no context
        
        # Build context from recent history
        history_text = "\n".join([
            f"Q: {turn['question']}\nA: {turn['answer']}"
            for turn in self.conversation_history[-3:]
        ])
        
        prompt = f"""
        Given conversation history and follow-up, rephrase follow-up 
        as standalone question with all necessary context.
        
        History: {history_text}
        Follow-up: {follow_up}
        
        Standalone Question:
        """
        
        return self.llm.invoke(prompt).strip()

# Example:
# Turn 1: "What is RAG?" → "What is RAG?" (no change)
# Turn 2: "How does it work?" → "How does RAG work?"
# Turn 3: "Benefits vs fine-tuning?" → "What are benefits of RAG vs fine-tuning?"
```

**2. Memory Strategies**

**Buffer Memory** (Simple)
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()  # Stores all messages
# Problem: Grows unbounded
```

**Window Memory** (Recommended)
```python
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=5)  # Last 5 exchanges
# Good for chatbots where recent context matters most
```

**Summary Memory** (For Long Conversations)
```python
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)
# Summarizes old turns: "User asked about RAG and learned..."
```

**Vector Memory** (Most Powerful)
```python
from langchain.memory import VectorStoreRetrieverMemory
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)
# Embeds each turn, retrieves semantically relevant past conversations
```

**3. Hybrid Memory System (Production)**

```python
class HybridMemory:
    def __init__(self, llm, vectorstore):
        # Short-term: Last 3 turns (always included)
        self.short_term = ConversationBufferWindowMemory(k=3)
        
        # Long-term: Semantic search over all history
        self.long_term = VectorStoreRetrieverMemory(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
        )
        
        # Summary: Condensed older conversation
        self.summary = ConversationSummaryMemory(llm=llm)
    
    def get_context(self, current_question):
        return {
            "recent": self.short_term.load_memory_variables({})["history"],
            "relevant": self.long_term.load_memory_variables(
                {"prompt": current_question}
            )["history"],
            "summary": self.summary.load_memory_variables({})["history"]
            if self.turn_count > 10 else None
        }
```

**4. Handling Complex Follow-ups**

```python
class AdvancedConversationalRAG:
    def query(self, user_question):
        # Classify intent
        intent = self._classify_intent(user_question)
        
        if intent == "clarification":
            # "What do you mean by that?"
            return self._handle_clarification(user_question)
        
        elif intent == "follow_up":
            # "Tell me more about the second point"
            return self._handle_follow_up(user_question)
        
        elif intent == "comparison":
            # "How does that compare to X?"
            return self._handle_comparison(user_question)
```

**Best Practices:**
1. **Always reformulate** contextual questions into standalone queries
2. **Use hybrid memory**: Short-term buffer + semantic long-term
3. **Implement intent classification** for different conversational patterns
4. **Track entities** mentioned across conversation
5. **Test edge cases**: Pronoun resolution, topic switches

---

## Question 10: How would you implement multi-lingual RAG?

**Answer:**

**Approaches:**

**1. Multi-lingual Embeddings (Recommended)**

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Maps all languages to same semantic space
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-mpnet-base-v2"
    # Supports 50+ languages
)

# OR Cohere (100+ languages)
from langchain_community.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

# Single vector store with mixed languages
vectorstore = Chroma.from_documents(
    docs_english + docs_spanish + docs_french,
    embeddings
)

# Query in any language retrieves relevant docs regardless of language
docs = retriever.get_relevant_documents("What is the return policy?")
# Can return English, Spanish, French docs if relevant
```

**How It Works:**
- "Hello world" (English) → [0.2, 0.8, ...]
- "Hola mundo" (Spanish) → [0.19, 0.82, ...]  
- "Bonjour monde" (French) → [0.21, 0.81, ...]
- Similar vectors! Semantic space aligned across languages

**2. Translation Layer**

```python
from deep_translator import GoogleTranslator
from langdetect import detect

class TranslationRAG:
    def query(self, user_question):
        # Detect language
        source_lang = detect(user_question)
        
        # Translate to English
        if source_lang != "en":
            translator = GoogleTranslator(source=source_lang, target="en")
            english_query = translator.translate(user_question)
        
        # Retrieve + generate in English
        docs = self.retriever.get_relevant_documents(english_query)
        answer_en = self.llm.invoke(f"Q: {english_query}\nContext: {docs}")
        
        # Translate back
        if source_lang != "en":
            translator_back = GoogleTranslator(source="en", target=source_lang)
            answer = translator_back.translate(answer_en)
        
        return answer
```

**3. Hybrid (Best Quality)**

```python
class HybridMultilingualRAG:
    def query(self, question, user_language):
        # Step 1: Retrieve using multi-lingual embeddings (no translation)
        docs = self.vectorstore.similarity_search(question, k=10)
        
        # Step 2: Rerank docs (translate query to doc language if different)
        reranked_docs = self._cross_lingual_rerank(question, docs, user_language)
        
        # Step 3: Translate docs to user language if needed
        context = self._prepare_context(reranked_docs[:5], user_language)
        
        # Step 4: Generate answer in user's language
        answer = self._generate_answer(question, context, user_language)
        
        return answer
```

**Comparison:**

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Multi-lingual Embeddings** | ✅ No translation errors<br/>✅ Fast<br/>✅ Handles code-switching | ❌ Slightly lower quality | General use, many languages |
| **Translation Layer** | ✅ Use best English models<br/>✅ Explicit control | ❌ Translation errors<br/>❌ Slow<br/>❌ Costly | English-primary systems |
| **Hybrid** | ✅ Best quality<br/>✅ Balanced approach | ❌ Most complex | Production high-quality |

**Recommendation:**
```python
# Start here (90% of cases)
stack = {
    "embeddings": "Cohere embed-multilingual-v3.0",
    "llm": "GPT-4 (native multilingual)",
    "vectorstore": "Single index",
    "fallback": "Translation for rare languages"
}
```

---
