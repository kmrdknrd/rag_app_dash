# RAG Processing Pipeline Documentation - app_hybrid_split.py

## Overview

The `app_hybrid_split.py` file implements a sophisticated hybrid RAG (Retrieval-Augmented Generation) system that combines semantic search with BM25 keyword search for enhanced document retrieval. This version features comprehensive progress tracking, multi-stage processing, and project-based organization.

## Visual Pipeline Schematic

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       HYBRID RAG PROCESSING PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

📄 PDF DOCUMENTS
     │
     ▼
┌─────────────────────┐    ┌─────────────────────┐
│   PDF PROCESSING    │────│   TEXT CHUNKING     │
│                     │    │                     │
│ • Docling           │    │ • RecursiveCharacter│
│ • ProgressTracker   │    │   TextSplitter      │
│ • ~45s per 1MB      │    │ • Configurable size │
│ • Stage tracking    │    │ • Overlap control   │
└─────────────────────┘    └─────────────────────┘
     │                              │
     ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐
│   PROJECT STORAGE   │    │   EMBEDDING GEN     │
│                     │    │                     │
│ • docs_pdf/         │    │ • BiEncoderPipeline │
│ • docs_md/          │    │ • Singleton pattern │
│ • embeddings/       │    │ • Model choice      │
│ • bm25/             │    │ • GPU acceleration  │
└─────────────────────┘    └─────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │   BM25 INDEX        │
                          │                     │
                          │ • BM25Okapi         │
                          │ • NLTK tokenization │
                          │ • Pickle caching    │
                          │ • Project-specific  │
                          └─────────────────────┘

════════════════════════════════════════════════════════════════════════════════════════

🔍 USER QUERY
     │ (QueryProgressTracker starts)
     ▼
┌─────────────────────┐
│   INITIALIZING      │
│                     │
│ • Stage 1/4 (5%)    │
│ • Setup retrieval   │
│ • Check hybrid mode │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   HYBRID ENABLED?   │
│                     │
│ • Check config      │
│ • Route query path  │
└─────────────────────┘
     │
     ├─────────── YES ─────────────┐–––––––––─── NO ───––––––┐
     ▼                             ▼                         ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   DENSE RETRIEVAL   │    │   SPARSE RETRIEVAL  │    │   DENSE ONLY        │
│                     │    │                     │    │                     │
│ • user_input →      │    │ • user_input →      │    │ • user_input →      │
│   retrieve_all()    │    │   tokenize()        │    │   retrieve_top_k()  │
│ • Cosine similarity │    │ • BM25 scoring      │    │ • Cosine similarity │
│ • ALL documents     │    │ • ALL documents     │    │ • Top-K documents   │
│ • Stage 2/4 (30%)   │    │ • Stage 2/4 (30%)   │    │ • Stage 2/4 (30%)   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                          │
           └─────────┬─────────────────┘                          │
                     ▼                                            │
            ┌─────────────────────┐                               │
            │   HYBRID FUSION     │                               │
            │                     │                               │
            │ • Score combination │                               │
            │ • dense_scores +    │                               │
            │   (bm25 * weight)   │                               │
            │ • minmax_scale()    │                               │
            │ • Top-K selection   │                               │
            └─────────────────────┘                               │
                     │                                            │
                     └─────────────┬──────────────────────────────┘
                                   ▼
                          ┌─────────────────────┐
                          │   CROSS-ENCODER     │
                          │     RERANKING       │
                          │                     │
                          │ • Stage 3/4 (25%)   │
                          │ • Query-doc pairs   │
                          │ • Hybrid reranking  │
                          │ • Top-N selection   │
                          └─────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │   CONTEXT ASSEMBLY  │
                          │                     │
                          │ • Document format   │
                          │ • Reference IDs     │
                          │ • Prompt injection  │
                          │ • History context   │
                          └─────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │   LLM GENERATION    │
                          │                     │
                          │ • Stage 4/4 (40%)   │
                          │ • Multi-provider    │
                          │ • Conversation mode │
                          │ • Response tracking │
                          └─────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │   POST-PROCESSING   │
                          │                     │
                          │ • Clickable refs    │
                          │ • Statistics update │
                          │ • Progress complete │
                          │ • UI formatting     │
                          └─────────────────────┘
                                   │
                                   ▼
                              💬 USER RESPONSE
```

## Key Components and Classes

### 1. Progress Tracking System

#### ProgressTracker (Lines 47-175)
**Purpose**: Centralized tracking for PDF processing pipeline.

**Stages and Weights**:
- `upload` (5%): File upload completion
- `pdf_processing` (40%): PDF to text conversion
- `embedding_init` (10%): Model initialization  
- `embedding_creation` (45%): Embedding generation

**Key Methods**:
- `set_stage()`: Initialize new processing stage
- `update_stage_progress()`: Update progress within stage
- `calculate_overall_progress()`: Compute weighted overall progress
- `log_message()`: Timestamped logging with console output

#### QueryProgressTracker (Lines 225-299)
**Purpose**: Real-time tracking for query processing stages.

**Stages and Weights**:
- `initializing` (5%): Query setup and validation
- `retrieving` (30%): Document retrieval (dense + sparse)
- `reranking` (25%): Cross-encoder reranking
- `generating` (40%): LLM response generation

**Features**:
- Real-time UI updates during query processing
- Detailed stage messaging with context
- Progress calculation with weighted stages

### 2. BM25 Integration System

#### BM25 Corpus Management (Lines 829-876)
**Core Functions**:

```python
def get_bm25_path(base_dir, project_name, model_name, chunk_size, chunk_overlap):
    """Project-specific BM25 storage path"""
    
def create_bm25_corpus(documents_embeddings, bm25_path):
    """Create and cache BM25 index from document embeddings"""
    
def load_bm25_corpus(bm25_path):
    """Load cached BM25 index for reuse"""
```

**Storage Structure**:
```
projects/[ProjectName]/embeddings/[model]/chunk_size_[X]/chunk_overlap_[Y]/bm25/bm25.pkl
```

**Process**:
1. Extract text from document embeddings
2. Tokenize using NLTK word tokenization
3. Create BM25Okapi object with tokenized corpus
4. Cache serialized corpus for future use
5. Load cached corpus to avoid recomputation

### 3. Hybrid Search Implementation

#### Hybrid Search Configuration (Lines 1341-1343, 1908-1947)
**Configuration Options**:
```python
'hybrid_search_config': {
    'enabled': False,           # Toggle hybrid search
    'bm25_weight': 0.3,        # Weight for retrieval fusion
    'bm25_weight_rerank': 0.1  # Weight for reranking fusion
}
```

**UI Controls**:
- Checkbox to enable/disable hybrid search
- Slider for BM25 weight in retrieval (0.0-1.0)
- Slider for BM25 weight in reranking (0.0-1.0)
- Real-time configuration updates

#### Hybrid Retrieval Process (Lines 2992-3062)
**Step-by-Step Process**:

1. **Check Hybrid Mode**: Verify if hybrid search is enabled
2. **Parallel Retrieval**: 
   - **Dense**: `user_input` → `retrieve_all()` → semantic similarity scores for ALL documents
   - **Sparse**: `user_input` → `tokenize()` → `bm25.get_scores()` → keyword scores for ALL documents
3. **BM25 Setup**: Load or create BM25 corpus for current project
4. **Score Normalization**: Apply minmax scaling to BM25 scores
5. **Score Fusion**: Combine scores using weighted formula:
   ```python
   combined_scores = dense_scores + bm25_scores_normalized * bm25_weight
   ```
6. **Top-K Selection**: Select highest-scoring documents for reranking

**Fallback Mechanism**:
- Falls back to dense-only search if BM25 creation fails
- Falls back to dense-only search if no current project
- Graceful degradation with informative logging

### 4. BiEncoderPipeline (Lines 1007-1150)

**Singleton Pattern with Instance Keys**:
```python
instance_key = (model_name, chunk_size, chunk_overlap)
```

**Supported Models**:
- `Snowflake/snowflake-arctic-embed-l-v2.0` (default)
- `BAAI/bge-m3`

**Key Methods**:
- `embed_documents()`: Generate and cache embeddings
- `retrieve_top_k()`: Standard top-K retrieval
- `retrieve_all()`: **New in hybrid version** - returns all documents with scores

**Enhanced Caching**:
- Project-specific embedding storage
- Configuration-aware cache paths
- Automatic cache validation and loading

### 5. CrossEncoderPipeline (Lines 1169-1340)

#### Hybrid Reranking Support (Lines 1202, 1231-1241, 1272-1280)
**Enhanced `rerank()` Method**:
```python
def rerank(self, query, documents, top_n=4, hybrid_rerank=False, bm25_weight=0.1):
```

**Hybrid Reranking Process**:
1. **Cross-Encoder Scoring**: Compute relevance scores for query-document pairs
2. **BM25 Integration**: Extract BM25 scores from document metadata
3. **Score Fusion**: Combine cross-encoder and BM25 scores:
   ```python
   final_score = cross_encoder_score + (bm25_score * bm25_weight)
   ```
4. **Ranking**: Sort documents by combined scores
5. **Top-N Selection**: Return highest-scoring documents

**Model Support**:
- `cross-encoder/ms-marco-MiniLM-L6-v2` (standard)
- `mixedbread-ai/mxbai-rerank-base-v2` (specialized)

### 6. Project Management Enhancement

#### Project-Specific BM25 Storage
**Directory Structure**:
```
projects/[ProjectName]/
├── docs_pdf/          # Original PDFs
├── docs_md/           # Processed text
└── embeddings/        # Model-specific data
    └── [model_name]/
        └── chunk_size_[X]/
            └── chunk_overlap_[Y]/
                ├── [doc_id]/      # Document embeddings
                │   ├── chunk_0.pkl
                │   └── chunk_1.pkl
                └── bm25/          # BM25 index (NEW)
                    └── bm25.pkl
```

**Benefits**:
- Isolated BM25 indices per project
- Configuration-specific caching
- Consistent with existing embedding storage
- Easy cleanup when deleting projects

### 7. Query Processing Pipeline (Lines 2950-3099)

#### Complete Query Flow:
1. **Initialization** (QueryProgressTracker starts)
   ```python
   query_progress_tracker.set_stage("initializing", detail="Setting up retrieval...")
   ```

2. **Retrieval Stage** (30% weight)
   - Check if hybrid search is enabled
   - **If Hybrid Enabled**:
     - Perform dense retrieval: `user_input` → `retrieve_all()` → all documents with semantic scores
     - Load/create BM25 corpus 
     - Perform sparse retrieval: `user_input` → tokenize → `bm25.get_scores()` → all documents with BM25 scores
     - Normalize BM25 scores and fuse with dense scores
     - Select top-K documents based on combined scores
   - **If Dense Only**: 
     - Perform standard dense retrieval: `user_input` → `retrieve_top_k()`

3. **Reranking Stage** (25% weight)
   ```python
   query_progress_tracker.set_stage("reranking", detail=f"Reranking to top {top_n} documents")
   ```
   - Perform hybrid reranking if enabled (using separate BM25 weight)
   - Or standard cross-encoder reranking
   - Select final top-N documents

4. **Generation Stage** (40% weight)
   ```python
   query_progress_tracker.set_stage("generating", detail="Generating response with retrieved context")
   ```
   - Assemble context from reranked documents
   - Format with document IDs for references
   - Generate LLM response with context

### 8. UI Integration and Real-time Updates

#### Hybrid Search Controls (Lines 1908-1947)
**Configuration Panel**:
- Toggle for enabling hybrid search
- Weight sliders with real-time updates
- Descriptive help text for parameters
- Bootstrap styling for consistency

#### Progress Display Integration
**Query Progress UI** (Lines 1633-1644):
- Real-time progress bar updates
- Stage-specific status messages
- Detailed progress information
- Automatic UI refresh during processing

#### Statistics Tracking Enhancement
**PersonalStatistics Integration**:
- Track hybrid vs. dense-only queries
- Monitor BM25 corpus creation events
- Record retrieval and reranking performance
- Document reference tracking for analytics

## Key Architectural Improvements

### 1. **True Hybrid Retrieval**
- User query processed through BOTH dense and sparse retrieval simultaneously
- Semantic similarity scores from bi-encoder on query
- BM25 keyword scores from same query tokenization
- Score fusion with configurable weighting
- Graceful fallback to dense-only search

### 2. **Enhanced Progress Tracking**
- Multi-stage progress tracking for queries
- Real-time UI updates during processing
- Detailed logging with timestamps
- User-friendly progress messages

### 3. **Hybrid Configuration Management**
- Runtime configuration updates
- Separate weights for retrieval and reranking
- UI-driven parameter adjustment
- Session-persistent settings

### 4. **Performance Optimizations**
- BM25 corpus caching and reuse
- Singleton pattern for model management
- Batch processing for large document sets
- Memory-efficient score computation

## Usage Patterns

### Basic Hybrid Search
1. Enable hybrid search in configuration panel
2. Adjust BM25 weights (retrieval: 0.3, reranking: 0.1 typical)
3. Upload documents (automatically creates BM25 index)
4. Query processing uses both semantic and keyword matching on the same query

### Dense-Only Fallback
- Automatically falls back if BM25 creation fails
- User can disable hybrid search for pure semantic search
- Maintains compatibility with original dense-only pipeline

### Performance Monitoring
- Real-time progress tracking during query processing
- Detailed logs for debugging and optimization
- Statistics tracking for usage analytics
- Error handling with graceful degradation

This hybrid implementation represents a sophisticated RAG system that combines the best of semantic and keyword search by processing each user query through both retrieval methods simultaneously, then intelligently fusing the results for optimal document retrieval performance.