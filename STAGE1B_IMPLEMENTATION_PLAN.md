# Stage 1B: Experts Module Implementation Plan

## Architecture Overview

**Document Processing: Docling**
- IBM's open-source document processing framework
- Advanced PDF parsing with layout understanding
- Structured content extraction (text, tables, images)
- Multiple output formats (JSON, Markdown, DoclingDocument)

**Vector Database: ChromaDB**
- Local-first vector database with Python integration
- Built-in embedding functions (OpenAI, sentence-transformers)
- Efficient similarity search and filtering
- Persistent storage with SQLite backend

**Embeddings Strategy: OpenAI Ada-002 + Local Fallback**
- Primary: OpenAI text-embedding-ada-002 (high quality)
- Fallback: sentence-transformers/all-MiniLM-L6-v2 (local, free)
- Chunk-level embeddings with metadata preservation
- Cost optimization through intelligent caching

**Integration Approach: Parallel Development**
- New experts module alongside existing authors module
- Shared fine-tuning infrastructure (`train.py`, `adapters/`)
- Gradual migration tools for authors → experts conversion
- Zero breaking changes to existing functionality

## Implementation Structure

```
simtune/
├── cli/commands/
│   ├── expert.py          # Expert profile management
│   ├── knowledge.py       # Knowledge base operations
│   └── author.py          # Preserved as-is
├── core/
│   ├── knowledge/         # PDF processing and management
│   │   ├── __init__.py
│   │   ├── processor.py   # Docling integration
│   │   ├── chunker.py     # Document chunking strategies
│   │   └── validator.py   # Document quality checks
│   ├── embeddings/        # Vector database operations
│   │   ├── __init__.py
│   │   ├── generator.py   # Embedding creation (OpenAI + local)
│   │   ├── store.py       # ChromaDB integration
│   │   └── retriever.py   # Similarity search and context retrieval
│   ├── experts/           # Expert-specific business logic
│   │   ├── __init__.py
│   │   ├── manager.py     # Expert profile management
│   │   ├── dataset_generator.py  # Training data from knowledge base
│   │   └── migration.py   # Author → Expert conversion
│   └── [existing modules] # Preserved
├── data/
│   ├── experts/<expert_id>/     # Expert profiles and knowledge bases
│   │   ├── profile.json         # ExpertProfile data
│   │   ├── domain_config.yml    # Domain expertise configuration
│   │   ├── knowledge_base/      # Document storage and processing
│   │   │   ├── documents/       # Original PDFs
│   │   │   ├── processed/       # Docling output (JSON/MD)
│   │   │   ├── chunks/          # Document chunks with metadata
│   │   │   └── vectors.db       # ChromaDB database
│   │   ├── train.jsonl          # Generated training dataset
│   │   ├── content/             # Generated content with source attribution
│   │   └── models.json          # Fine-tune job metadata
│   └── authors/                 # Preserved as-is
└── requirements.txt             # Updated with new dependencies
```

## Core Functionality Implementation

### 1. Expert Profile Management
**New Data Models:**
```python
class DomainConfiguration(BaseModel):
    domain: str = Field(description="Primary domain expertise")
    subdomain: List[str] = Field(default_factory=list)
    expertise_level: str = Field(default="intermediate")  # beginner, intermediate, expert
    focus_areas: List[str] = Field(default_factory=list)
    citation_style: str = Field(default="apa")  # apa, mla, chicago, ieee
    response_style: str = Field(default="comprehensive")  # brief, comprehensive, detailed
    
class ExpertProfile(BaseModel):
    expert_id: str = Field(description="Unique identifier for the expert")
    name: str = Field(description="Display name for the expert")
    description: str = Field(default="", description="Brief description")
    domain_config: DomainConfiguration = Field(default_factory=DomainConfiguration)
    knowledge_base_stats: KnowledgeBaseStats = Field(default_factory=KnowledgeBaseStats)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
```

### 2. PDF Document Processing Pipeline
**Step 1: Document Import**
- CLI command: `simtune knowledge import <expert_id> <pdf_path>`
- File validation (PDF format, size limits, corruption checks)
- Organized storage in `data/experts/<expert_id>/knowledge_base/documents/`
- Metadata extraction (title, author, creation date, page count)

**Step 2: Docling Processing**
- CLI command: `simtune knowledge process <expert_id>`
- Batch processing with progress tracking
- Multiple output formats (JSON for structure, Markdown for readability)
- Error handling for corrupted or complex documents
- Processing statistics and quality metrics

### 3. Document Chunking and Preparation
**Intelligent Chunking Strategy:**
```python
class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_structure(self, doc: DoclingDocument) -> List[DocumentChunk]:
        """Chunk by document structure (sections, paragraphs)"""
    
    def chunk_by_semantic_similarity(self, text: str) -> List[DocumentChunk]:
        """Chunk by semantic coherence using embeddings"""
    
    def chunk_by_fixed_size(self, text: str) -> List[DocumentChunk]:
        """Fallback: fixed-size chunks with overlap"""
```

### 4. Vector Database Integration
**ChromaDB Implementation:**
- Expert-specific collections: `expert_{expert_id}_knowledge`
- Chunk-level embeddings with rich metadata
- Hybrid search: semantic similarity + keyword filtering
- Efficient storage with automatic persistence

**Embedding Generation:**
```python
class EmbeddingGenerator:
    def __init__(self, provider: str = "openai"):
        self.provider = provider  # "openai" or "local"
    
    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[np.ndarray]:
        """Generate embeddings with automatic fallback"""
    
    def estimate_cost(self, text_length: int) -> float:
        """Cost estimation for OpenAI embeddings"""
```

### 5. Training Dataset Generation
**Knowledge-to-Training Pipeline:**
```python
class KnowledgeDatasetGenerator:
    def generate_qa_pairs(self, expert_id: str, num_examples: int = 100) -> List[TrainingExample]:
        """Generate question-answer pairs from knowledge base"""
    
    def generate_explanation_examples(self, chunks: List[DocumentChunk]) -> List[TrainingExample]:
        """Create explanation-style training examples"""
    
    def generate_citation_examples(self, chunks: List[DocumentChunk]) -> List[TrainingExample]:
        """Create examples with proper source attribution"""
```

### 6. Knowledge-Based Query System
**Context-Aware Retrieval:**
- Similarity search with configurable thresholds
- Source attribution and citation generation
- Multi-document synthesis capabilities
- Query expansion and refinement

## CLI Command Implementation

### Expert Management (`cli/commands/expert.py`)
```bash
# Create new domain expert
simtune expert create <expert_id> --domain "Machine Learning" --interactive

# List all experts
simtune expert list

# Show expert details with knowledge base statistics
simtune expert show <expert_id>

# Convert author profile to expert profile
simtune expert migrate <author_id> --domain "Writing" --preserve-data

# Delete expert (with confirmation)
simtune expert delete <expert_id>
```

### Knowledge Base Management (`cli/commands/knowledge.py`)
```bash
# Import PDF documents
simtune knowledge import <expert_id> <pdf_path> [--batch-directory]

# Process documents with Docling
simtune knowledge process <expert_id> [--reprocess-all]

# Generate vector embeddings
simtune knowledge embed <expert_id> [--provider openai|local]

# Search knowledge base
simtune knowledge search <expert_id> "machine learning algorithms" [--limit 5]

# Generate training dataset from knowledge base
simtune knowledge generate-dataset <expert_id> [--examples 200] [--quality-filter]

# Knowledge base statistics and health check
simtune knowledge stats <expert_id>

# Export knowledge base (for backup/sharing)
simtune knowledge export <expert_id> --format [json|markdown|archive]
```

## Technical Implementation Details

### 1. Document Processing Service
```python
class DocumentProcessor:
    def __init__(self, expert_id: str):
        self.expert_id = expert_id
        self.storage = ExpertStorage(expert_id)
        self.docling_processor = DoclingProcessor()
    
    async def process_document(self, pdf_path: Path) -> ProcessingResult:
        """Process single PDF with Docling"""
        
    async def batch_process(self, progress_callback=None) -> BatchProcessingResult:
        """Process all unprocessed documents"""
        
    def validate_document(self, pdf_path: Path) -> ValidationResult:
        """Validate PDF before processing"""
```

### 2. Vector Database Service
```python
class VectorStore:
    def __init__(self, expert_id: str):
        self.expert_id = expert_id
        self.collection = self._get_or_create_collection()
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks with embeddings"""
    
    def similarity_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Semantic similarity search"""
    
    def hybrid_search(self, query: str, filters: Dict) -> List[SearchResult]:
        """Combine semantic and metadata filtering"""
```

### 3. Expert Storage Extension
```python
class ExpertStorage:
    def __init__(self, expert_id: str):
        self.expert_id = expert_id
        self.expert_dir = settings.experts_dir / expert_id
        # Similar pattern to AuthorStorage but for experts
    
    def save_profile(self, profile: ExpertProfile) -> None:
        """Save expert profile and domain configuration"""
    
    def load_profile(self) -> Optional[ExpertProfile]:
        """Load expert profile with validation"""
    
    def save_knowledge_stats(self, stats: KnowledgeBaseStats) -> None:
        """Save knowledge base statistics"""
```

### 4. Migration Service
```python
class AuthorToExpertMigrator:
    def migrate_author(self, author_id: str, domain: str) -> ExpertProfile:
        """Convert author profile to expert profile"""
    
    def preserve_training_data(self, author_id: str, expert_id: str) -> None:
        """Preserve existing training data"""
    
    def generate_migration_report(self) -> MigrationReport:
        """Report on migration success and issues"""
```

## Dependency Management

### New Requirements
```txt
# PDF Processing
docling>=1.0.0
pypdf2>=3.0.0

# Vector Database
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Document Processing
python-magic>=0.4.27
filetype>=1.2.0

# Performance
numpy>=1.24.0
scikit-learn>=1.3.0

# Optional: Advanced chunking
nltk>=3.8.1
spacy>=3.5.0
```

### Environment Configuration
```env
# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY=./data/vectors
CHROMA_COLLECTION_PREFIX=simtune_

# Embedding Configuration  
EMBEDDING_PROVIDER=openai  # openai | local
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_BATCH_SIZE=100

# Document Processing
MAX_DOCUMENT_SIZE_MB=50
DOCLING_TIMEOUT_SECONDS=300
PARALLEL_PROCESSING_WORKERS=4
```

## Integration with Existing Infrastructure

### 1. Fine-Tuning Reuse
- **Complete Compatibility**: Generated datasets use same TrainingExample format
- **Shared Commands**: `simtune train start <expert_id>` works identically
- **Model Management**: Same ModelMetadata and job tracking
- **Provider Abstraction**: Same OpenAI adapter integration

### 2. Content Generation Enhancement
```python
# Enhanced generate.py for context-aware generation
@generate_app.command("query")
def query_expert(
    expert_id: str,
    prompt: str = typer.Option(..., "--prompt", "-p"),
    with_sources: bool = typer.Option(False, "--with-sources"),
    context_limit: int = typer.Option(5, "--context-limit")
):
    """Generate response with knowledge base context"""
```

### 3. Storage System Extension
- **Parallel Structure**: `data/experts/` alongside `data/authors/`
- **Shared Utilities**: Reuse markdown_utils, validation helpers
- **Consistent Patterns**: Same JSON/YAML/JSONL approach

## Testing Strategy

### 1. Unit Tests
```python
# tests/unit/core/knowledge/test_processor.py
def test_docling_pdf_processing(sample_pdf):
    processor = DocumentProcessor("test_expert")
    result = processor.process_document(sample_pdf)
    assert result.success
    assert len(result.chunks) > 0

# tests/unit/core/embeddings/test_generator.py  
def test_embedding_generation_openai(mock_openai_client):
    generator = EmbeddingGenerator("openai")
    embeddings = generator.generate_embeddings(sample_chunks)
    assert len(embeddings) == len(sample_chunks)
```

### 2. Integration Tests
```python
# tests/integration/test_expert_workflow.py
def test_complete_expert_workflow():
    # Create expert → Import PDFs → Process → Embed → Generate dataset → Train
    expert_id = "test_ml_expert"
    workflow = ExpertWorkflow(expert_id)
    result = workflow.run_complete_pipeline()
    assert result.success
```

### 3. CLI Tests  
```python
# tests/integration/test_expert_cli.py
def test_expert_commands():
    runner = CliRunner()
    
    # Test expert creation
    result = runner.invoke(expert_app, ["create", "test_expert", "--domain", "AI"])
    assert result.exit_code == 0
    
    # Test knowledge import
    result = runner.invoke(knowledge_app, ["import", "test_expert", "sample.pdf"])
    assert result.exit_code == 0
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- **Core Data Models**: ExpertProfile, DomainConfiguration, DocumentChunk
- **Basic CLI Structure**: expert.py and knowledge.py command scaffolding  
- **Storage System**: ExpertStorage class with save/load functionality
- **PDF Import**: Basic file handling and validation

### Phase 2: Document Processing (Weeks 3-4)
- **Docling Integration**: PDF processing pipeline with error handling
- **Document Chunking**: Multiple chunking strategies with quality metrics
- **Processing CLI**: `simtune knowledge process` with progress tracking
- **Unit Tests**: Comprehensive coverage of document processing

### Phase 3: Vector Database (Weeks 5-6)
- **ChromaDB Integration**: Vector storage and retrieval system
- **Embedding Generation**: OpenAI + local embedding providers
- **Search Functionality**: `simtune knowledge search` with ranking
- **Performance Optimization**: Batch processing and caching

### Phase 4: Dataset Generation (Weeks 7-8)
- **Training Data Pipeline**: Knowledge → TrainingExample conversion
- **Quality Filtering**: Relevance scoring and deduplication
- **Integration Testing**: End-to-end workflow validation
- **Fine-tuning Integration**: Verify compatibility with existing train.py

### Phase 5: Migration & Polish (Weeks 9-10)
- **Migration Tools**: Author → Expert conversion utilities
- **CLI Enhancements**: Rich formatting, progress bars, error handling
- **Documentation**: Comprehensive usage guides and API docs
- **Performance Tuning**: Memory optimization and processing speed

### Phase 6: Production Readiness (Weeks 11-12)
- **Error Recovery**: Robust handling of edge cases and failures
- **Monitoring**: Knowledge base health checks and statistics
- **Security**: Input validation and safe file handling
- **Final Testing**: Stress testing with large document collections

## Migration Strategy

### 1. Author → Expert Conversion
```python
# Automated migration tool
class MigrationTool:
    def convert_author_to_expert(self, author_id: str, domain: str) -> ExpertProfile:
        """Convert existing author to domain expert"""
        
    def preserve_training_history(self, author_id: str, expert_id: str) -> None:
        """Maintain fine-tuning job history"""
        
    def generate_migration_summary(self) -> Dict:
        """Report on what was preserved/changed"""
```

### 2. Gradual Adoption Path
- **Phase 1**: Experts alongside authors (both functional)
- **Phase 2**: Author profiles gain "upgrade to expert" option  
- **Phase 3**: New users guided toward experts by default
- **Phase 4**: Authors marked as "legacy" but still supported

### 3. Data Preservation Guarantees
- **Zero Data Loss**: All author profiles, datasets, and models preserved
- **Backward Compatibility**: Existing CLI commands continue to work
- **Rollback Safety**: Migration process is reversible
- **Export Options**: Complete data export before any changes

## Success Metrics

### 1. Technical Performance
- **PDF Processing**: < 30 seconds per typical research paper
- **Embedding Generation**: < 5 seconds per 1000 tokens (OpenAI)
- **Search Latency**: < 500ms for knowledge base queries
- **Memory Usage**: < 2GB for 100-document knowledge base

### 2. Quality Metrics
- **Processing Success Rate**: > 95% for well-formed PDFs
- **Embedding Quality**: Semantic search accuracy > 80%
- **Dataset Relevance**: Human evaluation score > 4/5
- **Fine-tuning Effectiveness**: Comparable results to author module

### 3. User Experience
- **Setup Time**: < 10 minutes from expert creation to first query
- **Error Recovery**: Clear error messages with actionable guidance
- **Documentation Coverage**: Complete CLI help and user guides
- **Migration Success**: 100% successful author → expert conversions

This comprehensive implementation plan provides a robust foundation for transforming Simtune from a writing style fine-tuning tool into a domain knowledge expert platform while preserving all existing functionality and providing clear migration paths.