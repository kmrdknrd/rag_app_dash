# 8devices RAG Chatbot (v.0.1.0)

A Retrieval-Augmented Generation (RAG) chatbot application built with Dash that allows users to upload PDF documents and ask questions about them using local AI models.

## Features

### 🔒 **Privacy**
- All models run locally on your machine
- Documents are stored locally - no data sent to external servers
- Complete privacy and data security

### 📄 **PDF Document Processing**
- Upload multiple PDF files via drag-and-drop interface
- Automatic text extraction using Docling
- Smart document chunking with configurable parameters
- Check for previously processed documents to resume sessions

### 🧠 **Advanced RAG Pipeline**
- **Bi-encoder retrieval** using Snowflake Arctic or BGE-M3 models
- **Cross-encoder reranking** for improved relevance
- **Configurable chunk sizes** (512, 1024, 2048, 4096 tokens)
- **Adjustable retrieval counts** (50, 100, 200 documents)
- **Dynamic reranking** based on chunk size constraints

### 🎯 **Flexible Prompt Strategies**
Choose from 4 different prompt types with varying knowledge constraints:
- **Strict**: Only uses uploaded documents, no external knowledge
- **Moderate**: Primarily documents, supplements with LLM knowledge when needed  
- **Loose**: Uses documents as starting point, freely combines with broader knowledge
- **Simple**: Document-focused with minimal constraints

### 💬 **Interactive Chat Interface**
- Real-time chat with document-aware responses
- RAG mode toggle for document-based vs. general chat
- Live processing log to monitor system operations
- Document reference citations in responses

### ⚙️ **Advanced Configuration**
- Collapsible advanced settings panel
- Model selection for both bi-encoder and cross-encoder
- Configurable chunking and retrieval parameters
- Prompt format preview functionality

## Installation

### Prerequisites
- Python 3.10 or higher
- Ollama installed and running locally (for LLM inference)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kmrdknrd/rag_app_dash.git
   cd dash_rag_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama:**
   ```bash
   # Install Ollama (visit https://ollama.ai for platform-specific instructions)
   
   # Pull the required model
   ollama pull qwen3:0.6b
   ```

4. **Run the application:**
   ```bash
   python app_six.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:8060`

## Usage

### Quick Start
1. **Enable RAG Mode** using the toggle at the top
2. **Upload PDF files** by dragging and dropping into the upload area
3. **Wait for processing** - monitor progress in the Processing Log
4. **Start chatting** once processing is complete

### Advanced Usage
1. **Configure RAG settings** using the "Advanced RAG Configuration" panel:
   - Choose bi-encoder model and chunking parameters
   - Select cross-encoder model and reranking count
   - Adjust document retrieval numbers

2. **Select prompt strategy** based on your needs:
   - Use "Strict" for fact-checking scenarios
   - Use "Moderate" for general Q&A with safety
   - Use "Loose" for creative or exploratory queries
   - Use "Simple" for streamlined document interaction

3. **Resume previous sessions** by clicking "Check for processed PDFs"

## Architecture

### Key Components
- **PDF Processors**: Docling and Marker for robust text extraction
- **Embedders**: Sentence Transformers for semantic search
- **Rerankers**: Cross-encoders for relevance refinement  
- **LLM**: Ollama-based local language model
- **Interface**: Dash with Bootstrap components

## Configuration Options

### Bi-Encoder Models
- `Snowflake/snowflake-arctic-embed-l-v2.0` (default)
- `BAAI/bge-m3`

### Cross-Encoder Models  
- `cross-encoder/ms-marco-MiniLM-L6-v2` (default)
- `mixedbread-ai/mxbai-rerank-base-v2`

### Chunking Parameters
- **Chunk Size**: 512, 1024, 2048, 4096 tokens
- **Chunk Overlap**: 0, 128, 256 tokens
- **Retrieval Count**: 50, 100, 200 documents
- **Reranking Count**: Automatically adjusted based on chunk size

## Performance Notes

- Processing a 1MB document takes approximately 45 seconds
- First-time model loading may take several minutes
- Keep the application window open during processing
- Local models provide privacy but are slower than cloud APIs

## Troubleshooting

### Common Issues
1. **Slow processing**: Expected behavior due to local models
2. **Models not loading**: Ensure sufficient RAM and disk space
3. **Ollama connection errors**: Verify Ollama is running locally
4. **PDF processing failures**: Try different PDF files or converters

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and documents
- **CPU**: Modern multi-core processor recommended

## Development

### Adding Features
The modular design allows easy extension of:
- New PDF processors
- Additional embedding models
- Custom prompt templates
- Enhanced UI components

## License

This project is in alpha development phase. For feedback and contributions, contact: konradas.m@8devices.com

## Roadmap

### Coming Soon:
- **Hyperlinks** to source documents in LLM responses
- **Multiple LLM options** (local and OpenAI API)
- **Persistent database** storage instead of memory-based
- **Enhanced document management**
- **Performance optimizations**

---

**Note**: This application is in active development and may change rapidly. Current focus is demonstrating RAG capabilities with local models while maintaining complete data privacy.