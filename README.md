# 8devices RAG Chatbot (v.0.2.0)

A powerful Retrieval-Augmented Generation (RAG) chatbot application built with Dash that allows users to upload PDF documents and ask questions about them using local AI models or cloud APIs. Features complete project management, multi-model support, and enhanced conversational capabilities.

## Features

### 🔒 **Privacy-First Design**
- All models can run locally on your machine
- Documents are stored locally - no data sent to external servers
- Complete privacy and data security for local models
- Optional cloud API integration for enhanced performance

### 📁 **Project Management System**
- **Multi-project organization** - Create separate projects for different document collections
- **Project-specific storage** - Documents organized in `projects/[ProjectName]/` structure
- **Automatic project creation** - Seamless project setup and management
- **Project-based PDF serving** - Secure document access with clickable references

### 📄 **Advanced PDF Document Processing**
- Upload multiple PDF files via drag-and-drop interface
- Automatic text extraction using Docling and Marker
- Smart document chunking with configurable parameters
- Project-specific document processing and resume functionality
- Enhanced file validation and error handling

### 🧠 **Enhanced RAG Pipeline**
- **Bi-encoder retrieval** using Snowflake Arctic or BGE-M3 models
- **Cross-encoder reranking** for improved relevance
- **Configurable chunk sizes** (512, 1024, 2048, 4096 tokens)
- **Adjustable retrieval counts** (50, 100, 200 documents)
- **Dynamic reranking** based on chunk size constraints
- **Project-aware document linking** with clickable PDF references

### 🎯 **Flexible Prompt Strategies**
Choose from 4 different prompt types with varying knowledge constraints:
- **Strict**: Only uses uploaded documents, no external knowledge
- **Moderate**: Primarily documents, supplements with LLM knowledge when needed  
- **Loose**: Uses documents as starting point, freely combines with broader knowledge
- **Simple**: Document-focused with minimal constraints

### 💬 **Enhanced Chat Interface**
- **Conversation modes**: Single-turn (independent) or multi-turn (8-turn history)
- **Real-time chat** with document-aware responses
- **RAG mode toggle** for document-based vs. general chat
- **Enhanced content rendering** with clickable links and formatting
- **Live processing log** to monitor system operations
- **Document reference citations** with clickable PDF links

### 🤖 **Multi-Model Support**
- **Local Ollama models** (privacy-focused)
- **OpenAI models** (GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano, o3, o4-mini)
- **Google Gemini models** (Gemini 2.5 Pro, Flash, Flash-Lite)
- **Unified API key management** for cloud providers
- **Dynamic model switching** without restart

### ⚙️ **Advanced Configuration**
- Collapsible advanced settings panel
- Model selection for bi-encoder, cross-encoder, and LLM
- Configurable chunking and retrieval parameters
- Conversation mode selection
- Prompt format preview functionality

## Installation

### Prerequisites
- Python 3.10 or higher
- Ollama installed and running locally (for local LLM inference)
- OpenAI API key (optional, for OpenAI models)
- Google Gemini API key (optional, for Gemini models)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kmrdknrd/rag_app_dash.git
   cd rag_app_dash
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama (for local models):**
   ```bash
   # Install Ollama (visit https://ollama.ai for platform-specific instructions)
   
   # Pull recommended models
   ollama pull qwen3:1.7b
   ollama pull cogito:3b
   ```

4. **Run the application:**
   ```bash
   python app_projects.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:8061`

## Usage

### Quick Start
1. **Create or select a project** using the project dropdown
2. **Enable RAG Mode** using the toggle at the top
3. **Upload PDF files** by dragging and dropping into the upload area
4. **Wait for processing** - monitor progress in the Processing Log
5. **Start chatting** once processing is complete

### Advanced Usage

#### Project Management
1. **Create new projects** for different document collections
2. **Switch between projects** using the project dropdown
3. **Delete projects** when no longer needed (removes all associated files)

#### Model Configuration
1. **Select LLM model**:
   - Choose from local Ollama models (privacy-focused, slower)
   - Or select OpenAI models (faster, requires API key)
   - Or select Google Gemini models (fast, cost-effective, requires API key)
   - Enter your API key when prompted for cloud models

2. **Configure RAG settings** using the "Advanced RAG Configuration" panel:
   - Choose bi-encoder model and chunking parameters
   - Select cross-encoder model and reranking count
   - Adjust document retrieval numbers

#### Conversation Settings
1. **Select conversation mode**:
   - **Single-turn**: Each query is independent
   - **Multi-turn**: Maintains 8-turn conversation history

2. **Select prompt strategy** based on your needs:
   - Use "Strict" for fact-checking scenarios
   - Use "Moderate" for general Q&A with safety
   - Use "Loose" for creative or exploratory queries
   - Use "Simple" for streamlined document interaction

#### Document Management
1. **Resume previous sessions** by selecting an existing project
2. **Check for processed PDFs** to reload existing documents
3. **Click document references** in responses to view source PDFs

## Architecture

### Key Components
- **PDF Processors**: Docling and Marker for robust text extraction
- **Embedders**: Sentence Transformers for semantic search
- **Rerankers**: Cross-encoders for relevance refinement  
- **LLM**: Multi-provider support (Ollama, OpenAI, Gemini)
- **Interface**: Dash with Bootstrap components
- **Project Management**: File-based project organization

### Project Structure
```
projects/
├── [ProjectName]/
│   ├── docs_pdf/          # Original PDF files
│   ├── docs_md/           # Processed markdown text
│   └── embeddings/        # Stored embeddings by model/config
│       └── [model_name]/
│           └── chunk_size_[X]/
│               └── chunk_overlap_[Y]/
│                   └── [doc_id]/
│                       ├── chunk_0.pkl
│                       └── chunk_1.pkl
```

## Configuration Options

### Bi-Encoder Models
- `Snowflake/snowflake-arctic-embed-l-v2.0` (default)
- `BAAI/bge-m3`

### Cross-Encoder Models  
- `cross-encoder/ms-marco-MiniLM-L6-v2` (default)
- `mixedbread-ai/mxbai-rerank-base-v2`

### LLM Models
#### Local Ollama Models
- `qwen3:1.7b`
- `qwen3:4b`, `qwen3:8b`
- `cogito:3b`, `cogito:8b`
- `deepseek-r1:1.5b`, `deepseek-r1:7b`, `deepseek-r1:8b`

#### OpenAI Models (requires API key)
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `o3`
- `o4-mini`

#### Google Gemini Models (requires API key)
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite-preview-06-17` (default)

### Chunking Parameters
- **Chunk Size**: 512, 1024, 2048, 4096 tokens
- **Chunk Overlap**: 0, 128, 256 tokens
- **Retrieval Count**: 50, 100, 200 documents
- **Reranking Count**: Automatically adjusted based on chunk size

## Performance Notes

- Processing a 1MB document takes approximately 45 seconds
- First-time model loading may take several minutes
- Local models provide privacy but are slower than cloud APIs
- Cloud APIs (OpenAI/Gemini) offer faster responses but require internet connection
- Project-specific caching improves performance for repeated use

## API Key Setup

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an API key
3. Enter the key in the application when prompted

### Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Enter the key in the application when prompted

## Troubleshooting

### Common Issues
1. **Slow processing**: Expected behavior, especially with local models
2. **Models not loading**: Ensure sufficient RAM and disk space
3. **Ollama connection errors**: Verify Ollama is running locally
4. **PDF processing failures**: Try different PDF files or check logs in `app_log.txt`
5. **API key errors**: Verify your API key is correct and has sufficient credits

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and documents
- **CPU**: Modern multi-core processor recommended
- **Internet**: Required for cloud models (OpenAI/Gemini)

## Logging and Debugging

- All operations are logged to `app_log.txt`
- Processing logs are displayed in real-time in the UI
- Error messages include detailed information for troubleshooting

## Development

### Adding Features
The modular design allows easy extension of:
- New PDF processors
- Additional embedding models
- Custom prompt templates
- Enhanced UI components
- New cloud model providers

### Key Files
- `app_projects.py` - Main application file
- `CLAUDE.md` - Development guidelines and architecture documentation
- `app_log.txt` - Application logs
- `projects/` - Project data storage

## License

This project is in active development. For feedback and contributions, contact: konradas.m@8devices.com

## Changelog

### v0.2.0 (Current)
- **NEW**: Complete project management system
- **NEW**: Google Gemini model integration
- **NEW**: Multi-turn conversation support
- **NEW**: Enhanced HTML content rendering
- **NEW**: Project-aware document linking
- **NEW**: File-based logging system
- **IMPROVED**: Better error handling and recovery
- **IMPROVED**: Enhanced UI with dynamic controls
- **IMPROVED**: Optimized memory management

### v0.1.0
- Initial release with basic RAG functionality
- Local Ollama model support
- Basic OpenAI integration
- Single-turn conversations

---

**Note**: This application continues to evolve rapidly. The current focus is on providing a comprehensive RAG solution with flexible model support while maintaining user privacy and ease of use.