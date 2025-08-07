# Project management utilities for RAG application
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
import nltk
from rank_bm25 import BM25Okapi

def get_project_directories(base_dir, project_name):
    """Get project-specific directory paths"""
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    project_base = base_dir / "projects" / project_name
    return {
        'docs_pdf': project_base / "docs_pdf",
        'docs_md': project_base / "docs_md", 
        'embeddings': project_base / "embeddings"
    }

def get_available_projects(base_dir):
    """Get list of available projects"""
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    projects_dir = base_dir / "projects"
    if not projects_dir.exists():
        return []
    
    projects = []
    for project_dir in projects_dir.iterdir():
        if project_dir.is_dir():
            projects.append(project_dir.name)
    return sorted(projects)

def create_project(base_dir, project_name):
    """Create a new project with necessary directories"""
    if not project_name or not project_name.strip():
        raise ValueError("Project name cannot be empty")
    
    # Sanitize project name
    project_name = project_name.strip().replace("/", "_").replace("\\", "_")
    
    project_dirs = get_project_directories(base_dir, project_name)
    
    # Create all project directories
    for dir_path in project_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return project_name

def delete_project(base_dir, project_name):
    """Delete a project and all its data"""
    if not project_name:
        raise ValueError("Project name cannot be empty")
    
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    project_base = base_dir / "projects" / project_name
    if project_base.exists():
        shutil.rmtree(project_base)

def get_bm25_path(base_dir, project_name, model_name, chunk_size, chunk_overlap):
    """Get BM25 storage path for a specific configuration"""
    project_dirs = get_project_directories(base_dir, project_name)
    bm25_path = project_dirs['embeddings'] / model_name.split("/")[-1] / f"chunk_size_{chunk_size}" / f"chunk_overlap_{chunk_overlap}" / "bm25"
    return bm25_path / "bm25.pkl"

def create_bm25_corpus(documents_embeddings, bm25_path):
    """Create and save BM25 corpus from document embeddings"""
    try:
        # Create BM25 directory if it doesn't exist
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract text from documents
        corpus = [doc["text"] for doc in documents_embeddings]
        
        # Tokenize corpus
        tokenized_corpus = [nltk.word_tokenize(doc) for doc in corpus]
        
        # Create BM25 object
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save BM25 corpus
        with open(bm25_path, "wb") as f:
            pickle.dump(tokenized_corpus, f)
        
        return bm25, tokenized_corpus
    
    except Exception as e:
        return None, None

def load_bm25_corpus(bm25_path):
    """Load BM25 corpus from file"""
    try:
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                tokenized_corpus = pickle.load(f)
            
            bm25 = BM25Okapi(tokenized_corpus)
            return bm25, tokenized_corpus
        else:
            return None, None
    
    except Exception as e:
        return None, None

def build_conversation_history(messages_json, max_turns=8):
    """Build conversation history from messages JSON, limited to max_turns to avoid token limits"""
    try:
        messages = json.loads(messages_json)
        
        # Get the most recent messages (excluding the current user input which will be added later)
        # Use max_turns*2 because each turn has both user and assistant messages
        recent_messages = messages[-max_turns*2:] if len(messages) > max_turns*2 else messages
        
        # Build conversation history string
        history_parts = []
        for msg in recent_messages:
            if msg['role'] == 'user':
                history_parts.append(f"Human: {msg['content']}")
            else:
                history_parts.append(f"Assistant: {msg['content']}")
        
        return "\n\n".join(history_parts) if history_parts else ""
    except:
        return ""

def build_openai_conversation_history(messages_json, max_turns=8):
    """Build OpenAI-formatted conversation history from messages JSON, limited to avoid token limits"""
    try:
        messages = json.loads(messages_json)
        
        # Get the most recent messages (excluding the current user input which will be added later)
        # Use max_turns*2 because each turn has both user and assistant messages
        recent_messages = messages[-max_turns*2:] if len(messages) > max_turns*2 else messages
        
        # Build OpenAI messages format
        openai_messages = []
        for msg in recent_messages:
            if msg['role'] == 'user':
                openai_messages.append({"role": "user", "content": msg['content']})
            else:
                openai_messages.append({"role": "assistant", "content": msg['content']})
        
        return openai_messages
    except:
        return []

def build_gemini_conversation_history(messages_json, max_turns=8):
    """Build Gemini-formatted conversation history from messages JSON, limited to avoid token limits"""
    try:
        messages = json.loads(messages_json)
        
        # Get the most recent messages (excluding the current user input which will be added later)
        # Use max_turns*2 because each turn has both user and assistant messages
        recent_messages = messages[-max_turns*2:] if len(messages) > max_turns*2 else messages
        
        # Build Gemini chat history format
        gemini_history = []
        for msg in recent_messages:
            if msg['role'] == 'user':
                gemini_history.append({
                    "role": "user",
                    "parts": [{"text": msg['content']}]
                })
            else:
                gemini_history.append({
                    "role": "model", 
                    "parts": [{"text": msg['content']}]
                })
        
        return gemini_history
    except:
        return []

def collect_feedback_information(user_query, llm_response, user_comment, retrieved_chunks=None, session_data=None):
    """Collect comprehensive feedback information for sharing"""
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get current project and RAG configuration
    current_project = session_data.get('current_project', 'No project selected') if session_data else 'No project selected'
    rag_config = session_data.get('bi_encoder_config', {}) if session_data else {}
    cross_encoder_config = session_data.get('cross_encoder_config', {}) if session_data else {}
    llm_config = session_data.get('llm_config', {}) if session_data else {}
    hybrid_config = session_data.get('hybrid_search_config', {}) if session_data else {}
    
    # Get project documents information
    documents_info = "No documents information available"
    embedding_status = "No embedding information available"
    
    if session_data and current_project and current_project != 'No project selected':
        try:
            # Get project directories
            project_dirs = get_project_directories(Path.cwd(), current_project)
            pdf_dir = project_dirs['docs_pdf']
            
            # List PDF files
            pdf_files = []
            if os.path.exists(pdf_dir):
                pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            
            documents_info = f"Project: {current_project}\nPDF files ({len(pdf_files)}): {', '.join(pdf_files) if pdf_files else 'None'}"
            
            # Check embedding status
            embeddings_exist = len(session_data.get('embeddings', [])) > 0
            embedding_status = f"Embeddings loaded: {'Yes' if embeddings_exist else 'No'}"
            if embeddings_exist:
                embedding_status += f" ({len(session_data['embeddings'])} chunks)"
            
        except Exception as e:
            documents_info = f"Error retrieving document info: {str(e)}"
    
    # Format retrieved chunks information
    chunks_info = "No chunks retrieved (direct chat mode)"
    if retrieved_chunks:
        chunks_info = f"Retrieved chunks: {len(retrieved_chunks)}\n"
        for i, chunk in enumerate(retrieved_chunks[:3]):  # Show first 3 chunks
            chunks_info += f"\nChunk {i+1}:\n"
            chunks_info += f"Text preview: {chunk.get('text', '')[:200]}...\n"
            chunks_info += f"Document: {chunk.get('original_doc_id', 'Unknown')}\n"
            chunks_info += f"Page: {chunk.get('page', 'Unknown')}\n"
            chunks_info += f"Similarity: {chunk.get('similarity', 'N/A')}\n"
            if 'rerank_score' in chunk:
                chunks_info += f"Rerank score: {chunk.get('rerank_score', 'N/A')}\n"
    
    return {
        'timestamp': timestamp,
        'user_query': user_query,
        'llm_response': llm_response,
        'user_comment': user_comment,
        'documents_info': documents_info,
        'embedding_status': embedding_status,
        'chunks_info': chunks_info,
        'configs': {
            'rag_config': rag_config,
            'cross_encoder_config': cross_encoder_config,
            'llm_config': llm_config,
            'hybrid_config': hybrid_config
        }
    }