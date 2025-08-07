# Session state management and helper functions

session_data = {
    'embeddings': [],
    'doc_paths': [],
    'pdf_processor': None,
    'bi_encoder': None,
    'cross_encoder': None,
    'llm': None,
    'llm_client': None,
    'dir': None,
    'initialized': False,
    'projects': [],  # List of available projects
    'current_project': None,  # Currently selected project
    'bi_encoder_config': {
        'model_name': 'Snowflake/snowflake-arctic-embed-l-v2.0',
        'chunk_size': 2048,
        'chunk_overlap': 128,
        'retrieval_count': 50
    },
    'cross_encoder_config': {
        'model_name': 'cross-encoder/ms-marco-MiniLM-L6-v2',
        'top_n': 8
    },
    'hybrid_search_config': {
        'enabled': True,
        'bm25_weight': 0.1,
        'bm25_weight_rerank': 0.9
    },
    'llm_config': {
        'model_name': 'gemini-2.5-flash-lite-preview-06-17',
        'api_key': None
    },
    'rag_mode': True,  # Default to RAG mode
    'prompt_type': 'loose',  # Default prompt type
    'conversation_mode': 'single'  # Default conversation mode (single/multi)
}

def validate_session_state():
    """Validate that session data is properly initialized"""
    return session_data.get('dir') is not None and session_data.get('initialized', False)

def reset_models():
    """Reset all model instances to force reinitialization"""
    session_data['bi_encoder'] = None
    session_data['cross_encoder'] = None
    session_data['llm'] = None
    session_data['llm_client'] = None

def ensure_project_setup():
    """Ensure that a project is selected and directories exist"""
    from utils.project_management import get_project_directories
    
    if not session_data.get('current_project'):
        return False, "No project selected. Please create or select a project first."
    
    try:
        dirs = get_project_directories(session_data['current_project'])
        if not dirs:
            return False, f"Project directory setup failed for {session_data['current_project']}"
        session_data['dir'] = dirs
        return True, ""
    except Exception as e:
        return False, f"Error setting up project directories: {str(e)}"