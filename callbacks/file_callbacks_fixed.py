"""
File processing callbacks for app_clean_modular.py
Handles document upload, processing, and file management functionality.
"""

import os
import base64
import time
import traceback
import pickle
from pathlib import Path
from dash import callback, Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from config.session_state import session_data, validate_session_state, ensure_project_setup
from models.pdf_processor import PdfProcessor
from models.bi_encoder import BiEncoderPipeline
from models.cross_encoder import CrossEncoderPipeline
from utils.project_management import get_project_directories, create_project, delete_project
from utils.content_processing import save_processed_documents, load_processed_documents, get_bm25_path, create_bm25_corpus
from utils.logging_setup import log_message


def register_file_callbacks(app, progress_tracker, personal_stats):
    """Register all file processing related callbacks"""
    
    @app.callback(
        [Output('upload-status', 'children'),
         Output('processing-stage', 'children')],
        [Input('upload-pdf', 'contents')],
        [State('upload-pdf', 'filename'),
         State('project-data', 'data')],
        prevent_initial_call=True
    )
    def handle_enhanced_file_upload(contents, filenames, project_data):
        """Handle file upload with enhanced processing and feedback"""
        if contents is None:
            raise PreventUpdate
        
        # Check if a project is selected
        current_project = project_data.get('current_project')
        if not current_project:
            return (
                dbc.Alert("Please select or create a project first.", color="warning", dismissable=True),
                'error'
            )
        
        try:
            progress_tracker.reset()
            progress_tracker.log_message("Starting file upload and processing...")
            
            # Stage 1: Upload files
            progress_tracker.set_stage("upload", len(contents))
            
            # Get project directories
            project_dirs = get_project_directories(session_data['dir'], current_project)
            
            # Handle both single and multiple file uploads
            if not isinstance(contents, list):
                contents = [contents]
                filenames = [filenames]
            
            # Process uploaded files
            uploaded_files = []
            pdf_files = []
            
            for i, (content, filename) in enumerate(zip(contents, filenames)):
                if not filename.lower().endswith('.pdf'):
                    progress_tracker.log_message(f"Skipping non-PDF file: {filename}")
                    continue
                    
                # Decode and save PDF
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                
                pdf_path = os.path.join(project_dirs['docs_pdf'], filename)
                with open(pdf_path, 'wb') as f:
                    f.write(decoded)
                
                uploaded_files.append({
                    'filename': filename,
                    'path': pdf_path,
                    'size': len(decoded)
                })
                pdf_files.append(Path(pdf_path))
                
                progress_tracker.log_message(f"Uploaded: {filename} ({len(decoded):,} bytes)")
                progress_tracker.update_stage_progress(i + 1, filename)
            
            if not uploaded_files:
                return dbc.Alert("No valid PDF files found", color="warning", dismissable=True), 'error'
            
            # Stage 3: Initialize models
            progress_tracker.set_stage("embedding_init", 1)
            
            # Initialize PDF processor if needed
            if not session_data.get('pdf_processor'):
                session_data['pdf_processor'] = PdfProcessor()
            
            # Initialize BiEncoder if needed
            if not session_data.get('bi_encoder'):
                bi_encoder_config = session_data.get('bi_encoder_config', {})
                session_data['bi_encoder'] = BiEncoderPipeline(
                    model_name=bi_encoder_config.get('model_name', 'Snowflake/snowflake-arctic-embed-l-v2.0'),
                    chunk_size=bi_encoder_config.get('chunk_size', 2048),
                    chunk_overlap=bi_encoder_config.get('chunk_overlap', 128)
                )
            
            # Initialize cross-encoder
            if not session_data.get('cross_encoder'):
                cross_encoder_config = session_data['cross_encoder_config']
                session_data['cross_encoder'] = CrossEncoderPipeline(
                    model_name=cross_encoder_config['model_name']
                )
            
            progress_tracker.update_stage_progress(1, "Models initialized")
            
            # Stage 2: Process PDFs and generate embeddings
            all_page_chunks = []
            doc_ids = []
            
            progress_tracker.set_stage('pdf_processing', len(pdf_files))
            
            for pdf_file in pdf_files:
                progress_tracker.log_message(f"Processing PDF: {pdf_file.name}")
                
                # Import page-aware chunking
                from page_aware_chunker import chunk_pdf_with_pages
                
                # Create page-aware chunks directly from PDF file
                bi_encoder_config = session_data.get('bi_encoder_config', {})
                chunk_size = bi_encoder_config.get('chunk_size', 2048)
                chunk_overlap = bi_encoder_config.get('chunk_overlap', 128)
                page_chunks, processed_text = chunk_pdf_with_pages(str(pdf_file), chunk_size, chunk_overlap)
                all_page_chunks.extend(page_chunks)
                doc_ids.extend([pdf_file.stem] * len(page_chunks))
                
                progress_tracker.log_message(f"Created {len(page_chunks)} chunks from {pdf_file.name}")
            
            # Stage 4: Generate embeddings
            progress_tracker.set_stage('embedding_creation', len(doc_ids))
            progress_tracker.log_message("Generating embeddings...")
            
            embeddings_dir = project_dirs['embeddings']
            
            # Use page-aware embedding creation
            embeddings = session_data['bi_encoder'].embed_page_aware_documents(
                all_page_chunks, 
                doc_ids=doc_ids,
                save_path=embeddings_dir
            )
            session_data['embeddings'] = embeddings
            
            # Create BM25 corpus for hybrid search
            try:
                bi_encoder_config = session_data['bi_encoder_config']
                bm25_path = get_bm25_path(
                    session_data['dir'], 
                    current_project,
                    bi_encoder_config['model_name'],
                    bi_encoder_config['chunk_size'],
                    bi_encoder_config['chunk_overlap']
                )
                
                progress_tracker.log_message("Creating BM25 corpus for hybrid search...")
                bm25, tokenized_corpus = create_bm25_corpus(embeddings, bm25_path)
                if bm25 is not None:
                    progress_tracker.log_message("BM25 corpus created successfully")
                else:
                    progress_tracker.log_message("Warning: BM25 corpus creation failed")
            except Exception as e:
                progress_tracker.log_message(f"Warning: BM25 corpus creation failed: {e}")
            
            # Store doc paths for display
            session_data['doc_paths'] = [f.stem for f in pdf_files]
            session_data['initialized'] = True
            
            # Complete processing
            progress_tracker.complete()
            
            # Track the upload statistics
            processing_time = time.time() - progress_tracker.session_start_time
            personal_stats.track_document_upload(len(uploaded_files), processing_time)
            personal_stats.track_button_click('upload')
            
            return (
                dbc.Alert(f"Processing complete! Uploaded {len(uploaded_files)} PDF files. Ready to chat.", color="success", dismissable=True),
                'ready'
            )
                
        except Exception as e:
            progress_tracker.log_message(f"Error occurred: {str(e)}")
            personal_stats.track_error('upload_errors')
            return (
                dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True),
                'error'
            )

    @app.callback(
        Output('clear-docs-modal', 'is_open'),
        [Input('clear-docs-btn', 'n_clicks'),
         Input('clear-docs-confirm', 'n_clicks'),
         Input('clear-docs-cancel', 'n_clicks')],
        [State('clear-docs-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_clear_docs_modal(clear_btn, confirm_btn, cancel_btn, is_open):
        """Toggle the clear documents confirmation modal"""
        if clear_btn or confirm_btn or cancel_btn:
            return not is_open
        return is_open

    @app.callback(
        [Output('upload-status', 'children', allow_duplicate=True),
         Output('processing-stage', 'children', allow_duplicate=True)],
        [Input('clear-docs-confirm', 'n_clicks')],
        [State('processing-stage', 'children')],
        prevent_initial_call=True
    )
    def handle_clear_documents_confirm(n_clicks, processing_stage):
        """Clear all documents from the current project"""
        if not n_clicks:
            raise PreventUpdate
        
        try:
            current_project = session_data.get('current_project')
            if not current_project:
                return dbc.Alert("No project selected", color="warning", dismissable=True), 'not_ready'
            
            # Clear session data
            session_data['embeddings'] = []
            session_data['doc_paths'] = []
            
            # Clear project files
            project_dirs = get_project_directories(session_data['dir'], current_project)
            
            # Clear PDF files
            if os.path.exists(project_dirs['docs_pdf']):
                for filename in os.listdir(project_dirs['docs_pdf']):
                    if filename.endswith('.pdf'):
                        os.remove(os.path.join(project_dirs['docs_pdf'], filename))
            
            # Clear embeddings directory
            if os.path.exists(project_dirs['embeddings']):
                import shutil
                shutil.rmtree(project_dirs['embeddings'])
                os.makedirs(project_dirs['embeddings'], exist_ok=True)
            
            personal_stats.track_button_click('clear_documents')
            log_message(f"Cleared all documents from project: {current_project}")
            
            return (
                dbc.Alert("All documents cleared from current project", color="info", dismissable=True),
                'not_ready'
            )
        
        except Exception as e:
            error_msg = f"Error clearing documents: {str(e)}"
            log_message(error_msg)
            personal_stats.track_error('clear_documents_errors')
            return (
                dbc.Alert(error_msg, color="danger", dismissable=True),
                'error'
            )

    @app.callback(
        Output('pdf-modal', 'is_open'),
        [Input('pdf-modal-close', 'n_clicks'),
         Input('pdf-modal-backdrop', 'n_clicks')],
        [State('pdf-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_pdf_modal(close_clicks, backdrop_clicks, is_open):
        """Toggle PDF viewer modal"""
        if close_clicks or backdrop_clicks:
            return False
        return is_open

    @app.callback(
        [Output('upload-status', 'children', allow_duplicate=True),
         Output('processing-stage', 'children', allow_duplicate=True)],
        Input('check-processed-button', 'n_clicks'),
        State('project-data', 'data'),
        prevent_initial_call=True
    )
    def check_processed_pdfs(n_clicks, project_data):
        """Check for processed PDFs in the current project"""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        
        from models.progress_tracking import check_progress_tracker
        
        # Reset and start check progress tracking
        check_progress_tracker.reset()
        check_progress_tracker.set_step(1, "Checking for processed PDFs", "Initializing...")
        
        try:
            current_project = project_data.get('current_project')
            if not current_project:
                return (
                    dbc.Alert("Please select or create a project first.", color="warning", dismissable=True),
                    'error'
                )
            
            # Get project directories
            project_dirs = get_project_directories(session_data['dir'], current_project)
            
            check_progress_tracker.set_step(1, "Checking for processed PDFs", f"Project: {current_project}")
            
            pdf_files = []
            if os.path.exists(project_dirs['docs_pdf']):
                pdf_files = [f for f in os.listdir(project_dirs['docs_pdf']) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                check_progress_tracker.reset()
                return (
                    dbc.Alert("No PDF files found in the current project.", color="info", dismissable=True),
                    'idle'
                )
            
            check_progress_tracker.set_step(2, "Found processed documents", f"Found {len(pdf_files)} documents")
            
            check_progress_tracker.set_step(3, "Initializing models", "Preparing models...")
            
            # Initialize bi-encoder if needed
            if not session_data.get('bi_encoder'):
                check_progress_tracker.set_step(4, "Initializing bi-encoder", "Loading embedding model...")
                bi_encoder_config = session_data.get('bi_encoder_config', {})
                session_data['bi_encoder'] = BiEncoderPipeline(
                    model_name=bi_encoder_config.get('model_name', 'Snowflake/snowflake-arctic-embed-l-v2.0'),
                    chunk_size=bi_encoder_config.get('chunk_size', 2048),
                    chunk_overlap=bi_encoder_config.get('chunk_overlap', 128)
                )
            else:
                check_progress_tracker.set_step(4, "Bi-encoder ready", "Model already loaded")
            
            # Initialize cross-encoder if needed
            if not session_data.get('cross_encoder'):
                check_progress_tracker.set_step(5, "Initializing cross-encoder", "Loading reranking model...")
                cross_encoder_config = session_data['cross_encoder_config']
                session_data['cross_encoder'] = CrossEncoderPipeline(
                    model_name=cross_encoder_config['model_name']
                )
            else:
                check_progress_tracker.set_step(5, "Cross-encoder ready", "Model already loaded")
            
            check_progress_tracker.set_step(6, "Loading embeddings", f"Loading embeddings for {len(pdf_files)} documents...")
            
            # Load existing embeddings (using working version approach)
            try:
                bi_encoder_config = session_data.get('bi_encoder_config', {})
                model_name = bi_encoder_config.get('model_name', 'Snowflake/snowflake-arctic-embed-l-v2.0').split("/")[-1]
                chunk_size = bi_encoder_config.get('chunk_size', 2048)
                chunk_overlap = bi_encoder_config.get('chunk_overlap', 128)
                
                embeddings_dir = Path(project_dirs['embeddings'])
                doc_ids = [os.path.splitext(f)[0] for f in pdf_files]
                
                # Load embeddings from saved pickle files (same as working version)
                embeddings = []
                
                for doc_id in doc_ids:
                    doc_dir = embeddings_dir / model_name / f"chunk_size_{chunk_size}" / f"chunk_overlap_{chunk_overlap}" / doc_id
                    if doc_dir.exists():
                        # Load all chunks for this document
                        chunk_files = sorted(doc_dir.glob("chunk_*.pkl"))
                        for chunk_file in chunk_files:
                            with open(chunk_file, "rb") as f:
                                chunk_data = pickle.load(f)
                                embeddings.append(chunk_data)
                
                if embeddings:
                    print(f"Loaded {len(embeddings)} embeddings from disk")
                    session_data['embeddings'] = embeddings
                    session_data['doc_paths'] = doc_ids
                    session_data['initialized'] = True
                    
                    # Create BM25 corpus for hybrid search
                    try:
                        from utils.content_processing import get_bm25_path, create_bm25_corpus
                        bm25_path = get_bm25_path(
                            session_data['dir'], 
                            current_project,
                            bi_encoder_config['model_name'],
                            bi_encoder_config['chunk_size'],
                            bi_encoder_config['chunk_overlap']
                        )
                        bm25, tokenized_corpus = create_bm25_corpus(embeddings, bm25_path)
                        if bm25 is not None:
                            print("BM25 corpus loaded successfully")
                    except Exception as e:
                        print(f"Warning: BM25 corpus creation failed: {e}")
                    
                    check_progress_tracker.complete()
                    personal_stats.track_button_click('check_processed')
                    
                    return (
                        dbc.Alert(f"Successfully loaded {len(embeddings)} processed document chunks from {len(pdf_files)} PDF files. Ready to chat!", color="success", dismissable=True),
                        'ready'
                    )
                else:
                    check_progress_tracker.reset()
                    return (
                        dbc.Alert("No processed embeddings found. Please upload and process PDFs first.", color="warning", dismissable=True),
                        'not_ready'
                    )
                    
            except Exception as e:
                check_progress_tracker.reset()
                personal_stats.track_error('check_processed_errors')
                return (
                    dbc.Alert(f"Error loading processed documents: {str(e)}", color="danger", dismissable=True),
                    'error'
                )
                
        except Exception as e:
            check_progress_tracker.set_step(4, "Error", f"Error: {str(e)}")
            personal_stats.track_error('check_processed_errors')
            return (
                dbc.Alert(f"Error checking processed PDFs: {str(e)}", color="danger", dismissable=True),
                'error'
            )

    print("File processing callbacks registered successfully")