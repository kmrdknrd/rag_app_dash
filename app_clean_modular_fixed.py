# Modular RAG Chatbot Application
# All functionality from app_clean.py organized into modular structure

# Standard library imports
import base64
import json
import os
import pickle
import re
import shutil
import logging
import sys
from datetime import datetime
from pathlib import Path
from html.parser import HTMLParser
import threading
import time
import socket
import fcntl
import atexit

# Third-party imports
import dash
import dash_bootstrap_components as dbc
import numpy as np
import torch
import nltk
from dash import dcc, html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
from docling.document_converter import DocumentConverter
from FlagEmbedding import FlagLLMReranker
from page_aware_chunker import chunk_pdf_with_pages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from mxbai_rerank import MxbaiRerankV2
from natsort import natsorted
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import google.generativeai as genai
import flask

# Import all modular components
from config.session_state import session_data, validate_session_state, reset_models, ensure_project_setup
from config.settings import STARTUP_MESSAGE, STARTUP_SUCCESS, create_error_alert, create_progress_card
from config.prompt_templates import PROMPT_INSTRUCTIONS
from models.progress_tracking import progress_tracker, check_progress_tracker, query_progress_tracker
from models.statistics import personal_stats
from models.pdf_processor import PdfProcessor
from models.bi_encoder import BiEncoderPipeline
from models.cross_encoder import CrossEncoderPipeline
from utils.project_management import get_project_directories, get_available_projects, create_project, delete_project
from utils.content_processing import parse_html_content
from utils.logging_setup import setup_logging
from utils.port_management import PortManager, start_app
from ui.layout import create_main_layout
from ui.modals import create_modals
from ui.components import create_error_alert, create_progress_card
from callbacks.progress_callbacks import register_progress_callbacks
from callbacks.project_callbacks_fixed import register_project_callbacks
from callbacks.config_callbacks_fixed import register_config_callbacks
from callbacks.file_callbacks_fixed import register_file_callbacks
from callbacks.chat_callbacks_fixed import register_chat_callbacks
from callbacks.feedback_callbacks import register_feedback_callbacks

# Initialize logging
logger, original_stdout = setup_logging()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Configure static files serving for PDFs
@app.server.route('/docs_pdf/<project>/<path:filename>')
def serve_pdf(project, filename):
    """Serve PDF files from project-specific directories"""
    base_dir = Path.cwd()
    project_dirs = get_project_directories(base_dir, project)
    docs_dir = project_dirs['docs_pdf']
    return flask.send_from_directory(str(docs_dir), filename)

# Create the application layout
app.layout = html.Div([
    create_main_layout(session_data, STARTUP_MESSAGE, STARTUP_SUCCESS),
    *create_modals()
])

# Register all callbacks
def register_all_callbacks():
    """Register all application callbacks"""
    
    # Progress tracking callbacks
    register_progress_callbacks(app, progress_tracker, check_progress_tracker, query_progress_tracker)
    
    # Project management callbacks
    register_project_callbacks(app, session_data, personal_stats, progress_tracker, check_progress_tracker)
    
    # Configuration and modal callbacks
    register_config_callbacks(app, session_data, personal_stats, PROMPT_INSTRUCTIONS)
    
    # File processing callbacks
    register_file_callbacks(app, progress_tracker, personal_stats)
    
    # Chat processing callbacks
    register_chat_callbacks(app, personal_stats, query_progress_tracker)
    
    # Feedback system callbacks
    register_feedback_callbacks(app, personal_stats)

# Register all callbacks
register_all_callbacks()

# Main application entry point
if __name__ == '__main__':
    try:
        port = start_app()
        if port:
            print(f"Starting RAG Chatbot on port {port}")
            print(f"Access the application at: http://localhost:{port}")
            print(f"Port allocation logged to: port_log.txt")
            print(f"Application logs written to: app_log.txt")
            
            app.run(
                debug=False,
                host='0.0.0.0',
                port=port
            )
        else:
            print("Failed to allocate port. Exiting.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)