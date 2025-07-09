# Standard library imports
import base64
import io
import json
import os
import pickle
import re
import shutil
import tempfile
import uuid
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from html.parser import HTMLParser

# Third-party imports
import dash
import dash_bootstrap_components as dbc
import difflib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import torch
from dash import dcc, html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
from datasets import load_dataset
from docling.document_converter import DocumentConverter
from FlagEmbedding import FlagLLMReranker, LayerWiseFlagLLMReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from mxbai_rerank import MxbaiRerankV2
from natsort import natsorted
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import google.generativeai as genai

# Enhanced content parser for rendering formatted text and links in Dash
def parse_html_content(content):
    """Parse content and convert to properly formatted Dash components"""
    if not content:
        return []
    
    # Split content into lines to preserve line breaks
    lines = content.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            # Empty line - add a line break
            result.append(html.Br())
            continue
        
        # Process the line for links and formatting
        line_components = []
        
        # Enhanced regex to find HTML links
        link_pattern = r'<a href=[\'"]([^\'"]*)[\'"]>([^<]*)</a>'
        
        last_end = 0
        for match in re.finditer(link_pattern, line):
            # Add text before the link
            if match.start() > last_end:
                text_before = line[last_end:match.start()]
                if text_before:
                    line_components.append(text_before)
            
            # Add the link as html.A component
            href = match.group(1)
            link_text = match.group(2)
            line_components.append(html.A(link_text, href=href, target="_blank", 
                                        style={"color": "#0066cc", "textDecoration": "underline"}))
            
            last_end = match.end()
        
        # Add remaining text after the last link
        if last_end < len(line):
            remaining_text = line[last_end:]
            if remaining_text:
                line_components.append(remaining_text)
        
        # If no links found, use original line
        if not line_components:
            line_components = [line]
        
        # Process line for bold formatting - check for **text** pattern
        processed_components = []
        for component in line_components:
            if isinstance(component, str):
                # Look for **bold** patterns
                bold_pattern = r'\*\*([^*]+)\*\*'
                bold_matches = list(re.finditer(bold_pattern, component))
                
                if bold_matches:
                    last_end = 0
                    for match in bold_matches:
                        # Add text before bold
                        if match.start() > last_end:
                            text_before = component[last_end:match.start()]
                            if text_before:
                                processed_components.append(text_before)
                        
                        # Add bold text
                        bold_text = match.group(1)
                        processed_components.append(html.B(bold_text))
                        
                        last_end = match.end()
                    
                    # Add remaining text after last bold
                    if last_end < len(component):
                        remaining_text = component[last_end:]
                        if remaining_text:
                            processed_components.append(remaining_text)
                else:
                    processed_components.append(component)
            else:
                processed_components.append(component)
        
        # Check if this line looks like a header
        line_text = ''.join([str(c) for c in processed_components if isinstance(c, str)])
        if line_text.strip().startswith('#') or line_text.strip().endswith(':'):
            # Make entire line bold
            result.append(html.B(processed_components))
        elif line_text.strip().startswith('- ') or line_text.strip().startswith('* '):
            # Format as list item, removing the bullet
            if isinstance(processed_components[0], str):
                processed_components[0] = processed_components[0][2:]
            result.append(html.Li(processed_components))
        else:
            # Regular text
            result.extend(processed_components)
        
        # Add line break after each line except the last one
        if i < len(lines) - 1:
            result.append(html.Br())
    
    return result

# Setup logging to file
LOG_FILE = 'app_log.txt'
with open(LOG_FILE, 'w') as f:
    f.write('')

# Custom logging handler to write to file
class FileLogHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
    
    def emit(self, record):
        with open(self.filename, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {self.format(record)}\n")

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = FileLogHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(file_handler)

# Redirect print to logging
class PrintLogger:
    def write(self, message):
        if message.strip():
            logger.info(message.strip())
    
    def flush(self):
        pass

# Replace print with logging
original_stdout = sys.stdout
sys.stdout = PrintLogger()

# Project management helper functions
def get_project_directories(base_dir, project_name):
    """Get project-specific directory paths"""
    project_base = Path(base_dir) / "projects" / project_name
    return {
        'docs_pdf': project_base / "docs_pdf",
        'docs_md': project_base / "docs_md", 
        'embeddings': project_base / "embeddings"
    }

def get_available_projects(base_dir):
    """Get list of available projects"""
    projects_dir = Path(base_dir) / "projects"
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
    
    print(f"Created project: {project_name}")
    return project_name

def delete_project(base_dir, project_name):
    """Delete a project and all its data"""
    if not project_name:
        raise ValueError("Project name cannot be empty")
    
    project_base = Path(base_dir) / "projects" / project_name
    if project_base.exists():
        shutil.rmtree(project_base)
        print(f"Deleted project: {project_name}")
    else:
        print(f"Project {project_name} does not exist")

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

def stream_openai_response(messages, model_name, client, message_id):
    """Stream OpenAI response and store chunks in global state"""
    global streaming_state
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=1024
        )
        
        with streaming_state['lock']:
            streaming_state['chunks'] = []
            streaming_state['active'] = True
            streaming_state['message_id'] = message_id
            streaming_state['complete'] = False
            streaming_state['error'] = None
        
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                
                with streaming_state['lock']:
                    streaming_state['chunks'].append(content)
        
        with streaming_state['lock']:
            streaming_state['complete'] = True
            streaming_state['active'] = False
            
        return full_response
        
    except Exception as e:
        with streaming_state['lock']:
            streaming_state['error'] = str(e)
            streaming_state['complete'] = True
            streaming_state['active'] = False
        return ""

def stream_gemini_response(prompt, model_name, client, message_id):
    """Stream Gemini response and store chunks in global state"""
    global streaming_state
    
    try:
        response = client.generate_content(
            prompt,
            stream=True
        )
        
        with streaming_state['lock']:
            streaming_state['chunks'] = []
            streaming_state['active'] = True
            streaming_state['message_id'] = message_id
            streaming_state['complete'] = False
            streaming_state['error'] = None
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                
                with streaming_state['lock']:
                    streaming_state['chunks'].append(chunk.text)
        
        with streaming_state['lock']:
            streaming_state['complete'] = True
            streaming_state['active'] = False
            
        return full_response
        
    except Exception as e:
        with streaming_state['lock']:
            streaming_state['error'] = str(e)
            streaming_state['complete'] = True
            streaming_state['active'] = False
        return ""

class PdfProcessor:
    def __init__(self, converter="docling"):
        """Initialize PDF converter with pre-loaded model"""
        if converter not in ["marker", "docling"]:
            raise ValueError("Invalid converter. Please choose 'marker' or 'docling'.")
        
        self.converter_type = converter
        if converter == "marker":
            self.converter = PdfConverter(artifact_dict=create_model_dict())
        elif converter == "docling":
            self.converter = DocumentConverter()
            self.converter.initialize_pipeline("pdf")
    
    def process_document(self, doc_path, save_path=None):
        """Process document using pre-loaded model"""
        # Check if path is to a single file or a directory
        # Process a single file or all files in a directory
        all_texts = []
        
        # Use the file directly if it's a single file, otherwise get all PDFs in directory
        files = [doc_path] if doc_path.is_file() else list(doc_path.glob("*.pdf"))
        
        if save_path is not None:
            if not os.path.exists(save_path):
                # Create save path if it doesn't exist
                print(f"Creating save directory: {save_path}")
                os.makedirs(save_path)
        
        # Process each file
        for i, file in enumerate(files):
            # Skip if file already processed
            if save_path is not None and os.path.exists(save_path / f"{file.stem}.md"):
                print(f"Skipping {file.stem} because .md already exists")
                # Load .md from disk
                with open(save_path / f"{file.stem}.md", "r") as f:
                    text = f.read()
                all_texts.append(text)
                continue
            
            print(f"Processing {file.stem} ({i+1}/{len(files)})")
            
            if self.converter_type == "marker":
                rendered_doc = self.converter(str(file))
                text, _, images = text_from_rendered(rendered_doc)
            elif self.converter_type == "docling":
                rendered_doc = self.converter.convert(file).document
                text = rendered_doc.export_to_markdown()
            all_texts.append(text)
            
            if save_path is not None:  # save to .md's
                with open(save_path / f"{file.stem}.md", "w") as f:
                    f.write(text)
        
        # For single files, return the text as a list with one item
        return all_texts[0:1] if doc_path.is_file() else all_texts 

class BiEncoderPipeline:
    _instances = {}  # Class variable to store instances
    
    def __new__(cls, 
                 model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                 chunk_size=1024,
                 chunk_overlap=0):
        # Create a unique key for this model configuration
        instance_key = f"{model_name}_{chunk_size}_{chunk_overlap}"
        
        # If an instance with this configuration doesn't exist, create it
        if instance_key not in cls._instances:
            cls._instances[instance_key] = super(BiEncoderPipeline, cls).__new__(cls)
        
        return cls._instances[instance_key]
    
    def __init__(self, 
                 model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                 chunk_size=1024,
                 chunk_overlap=0):
        # Skip initialization if this instance was already initialized
        if hasattr(self, 'initialized'):
            return
            
        print(f"Initializing BiEncoderPipeline with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.initialized = True

    def embed_documents(self, doc_texts, doc_ids = None, save_path = None):
        """Embed documents using pre-loaded models"""
        # If string given (i.e., one document, big string), and not list (i.e., multiple documents or single document but list), make it a list
        if not isinstance(doc_texts, list):
            doc_texts = [doc_texts]
        
        if save_path is not None:
            actual_save_path = Path(save_path) / self.model_name.split("/")[-1] / f"chunk_size_{self.chunk_size}" / f"chunk_overlap_{self.chunk_overlap}"
            print(f"Actual save path: {actual_save_path}")
            # Create save path if it doesn't exist
            print(f"Creating save directory: {actual_save_path}")
            if not os.path.exists(actual_save_path):
                os.makedirs(actual_save_path)

        # Process each text in the list
        all_chunks = []
        all_vectors = []
        for i, doc in tqdm(enumerate(doc_texts), total=len(doc_texts), desc="Embedding documents"):
            # Create a directory for the document if it doesn't exist
            if save_path and doc_ids:
                doc_dir = actual_save_path / doc_ids[i]
                if not os.path.exists(doc_dir):
                    os.makedirs(doc_dir)
            
            # Split the document into chunks to check if all are already saved
            doc_chunks = self.text_splitter.split_text(doc)
            
            # Check if all chunks for this document are already saved
            if save_path is not None and doc_ids:
                all_chunks_exist = all(
                    os.path.exists(f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl") 
                    for j in range(len(doc_chunks))
                )
                
                if all_chunks_exist:
                    print(f"Skipping {doc_ids[i]} because all chunks already exist")
                    # Load existing chunks from disk
                    loaded_chunks = []
                    loaded_vectors = []
                    for j in range(len(doc_chunks)):
                        with open(f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl", "rb") as f:
                            chunk_data = pickle.load(f)
                            loaded_chunks.append(chunk_data["text"])
                            loaded_vectors.append(chunk_data["vector"])
                    
                    all_chunks.append(loaded_chunks)
                    all_vectors.append(loaded_vectors)
                    continue
            
            # Store the chunks and their embeddings
            all_chunks.append(doc_chunks)
            all_vectors.append(self.model.encode(doc_chunks))

        # Create results list; each element is a dict with the chunk text, its vector, its index, and the overall document index
        results = []
        for i, doc_chunks in enumerate(all_chunks): # For each document
            for j, chunk in enumerate(doc_chunks): # For each chunk
                results.append({
                    "text": chunk,
                    "vector": all_vectors[i][j],
                    "original_doc_id": doc_ids[i] if doc_ids is not None else None,
                    "doc_idx": i,
                    "chunk_idx": j
                })
            
                if save_path is not None and doc_ids:
                    # Only save if the chunk doesn't already exist
                    chunk_path = f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl"
                    if not os.path.exists(chunk_path):
                        with open(chunk_path, "wb") as f:
                            pickle.dump(results[-1], f)

        return results
    
    def retrieve_top_k(self, query, documents_embeddings, top_k=50):
        """Retrieve top k embeddings using cosine similarity to the query"""
        
        # Embed query
        query_vector = self.model.encode([query])   
        
        # Get embeddings from documents dicts
        stored_vectors = np.array([item["vector"] for item in documents_embeddings])
        
        # Compute similarities between query and stored vectors
        similarities = cosine_similarity(query_vector, stored_vectors).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                **documents_embeddings[i],
                "similarity": float(similarities[i])
                # "original_id": i
            }
            for i in top_indices
        ]

class CrossEncoderPipeline:
    _instances = {}  # Class variable to store instances
    
    def __new__(cls, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device=None):
        # Create a unique key for this model configuration
        instance_key = f"{model_name}_{device}"
        
        # If an instance with this configuration doesn't exist, create it
        if instance_key not in cls._instances:
            cls._instances[instance_key] = super(CrossEncoderPipeline, cls).__new__(cls)
        
        return cls._instances[instance_key]
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device=None):
        # Skip initialization if this instance was already initialized
        if hasattr(self, 'initialized'):
            return
            
        print(f"Initializing CrossEncoderPipeline with model: {model_name}")
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        
        if "mxbai" in model_name:
            self.model = MxbaiRerankV2(model_name, device="cpu")
            self.model_type = "mxbai"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model_type = "cross_encoder"
        
        self.initialized = True
    
    def rerank(self, query, documents, top_n=4):
        """Rerank documents using cross-encoder"""
        
        if self.model_type == "mxbai":
            # Get texts out of documents dicts
            texts = [doc["text"] for doc in documents]
            
            # rerank in halves to avoid memory issues
            rerankings_half_1 = self.model.rank(query, texts[:25], top_k=top_n)  # Rerank the chunks  
            rerankings_half_2 = self.model.rank(query, texts[25:], top_k=top_n)  # Rerank the chunks
            for d in rerankings_half_2:
                d.index += 25
            
            # combine rerankings
            rerankings = rerankings_half_1 + rerankings_half_2

            # sort rerankings by score and keep only top_n
            rerankings = sorted(rerankings, key=lambda x: x.score, reverse=True)[:top_n]
            
            reranked_scores = [d.score for d in rerankings]
            reranked_indices = [d.index for d in rerankings]
            reranked_results = [
                {
                    **{k: v for k, v in documents[reranked_indices[i]].items() if k not in ["vector", "match_types"]},
                    "rerank_score": reranked_scores[i]
                }
                for i in range(min(top_n, len(reranked_indices)))
            ]
            
            return reranked_results
        else:
            # Original cross-encoder logic
            texts = [doc["text"] for doc in documents]
            
            # Tokenize inputs
            inputs = self.tokenizer(
                [query] * len(texts), # Repeat query for each document
                text_pair=texts, # Pair query with each document (i.e., query + chunk 1, query + chunk 2, ...)
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Compute logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert logits to scores
            scores = torch.sigmoid(logits).squeeze().cpu().numpy() # Convert to numpy array
            
            # Create results list
            results = []
            for idx, doc in enumerate(documents):
                results.append({
                    **doc,
                    "rerank_score": float(scores[idx])
                })
            
            results_sorted = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            return results_sorted[:top_n]

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Configure static files serving for PDFs
import flask
@app.server.route('/docs_pdf/<project>/<path:filename>')
def serve_pdf(project, filename):
    # Get project-specific directory
    base_dir = Path.cwd()
    project_dirs = get_project_directories(base_dir, project)
    docs_dir = project_dirs['docs_pdf']
    return flask.send_from_directory(str(docs_dir), filename)

# Global variables to store session data
# In production, use a proper session management system

# Define prompt instructions once to avoid duplication
PROMPT_INSTRUCTIONS = {
    'strict': """Answer the user's QUERY using the text in DOCUMENTS.
Keep your answer grounded in the facts of the DOCUMENTS.
Reference the IDs of the DOCUMENTS in your response in the format <DOCUMENT1: ID1> <DOCUMENT2: ID2> <DOCUMENT3: ID3> etc.
If the DOCUMENTS don't contain the facts to answer the QUERY, your best response is "the materials do not appear to be sufficient to provide a good answer." """,
    
    'moderate': """You are a helpful assistant that uses the text in DOCUMENTS to answer the user's QUERY.
You are given a user's QUERY and a list of DOCUMENTS that are retrieved from a vector database based on the QUERY.
Use the DOCUMENTS as supplementary information to answer the QUERY.
Reference the IDs of the DOCUMENTS in your response, i.e. "The answer is based on the following documents: <DOCUMENT1: ID1> <DOCUMENT2: ID2> <DOCUMENT3: ID3> etc."
If the DOCUMENTS don't contain the facts to answer the QUERY, your best response is "the materials do not appear to be sufficient to provide a good answer." """,
    
    'loose': """You are a helpful assistant that answers the user's QUERY.
To help you answer the QUERY, you are given a list of DOCUMENTS that are retrieved from a vector database based on the QUERY.
Use the DOCUMENTS as supplementary information to answer the QUERY.
Reference the IDs of the DOCUMENTS in your response, i.e. "The answer is based on the following documents: <DOCUMENT1: ID1> <DOCUMENT2: ID2> <DOCUMENT3: ID3> etc." """,
    
    'simple': """Answer the user's QUERY using the text in DOCUMENTS.
Keep your answer grounded in the facts of the DOCUMENTS.
Reference the IDs of the DOCUMENTS in your response in the format <DOCUMENT1: ID1> <DOCUMENT2: ID2> <DOCUMENT3: ID3> etc."""
}

session_data = {
    'embeddings': [],
    'docs_md': [],
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
    'llm_config': {
        'model_name': 'gemini-2.5-flash-lite-preview-06-17',
        'api_key': None
    },
    'rag_mode': True,  # Default to RAG mode
    'prompt_type': 'strict',  # Default prompt type
    'conversation_mode': 'single'  # Default conversation mode (single/multi)
}

# Global state for streaming responses
streaming_state = {
    'active': False,
    'message_id': None,
    'chunks': [],
    'complete': False,
    'error': None,
    'thread': None,
    'lock': threading.Lock()
}


# Startup message
startup_message = f"""Naudojatės 8devices RAG Chatbot (v.0.2.0).
Turėkit omeny, kad ši aplikacija yra alfa versijoje, ir yra greitai atnaujinama. Dabartinis tikslas yra parodyti, kaip galima naudoti Retrieval-Augmented Generation (RAG) su PDF dokumentais, kad praturtinti LLM atsakymus. Visi modeliai yra lokalūs, ir dokumentai yra išsaugomi jūsų kompiuteryje, tad jūsų duomenys nėra perduodami jokiam serveriui. Dėl to, ši aplikacija veikia lėtai.

NAUJA: 
- Kurti skirtingus projektus (pvz., 'Academic', 'Work'), kad atskirai organizuoti savo dokumentus.
- Hyperlinks į naudotus šaltinius LLM atsakymuose.
- OpenAI modeliai

Jei turite kokių nors pastabų, galite jas pateikti adresu: konradas.m@8devices.com

Greitai:
- Hyperlinks į konkrečius puslapius LLM atsakymuose naudotuose šaltiniuose
- Įkelti PDF dokumentus į saugią duomenų bazę, o ne į atmintį
"""
startup_success = True

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("8devices RAG Chatbot (v.0.2.0)", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # System status row
    dbc.Row([
        dbc.Col([
            html.Div(id='system-status', children=[
                dbc.Alert([
                    html.Pre(startup_message, style={
                        "whiteSpace": "pre-wrap",
                        "margin": "0",
                        "fontFamily": "inherit",
                        "fontSize": "inherit"
                    })
                ], color="success" if startup_success else "info",
                   className="mb-3")
            ])
        ])
    ]),
    
    # Upload PDFs with RAG Mode and Project Management
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Upload PDFs"),
                dbc.CardBody([
                    # RAG Mode checkbox on top row
                    dbc.Row([
                        dbc.Col([
                            dbc.Checkbox(
                                id='rag-mode-checkbox',
                                label="RAG mode (Retrieval-Augmented Generation)",
                                value=session_data['rag_mode'],
                                className="mb-1"
                            ),
                            html.Small("Enable RAG mode to use uploaded documents for context-aware responses. Disable for general chat mode.", 
                                      className="text-muted mb-2 d-block")
                        ])
                    ]),
                    # Project Management section
                    dbc.Row([
                        dbc.Col([
                            html.H6("Project Management", className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Current Project", className="mb-2"),
                                    dbc.Select(
                                        id='project-selector',
                                        options=[],  # Will be populated dynamically
                                        placeholder="Select a project...",
                                        value=None
                                    )
                                ], width=2, className="mb-2"),
                                dbc.Col([
                                    html.Label("Create New Project", className="mb-2"),
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id='new-project-input',
                                            placeholder='New project name...',
                                            type='text'
                                        ),
                                        dbc.Button(
                                            'Create',
                                            id='create-project-button',
                                            color='success',
                                        )
                                    ])
                                ], width=3, className="mb-2"),
                                dbc.Col([
                                    html.Label("Actions", className="mb-2"),
                                    html.Br(),
                                    dbc.Button(
                                        'Delete Project',
                                        id='delete-project-button',
                                        color='danger',
                                        size='sm',
                                        disabled=True
                                    )
                                ], width=2)
                            ]),
                            html.Div(id='project-status', className="mt-2")
                        ], width=12, id="project-management-col")
                    ], className="mt-3"),
                    # PDF Upload section (separate row below Project Management)
                    dbc.Row([
                        dbc.Col([
                            html.H6("PDF Upload", className="mb-2"),
                            dcc.Upload(
                                id='upload-pdf',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select PDF Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=True,
                                disabled=True  # Will be controlled by callbacks
                            ),
                            html.Small("Note: Processing a 1MB document takes approximately 45s to make it ready for chatting. Leave the window open while processing multiple documents.", 
                                      className="text-muted mt-2 d-block"),
                            dbc.Button(
                                "Check for processed PDFs",
                                id="check-processed-button",
                                className="mt-3 d-block",
                                n_clicks=0,
                                disabled=True  # Will be controlled by callbacks
                            ),
                            dbc.Button(
                                "View Processing Log",
                                id="view-log-button",
                                className="mt-2 d-block",
                                color="secondary",
                                outline=True,
                                size="sm",
                                n_clicks=0
                            ),
                            html.Div(id='upload-status', className="mt-3"),
                            dbc.Progress(id="progress-bar", value=0, className="mt-3", style={'display': 'none'}),
                            html.Div(id='file-list', className="mt-3", children="")
                        ], width=12, id="pdf-upload-col")
                    ], className="mt-2")
                ])
            ])
        ], width=12, id="upload-col")
    ], className="mb-4", id="upload-log-row"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Chat with your PDFs"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Model Name", className="mb-2"),
                            dbc.Select(
                                id='llm-model-input',
                                options=[
                                    {'label': 'Gemini models', 'value': '', 'disabled': True},
                                    {'label': 'Gemini 2.5 Pro', 'value': 'gemini-2.5-pro'},
                                    {'label': 'Gemini 2.5 Flash', 'value': 'gemini-2.5-flash'},
                                    {'label': 'Gemini 2.5 Flash-Lite', 'value': 'gemini-2.5-flash-lite-preview-06-17'},
                                    {'label': 'OpenAI Models', 'value': '', 'disabled': True},
                                    {'label': 'GPT-4.1', 'value': 'gpt-4.1'},
                                    {'label': 'GPT-4.1 Mini', 'value': 'gpt-4.1-mini'},
                                    {'label': 'GPT-4.1 Nano', 'value': 'gpt-4.1-nano'},
                                    {'label': 'o3', 'value': 'o3'},
                                    {'label': 'o4-mini', 'value': 'o4-mini'},
                                    {'label': 'Local Ollama Models', 'value': '', 'disabled': True},
                                    {'label': 'Qwen3 (1.7B)', 'value': 'qwen3:1.7b'},
                                    {'label': 'Qwen3 (4B)', 'value': 'qwen3:4b'},
                                    {'label': 'Qwen3 (8B)', 'value': 'qwen3:8b'},
                                    {'label': 'Cogito (3B)', 'value': 'cogito:3b'},
                                    {'label': 'Cogito (8B)', 'value': 'cogito:8b'},
                                    {'label': 'DeepSeek-R1 (1.5B)', 'value': 'deepseek-r1:1.5b'},
                                    {'label': 'DeepSeek-R1 (7B)', 'value': 'deepseek-r1:7b'},
                                    {'label': 'DeepSeek-R1 (8B)', 'value': 'deepseek-r1:8b'},
                                ],
                                value=session_data['llm_config']['model_name'],
                                disabled=not session_data['rag_mode']
                            )
                        ], width=2),
                        dbc.Col([
                            html.Label("OpenAI/Gemini API Key", className="mb-2"),
                            dbc.Input(
                                id='openai-api-key-input',
                                placeholder='Enter your OpenAI/Gemini API key...',
                                type='password',
                                disabled=not session_data['rag_mode'],
                                style={'display': 'none'},  # Hidden by default
                                className="mb-2"
                            ),
                            html.Small("Required for OpenAI/Gemini models. Your key is used only to authenticate with the API and is not stored anywhere.", 
                                      id='api-key-help-text',
                                      className="text-muted d-block mt-1", 
                                      style={'display': 'none'})
                        ], width=10, id="api-key-row")
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Advanced RAG Configuration",
                                id="advanced-rag-config-button",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="mb-3",
                                disabled=not session_data['rag_mode']
                            )
                        ], width=12)
                    ]),
                    html.Div(id='chat-history', style={
                        'height': '400px',
                        'overflowY': 'auto',
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'marginBottom': '10px'
                    }),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("Prompt Type ", className="me-1"),
                                dbc.Button(
                                    "(view)",
                                    id="view-prompt-link",
                                    color="link",
                                    size="sm",
                                    className="p-0 text-decoration-underline",
                                    style={"fontSize": "inherit", "verticalAlign": "baseline"}
                                )
                            ], className="mb-2"),
                            dbc.Select(
                                id='prompt-type-select',
                                options=[
                                    {'label': 'Strict', 'value': 'strict'},
                                    {'label': 'Moderate', 'value': 'moderate'},
                                    {'label': 'Loose', 'value': 'loose'},
                                    {'label': 'Simple', 'value': 'simple'}
                                ],
                                value=session_data['prompt_type'],
                                className="mb-2"
                            ),
                        ], width=2, id="prompt-type-col"),
                        dbc.Col([
                            html.Label("Conversation Mode", className="mb-2"),
                            dbc.Select(
                                id='conversation-mode-select',
                                options=[
                                    {'label': 'Single-turn', 'value': 'single'},
                                    {'label': 'Multi-turn', 'value': 'multi'}
                                ],
                                value=session_data['conversation_mode'],
                                className="mb-2"
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("Actions", className="mb-2"),
                            dbc.Button(
                                "Clear Chat",
                                id="clear-chat-button",
                                color="secondary",
                                size="sm",
                                className="d-block"
                            )
                        ], width=2)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Small(
                                "Strict: Only uses information from uploaded documents. LLM cannot use its pre-trained knowledge.",
                                id="prompt-explainer",
                                className="text-muted mb-2 d-block"
                            )
                        ])
                    ], id="prompt-explainer-row"),
                    dbc.Row([
                        dbc.Col([
                            html.Small(
                                "Single-turn: Each query is treated independently. The LLM does not remember previous conversation context.",
                                id="conversation-mode-explainer",
                                className="text-muted mb-3 d-block"
                            )
                        ])
                    ]),
                    dbc.InputGroup([
                        dbc.Input(
                            id='user-input',
                            placeholder='Ask a question about your documents...',
                            type='text',
                            disabled=not startup_success,
                            n_submit=0
                        ),
                        dbc.Button(
                            'Send',
                            id='send-button',
                            color='primary',
                            disabled=not startup_success
                        )
                    ], className="mb-5")
                ])
            ])
        ], width=12)
    ]),
    
    # Hidden div to store chat messages
    html.Div(id='chat-messages', style={'display': 'none'}, children='[]'),
    
    # Hidden div to store detailed processing state
    html.Div(id='processing-stage', style={'display': 'none'}, children='idle'),
    
    # Hidden div to store next stage
    html.Div(id='next-stage', style={'display': 'none'}, children='0'),
    
    # Hidden store for bi-encoder configuration
    dcc.Store(id='bi-encoder-config', data=session_data['bi_encoder_config']),
    
    # Hidden store for cross-encoder configuration
    dcc.Store(id='cross-encoder-config', data=session_data['cross_encoder_config']),
    
    # Hidden store for project data
    dcc.Store(id='project-data', data={'projects': [], 'current_project': None}),
    
    # Hidden store for streaming state
    dcc.Store(id='streaming-store', data={'active': False, 'message_id': None, 'content': ''}),
    
    # Auto-refresh interval for log display
    dcc.Interval(id="log-interval", interval=1000, n_intervals=0),
    
    # Streaming interval for real-time response updates
    dcc.Interval(id="streaming-interval", interval=200, n_intervals=0, disabled=True),
    
    # Modal for viewing prompt format
    dbc.Modal([
        dbc.ModalHeader("Prompt Format"),
        dbc.ModalBody([
            html.Pre(id="prompt-display", style={
                "whiteSpace": "pre-wrap",
                "backgroundColor": "#f8f9fa",
                "padding": "15px",
                "borderRadius": "5px",
                "fontSize": "14px",
                "maxHeight": "400px",
                "overflowY": "auto"
            })
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-prompt-modal", color="secondary")
        ])
    ], id="prompt-modal", size="lg", is_open=False),
    
    # Modal for viewing processing log
    dbc.Modal([
        dbc.ModalHeader("Processing Log"),
        dbc.ModalBody([
            html.Pre(id="log-display", style={
                "whiteSpace": "pre-wrap",
                "border": "1px solid #ddd",
                "padding": "15px",
                "minHeight": "300px",
                "backgroundColor": "#f8f9fa",
                "fontSize": "12px",
                "maxHeight": "500px",
                "overflowY": "auto",
                "borderRadius": "5px"
            })
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-log-modal", color="secondary")
        ])
    ], id="log-modal", size="lg", is_open=False),
    
    # Modal for Advanced RAG Configuration
    dbc.Modal([
        dbc.ModalHeader("Advanced RAG Configuration"),
        dbc.ModalBody([
            # Bi-encoder configuration
            html.H5("Bi-Encoder Configuration", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Model Name"),
                    dbc.Select(
                        id='model-name-input',
                        options=[
                            {'label': 'Snowflake/snowflake-arctic-embed-l-v2.0', 'value': 'Snowflake/snowflake-arctic-embed-l-v2.0'},
                            {'label': 'BAAI/bge-m3', 'value': 'BAAI/bge-m3'}
                        ],
                        value=session_data['bi_encoder_config']['model_name'],
                        disabled=not session_data['rag_mode']
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Chunk Size"),
                    dbc.Select(
                        id='chunk-size-input',
                        options=[
                            {'label': '512', 'value': 512},
                            {'label': '1024', 'value': 1024},
                            {'label': '2048', 'value': 2048},
                            {'label': '4096', 'value': 4096}
                        ],
                        value=session_data['bi_encoder_config']['chunk_size'],
                        disabled=not session_data['rag_mode']
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Chunk Overlap"),
                    dbc.Select(
                        id='chunk-overlap-input',
                        options=[
                            {'label': '0', 'value': 0},
                            {'label': '128', 'value': 128},
                            {'label': '256', 'value': 256}
                        ],
                        value=session_data['bi_encoder_config']['chunk_overlap'],
                        disabled=not session_data['rag_mode']
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Retrieved Documents"),
                    dbc.Select(
                        id='retrieval-count-input',
                        options=[
                            {'label': '50', 'value': 50},
                            {'label': '100', 'value': 100},
                            {'label': '200', 'value': 200}
                        ],
                        value=session_data['bi_encoder_config']['retrieval_count'],
                        disabled=not session_data['rag_mode']
                    )
                ], width=3)
            ], className="mb-4"),
            
            # Cross-encoder configuration
            html.H5("Cross-Encoder Configuration", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Model Name"),
                    dbc.Select(
                        id='cross-encoder-model-input',
                        options=[
                            {'label': 'cross-encoder/ms-marco-MiniLM-L6-v2', 'value': 'cross-encoder/ms-marco-MiniLM-L6-v2'},
                            {'label': 'mixedbread-ai/mxbai-rerank-base-v2', 'value': 'mixedbread-ai/mxbai-rerank-base-v2'}
                        ],
                        value=session_data['cross_encoder_config']['model_name'],
                        disabled=not session_data['rag_mode']
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Reranked Documents"),
                    dbc.Select(
                        id='cross-encoder-top-n-input',
                        options=[],  # Will be populated dynamically
                        value=session_data['cross_encoder_config']['top_n'],
                        disabled=not session_data['rag_mode']
                    )
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.Small("Note: If changing the RAG Configuration, all previously uploaded documents will be re-processed according to the new configuration. Only change this if you know what you are doing!", 
                              className="text-muted")
                ])
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-rag-config-modal", color="secondary")
        ])
    ], id="rag-config-modal", size="lg", is_open=False)
    
], fluid=True)

# Callback to initialize project list on app startup
@app.callback(
    [Output('project-data', 'data'),
     Output('project-selector', 'options')],
    Input('project-data', 'id'),  # Triggers on app startup
)
def initialize_projects(_):
    if session_data['dir'] is None:
        session_data['dir'] = Path.cwd()
    
    # Get available projects
    projects = get_available_projects(session_data['dir'])
    session_data['projects'] = projects
    
    project_options = [{'label': project, 'value': project} for project in projects]
    
    # Set default project to first available project if none is currently selected
    current_project = session_data.get('current_project')
    if not current_project and projects:
        current_project = projects[0]
        session_data['current_project'] = current_project
    
    project_data = {
        'projects': projects,
        'current_project': current_project
    }
    
    return project_data, project_options

# Callback to set default project selection
@app.callback(
    Output('project-selector', 'value', allow_duplicate=True),
    Input('project-data', 'data'),
    prevent_initial_call=True
)
def set_default_project(project_data):
    if project_data and project_data.get('projects'):
        current_project = project_data.get('current_project')
        if current_project:
            return current_project
    raise PreventUpdate

# Callback to create new project
@app.callback(
    [Output('project-status', 'children'),
     Output('new-project-input', 'value'),
     Output('project-data', 'data', allow_duplicate=True),
     Output('project-selector', 'options', allow_duplicate=True),
     Output('project-selector', 'value', allow_duplicate=True)],
    Input('create-project-button', 'n_clicks'),
    State('new-project-input', 'value'),
    State('project-data', 'data'),
    prevent_initial_call=True
)
def create_new_project(n_clicks, project_name, project_data):
    if n_clicks is None or n_clicks == 0 or not project_name:
        raise PreventUpdate
    
    try:
        if session_data['dir'] is None:
            session_data['dir'] = Path.cwd()
        
        # Check if project already exists
        existing_projects = project_data.get('projects', [])
        if project_name in existing_projects:
            return (
                dbc.Alert(f"Project '{project_name}' already exists.", color="warning"),
                "",
                project_data,
                [{'label': p, 'value': p} for p in existing_projects],
                project_data.get('current_project')
            )
        
        # Create the project
        sanitized_name = create_project(session_data['dir'], project_name)
        
        # Update project list
        updated_projects = existing_projects + [sanitized_name]
        session_data['projects'] = updated_projects
        session_data['current_project'] = sanitized_name
        
        updated_project_data = {
            'projects': updated_projects,
            'current_project': sanitized_name
        }
        
        project_options = [{'label': p, 'value': p} for p in updated_projects]
        
        return (
            dbc.Alert(f"Project '{sanitized_name}' created successfully!", color="success"),
            "",
            updated_project_data,
            project_options,
            sanitized_name
        )
        
    except Exception as e:
        return (
            dbc.Alert(f"Error creating project: {str(e)}", color="danger"),
            "",
            project_data,
            [{'label': p, 'value': p} for p in project_data.get('projects', [])],
            project_data.get('current_project')
        )

# Callback to handle project selection
@app.callback(
    [Output('project-data', 'data', allow_duplicate=True),
     Output('delete-project-button', 'disabled')],
    Input('project-selector', 'value'),
    State('project-data', 'data'),
    prevent_initial_call=True
)
def select_project(project_name, project_data):
    if project_name is None:
        raise PreventUpdate
    
    # Update current project
    session_data['current_project'] = project_name
    updated_project_data = {
        'projects': project_data.get('projects', []),
        'current_project': project_name
    }
    
    # Clear current session data for the new project
    session_data['embeddings'] = []
    session_data['docs_md'] = []
    session_data['doc_paths'] = []
    session_data['initialized'] = False
    
    # Get files in the selected project
    if session_data['dir'] is None:
        session_data['dir'] = Path.cwd()
    
    project_dirs = get_project_directories(session_data['dir'], project_name)
    docs_dir = project_dirs['docs_pdf']
    
    # file_list = []
    if docs_dir.exists():
        pdf_files = list(docs_dir.glob("*.pdf"))
        # file_list = html.Ul([html.Li(f.name) for f in pdf_files]) if pdf_files else ""
        session_data['doc_paths'] = [f.stem for f in pdf_files]
    
    return updated_project_data, False #, file_list

# Callback to delete project
@app.callback(
    [Output('project-status', 'children', allow_duplicate=True),
     Output('project-data', 'data', allow_duplicate=True),
     Output('project-selector', 'options', allow_duplicate=True),
     Output('project-selector', 'value', allow_duplicate=True),
     Output('delete-project-button', 'disabled', allow_duplicate=True)],
    Input('delete-project-button', 'n_clicks'),
    State('project-data', 'data'),
    prevent_initial_call=True
)
def delete_selected_project(n_clicks, project_data):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    current_project = project_data.get('current_project')
    if not current_project:
        raise PreventUpdate
    
    try:
        if session_data['dir'] is None:
            session_data['dir'] = Path.cwd()
        
        # Delete the project
        delete_project(session_data['dir'], current_project)
        
        # Update project list
        existing_projects = project_data.get('projects', [])
        updated_projects = [p for p in existing_projects if p != current_project]
        
        # Clear session data
        session_data['projects'] = updated_projects
        session_data['current_project'] = None
        session_data['embeddings'] = []
        session_data['docs_md'] = []
        session_data['doc_paths'] = []
        session_data['initialized'] = False
        
        updated_project_data = {
            'projects': updated_projects,
            'current_project': None
        }
        
        project_options = [{'label': p, 'value': p} for p in updated_projects]
        
        return (
            dbc.Alert(f"Project '{current_project}' deleted successfully!", color="success"),
            updated_project_data,
            project_options,
            None,
            True
            #""
        )
        
    except Exception as e:
        return (
            dbc.Alert(f"Error deleting project: {str(e)}", color="danger"),
            project_data,
            [{'label': p, 'value': p} for p in project_data.get('projects', [])],
            current_project,
            False
            #""
        )


# Callback to update log display
@app.callback(
    Output("log-display", "children"),
    Input("log-interval", "n_intervals")
)
def update_log_display(n):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            content = f.read()
            return content if content else "Waiting for processing to start..."
    return "Waiting for processing to start..."

# Callback to update bi-encoder configuration
@app.callback(
    Output('bi-encoder-config', 'data'),
    [Input('model-name-input', 'value'),
     Input('chunk-size-input', 'value'),
     Input('chunk-overlap-input', 'value'),
     Input('retrieval-count-input', 'value')]
)
def update_bi_encoder_config(model_name, chunk_size, chunk_overlap, retrieval_count):
    if model_name is None or chunk_size is None or chunk_overlap is None or retrieval_count is None:
        raise PreventUpdate
    
    # Ensure chunk_size, chunk_overlap, and retrieval_count are integers
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    retrieval_count = int(retrieval_count)
    
    # Update session data
    print(f"Updating bi-encoder configuration: model = {model_name}, chunk size = {chunk_size}, chunk overlap = {chunk_overlap}, retrieval count = {retrieval_count}")
    session_data['bi_encoder_config'] = {
        'model_name': model_name,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'retrieval_count': retrieval_count
    }
    
    # Reset bi-encoder to force reinitialization with new parameters
    session_data['bi_encoder'] = None
    
    return session_data['bi_encoder_config']

# Callback to update RAG mode
@app.callback(
    Output('rag-mode-checkbox', 'value'),
    Input('rag-mode-checkbox', 'value'),
    prevent_initial_call=True
)
def update_rag_mode(rag_mode):
    if rag_mode is None:
        raise PreventUpdate
    session_data['rag_mode'] = rag_mode
    print(f"RAG mode {'enabled' if rag_mode else 'disabled'}")
    return rag_mode

# Callback to update prompt type and explainer text
@app.callback(
    [Output('prompt-type-select', 'value'),
     Output('prompt-explainer', 'children')],
    Input('prompt-type-select', 'value'),
    prevent_initial_call=True
)
def update_prompt_type(prompt_type):
    if prompt_type is None:
        raise PreventUpdate
    
    session_data['prompt_type'] = prompt_type
    print(f"Prompt type changed to: {prompt_type}")
    
    # Define explainer texts
    explainers = {
        'strict': "Strict: Only uses information from uploaded documents. LLM cannot use its pre-trained knowledge.",
        'moderate': "Moderate: Primarily uses documents but may supplement with LLM's knowledge when documents are insufficient.",
        'loose': "Loose: Uses documents as starting point but freely combines with LLM's broader knowledge.",
        'simple': "Simple: Document-focused with minimal constraints on knowledge usage."
    }
    
    explainer_text = explainers.get(prompt_type, explainers['strict'])
    
    return prompt_type, explainer_text

# Callback to update conversation mode and explainer text
@app.callback(
    [Output('conversation-mode-select', 'value'),
     Output('conversation-mode-explainer', 'children')],
    Input('conversation-mode-select', 'value'),
    prevent_initial_call=True
)
def update_conversation_mode(conversation_mode):
    if conversation_mode is None:
        raise PreventUpdate
    
    session_data['conversation_mode'] = conversation_mode
    print(f"Conversation mode changed to: {conversation_mode}")
    
    # Define explainer texts
    explainers = {
        'single': "Single-turn: Each query is treated independently. The LLM does not remember previous conversation context.",
        'multi': "Multi-turn: The LLM remembers previous conversation context (last 8 turns) and can reference earlier messages."
    }
    
    explainer_text = explainers.get(conversation_mode, explainers['single'])
    
    return conversation_mode, explainer_text

# Callback to clear chat history
@app.callback(
    [Output('chat-history', 'children', allow_duplicate=True),
     Output('chat-messages', 'children', allow_duplicate=True)],
    Input('clear-chat-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_chat_history(n_clicks):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    print("Clearing chat history...")
    return [], '[]'


# Separate callback to control UI elements based on RAG mode and project selection
@app.callback(
    [Output('upload-pdf', 'disabled'),
     Output('advanced-rag-config-button', 'disabled'),
     Output('advanced-rag-config-button', 'style'),
     Output('upload-col', 'style'),
     Output('project-management-col', 'style'),
     Output('pdf-upload-col', 'style'),
     Output('prompt-type-col', 'style'),
     Output('prompt-explainer-row', 'style'),
     Output('model-name-input', 'disabled'),
     Output('chunk-size-input', 'disabled'),
     Output('chunk-overlap-input', 'disabled'),
     Output('retrieval-count-input', 'disabled'),
     Output('cross-encoder-model-input', 'disabled'),
     Output('cross-encoder-top-n-input', 'disabled'),
     Output('check-processed-button', 'disabled'),
     Output('prompt-type-select', 'disabled'),
     Output('conversation-mode-select', 'disabled'),
     Output('clear-chat-button', 'disabled'),
     Output('llm-model-input', 'disabled'),
     Output('openai-api-key-input', 'disabled')],
    [Input('rag-mode-checkbox', 'value'),
     Input('project-data', 'data')]
)
def control_ui_elements(rag_mode, project_data):
    if rag_mode is None:
        rag_mode = session_data['rag_mode']
    
    current_project = project_data.get('current_project') if project_data else None
    
    # Disable/enable components based on RAG mode and project selection
    rag_disabled = not rag_mode
    project_disabled = rag_mode and not current_project
    
    # Upload and check-processed button need both RAG mode and project
    upload_disabled = rag_disabled or project_disabled
    
    # RAG-specific components (chunking, retrieval, etc.) only need RAG mode
    rag_config_disabled = rag_disabled
    
    # LLM and API key should always be enabled for both RAG and non-RAG modes
    llm_disabled = False
    
    # Upload column should always be visible since it contains the RAG mode checkbox
    upload_col_style = {'display': 'block'}
    
    # Show/hide project management row based on RAG mode
    project_management_style = {'display': 'block', 'marginBottom': '1rem'} if rag_mode else {'display': 'none'}
    
    # Show/hide PDF upload section based on RAG mode
    pdf_upload_style = {'display': 'block'} if rag_mode else {'display': 'none'}
    
    # Show/hide Advanced RAG Configuration button based on RAG mode
    advanced_rag_config_style = {'display': 'block'} if rag_mode else {'display': 'none'}
    
    # Show/hide Prompt Type controls based on RAG mode
    prompt_type_style = {'display': 'block'} if rag_mode else {'display': 'none'}
    prompt_explainer_style = {'display': 'block'} if rag_mode else {'display': 'none'}
    
    # Conversation mode should always be enabled (not disabled based on RAG mode)
    conversation_mode_disabled = False
    
    return (upload_disabled,                    # 1. upload-pdf disabled
            rag_config_disabled,                # 2. advanced-rag-config-button disabled  
            advanced_rag_config_style,          # 3. advanced-rag-config-button style
            upload_col_style,                   # 4. upload-col style
            project_management_style,           # 5. project-management-col style
            pdf_upload_style,                   # 6. pdf-upload-col style
            prompt_type_style,                  # 7. prompt-type-col style
            prompt_explainer_style,             # 8. prompt-explainer-row style
            rag_config_disabled,                # 9. model-name-input disabled
            rag_config_disabled,                # 10. chunk-size-input disabled
            rag_config_disabled,                # 11. chunk-overlap-input disabled
            rag_config_disabled,                # 12. retrieval-count-input disabled
            rag_config_disabled,                # 13. cross-encoder-model-input disabled
            rag_config_disabled,                # 14. cross-encoder-top-n-input disabled
            upload_disabled,                    # 15. check-processed-button disabled
            rag_config_disabled,                # 16. prompt-type-select disabled
            conversation_mode_disabled,         # 17. conversation-mode-select disabled
            rag_config_disabled,                # 18. clear-chat-button disabled
            llm_disabled,                       # 19. llm-model-input disabled
            llm_disabled)                       # 20. openai-api-key-input disabled

# Callback to update cross-encoder top_n options based on chunk size
@app.callback(
    Output('cross-encoder-top-n-input', 'options'),
    Input('chunk-size-input', 'value')
)
def update_cross_encoder_options(chunk_size):
    if chunk_size is None:
        return []
    
    chunk_size = int(chunk_size)
    
    # Define options based on chunk size
    if chunk_size == 512:
        options = [4, 8, 16, 32]
    elif chunk_size == 1024:
        options = [4, 8, 16]
    elif chunk_size == 2048:
        options = [4, 8]
    elif chunk_size == 4096:
        options = [4]
    else:
        options = [4, 8]  # Default
    
    return [{'label': str(opt), 'value': opt} for opt in options]

# Callback to update cross-encoder configuration
@app.callback(
    Output('cross-encoder-config', 'data'),
    [Input('cross-encoder-model-input', 'value'),
     Input('cross-encoder-top-n-input', 'value')]
)
def update_cross_encoder_config(model_name, top_n):
    if model_name is None:
        raise PreventUpdate
    
    # Use existing top_n if not provided
    if top_n is None:
        top_n = session_data['cross_encoder_config'].get('top_n', 8)
    
    top_n = int(top_n)
    
    # Update session data
    print(f"Updating cross-encoder configuration: model = {model_name}, top_n = {top_n}")
    session_data['cross_encoder_config'] = {
        'model_name': model_name,
        'top_n': top_n
    }
    
    # Reset cross-encoder to force reinitialization with new parameters
    session_data['cross_encoder'] = None
    
    return session_data['cross_encoder_config']

# Callback to show/hide OpenAI API key input based on selected model
@app.callback(
    [Output('openai-api-key-input', 'style'),
     Output('api-key-help-text', 'style'),
     Output('api-key-row', 'style')],
    Input('llm-model-input', 'value')
)
def toggle_api_key_input(model_name):
    if model_name is None:
        raise PreventUpdate
    
    # Check if it's an OpenAI or Gemini model
    openai_models = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o3', 'o4-mini']
    gemini_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']
    is_openai_model = model_name in openai_models
    is_gemini_model = model_name in gemini_models
    needs_api_key = is_openai_model or is_gemini_model
    
    if needs_api_key:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callback to handle LLM model configuration changes
@app.callback(
    Output('llm-model-input', 'value'),
    Input('llm-model-input', 'value'),
    prevent_initial_call=True
)
def update_llm_config(model_name):
    if model_name is None:
        raise PreventUpdate
    
    session_data['llm_config']['model_name'] = model_name
    print(f"LLM model changed to: {model_name}")
    
    # Reset the LLM instance to force reinitialization with new model
    session_data['llm'] = None
    session_data['llm_client'] = None
    
    return model_name

# Callback to handle OpenAI API key changes
@app.callback(
    Output('openai-api-key-input', 'value'),
    Input('openai-api-key-input', 'value'),
    prevent_initial_call=True
)
def update_openai_api_key(api_key):
    if api_key is None:
        raise PreventUpdate
    
    session_data['llm_config']['api_key'] = api_key
    print("OpenAI API key updated")
    
    # Reset the LLM instance to force reinitialization with new API key
    session_data['llm'] = None
    session_data['llm_client'] = None
    
    return api_key

# Callback to open prompt modal and display prompt format
@app.callback(
    [Output("prompt-modal", "is_open"),
     Output("prompt-display", "children")],
    [Input("view-prompt-link", "n_clicks"),
     Input("close-prompt-modal", "n_clicks")],
    [State("prompt-modal", "is_open"),
     State("prompt-type-select", "value")],
    prevent_initial_call=True
)
def toggle_prompt_modal(view_clicks, close_clicks, is_open, prompt_type):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == "view-prompt-link":
        # Get prompt instructions based on selected type
        selected_instructions = PROMPT_INSTRUCTIONS.get(prompt_type, PROMPT_INSTRUCTIONS['strict'])
        
        # Create example prompt format
        example_prompt = f"""<QUERY>
[User's question will appear here]
</QUERY>

<INSTRUCTIONS>
{selected_instructions}
</INSTRUCTIONS>

<DOCUMENTS>
<DOCUMENT1: document_name_1>
TEXT:
[Retrieved document content 1...]
</DOCUMENT1: document_name_1>

<DOCUMENT2: document_name_2>
TEXT:
[Retrieved document content 2...]
</DOCUMENT2: document_name_2>

[Additional documents as needed...]
</DOCUMENTS>"""
        
        return True, example_prompt
    
    elif trigger_id == "close-prompt-modal":
        return False, ""
    
    return is_open, ""

# Callback to open log modal
@app.callback(
    Output("log-modal", "is_open"),
    [Input("view-log-button", "n_clicks"),
     Input("close-log-modal", "n_clicks")],
    State("log-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_log_modal(view_clicks, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == "view-log-button":
        return True
    elif trigger_id == "close-log-modal":
        return False
    
    return is_open

# Callback to open RAG config modal
@app.callback(
    Output("rag-config-modal", "is_open"),
    [Input("advanced-rag-config-button", "n_clicks"),
     Input("close-rag-config-modal", "n_clicks")],
    State("rag-config-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_rag_config_modal(open_clicks, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == "advanced-rag-config-button":
        return True
    elif trigger_id == "close-rag-config-modal":
        return False
    
    return is_open

# Callback to check for processed PDFs
@app.callback(
    [Output('upload-status', 'children', allow_duplicate=True),
     Output('processing-stage', 'children', allow_duplicate=True)],
    Input('check-processed-button', 'n_clicks'),
    State('project-data', 'data'),
    prevent_initial_call=True
)
def check_processed_pdfs(n_clicks, project_data):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    # Check if a project is selected
    current_project = project_data.get('current_project')
    if not current_project:
        return (
            dbc.Alert("Please select a project first.", color="warning"),
            'idle'
        )
    
    try:
        print(f"Checking for processed PDFs in project: {current_project}")
        
        # Set up directories
        if session_data['dir'] is None:
            session_data['dir'] = Path.cwd()
        
        # Get project-specific directories
        project_dirs = get_project_directories(session_data['dir'], current_project)
        docs_dir = project_dirs['docs_pdf']
        processed_dir = project_dirs['docs_md']
        embeddings_dir = project_dirs['embeddings']
        
        # Check if directories exist
        if not docs_dir.exists() or not processed_dir.exists() or not embeddings_dir.exists():
            print("No processed documents found - directories don't exist")
            return (
                dbc.Alert("No processed documents found. Please upload PDFs first.", color="warning"),
                'idle'
            )
        
        # Get current configuration
        config = session_data['bi_encoder_config']
        model_name = config['model_name'].split("/")[-1]
        chunk_size = config['chunk_size']
        chunk_overlap = config['chunk_overlap']
        
        # Check for processed files
        pdf_files = list(docs_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in docs_pdf directory")
            return (
                dbc.Alert("No PDF files found. Please upload PDFs first.", color="warning"),
                'idle'
            )
        
        # Check if markdown files exist for all PDFs
        processed_docs = []
        doc_ids = []
        missing_md = []
        
        for pdf_file in pdf_files:
            md_file = processed_dir / f"{pdf_file.stem}.md"
            if md_file.exists():
                with open(md_file, 'r') as f:
                    processed_docs.append(f.read())
                doc_ids.append(pdf_file.stem)
            else:
                missing_md.append(pdf_file.stem)
        
        if missing_md:
            print(f"Missing markdown files for: {missing_md}")
            return (
                dbc.Alert(f"Some PDFs need processing: {', '.join(missing_md)}. Please upload them again.", color="warning"),
                'idle'
            )
        
        # Check if embeddings exist for current configuration
        embeddings_path = embeddings_dir / model_name / f"chunk_size_{chunk_size}" / f"chunk_overlap_{chunk_overlap}"
        
        if not embeddings_path.exists():
            print(f"No embeddings found for current configuration at {embeddings_path}")
            return (
                dbc.Alert("No embeddings found for current configuration. Please upload PDFs to process them.", color="warning"),
                'idle'
            )
        
        # Check if all documents have embeddings
        missing_embeddings = []
        for doc_id in doc_ids:
            doc_embedding_dir = embeddings_path / doc_id
            if not doc_embedding_dir.exists():
                missing_embeddings.append(doc_id)
            else:
                # Check if at least one chunk file exists
                chunk_files = list(doc_embedding_dir.glob("chunk_*.pkl"))
                if not chunk_files:
                    missing_embeddings.append(doc_id)
        
        if missing_embeddings:
            print(f"Missing embeddings for: {missing_embeddings}")
            return (
                dbc.Alert(f"Embeddings missing for: {', '.join(missing_embeddings)}. Please re-upload these PDFs.", color="warning"),
                'idle'
            )
        
        print(f"Found processed documents: {doc_ids}")
        print("Initializing models...")
        
        # Initialize models if not already done
        if session_data['bi_encoder'] is None:
            print("Initializing bi-encoder...")
            session_data['bi_encoder'] = BiEncoderPipeline(
                model_name=config['model_name'],
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap']
            )
        
        if session_data['cross_encoder'] is None:
            print("Initializing cross-encoder...")
            cross_encoder_config = session_data['cross_encoder_config']
            session_data['cross_encoder'] = CrossEncoderPipeline(
                model_name=cross_encoder_config['model_name']
            )
        
        # Load embeddings
        print("Loading embeddings...")
        embeddings = session_data['bi_encoder'].embed_documents(
            processed_docs, 
            doc_ids=doc_ids,
            save_path=embeddings_dir
        )
        session_data['embeddings'] = embeddings
        session_data['docs_md'] = processed_docs
        session_data['doc_paths'] = doc_ids
        session_data['initialized'] = True
        
        print("System ready for chatting!")
        return (
            dbc.Alert(f"Found and loaded {len(doc_ids)} processed documents. Ready to chat!", color="success"),
            'ready'
        )
        
    except Exception as e:
        error_msg = f"Error checking processed PDFs: {str(e)}"
        print(error_msg)
        return (
            dbc.Alert(error_msg, color="danger"),
            'error'
        )

@app.callback(
    [Output('upload-status', 'children'),
    #  Output('file-list', 'children'),
     Output('processing-stage', 'children'),
     Output('progress-bar', 'value'),
     Output('progress-bar', 'style')],
    [Input('upload-pdf', 'contents')],
    [State('upload-pdf', 'filename'),
     State('project-data', 'data')]
)
def handle_file_upload(contents, filenames, project_data):
    if contents is None:
        raise PreventUpdate
    
    # Check if a project is selected
    current_project = project_data.get('current_project')
    if not current_project:
        return (
            dbc.Alert("Please select or create a project first.", color="warning"),
            #"",
            'idle',
            0,
            {'display': 'none'}
        )
    
    try:
        # Create directories and save files
        if session_data['dir'] is None:
            session_data['dir'] = Path.cwd()
        
        # Get project-specific directories
        project_dirs = get_project_directories(session_data['dir'], current_project)
        docs_dir = project_dirs['docs_pdf']
        processed_dir = project_dirs['docs_md']
        embeddings_dir = project_dirs['embeddings']
        
        # Create directories
        for d in [docs_dir, processed_dir, embeddings_dir]:
            if not d.exists():
                d.mkdir(parents=True)
            
        # Save uploaded PDFs
        uploaded_files = []
        for content, filename in zip(contents, filenames):
            if filename.endswith('.pdf'):
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                
                file_path = docs_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(decoded)
                uploaded_files.append(filename)
        
        if not uploaded_files:
            return (
                dbc.Alert("Please upload PDF files only.", color="warning"),
                #"",
                'idle',
                0,
                {'display': 'none'}
            )
        
        # Now process everything in sequence
        status_messages = []
        
        # Stage 1: Initialize PDF processor
        status_messages.append("Initializing PDF processor...")
        print("Initializing PDF processor...")
        if session_data['pdf_processor'] is None:
            print("Initializing PDF processor...")
            session_data['pdf_processor'] = PdfProcessor()
        
        # Stage 2: Process PDFs
        status_messages.append("Processing PDFs...")
        print("Processing PDFs...")
        pdf_files = list(docs_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            if not (processed_dir / f"{pdf_file.stem}.md").exists():
                print(f"Processing {pdf_file.stem}...")
                processed_docs = session_data['pdf_processor'].process_document(
                    pdf_file, 
                    save_path=processed_dir)
                session_data['docs_md'].extend(processed_docs)
        
        # Stage 3: Initialize embedder
        status_messages.append("Initializing embedder...")
        print("Initializing embedder...")
        if session_data['bi_encoder'] is None:
            print("Initializing bi-encoder...")
            config = session_data['bi_encoder_config']
            session_data['bi_encoder'] = BiEncoderPipeline(
                model_name=config['model_name'],
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap']
            )
            # Initialize cross-encoder
            if session_data['cross_encoder'] is None:
                cross_encoder_config = session_data['cross_encoder_config']
                session_data['cross_encoder'] = CrossEncoderPipeline(
                    model_name=cross_encoder_config['model_name']
                )
            # Set initialized flag to True
            session_data['initialized'] = True
        
        # Stage 4: Create embeddings
        status_messages.append("Creating embeddings...")
        print("Creating embeddings...")
        processed_docs = []
        doc_ids = []
        for md_file in processed_dir.glob("*.md"):
            with open(md_file, 'r') as f:
                processed_docs.append(f.read())
            doc_ids.append(md_file.stem)
        
        if processed_docs:
            embeddings = session_data['bi_encoder'].embed_documents(
                processed_docs, 
                doc_ids=doc_ids,
                save_path=embeddings_dir
            )
            session_data['embeddings'] = embeddings
        
        # Store doc paths for display
        session_data['doc_paths'] = [f.stem for f in pdf_files]
        
        # All done!
        print("Processing complete! Ready to chat.")
        return (
            dbc.Alert("Processing complete! Ready to chat.", color="success"),
            #html.Ul([html.Li(f"{doc_id}.pdf") for doc_id in session_data['doc_paths']]),
            'ready',
            100,
            {'display': 'block'}
        )
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return (
            dbc.Alert(f"Error: {str(e)}", color="danger"),
            #"",
            'error',
            0,
            {'display': 'none'}
        )

@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-messages', 'children'),
     Output('user-input', 'value')],
    [Input('send-button', 'n_clicks'),
     Input('user-input', 'n_submit')],
    [State('user-input', 'value'),
     State('chat-messages', 'children'),
     State('processing-stage', 'children'),
     State('project-data', 'data')]
)
def handle_chat(n_clicks, n_submit, user_input, messages_json, processing_state, project_data):
    print(f"Processing state: {processing_state}")
    if (n_clicks is None and n_submit is None) or not user_input:
        raise PreventUpdate
    
    # Check if we're in RAG mode and need a project
    if session_data['rag_mode']:
        current_project = project_data.get('current_project')
        if not current_project:
            # Return error message
            messages = json.loads(messages_json)
            messages.append({
                'role': 'assistant',
                'content': 'Please select a project first to use RAG mode.',
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            error_display = [dbc.Card([
                dbc.CardBody([
                    html.Small(messages[-1]['timestamp'], className="text-muted"),
                    html.P(messages[-1]['content'], className="mb-0")
                ])
            ], color="danger", outline=True, className="mb-2")]
            
            return error_display, json.dumps(messages), ""
        
        if processing_state != 'ready':
            raise PreventUpdate
    
    # Additional check to ensure system is properly initialized in RAG mode
    if session_data['rag_mode'] and not session_data.get('initialized', False):
        raise PreventUpdate
    
    # Check if OpenAI/Gemini model is selected but no API key is provided
    model_name = session_data['llm_config']['model_name']
    openai_models = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o3', 'o4-mini']
    gemini_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']
    needs_api_key = model_name in openai_models or model_name in gemini_models
    if needs_api_key and not session_data['llm_config'].get('api_key'):
        # Return error message
        messages = json.loads(messages_json)
        messages.append({
            'role': 'assistant',
            'content': 'Please enter your API key to use OpenAI/Gemini models.',
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        error_display = [dbc.Card([
            dbc.CardBody([
                html.Small(messages[-1]['timestamp'], className="text-muted"),
                html.P(messages[-1]['content'], className="mb-0")
            ])
        ], color="danger", outline=True, className="mb-2")]
        
        return error_display, json.dumps(messages), ""
    
    try:
        # Parse existing messages
        messages = json.loads(messages_json)
        
        # Add user message
        messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        if session_data['llm'] is None:
            model_name = session_data['llm_config']['model_name']
            print(f"Initializing LLM with model: {model_name}...")
            
            # Check if it's an OpenAI or Gemini model
            openai_models = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o3', 'o4-mini']
            gemini_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']
            
            if model_name in openai_models:
                api_key = session_data['llm_config'].get('api_key')
                if not api_key:
                    raise ValueError("OpenAI API key is required for OpenAI models")
                session_data['llm_client'] = OpenAI(api_key=api_key)
                session_data['llm'] = 'openai'  # Use string to indicate OpenAI
            elif model_name in gemini_models:
                api_key = session_data['llm_config'].get('api_key')
                if not api_key:
                    raise ValueError("Gemini API key is required for Gemini models")
                genai.configure(api_key=api_key)
                session_data['llm_client'] = genai.GenerativeModel(model_name)
                session_data['llm'] = 'gemini'  # Use string to indicate Gemini
            else:
                session_data['llm'] = ChatOllama(model=model_name)
        
        # Generate response based on RAG mode
        if session_data['rag_mode'] and session_data.get('embeddings'):
            # RAG mode: Retrieve and use relevant documents
            retrieval_count = session_data['bi_encoder_config']['retrieval_count']
            print(f"Retrieving top {retrieval_count} documents...")
            retrieved_docs = session_data['bi_encoder'].retrieve_top_k(
                user_input, 
                session_data['embeddings'], 
                top_k=retrieval_count
            )
            print("Reranking documents...")
            # Rerank documents
            top_n = session_data['cross_encoder_config']['top_n']
            print(f"Reranking to top {top_n} documents...")
            reranked_docs = session_data['cross_encoder'].rerank(
                user_input, 
                retrieved_docs, 
                top_n=top_n
            )
            print("Generating response with context...")
            # Generate response with context
            context_ids = [doc["original_doc_id"] for doc in reranked_docs]
            context_texts = [doc["text"] for doc in reranked_docs]
            context_texts_pretty = "\n".join([
                f"<DOCUMENT{i+1}: {context_ids[i]}>\nTEXT:\n{text}\n</DOCUMENT{i+1}: {context_ids[i]}>\n" 
                for i, text in enumerate(context_texts)
            ])
            
            # Get the selected prompt type and corresponding instructions
            prompt_type = session_data.get('prompt_type', 'strict')
            conversation_mode = session_data.get('conversation_mode', 'single')
            print(f"Using prompt type: {prompt_type}, conversation mode: {conversation_mode}")
            
            # Get the selected prompt instructions
            selected_instructions = PROMPT_INSTRUCTIONS.get(prompt_type, PROMPT_INSTRUCTIONS['strict'])
            
            # Build conversation history if in multi-turn mode
            conversation_history = ""
            if conversation_mode == 'multi':
                conversation_history = build_conversation_history(messages_json)
                if conversation_history:
                    conversation_history = f"""
            <CONVERSATION_HISTORY>
            {conversation_history}
            </CONVERSATION_HISTORY>
            """
            
            rag_prompt = f"""
            <QUERY>
            {user_input}
            </QUERY>

            <INSTRUCTIONS>
            {selected_instructions}
            </INSTRUCTIONS>
            {conversation_history}
            <DOCUMENTS>
            {context_texts_pretty}
            </DOCUMENTS>
            """
            
            print("Invoking LLM for RAG response...")
            print("\n", context_texts_pretty, "\n")
            
            # Handle different LLM types
            if session_data['llm'] == 'openai':
                model_name = session_data['llm_config']['model_name']
                client = session_data['llm_client']
                
                # Use proper conversation history format for OpenAI in multi-turn mode
                if conversation_mode == 'multi':
                    # Build messages with conversation history
                    messages = build_openai_conversation_history(messages_json)
                    
                    # Create system message with instructions and documents
                    system_message = f"""
                    <INSTRUCTIONS>
                    {selected_instructions}
                    </INSTRUCTIONS>
                    
                    <DOCUMENTS>
                    {context_texts_pretty}
                    </DOCUMENTS>
                    """
                    
                    # Insert system message at the beginning
                    messages.insert(0, {"role": "system", "content": system_message})
                    
                    # Add current user query
                    messages.append({"role": "user", "content": user_input})
                    
                    # Generate unique message ID for streaming
                    message_id = str(uuid.uuid4())
                    
                    # Start streaming in a separate thread
                    def stream_worker():
                        return stream_openai_response(messages, model_name, client, message_id)
                    
                    streaming_thread = threading.Thread(target=stream_worker)
                    streaming_thread.daemon = True
                    streaming_thread.start()
                    
                    # Wait for streaming to complete
                    streaming_thread.join()
                    
                    # Get final response
                    with streaming_state['lock']:
                        response_content = ''.join(streaming_state['chunks'])
                        if streaming_state['error']:
                            response_content = f"Error: {streaming_state['error']}"
                else:
                    # Single-turn mode - use the current format
                    # Generate unique message ID for streaming
                    message_id = str(uuid.uuid4())
                    
                    # Start streaming in a separate thread
                    def stream_worker():
                        return stream_openai_response([{"role": "user", "content": rag_prompt}], model_name, client, message_id)
                    
                    streaming_thread = threading.Thread(target=stream_worker)
                    streaming_thread.daemon = True
                    streaming_thread.start()
                    
                    # Wait for streaming to complete
                    streaming_thread.join()
                    
                    # Get final response
                    with streaming_state['lock']:
                        response_content = ''.join(streaming_state['chunks'])
                        if streaming_state['error']:
                            response_content = f"Error: {streaming_state['error']}"
                current_project = session_data.get('current_project', 'default')
                
                # Handle both old and new document reference formats
                for i, doc_name in enumerate(context_ids):
                    # Old format: DOCUMENT1, DOCUMENT2, etc.
                    response_content = response_content.replace(f"DOCUMENT{i+1}",
                                                                f"<a href='docs_pdf/{current_project}/{doc_name}.pdf'>[{i+1}]</a>")
                
                # New format: <document_id>X</document_id>
                doc_id_pattern = r'<document_id>(\d+)</document_id>'
                def replace_doc_id(match):
                    doc_num = int(match.group(1))
                    if 1 <= doc_num <= len(context_ids):
                        doc_name = context_ids[doc_num-1]
                        return f"<a href='docs_pdf/{current_project}/{doc_name}.pdf'>[{doc_num}]</a>"
                    return match.group(0)  # Return original if invalid
                
                response_content = re.sub(doc_id_pattern, replace_doc_id, response_content)
                
                print(response_content)
            elif session_data['llm'] == 'gemini':
                client = session_data['llm_client']
                
                # Generate unique message ID for streaming
                message_id = str(uuid.uuid4())
                
                # Start streaming in a separate thread
                def stream_worker():
                    return stream_gemini_response(rag_prompt, session_data['llm_config']['model_name'], client, message_id)
                
                streaming_thread = threading.Thread(target=stream_worker)
                streaming_thread.daemon = True
                streaming_thread.start()
                
                # Wait for streaming to complete
                streaming_thread.join()
                
                # Get final response
                with streaming_state['lock']:
                    response_content = ''.join(streaming_state['chunks'])
                    if streaming_state['error']:
                        response_content = f"Error: {streaming_state['error']}"
                current_project = session_data.get('current_project', 'default')
                
                # Handle both old and new document reference formats
                for i, doc_name in enumerate(context_ids):
                    # Old format: DOCUMENT1, DOCUMENT2, etc.
                    response_content = response_content.replace(f"DOCUMENT{i+1}",
                                                                f"<a href='docs_pdf/{current_project}/{doc_name}.pdf'>[{i+1}]</a>")
                
                # New format: <document_id>X</document_id>
                doc_id_pattern = r'<document_id>(\d+)</document_id>'
                def replace_doc_id(match):
                    doc_num = int(match.group(1))
                    if 1 <= doc_num <= len(context_ids):
                        doc_name = context_ids[doc_num-1]
                        return f"<a href='docs_pdf/{current_project}/{doc_name}.pdf'>[{doc_num}]</a>"
                    return match.group(0)  # Return original if invalid
                
                response_content = re.sub(doc_id_pattern, replace_doc_id, response_content)
                
                print(response_content)
            else:
                response = session_data['llm'].invoke(rag_prompt)
                response_content = response.content.split("</think>")[1] if "</think>" in response.content else response.content
                # Add project-specific PDF links for Ollama models too
                current_project = session_data.get('current_project', 'default')
                
                # Handle both old and new document reference formats
                for i, doc_name in enumerate(context_ids):
                    # Old format: DOCUMENT1, DOCUMENT2, etc.
                    response_content = response_content.replace(f"DOCUMENT{i+1}",
                                                                f"<a href='docs_pdf/{current_project}/{doc_name}.pdf'>[{i+1}]</a>")
                
                # New format: <document_id>X</document_id>
                doc_id_pattern = r'<document_id>(\d+)</document_id>'
                def replace_doc_id(match):
                    doc_num = int(match.group(1))
                    if 1 <= doc_num <= len(context_ids):
                        doc_name = context_ids[doc_num-1]
                        return f"<a href='docs_pdf/{current_project}/{doc_name}.pdf'>[{doc_num}]</a>"
                    return match.group(0)  # Return original if invalid
                
                response_content = re.sub(doc_id_pattern, replace_doc_id, response_content)
        else:
            # Non-RAG mode: Direct chat without document context
            conversation_mode = session_data.get('conversation_mode', 'single')
            print(f"Generating direct response in {conversation_mode} mode...")
            
            # Build conversation history if in multi-turn mode
            conversation_history = ""
            if conversation_mode == 'multi':
                conversation_history = build_conversation_history(messages_json)
                if conversation_history:
                    conversation_history = f"""
            <CONVERSATION_HISTORY>
            {conversation_history}
            </CONVERSATION_HISTORY>
            """
            
            chat_prompt = f"""
            <QUERY>
            {user_input}
            </QUERY>

            <INSTRUCTIONS>
            Provide a helpful and informative response to the user's query.
            Be concise but thorough in your explanation.
            </INSTRUCTIONS>
            {conversation_history}
            """
            
            print("Invoking LLM for direct response...")
            
            # Handle different LLM types
            if session_data['llm'] == 'openai':
                model_name = session_data['llm_config']['model_name']
                client = session_data['llm_client']
                
                # Use proper conversation history format for OpenAI in multi-turn mode
                if conversation_mode == 'multi':
                    # Build messages with conversation history
                    messages = build_openai_conversation_history(messages_json)
                    
                    # Create system message with instructions
                    system_message = """
                    <INSTRUCTIONS>
                    Provide a helpful and informative response to the user's query.
                    Be concise but thorough in your explanation.
                    </INSTRUCTIONS>
                    """
                    
                    # Insert system message at the beginning
                    messages.insert(0, {"role": "system", "content": system_message})
                    
                    # Add current user query
                    messages.append({"role": "user", "content": user_input})
                    
                    # Generate unique message ID for streaming
                    message_id = str(uuid.uuid4())
                    
                    # Start streaming in a separate thread
                    def stream_worker():
                        return stream_openai_response(messages, model_name, client, message_id)
                    
                    streaming_thread = threading.Thread(target=stream_worker)
                    streaming_thread.daemon = True
                    streaming_thread.start()
                    
                    # Wait for streaming to complete
                    streaming_thread.join()
                    
                    # Get final response
                    with streaming_state['lock']:
                        response_content = ''.join(streaming_state['chunks'])
                        if streaming_state['error']:
                            response_content = f"Error: {streaming_state['error']}"
                else:
                    # Single-turn mode - use the current format
                    # Generate unique message ID for streaming
                    message_id = str(uuid.uuid4())
                    
                    # Start streaming in a separate thread
                    def stream_worker():
                        return stream_openai_response([{"role": "user", "content": chat_prompt}], model_name, client, message_id)
                    
                    streaming_thread = threading.Thread(target=stream_worker)
                    streaming_thread.daemon = True
                    streaming_thread.start()
                    
                    # Wait for streaming to complete
                    streaming_thread.join()
                    
                    # Get final response
                    with streaming_state['lock']:
                        response_content = ''.join(streaming_state['chunks'])
                        if streaming_state['error']:
                            response_content = f"Error: {streaming_state['error']}"
            elif session_data['llm'] == 'gemini':
                client = session_data['llm_client']
                # Gemini uses the text-based conversation history format
                
                # Generate unique message ID for streaming
                message_id = str(uuid.uuid4())
                
                # Start streaming in a separate thread
                def stream_worker():
                    return stream_gemini_response(chat_prompt, session_data['llm_config']['model_name'], client, message_id)
                
                streaming_thread = threading.Thread(target=stream_worker)
                streaming_thread.daemon = True
                streaming_thread.start()
                
                # Wait for streaming to complete
                streaming_thread.join()
                
                # Get final response
                with streaming_state['lock']:
                    response_content = ''.join(streaming_state['chunks'])
                    if streaming_state['error']:
                        response_content = f"Error: {streaming_state['error']}"
            else:
                response = session_data['llm'].invoke(chat_prompt)
                response_content = response.content.split("</think>")[1] if "</think>" in response.content else response.content
        
        print("Processing response...")
        # Add assistant message
        messages.append({
            'role': 'assistant',
            'content': response_content,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Create chat history display
        chat_display = []
        for msg in messages:
            print(msg['content'])
            if msg['role'] == 'user':
                chat_display.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.Small(msg['timestamp'], className="text-muted"),
                            html.Div(parse_html_content(msg['content']), 
                                    className="mb-0", 
                                    style={"whiteSpace": "pre-wrap", "wordWrap": "break-word"})
                        ])
                    ], color="primary", outline=True, className="mb-2")
                )
            else:
                chat_display.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.Small(msg['timestamp'], className="text-muted"),
                            html.Div(parse_html_content(msg['content']), 
                                    className="mb-0", 
                                    style={"whiteSpace": "pre-wrap", "wordWrap": "break-word", "lineHeight": "1.5"})
                        ])
                    ], color="secondary", outline=True, className="mb-2")
                )
        
        print("Chat response generated successfully")
        return chat_display, json.dumps(messages), ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Chat error: {error_msg}")
        messages = json.loads(messages_json)
        messages.append({
            'role': 'assistant',
            'content': error_msg,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        return [dbc.Alert(error_msg, color="danger")], json.dumps(messages), ""

if __name__ == '__main__':
    print("Starting PDF RAG Chatbot with Projects application...")
    app.run(debug=True, port=8061)