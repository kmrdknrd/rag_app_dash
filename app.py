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
from datetime import datetime
from pathlib import Path

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

# Global variables to store session data
# In production, use a proper session management system
session_data = {
    'embeddings': [],
    'docs_md': [],
    'doc_paths': [],
    'pdf_processor': None,
    'bi_encoder': None,
    'cross_encoder': None,
    'llm': None,
    'dir': None,
    'initialized': False,
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
    'rag_mode': True,  # Default to RAG mode
    'prompt_type': 'strict'  # Default prompt type
}


# Startup message
startup_message = f"""Naudojatės 8devices RAG Chatbot (v.0.1.0).
Turėkit omeny, kad ši aplikacija yra alfa versijoje, ir yra greitai atnaujinama. Dabartinis tikslas yra parodyti, kaip galima naudoti Retrieval-Augmented Generation (RAG) su PDF dokumentais, kad praturtinti LLM atsakymus. Visi modeliai yra lokalūs, ir dokumentai yra išsaugomi jūsų kompiuteryje, tad jūsų duomenys nėra perduodami jokiam serveriui. Dėl to, ši aplikacija veikia lėtai.
Jei turite kokių nors pastabų, galite jas pateikti adresu: konradas.m@8devices.com

Greitai:
- Hyperlinks į naudotus šaltinius LLM atsakymuose
- Pasirinkti skirtingus LLM (tiek lokalius, tiek per OpenAI API)
- Įkelti PDF dokumentus į saugią duomenų bazę, o ne į atmintį
"""
startup_success = True

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("8devices RAG Chatbot (v.0.1.0)", className="text-center mb-4"),
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
    
    # RAG Mode toggle at the top
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Mode Selection"),
                dbc.CardBody([
                    dbc.Checkbox(
                        id='rag-mode-checkbox',
                        label="RAG mode (Retrieval-Augmented Generation)",
                        value=session_data['rag_mode'],
                        className="mb-2"
                    ),
                    html.Small("Enable RAG mode to use uploaded documents for context-aware responses. Disable for general chat mode.", 
                              className="text-muted")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Advanced RAG Configuration (collapsible)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Button(
                        [
                            "Advanced RAG Configuration",
                            html.Span("▼", id="triangle-icon", style={"marginLeft": "8px"})
                        ],
                        id="advanced-config-collapse-button",
                        className="mb-0",
                        color="link",
                        n_clicks=0,
                        disabled=not session_data['rag_mode'],
                        style={"textDecoration": "none", "color": "inherit", "border": "none", "padding": "0"}
                    )
                ]),
                dbc.Collapse([
                    dbc.CardBody([
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
                            ], width=4)
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
                            ], width=6),
                            dbc.Col([
                                html.Label("Reranked Documents"),
                                dbc.Select(
                                    id='cross-encoder-top-n-input',
                                    options=[],  # Will be populated dynamically
                                    value=session_data['cross_encoder_config']['top_n'],
                                    disabled=not session_data['rag_mode']
                                )
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Hr(),
                                html.Small("Note: If changing the RAG Configuration, all previously uploaded documents will be re-processed according to the new configuration.", 
                                  className="text-muted")
                            ])
                        ])
                    ])
                ],
                id="advanced-config-collapse",
                is_open=False
                )
            ])
        ], width=12)
    ], className="mb-4", id="advanced-config-row"),
    
    # Upload PDFs and Processing Log in same row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Upload PDFs"),
                dbc.CardBody([
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
                        disabled=not session_data['rag_mode']
                    ),
                    html.Small("Note: Processing a 1MB document takes approximately 45s to make it ready for chatting. Leave the window open while processing multiple documents.", 
                              className="text-muted mt-2 d-block"),
                    dbc.Button(
                        "Check for processed PDFs",
                        id="check-processed-button",
                        # color="info",
                        className="mt-3",
                        n_clicks=0,
                        disabled=not session_data['rag_mode']
                    ),
                    html.Div(id='upload-status', className="mt-3"),
                    dbc.Progress(id="progress-bar", value=0, className="mt-3", style={'display': 'none'}),
                    html.Div(id='file-list', className="mt-3", children=[
                        html.Ul([html.Li(f"{doc_id}.pdf") for doc_id in session_data['doc_paths']]) 
                        if session_data['doc_paths'] else ""
                    ])
                ])
            ])
        ], width=6, id="upload-col"),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Processing Log"),
                dbc.CardBody([
                    html.Pre(id="log-display", 
                            style={
                                "whiteSpace": "pre-wrap", 
                                "border": "1px solid black", 
                                "padding": "10px", 
                                "minHeight": "200px",
                                "backgroundColor": "#f8f9fa",
                                "fontSize": "12px",
                                "maxHeight": "300px",
                                "overflowY": "auto"
                            })
                ])
            ])
        ], width=6, id="log-col")
    ], className="mb-4", id="upload-log-row"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Chat with your PDFs"),
                dbc.CardBody([
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
                            # html.Small(
                            #     "Only uses information from uploaded documents. LLM cannot use its pre-trained knowledge.",
                            #     id="prompt-explainer",
                            #     className="text-muted mb-3 d-block"
                            # )
                        ], width=2)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Small(
                                "Only uses information from uploaded documents. LLM cannot use its pre-trained knowledge.",
                                id="prompt-explainer",
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
    
    # Auto-refresh interval for log display
    dcc.Interval(id="log-interval", interval=1000, n_intervals=0),
    
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
    ], id="prompt-modal", size="lg", is_open=False)
    
], fluid=True)

# Callback to toggle advanced configuration collapse and update triangle icon
@app.callback(
    [Output("advanced-config-collapse", "is_open"),
     Output("triangle-icon", "children")],
    Input("advanced-config-collapse-button", "n_clicks"),
    State("advanced-config-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks and n_clicks > 0:
        new_state = not is_open
        triangle = "▲" if new_state else "▼"
        return new_state, triangle
    return is_open, "▼"

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
        'strict': "Only uses information from uploaded documents. LLM cannot use its pre-trained knowledge.",
        'moderate': "Primarily uses documents but may supplement with LLM's knowledge when documents are insufficient.",
        'loose': "Uses documents as starting point but freely combines with LLM's broader knowledge.",
        'simple': "Document-focused with minimal constraints on knowledge usage."
    }
    
    explainer_text = explainers.get(prompt_type, explainers['strict'])
    
    return prompt_type, explainer_text


# Separate callback to control UI elements based on RAG mode
@app.callback(
    [Output('upload-pdf', 'disabled'),
     Output('advanced-config-collapse-button', 'disabled'),
     Output('advanced-config-row', 'style'),
     Output('upload-col', 'style'),
     Output('log-col', 'width'),
     Output('model-name-input', 'disabled'),
     Output('chunk-size-input', 'disabled'),
     Output('chunk-overlap-input', 'disabled'),
     Output('retrieval-count-input', 'disabled'),
     Output('cross-encoder-model-input', 'disabled'),
     Output('cross-encoder-top-n-input', 'disabled'),
     Output('check-processed-button', 'disabled'),
     Output('prompt-type-select', 'disabled')],
    Input('rag-mode-checkbox', 'value')
)
def control_ui_elements(rag_mode):
    if rag_mode is None:
        rag_mode = session_data['rag_mode']
    
    # Disable/enable components based on RAG mode
    disabled = not rag_mode
    
    # Show/hide advanced config row based on RAG mode
    advanced_config_style = {'display': 'block', 'marginBottom': '1.5rem'} if rag_mode else {'display': 'none'}
    
    # Show/hide upload column and adjust log column width
    if rag_mode:
        upload_col_style = {'display': 'block'}
        log_col_width = 6
    else:
        upload_col_style = {'display': 'none'}
        log_col_width = 12
    
    return disabled, disabled, advanced_config_style, upload_col_style, log_col_width, disabled, disabled, disabled, disabled, disabled, disabled, disabled, disabled

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
        prompt_instructions = {
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
        
        selected_instructions = prompt_instructions.get(prompt_type, prompt_instructions['strict'])
        
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

# Callback to check for processed PDFs
@app.callback(
    [Output('upload-status', 'children', allow_duplicate=True),
     Output('processing-stage', 'children', allow_duplicate=True)],
    Input('check-processed-button', 'n_clicks'),
    prevent_initial_call=True
)
def check_processed_pdfs(n_clicks):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    try:
        print("Checking for processed PDFs...")
        
        # Set up directories
        if session_data['dir'] is None:
            session_data['dir'] = Path.cwd()
        
        dir = session_data['dir']
        docs_dir = dir / "docs_pdf"
        processed_dir = dir / "docs_md"
        embeddings_dir = dir / "embeddings"
        
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
            dbc.Alert(f"Found and loaded {len(doc_ids)} processed documents: {', '.join(doc_ids)}. Ready to chat!", color="success"),
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
     Output('file-list', 'children'),
     Output('processing-stage', 'children'),
     Output('progress-bar', 'value'),
     Output('progress-bar', 'style')],
    [Input('upload-pdf', 'contents')],
    [State('upload-pdf', 'filename')]
)
def handle_file_upload(contents, filenames):
    if contents is None:
        raise PreventUpdate
    
    try:
        # Create directories and save files
        if session_data['dir'] is None:
            session_data['dir'] = Path.cwd()
        
        dir = session_data['dir']
        docs_dir = dir / "docs_pdf"
        processed_dir = dir / "docs_md"
        embeddings_dir = dir / "embeddings"
        
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
                "",
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
            html.Ul([html.Li(f"{doc_id}.pdf") for doc_id in session_data['doc_paths']]),
            'ready',
            100,
            {'display': 'block'}
        )
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return (
            dbc.Alert(f"Error: {str(e)}", color="danger"),
            "",
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
     State('processing-stage', 'children')]
)
def handle_chat(n_clicks, n_submit, user_input, messages_json, processing_state):
    print(f"Processing state: {processing_state}")
    if (n_clicks is None and n_submit is None) or not user_input or processing_state != 'ready':
        raise PreventUpdate
    
    # Additional check to ensure system is properly initialized
    if not session_data.get('initialized', False):
        raise PreventUpdate
    
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
            print("Initializing LLM...")
            session_data['llm'] = ChatOllama(model="qwen3:0.6b")
        
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
            print(f"Using prompt type: {prompt_type}")
            
            # Define instruction templates
            prompt_instructions = {
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
            
            selected_instructions = prompt_instructions.get(prompt_type, prompt_instructions['strict'])
            
            rag_prompt = f"""
            <QUERY>
            {user_input}
            </QUERY>

            <INSTRUCTIONS>
            {selected_instructions}
            </INSTRUCTIONS>
            
            <DOCUMENTS>
            {context_texts_pretty}
            </DOCUMENTS>
            """
            
            print("Invoking LLM for RAG response...")
            response = session_data['llm'].invoke(rag_prompt)
        else:
            # Non-RAG mode: Direct chat without document context
            print("Generating direct response...")
            chat_prompt = f"""
            <QUERY>
            {user_input}
            </QUERY>

            <INSTRUCTIONS>
            Provide a helpful and informative response to the user's query.
            Be concise but thorough in your explanation.
            </INSTRUCTIONS>
            """
            
            print("Invoking LLM for direct response...")
            response = session_data['llm'].invoke(chat_prompt)
        
        print("Processing response...")
        # Add assistant message
        messages.append({
            'role': 'assistant',
            'content': response.content.split("</think>")[1] if "</think>" in response.content else response.content,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Create chat history display
        chat_display = []
        for msg in messages:
            if msg['role'] == 'user':
                chat_display.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.Small(msg['timestamp'], className="text-muted"),
                            html.P(msg['content'], className="mb-0")
                        ])
                    ], color="primary", outline=True, className="mb-2")
                )
            else:
                chat_display.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.Small(msg['timestamp'], className="text-muted"),
                            html.P(msg['content'], className="mb-0")
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
    print("Starting PDF RAG Chatbot application...")
    app.run(debug=True, port=8060)