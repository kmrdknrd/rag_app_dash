# Modal dialogs for the RAG application
import dash_bootstrap_components as dbc
from dash import html

def create_modals():
    """Create all modal dialogs for the application"""
    return [
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
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "maxHeight": "400px",
                    "overflowY": "auto",
                    "backgroundColor": "#f8f9fa"
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
                html.H6("Bi-Encoder Configuration", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Model", className="mb-2"),
                        dbc.Select(
                            id='bi-encoder-model-select',
                            options=[
                                {'label': 'Snowflake Arctic Embed L v2.0', 'value': 'Snowflake/snowflake-arctic-embed-l-v2.0'},
                                {'label': 'BGE-M3', 'value': 'BAAI/bge-m3'}
                            ],
                            value='Snowflake/snowflake-arctic-embed-l-v2.0',  # Default value
                            className="mb-3"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Chunk Size", className="mb-2"),
                        dbc.Select(
                            id='chunk-size-select',
                            options=[
                                {'label': '512 tokens', 'value': 512},
                                {'label': '1024 tokens', 'value': 1024},
                                {'label': '2048 tokens', 'value': 2048},
                                {'label': '4096 tokens', 'value': 4096}
                            ],
                            value=2048,  # Default value
                            className="mb-3"
                        )
                    ], width=3),
                    dbc.Col([
                        html.Label("Chunk Overlap", className="mb-2"),
                        dbc.Select(
                            id='chunk-overlap-select',
                            options=[
                                {'label': '0 tokens', 'value': 0},
                                {'label': '128 tokens', 'value': 128},
                                {'label': '256 tokens', 'value': 256}
                            ],
                            value=128,  # Default value
                            className="mb-3"
                        )
                    ], width=3)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Retrieval Count", className="mb-2"),
                        dbc.Select(
                            id='retrieval-count-select',
                            options=[
                                {'label': '50 documents', 'value': 50},
                                {'label': '100 documents', 'value': 100},
                                {'label': '200 documents', 'value': 200}
                            ],
                            value=50,  # Default value
                            className="mb-3"
                        )
                    ], width=6)
                ]),
                html.Hr(),
                html.H6("Cross-Encoder Configuration", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Cross-Encoder Model", className="mb-2"),
                        dbc.Select(
                            id='cross-encoder-model-select',
                            options=[
                                {'label': 'MS Marco MiniLM-L6-v2', 'value': 'cross-encoder/ms-marco-MiniLM-L6-v2'},
                                {'label': 'Mxbai Rerank Base v2', 'value': 'mixedbread-ai/mxbai-rerank-base-v2'}
                            ],
                            value='cross-encoder/ms-marco-MiniLM-L6-v2',  # Default value
                            className="mb-3"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Reranking Count", className="mb-2"),
                        dbc.Select(
                            id='reranking-count-select',
                            options=[],  # Will be populated dynamically
                            value=8,  # Default value
                            className="mb-3"
                        )
                    ], width=6)
                ]),
                html.Hr(),
                html.H6("Hybrid Search Configuration", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Checklist(
                            id='hybrid-search-enabled',
                            options=[
                                {'label': 'Enable Hybrid Search (Dense + Sparse)', 'value': 'enabled'}
                            ],
                            value=[],  # Default value
                            className="mb-3"
                        )
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("BM25 Weight (Retrieval)", className="mb-2"),
                        dbc.Input(
                            id='bm25-weight-input',
                            type='number',
                            value=0.1,  # Default value
                            min=0,
                            max=1,
                            step=0.1,
                            className="mb-3"
                        ),
                        html.Small("Weight for BM25 scores in retrieval (0.0-1.0)", className="form-text text-muted")
                    ], width=6),
                    dbc.Col([
                        html.Label("BM25 Weight (Reranking)", className="mb-2"),
                        dbc.Input(
                            id='bm25-weight-rerank-input',
                            type='number',
                            value=0.9,  # Default value
                            min=0,
                            max=1,
                            step=0.1,
                            className="mb-3"
                        ),
                        html.Small("Weight for BM25 scores in reranking (0.0-1.0)", className="form-text text-muted")
                    ], width=6)
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-rag-config-modal", color="secondary")
            ])
        ], id="rag-config-modal", size="lg", is_open=False),
        
        # Feedback Modal
        dbc.Modal([
            dbc.ModalHeader([
                dbc.ModalTitle("Create Response Feedback")
            ]),
            dbc.ModalBody([
                html.Div(id="feedback-status", children=""),
                html.H6("Please provide feedback on what's wrong with this response:"),
                dbc.Textarea(
                    id="feedback-comment",
                    placeholder="Describe what's wrong with the response, what you expected, or any other feedback...",
                    rows=4,
                    style={"margin-bottom": "15px"}
                ),
                html.Hr(),
                html.H6("Technical Information (automatically included):"),
                html.Div([
                    html.P("✓ User query", className="mb-1"),
                    html.P("✓ LLM response", className="mb-1"),
                    html.P("✓ Current RAG configuration", className="mb-1"),
                    html.P("✓ Documents in project", className="mb-1"),
                    html.P("✓ Embedding status", className="mb-1"),
                    html.P("✓ Retrieved chunks", className="mb-1"),
                ], style={"font-size": "14px", "color": "#666"})
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-feedback-button", color="secondary", className="me-2"),
                dbc.Button("Create Feedback", id="submit-feedback-button", color="primary")
            ])
        ], id="feedback-modal", size="lg", is_open=False),
        
        # Hidden divs to store feedback data
        html.Div(id="feedback-data", style={"display": "none"}),
    ]