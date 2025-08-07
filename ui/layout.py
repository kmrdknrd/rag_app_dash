# Main application layout for the RAG chatbot
import dash_bootstrap_components as dbc
from dash import dcc, html
from pathlib import Path
from utils.project_management import get_available_projects

def create_main_layout(session_data, startup_message, startup_success):
    """Create the main application layout"""
    
    # Initialize project data at layout creation (same as working version)
    if not session_data.get('dir'):
        session_data['dir'] = Path.cwd()
    
    # Get available projects and populate session data
    projects = get_available_projects(session_data['dir'])
    session_data['projects'] = projects
    
    # Set default current project if none exists
    if not session_data.get('current_project') and projects:
        session_data['current_project'] = projects[0]
    
    # Create project options for dropdown
    project_options = [{'label': project, 'value': project} for project in projects]
    default_project = session_data.get('current_project')
    
    # Create initial project data
    initial_project_data = {
        'projects': projects,
        'current_project': default_project
    }
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("8devices RAG Chatbot (v0.5.1)", className="text-center mb-2 mt-2"),
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
                       className="mb-3", dismissable=True)
                ])
            ])
        ]),
        
        # Enhanced Progress Bar Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Upload Progress", className="card-title"),
                        dbc.Progress(
                            id="detailed-progress-bar",
                            value=0,
                            striped=True,
                            animated=True,
                            color="success",
                            style={"height": "25px", "margin-bottom": "10px"}
                        ),
                        html.Div(id="progress-status", children="Ready to upload files", 
                                style={"font-size": "14px", "color": "#666"}),
                        html.Div(id="current-file-status", children="", 
                                style={"font-size": "12px", "color": "#888", "margin-top": "5px"}),
                    ])
                ], style={"margin-bottom": "20px"})
            ], width=12, id="progress-container", style={"display": "none"})
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
                                            options=project_options,
                                            placeholder="Select a project...",
                                            value=default_project
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
                                        html.A('Click to Select PDF Files')
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
                                # Check processed PDFs progress bar
                                html.Div([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Check Progress", className="card-title"),
                                            dbc.Progress(
                                                id="check-processed-progress-bar",
                                                value=0,
                                                striped=True,
                                                animated=True,
                                                color="info",
                                                style={"height": "20px", "margin-bottom": "10px"}
                                            ),
                                            html.Div(id="check-progress-status", children="Ready to check for processed PDFs", 
                                                    style={"font-size": "14px", "color": "#666"}),
                                            html.Div(id="check-current-status", children="", 
                                                    style={"font-size": "12px", "color": "#888", "margin-top": "5px"}),
                                        ])
                                    ], style={"margin-bottom": "15px", "margin-top": "15px"})
                                ], id="check-progress-container", style={"display": "none"}),
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
                        # Query Processing Progress Bar
                        html.Div([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Query Processing", className="card-title"),
                                    dbc.Progress(
                                        id="query-progress-bar",
                                        value=0,
                                        striped=True,
                                        animated=True,
                                        color="primary",
                                        style={"height": "20px", "margin-bottom": "10px"}
                                    ),
                                    html.P(id="query-progress-status", children="Ready to process query", className="mb-1 small"),
                                    html.P(id="query-progress-detail", children="", className="mb-0 small text-muted")
                                ])
                            ])
                        ], id="query-progress-container", style={'display': 'none', 'margin-bottom': '10px'}),
                        
                        # Feedback status area
                        html.Div(id="feedback-global-status", children="", style={'margin-bottom': '10px'}),
                        
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
                                    className="d-block mb-2"
                                ),
                                dbc.Button(
                                    "Log Personal Statistics",
                                    id="log-stats-button",
                                    color="info",
                                    size="sm",
                                    className="d-block",
                                    outline=True
                                )
                            ], width=2)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Small(
                                    "Loose: Uses documents as starting point but freely combines with LLM's broader knowledge.",
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
                        ], className="mb-3"),
                        
                        # Statistics logging status
                        html.Div(id='stats-status', className="mb-3")
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
        
        # Hidden store for hybrid search configuration
        dcc.Store(id='hybrid-search-config', data=session_data['hybrid_search_config']),
        
        # Hidden store for project data
        dcc.Store(id='project-data', data=initial_project_data),
        
        # Hidden store for triggering chat processing
        dcc.Store(id='chat-processing-trigger', data=None),
        
        # Auto-refresh interval for log display
        dcc.Interval(id="log-interval", interval=1000, n_intervals=0),
        
        # Progress update interval (faster for smoother progress bar)
        dcc.Interval(id="progress-interval", interval=500, n_intervals=0),
        
    ], fluid=True)