# Configuration and modal callbacks for the RAG application
from datetime import datetime
from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

def register_config_callbacks(app, session_data, personal_stats, PROMPT_INSTRUCTIONS):
    """Register all configuration and modal callbacks"""
    
    @app.callback(
        [Output('log-modal', 'is_open'),
         Output('rag-config-modal', 'is_open'),
         Output('prompt-modal', 'is_open')],
        [Input('view-log-button', 'n_clicks'),
         Input('close-log-modal', 'n_clicks'),
         Input('advanced-rag-config-button', 'n_clicks'),
         Input('close-rag-config-modal', 'n_clicks'),
         Input('view-prompt-link', 'n_clicks'),
         Input('close-prompt-modal', 'n_clicks')],
        [State('log-modal', 'is_open'),
         State('rag-config-modal', 'is_open'),
         State('prompt-modal', 'is_open')]
    )
    def toggle_modals(view_log_clicks, close_log_clicks, 
                      rag_config_clicks, close_rag_config_clicks,
                      view_prompt_clicks, close_prompt_clicks,
                      log_is_open, rag_is_open, prompt_is_open):
        """Toggle modal visibility"""
        ctx = callback_context
        if not ctx.triggered:
            return False, False, False
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'view-log-button':
            personal_stats.track_button_click('view_log')
            return True, False, False
        elif button_id == 'close-log-modal':
            return False, False, False
        elif button_id == 'advanced-rag-config-button':
            personal_stats.track_button_click('view_config')
            return False, True, False
        elif button_id == 'close-rag-config-modal':
            return False, False, False
        elif button_id == 'view-prompt-link':
            personal_stats.track_button_click('view_prompt')
            return False, False, True
        elif button_id == 'close-prompt-modal':
            return False, False, False
        
        return log_is_open, rag_is_open, prompt_is_open

    @app.callback(
        Output('prompt-display', 'children'),
        [Input('prompt-type-select', 'value')]
    )
    def update_prompt_display(prompt_type):
        """Update prompt display based on selected type"""
        return PROMPT_INSTRUCTIONS.get(prompt_type, "No prompt selected")

    @app.callback(
        Output('bi-encoder-config', 'data'),
        [Input('bi-encoder-model-select', 'value'),
         Input('chunk-size-select', 'value'),
         Input('chunk-overlap-select', 'value'),
         Input('retrieval-count-select', 'value')]
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
        
        # Track config change
        personal_stats.track_config_change()
        
        # Reset bi-encoder to force reinitialization with new parameters
        session_data['bi_encoder'] = None
        
        return session_data['bi_encoder_config']

    @app.callback(
        Output('cross-encoder-config', 'data'),
        [Input('cross-encoder-model-select', 'value'),
         Input('reranking-count-select', 'value')]
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
        
        # Track config change
        personal_stats.track_config_change()
        
        # Reset cross-encoder to force reinitialization with new parameters
        session_data['cross_encoder'] = None
        
        return session_data['cross_encoder_config']

    @app.callback(
        Output('reranking-count-select', 'options'),
        Input('chunk-size-select', 'value')
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

    @app.callback(
        Output('hybrid-search-config', 'data'),
        [Input('hybrid-search-enabled', 'value'),
         Input('bm25-weight-input', 'value'),
         Input('bm25-weight-rerank-input', 'value')]
    )
    def update_hybrid_search_config(enabled_value, bm25_weight, bm25_weight_rerank):
        if bm25_weight is None or bm25_weight_rerank is None:
            raise PreventUpdate
        
        enabled = 'enabled' in (enabled_value or [])
        
        # Update session data
        print(f"Updating hybrid search configuration: enabled = {enabled}, bm25_weight = {bm25_weight}, bm25_weight_rerank = {bm25_weight_rerank}")
        session_data['hybrid_search_config'] = {
            'enabled': enabled,
            'bm25_weight': float(bm25_weight),
            'bm25_weight_rerank': float(bm25_weight_rerank)
        }
        
        # Track config change
        personal_stats.track_config_change()
        
        return session_data['hybrid_search_config']

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

    @app.callback(
        Output('openai-api-key-input', 'value'),
        Input('openai-api-key-input', 'value'),
        prevent_initial_call=True
    )
    def update_openai_api_key(api_key):
        if api_key is None:
            raise PreventUpdate
        
        session_data['llm_config']['api_key'] = api_key
        print("OpenAI/Gemini API key updated")
        
        # Reset the LLM instance to force reinitialization with new API key
        session_data['llm'] = None
        session_data['llm_client'] = None
        
        return api_key

    @app.callback(
        Output('rag-mode-checkbox', 'value'),
        Input('rag-mode-checkbox', 'value'),
        prevent_initial_call=True
    )
    def update_rag_mode(rag_mode):
        if rag_mode is None:
            raise PreventUpdate
        session_data['rag_mode'] = rag_mode
        personal_stats.track_rag_mode_toggle()
        print(f"RAG mode {'enabled' if rag_mode else 'disabled'}")
        return rag_mode

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

    @app.callback(
        [Output('upload-pdf', 'disabled', allow_duplicate=True),
         Output('advanced-rag-config-button', 'disabled'),
         Output('advanced-rag-config-button', 'style'),
         Output('upload-col', 'style'),
         Output('project-management-col', 'style'),
         Output('pdf-upload-col', 'style'),
         Output('prompt-type-col', 'style'),
         Output('prompt-explainer-row', 'style'),
         Output('bi-encoder-model-select', 'disabled'),
         Output('chunk-size-select', 'disabled'),
         Output('chunk-overlap-select', 'disabled'),
         Output('retrieval-count-select', 'disabled'),
         Output('cross-encoder-model-select', 'disabled'),
         Output('reranking-count-select', 'disabled'),
         Output('hybrid-search-enabled', 'disabled'),
         Output('bm25-weight-input', 'disabled'),
         Output('bm25-weight-rerank-input', 'disabled'),
         Output('check-processed-button', 'disabled', allow_duplicate=True),
         Output('prompt-type-select', 'disabled'),
         Output('conversation-mode-select', 'disabled'),
         Output('clear-chat-button', 'disabled'),
         Output('llm-model-input', 'disabled'),
         Output('openai-api-key-input', 'disabled')],
        [Input('rag-mode-checkbox', 'value'),
         Input('project-data', 'data')],
        prevent_initial_call='initial_duplicate'  # FIXED: Allow initial call with duplicate outputs
    )
    def control_ui_elements(rag_mode, project_data):
        # FIXED: Handle initial load case properly
        if rag_mode is None:
            rag_mode = session_data.get('rag_mode', True)  # Default to True if not set
        
        # FIXED: Better handling of project_data when it might be None
        current_project = None
        if project_data and isinstance(project_data, dict):
            current_project = project_data.get('current_project')
        elif session_data.get('current_project'):
            current_project = session_data['current_project']
        
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
        
        # Project management should be visible only in RAG mode
        project_management_style = {'display': 'block' if rag_mode else 'none'}
        
        # PDF upload should be visible only in RAG mode
        pdf_upload_style = {'display': 'block' if rag_mode else 'none'}
        
        # Prompt type should be visible only in RAG mode
        prompt_type_style = {'display': 'block' if rag_mode else 'none'}
        
        # Prompt explainer should be visible only in RAG mode
        prompt_explainer_style = {'display': 'block' if rag_mode else 'none'}
        
        # RAG config button style
        rag_config_button_style = {'display': 'block' if rag_mode else 'none'}
        
        # Conversation mode is always enabled
        conversation_mode_disabled = False
        
        return (
            upload_disabled,                        # upload-pdf disabled
            rag_config_disabled,                    # advanced-rag-config-button disabled
            rag_config_button_style,                # advanced-rag-config-button style
            upload_col_style,                       # upload-col style
            project_management_style,               # project-management-col style
            pdf_upload_style,                       # pdf-upload-col style
            prompt_type_style,                      # prompt-type-col style
            prompt_explainer_style,                 # prompt-explainer-row style
            rag_config_disabled,                    # bi-encoder-model-select disabled
            rag_config_disabled,                    # chunk-size-select disabled
            rag_config_disabled,                    # chunk-overlap-select disabled
            rag_config_disabled,                    # retrieval-count-select disabled
            rag_config_disabled,                    # cross-encoder-model-select disabled
            rag_config_disabled,                    # reranking-count-select disabled
            rag_config_disabled,                    # hybrid-search-enabled disabled
            rag_config_disabled,                    # bm25-weight-input disabled
            rag_config_disabled,                    # bm25-weight-rerank-input disabled
            upload_disabled,                        # check-processed-button disabled
            rag_config_disabled,                    # prompt-type-select disabled
            conversation_mode_disabled,             # conversation-mode-select disabled
            rag_config_disabled,                    # clear-chat-button disabled
            llm_disabled,                           # llm-model-input disabled
            llm_disabled                            # openai-api-key-input disabled
        )

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
        
        explainer_text = explainers.get(prompt_type, explainers['loose'])
        
        return prompt_type, explainer_text

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

    @app.callback(
        Output('stats-status', 'children'),
        Input('log-stats-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def log_personal_statistics(n_clicks):
        """Generate and save personal statistics report"""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        
        try:
            # Track this button click
            personal_stats.track_button_click('log_stats')
            
            # Generate statistics report
            report = personal_stats.get_statistics_report()
            
            # Save to file
            with open('personal_log.txt', 'w') as f:
                f.write(report)
            
            print("Personal statistics logged to personal_log.txt")
            
            return dbc.Alert([
                html.Strong("Statistics Logged Successfully!"),
                html.Br(),
                "Your personal usage statistics have been saved to 'personal_log.txt'. The file contains detailed information about your RAG chatbot usage patterns, including query statistics, model usage, button clicks, and more.",
                html.Br(),
                html.Small(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", className="text-muted")
            ], color="success", dismissable=True)
            
        except Exception as e:
            error_msg = f"Error logging statistics: {str(e)}"
            print(error_msg)
            personal_stats.track_error('general_errors')
            
            return dbc.Alert([
                html.Strong("Error Logging Statistics"),
                html.Br(),
                f"Failed to save statistics: {str(e)}"
            ], color="danger", dismissable=True)

    # Remove duplicate callbacks (they were causing conflicts)
    # Removed duplicate update_openai_api_key callback
    # Removed duplicate update_rag_mode callback
    # Removed duplicate toggle_api_key_input callback
    # Removed duplicate update_prompt_display callback
    # Removed duplicate update_prompt_type callback
    # Removed duplicate update_conversation_mode callback