"""
Chat processing callbacks for app_clean_modular.py
Handles user input, chat processing, and conversation management.
"""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from dash import callback, Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import nltk
import numpy as np
from sklearn.preprocessing import minmax_scale

from config.session_state import session_data, validate_session_state
from config.prompt_templates import PROMPT_INSTRUCTIONS
from models.cross_encoder import CrossEncoderPipeline
from utils.content_processing import create_error_alert, parse_html_content, get_bm25_path, load_bm25_corpus
from utils.logging_setup import log_message

# Import LLM clients

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
from langchain_ollama import ChatOllama
from openai import OpenAI
import google.generativeai as genai


def register_chat_callbacks(app, personal_stats, query_progress_tracker):
    """Register all chat processing callbacks"""
    
    @app.callback(
        [Output('chat-history', 'children'),
         Output('chat-messages', 'children'),
         Output('user-input', 'value'),
         Output('chat-processing-trigger', 'data')],
        [Input('send-button', 'n_clicks'),
         Input('user-input', 'n_submit'),
         Input('clear-chat-button', 'n_clicks')],
        [State('user-input', 'value'),
         State('chat-messages', 'children'),
         State('rag-mode-checkbox', 'value'),
         State('prompt-type-select', 'value'),
         State('conversation-mode-select', 'value'),
         State('project-data', 'data')],
        prevent_initial_call=True
    )
    def handle_user_input(send_clicks, submit_enter, clear_clicks, user_input, messages_json, rag_mode, prompt_type, conversation_mode, project_data):
        """Handle user input and prepare for processing"""
        # Check which input triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Handle clear chat button
        if trigger_id == 'clear-chat-button':
            # Clear conversation history
            if session_data.get('conversation_history'):
                session_data['conversation_history'] = []
            
            # Track the clear action
            personal_stats.track_button_click('clear_chat')
            
            return [], '[]', '', None  # Clear chat history, messages, input, and don't trigger processing
        
        if trigger_id in ['send-button', 'user-input'] and user_input and user_input.strip():
            try:
                # Validate system state
                if not validate_session_state():
                    return [create_error_alert("System not initialized")], no_update, no_update, no_update
                
                # Parse existing messages
                messages = json.loads(messages_json) if messages_json else []
                
                # Add user message
                user_message = {
                    'role': 'user',
                    'content': user_input.strip(),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                messages.append(user_message)
                
                # Create chat display elements
                chat_display = []
                for msg in messages:
                    if msg['role'] == 'user':
                        chat_display.append(
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.Strong("You", className="text-primary"),
                                        html.Small(f" at {msg['timestamp']}", className="text-muted ms-2")
                                    ]),
                                    html.Div(msg['content'], 
                                            className="mb-0", 
                                            style={"whiteSpace": "pre-wrap", "wordWrap": "break-word", "lineHeight": "1.5"})
                                ])
                            ], color="primary", outline=True, className="mb-2")
                        )
                    else:  # assistant
                        chat_display.append(
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.Strong("Assistant", className="text-success"),
                                        html.Small(f" at {msg['timestamp']}", className="text-muted ms-2")
                                    ]),
                                    html.Div(parse_html_content(msg['content']), 
                                            className="mb-0", 
                                            style={"whiteSpace": "pre-wrap", "wordWrap": "break-word", "lineHeight": "1.5"})
                                ])
                            ], color="secondary", outline=True, className="mb-2")
                        )
                
                # Trigger the processing callback with the updated messages
                trigger_data = {
                    'messages': json.dumps(messages),
                    'timestamp': time.time(),
                    'user_input': user_input,
                    'rag_mode': rag_mode,
                    'prompt_type': prompt_type,
                    'conversation_mode': conversation_mode,
                    'project_data': project_data
                }
                
                return chat_display, json.dumps(messages), "", trigger_data
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                log_message(f"User input error: {error_msg}")
                
                messages = json.loads(messages_json) if messages_json else []
                messages.append({
                    'role': 'assistant',
                    'content': error_msg,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                return [create_error_alert(error_msg)], json.dumps(messages), "", no_update
        
        return no_update, no_update, no_update, no_update

    @app.callback(
        [Output('chat-history', 'children', allow_duplicate=True),
         Output('chat-messages', 'children', allow_duplicate=True)],
        [Input('chat-processing-trigger', 'data')],
        prevent_initial_call=True
    )
    def handle_chat_processing(trigger_data):
        """Handle the actual chat processing with RAG and LLM"""
        if not trigger_data:
            raise PreventUpdate
        
        try:
            # Extract data from trigger
            messages = json.loads(trigger_data['messages'])
            user_input = trigger_data['user_input']
            rag_mode = trigger_data['rag_mode']
            prompt_type = trigger_data['prompt_type']
            conversation_mode = trigger_data['conversation_mode']
            project_data = trigger_data['project_data']
            
            # Initialize query progress tracking
            query_progress_tracker.reset()
            query_progress_tracker.set_stage("initializing", detail="Setting up query processing")
            
            # Track model usage for every query
            model_name = session_data['llm_config']['model_name']
            personal_stats.track_model_usage(model_name)
            
            if session_data['llm'] is None:
                log_message(f"Initializing LLM with model: {model_name}...")
                
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
                hybrid_enabled = session_data['hybrid_search_config']['enabled']
                
                # Update progress to retrieval stage  
                query_progress_tracker.set_stage("retrieving", detail=f"Retrieving {retrieval_count} relevant documents")
                
                if hybrid_enabled:
                    log_message("Hybrid search enabled - retrieving all documents for hybrid scoring...")
                    
                    # Get all documents with dense scores
                    all_dense_results = session_data['bi_encoder'].retrieve_all(
                        user_input, 
                        session_data['embeddings']
                    )
                    
                    # Load BM25 corpus
                    bi_encoder_config = session_data['bi_encoder_config']
                    current_project = session_data.get('current_project')
                    bm25_path = get_bm25_path(
                        session_data['dir'], 
                        current_project,
                        bi_encoder_config['model_name'],
                        bi_encoder_config['chunk_size'],
                        bi_encoder_config['chunk_overlap']
                    )
                    
                    # Try to load existing BM25 corpus
                    bm25, tokenized_corpus = load_bm25_corpus(bm25_path)
                    
                    # If BM25 corpus doesn't exist, create it
                    if bm25 is None:
                        log_message("Creating BM25 corpus...")
                        from utils.content_processing import create_bm25_corpus
                        bm25, tokenized_corpus = create_bm25_corpus(session_data['embeddings'], bm25_path)
                    
                    if bm25 is not None:
                        log_message("Performing BM25 scoring...")
                        query_tokens = nltk.word_tokenize(user_input)
                        bm25_scores = bm25.get_scores(query_tokens)
                        
                        # Normalize BM25 scores using sklearn (like app_clean.py)
                        bm25_scores_normalized = minmax_scale(bm25_scores)
                        
                        # Combine dense and sparse scores (exact copy from app_clean.py)
                        bm25_weight = session_data['hybrid_search_config']['bm25_weight']
                        dense_scores = np.array([doc["similarity"] for doc in all_dense_results])
                        combined_scores = dense_scores + bm25_scores_normalized * bm25_weight
                        
                        # Get top k documents based on combined scores (exact copy from app_clean.py)
                        top_indices = np.argsort(combined_scores)[-retrieval_count:][::-1]
                        top_docs = []
                        for i in top_indices:
                            doc = all_dense_results[i].copy()
                            doc["bm25_score"] = bm25_scores_normalized[i]
                            doc["retrieval_score"] = combined_scores[i]
                            top_docs.append(doc)
                        
                        log_message(f"Hybrid search completed - retrieved {len(top_docs)} documents")
                    else:
                        log_message("BM25 corpus creation failed, falling back to dense-only search...")
                        dense_results = session_data['bi_encoder'].retrieve_top_k(user_input, session_data['embeddings'], top_k=retrieval_count)
                        top_docs = dense_results
                        log_message(f"Dense fallback completed - retrieved {len(top_docs)} documents")
                else:
                    # Pure dense retrieval (non-hybrid mode)
                    log_message(f"Dense search - retrieving top {retrieval_count} documents...")
                    dense_results = session_data['bi_encoder'].retrieve_top_k(user_input, session_data['embeddings'], top_k=retrieval_count)
                    top_docs = dense_results
                    log_message(f"Dense search completed - retrieved {len(top_docs)} documents")
                
                # Rerank documents if cross-encoder is available
                if session_data.get('cross_encoder'):
                    top_n = session_data['cross_encoder_config']['top_n']
                    query_progress_tracker.set_stage("reranking", detail=f"Reranking to top {top_n} documents")
                    log_message("Reranking documents with cross-encoder...")
                    
                    # Use hybrid reranking if hybrid search is enabled
                    hybrid_enabled = session_data['hybrid_search_config']['enabled']
                    if hybrid_enabled:
                        bm25_weight_rerank = session_data['hybrid_search_config']['bm25_weight_rerank']
                        top_docs = session_data['cross_encoder'].rerank(
                            user_input, 
                            top_docs, 
                            top_n=top_n,
                            hybrid_rerank=True,
                            bm25_weight=bm25_weight_rerank
                        )
                    else:
                        top_docs = session_data['cross_encoder'].rerank(
                            user_input, 
                            top_docs, 
                            top_n=top_n
                        )
                    log_message(f"Reranking completed - final count: {len(top_docs)}")
                
                # Prepare context for LLM
                context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(top_docs)])
                
                # Update progress to generation stage
                query_progress_tracker.set_stage("generating", detail="Generating response with retrieved context")
                
                # Prepare conversation history for multi-turn mode
                conversation_context = ""
                if conversation_mode and len(messages) > 1:
                    # Include last 8 messages (4 turns) for context
                    recent_messages = messages[-9:-1]  # Exclude the current user message
                    conversation_history = []
                    for msg in recent_messages:
                        role = "Human" if msg['role'] == 'user' else "Assistant"
                        conversation_history.append(f"{role}: {msg['content']}")
                    
                    if conversation_history:
                        conversation_context = f"\n\nPrevious conversation:\n{chr(10).join(conversation_history)}\n"
                
                # Create prompt with retrieved context
                prompt_template = PROMPT_INSTRUCTIONS.get(prompt_type, PROMPT_INSTRUCTIONS['moderate'])
                full_prompt = f"""{prompt_template}

DOCUMENTS:
{context}{conversation_context}

QUERY: {user_input}

Please provide a comprehensive answer based on the provided documents."""
                
                personal_stats.track_query(user_input, rag_mode)
                personal_stats.track_prompt_type(prompt_type)
                personal_stats.track_conversation_mode(conversation_mode)
                personal_stats.track_button_click('send')
                
            else:
                # Non-RAG mode: Direct LLM query
                query_progress_tracker.set_stage("generating", detail=f"Generating direct response in {conversation_mode} mode")
                
                conversation_context = ""
                if conversation_mode and len(messages) > 1:
                    recent_messages = messages[-9:-1]
                    conversation_history = []
                    for msg in recent_messages:
                        role = "Human" if msg['role'] == 'user' else "Assistant"
                        conversation_history.append(f"{role}: {msg['content']}")
                    
                    if conversation_history:
                        conversation_context = f"Previous conversation:\n{chr(10).join(conversation_history)}\n\n"
                
                full_prompt = f"{conversation_context}Human: {user_input}\n\nAssistant:"
                personal_stats.track_query(user_input, rag_mode)
                personal_stats.track_prompt_type(prompt_type)
                personal_stats.track_conversation_mode(conversation_mode)
                personal_stats.track_button_click('send')
            
            # Generate response based on model type
            start_time = time.time()
            
            if session_data['llm'] == 'openai':
                # OpenAI API call with proper conversation history handling
                client = session_data['llm_client']
                
                # Use proper conversation history format for OpenAI in multi-turn mode
                if conversation_mode and conversation_mode != 'single':
                    # Build messages with conversation history
                    openai_messages = build_openai_conversation_history(json.dumps(messages[:-1]))
                    
                    if session_data['rag_mode'] and session_data.get('embeddings'):
                        # RAG mode: Create system message with instructions and documents
                        prompt_template = PROMPT_INSTRUCTIONS.get(prompt_type, PROMPT_INSTRUCTIONS['moderate'])
                        system_message = f"""
                        <INSTRUCTIONS>
                        {prompt_template}
                        </INSTRUCTIONS>
                        
                        <DOCUMENTS>
                        {context}
                        </DOCUMENTS>
                        """
                    else:
                        # Non-RAG mode: Create system message with basic instructions
                        system_message = """
                        <INSTRUCTIONS>
                        Provide a helpful and informative response to the user's query.
                        Be concise but thorough in your explanation.
                        </INSTRUCTIONS>
                        """
                    
                    # Insert system message at the beginning
                    openai_messages.insert(0, {"role": "system", "content": system_message})
                    
                    # Add current user query
                    openai_messages.append({"role": "user", "content": user_input})
                    
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=openai_messages,
                        max_tokens=2000,
                        temperature=0.7
                    )
                else:
                    # Single-turn mode - use the full_prompt format
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=2000,
                        temperature=0.7
                    )
                
                ai_response = response.choices[0].message.content
                
            elif session_data['llm'] == 'gemini':
                # Gemini API call with proper conversation history handling
                client = session_data['llm_client']
                
                # Use proper conversation history format for Gemini in multi-turn mode
                if conversation_mode and conversation_mode != 'single':
                    # Build Gemini chat history
                    gemini_history = build_gemini_conversation_history(json.dumps(messages[:-1]))
                    
                    if session_data['rag_mode'] and session_data.get('embeddings'):
                        # RAG mode: Create system instruction with RAG prompt components
                        prompt_template = PROMPT_INSTRUCTIONS.get(prompt_type, PROMPT_INSTRUCTIONS['moderate'])
                        system_instruction = f"""
                        <INSTRUCTIONS>
                        {prompt_template}
                        </INSTRUCTIONS>
                        
                        <DOCUMENTS>
                        {context}
                        </DOCUMENTS>
                        """
                    else:
                        # Non-RAG mode: Create system instruction for direct chat
                        system_instruction = """
                        <INSTRUCTIONS>
                        Provide a helpful and informative response to the user's query.
                        Be concise but thorough in your explanation.
                        </INSTRUCTIONS>
                        """
                    
                    # Start a chat with history for multi-turn conversation
                    chat = client.start_chat(history=gemini_history)
                    
                    # Generate response with current query and system context
                    full_prompt_with_context = f"{system_instruction}\n\n<QUERY>\n{user_input}\n</QUERY>"
                    response = chat.send_message(full_prompt_with_context)
                    ai_response = response.text.strip()
                else:
                    # Single-turn mode - use the generate_content format
                    response = client.generate_content(full_prompt)
                    ai_response = response.text
                
            else:
                # Ollama/LangChain
                ai_response = session_data['llm'].invoke(full_prompt)
            
            response_time = time.time() - start_time
            
            # Print the raw LLM response to terminal before processing
            print(f"\n{'='*50}")
            print("RAW LLM RESPONSE:")
            print(f"{'='*50}")
            print(ai_response)
            print(f"{'='*50}\n")
            
            # Add project-specific PDF links (exact copy from working version)
            if session_data['rag_mode'] and session_data.get('embeddings'):
                current_project = session_data.get('current_project', 'default')
                
                # Extract context info exactly like working version (lines 3394-3396)
                context_ids = [doc["original_doc_id"] for doc in top_docs]
                context_pages = [doc["page"] for doc in top_docs]
                
                # Handle both old and new document reference formats (exact copy from working version)
                import re
                enhanced_response_text = ai_response
                for i, doc_name in enumerate(context_ids):
                    # Old format: DOCUMENT1, DOCUMENT2, etc.
                    enhanced_response_text = enhanced_response_text.replace(f"DOCUMENT{i+1}",
                                                                f"<a href='docs_pdf/{current_project}/{doc_name}.pdf#page={context_pages[i]}'>[{i+1}]</a>")
                
                # New format: <document_id>X</document_id>
                doc_id_pattern = r'<document_id>(\d+)</document_id>'
                def replace_doc_id(match):
                    doc_num = int(match.group(1))
                    if 1 <= doc_num <= len(context_ids):
                        doc_name = context_ids[doc_num-1]
                        return f"<a href='docs_pdf/{current_project}/{doc_name}.pdf#page={context_pages[doc_num-1]}'>[{doc_num}]</a>"
                    return match.group(0)  # Return original if invalid
                
                enhanced_response_text = re.sub(doc_id_pattern, replace_doc_id, enhanced_response_text)
            else:
                enhanced_response_text = ai_response
            
            # Track response statistics
            personal_stats.track_response(ai_response, response_time)
            
            # Add assistant response to messages
            assistant_message = {
                'role': 'assistant',
                'content': enhanced_response_text,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            messages.append(assistant_message)
            
            # Update conversation history in session
            session_data['conversation_history'] = messages
            
            # Complete query processing
            query_progress_tracker.complete()
            
            # Create updated chat display
            chat_display = []
            for msg in messages:
                if msg['role'] == 'user':
                    chat_display.append(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Strong("You", className="text-primary"),
                                    html.Small(f" at {msg['timestamp']}", className="text-muted ms-2")
                                ]),
                                html.Div(msg['content'], 
                                        className="mb-0", 
                                        style={"whiteSpace": "pre-wrap", "wordWrap": "break-word", "lineHeight": "1.5"})
                            ])
                        ], color="primary", outline=True, className="mb-2")
                    )
                else:  # assistant
                    chat_display.append(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Strong("Assistant", className="text-success"),
                                    html.Small(f" at {msg['timestamp']}", className="text-muted ms-2")
                                ]),
                                html.Div(parse_html_content(msg['content']), 
                                        className="mb-0", 
                                        style={"whiteSpace": "pre-wrap", "wordWrap": "break-word", "lineHeight": "1.5"})
                            ])
                        ], color="secondary", outline=True, className="mb-2")
                    )
            
            return chat_display, json.dumps(messages)
            
        except Exception as e:
            error_msg = f"Chat processing error: {str(e)}"
            log_message(f"Chat processing error: {traceback.format_exc()}")
            personal_stats.track_error('chat_errors')
            
            # Add error message to conversation
            messages = json.loads(trigger_data['messages'])
            error_message = {
                'role': 'assistant', 
                'content': error_msg,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            messages.append(error_message)
            
            return [create_error_alert(error_msg)], json.dumps(messages)

    print("Chat processing callbacks registered successfully")