"""
Feedback system callbacks for app_clean_modular.py
Handles feedback modal and data collection functionality.
"""

import os
import json
import traceback
from datetime import datetime
from dash import callback, Input, Output, State, callback_context, no_update, html, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from config.session_state import session_data
from utils.logging_setup import log_message


def collect_feedback_information(user_query, llm_response, user_comment, retrieved_chunks=None):
    """Collect comprehensive feedback information"""
    feedback_text = f"""# RAG Chatbot Feedback Report

## User Information
- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Session ID: {id(session_data)}

## Query Information
- **User Query:** {user_query}

## Response Information
- **LLM Response:** {llm_response}

## User Feedback
- **Comment:** {user_comment}

## System Configuration
- **LLM Model:** {session_data.get('llm_config', {}).get('model_name', 'Unknown')}
- **BiEncoder Model:** {session_data.get('bi_encoder_config', {}).get('model_name', 'Unknown')}
- **CrossEncoder Model:** {session_data.get('cross_encoder_config', {}).get('model_name', 'Unknown')}
- **Chunk Size:** {session_data.get('bi_encoder_config', {}).get('chunk_size', 'Unknown')}
- **Retrieval Count:** {session_data.get('bi_encoder_config', {}).get('retrieval_count', 'Unknown')}
- **RAG Mode:** {session_data.get('rag_mode', 'Unknown')}
- **Prompt Type:** {session_data.get('prompt_type', 'Unknown')}
- **Conversation Mode:** {session_data.get('conversation_mode', 'Unknown')}
- **Current Project:** {session_data.get('current_project', 'Unknown')}

## Retrieved Documents Context
"""
    
    if retrieved_chunks and len(retrieved_chunks) > 0:
        feedback_text += f"- **Number of Retrieved Chunks:** {len(retrieved_chunks)}\n\n"
        for i, chunk in enumerate(retrieved_chunks[:5]):  # Limit to first 5 chunks
            feedback_text += f"### Retrieved Document {i+1}\n"
            feedback_text += f"- **Content:** {chunk.page_content[:200]}...\n"
            if hasattr(chunk, 'metadata'):
                feedback_text += f"- **Metadata:** {chunk.metadata}\n"
            feedback_text += "\n"
    else:
        feedback_text += "- No retrieved chunks available (Non-RAG mode or no documents loaded)\n\n"
    
    feedback_text += "\n---\nThis feedback was generated automatically by the RAG Chatbot system.\n"
    
    return feedback_text


def save_feedback_to_file(feedback_text):
    """Save feedback to a timestamped file in the feedback directory"""
    try:
        # Create feedback directory if it doesn't exist
        os.makedirs('feedback', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_{timestamp}.md"
        filepath = os.path.join('feedback', filename)
        
        # Write feedback to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(feedback_text)
        
        return filepath
    except Exception as e:
        log_message(f"Error saving feedback: {str(e)}")
        return None


def register_feedback_callbacks(app, personal_stats):
    """Register all feedback system callbacks"""
    
    @app.callback(
        [Output('feedback-modal', 'is_open'),
         Output('feedback-data', 'children'),
         Output('feedback-status', 'children'),
         Output('feedback-global-status', 'children')],
        [Input({'type': 'share-button', 'index': ALL}, 'n_clicks'),
         Input('cancel-feedback-button', 'n_clicks'),
         Input('submit-feedback-button', 'n_clicks')],
        [State('feedback-modal', 'is_open'),
         State('feedback-comment', 'value'),
         State('chat-messages', 'children'),
         State('feedback-data', 'children')],
        prevent_initial_call=True
    )
    def handle_feedback_modal(share_clicks, cancel_clicks, submit_clicks, is_open, comment, messages_json, feedback_data_str):
        """Handle opening/closing feedback modal and collecting feedback data"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        # Handle share button clicks (open modal)
        if 'share-button' in trigger_id:
            # Extract the message index from the trigger
            trigger_data = json.loads(trigger_id.split('.')[0])
            message_index = trigger_data['index']
            
            try:
                # Parse messages and extract the specific message
                messages = json.loads(messages_json) if messages_json else []
                
                if 0 <= message_index < len(messages):
                    target_message = messages[message_index]
                    
                    # Find the corresponding user query (usually the previous message)
                    user_query = "Unknown query"
                    if message_index > 0 and messages[message_index - 1]['role'] == 'user':
                        user_query = messages[message_index - 1]['content']
                    
                    # Get retrieved chunks if available (from session data)
                    retrieved_chunks = None
                    if session_data.get('rag_mode') and session_data.get('embeddings'):
                        # Get the most recent retrieval results
                        # Note: This is a simplified approach - in a full implementation,
                        # you might want to store retrieval results per query
                        try:
                            retrieval_count = session_data['bi_encoder_config']['retrieval_count']
                            retrieved_chunks = session_data['bi_encoder'].retrieve_top_k(user_query, k=retrieval_count)
                            retrieved_chunks = [result[0] for result in retrieved_chunks]  # Extract documents
                        except:
                            retrieved_chunks = None
                    
                    # Prepare feedback data
                    feedback_data = {
                        'message_index': message_index,
                        'message_content': target_message['content'],
                        'user_query': user_query,
                        'retrieved_chunks': retrieved_chunks
                    }
                    
                    personal_stats.track_button_click('feedback_opened')
                    return True, json.dumps(feedback_data), "", ""
                else:
                    error_alert = dbc.Alert("Invalid message index", color="danger", dismissable=True)
                    return True, "", error_alert, ""
            
            except Exception as e:
                log_message(f"Error opening feedback modal: {str(e)}")
                personal_stats.track_error('feedback_errors')
                error_alert = dbc.Alert(f"Error opening feedback: {str(e)}", color="danger", dismissable=True)
                return True, "", error_alert, ""
        
        # Handle cancel button
        elif 'cancel-feedback-button' in trigger_id:
            personal_stats.track_button_click('feedback_cancelled')
            return False, "", "", ""
        
        # Handle submit button
        elif 'submit-feedback-button' in trigger_id:
            if comment and comment.strip():
                try:
                    # Use the feedback data from state
                    if feedback_data_str:
                        feedback_data = json.loads(feedback_data_str)
                        
                        # Collect comprehensive feedback information
                        feedback_text = collect_feedback_information(
                            user_query=feedback_data.get('user_query', 'Unknown query'),
                            llm_response=feedback_data.get('message_content', 'Unknown response'),
                            user_comment=comment.strip(),
                            retrieved_chunks=feedback_data.get('retrieved_chunks')
                        )
                        
                        # Save feedback to file
                        filepath = save_feedback_to_file(feedback_text)
                        
                        if filepath:
                            log_message(f"Feedback saved to: {filepath}")
                            personal_stats.track_button_click('feedback_submitted')
                            # Show success message in global status area (outside modal)
                            success_alert = dbc.Alert([
                                html.Strong("Feedback Saved Successfully!"),
                                html.Br(),
                                f"Your feedback has been saved to: {os.path.basename(filepath)}",
                                html.Br(),
                                html.Small("Send feedback to konradas.m@8devices.com. Thank you for helping improve the system!", className="text-muted")
                            ], color="success", dismissable=True)
                            return False, "", "", success_alert
                        else:
                            log_message("Failed to save feedback")
                            personal_stats.track_error('feedback_errors')
                            error_alert = dbc.Alert("Failed to save feedback. Please check the logs.", color="danger", dismissable=True)
                            return True, feedback_data_str, error_alert, ""
                    else:
                        error_alert = dbc.Alert("No feedback data available. Please try again.", color="danger", dismissable=True)
                        return True, "", error_alert, ""
                    
                except Exception as e:
                    log_message(f"Error submitting feedback: {str(e)}")
                    personal_stats.track_error('feedback_errors')
                    error_alert = dbc.Alert(f"Error submitting feedback: {str(e)}", color="danger", dismissable=True)
                    return True, feedback_data_str, error_alert, ""
            else:
                # Show error if no comment provided
                error_alert = dbc.Alert("Please provide a comment before submitting feedback.", color="warning", dismissable=True)
                return True, feedback_data_str, error_alert, ""
        
        raise PreventUpdate

    @app.callback(
        [Output('feedback-comment', 'value'),
         Output('feedback-status', 'children', allow_duplicate=True)],
        Input('feedback-modal', 'is_open'),
        prevent_initial_call=True
    )
    def clear_feedback_comment(is_open):
        """Clear the feedback comment and status when modal opens"""
        if is_open:
            return "", ""
        raise PreventUpdate

    print("Feedback system callbacks registered successfully")