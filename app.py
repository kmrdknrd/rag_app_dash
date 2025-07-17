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

# Enhanced Progress Tracking System
class ProgressTracker:
    """Centralized progress tracking for PDF processing"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all progress tracking"""
        self.total_files = 0
        self.current_stage = "idle"
        self.stage_progress = 0
        self.stage_total = 0
        self.current_file = ""
        self.current_file_index = 0
        self.overall_progress = 0
        self.stage_weights = {
            "upload": 5,
            "pdf_processing": 40,
            "embedding_init": 10,
            "embedding_creation": 45
        }
        self.stage_messages = {
            "upload": "Uploading files...",
            "pdf_processing": "Converting PDFs to text...",
            "embedding_init": "Initializing AI models...",
            "embedding_creation": "Creating document embeddings..."
        }
        self.detailed_log = []
        self.session_start_time = time.time()
    
    def set_stage(self, stage, total_items=1):
        """Set the current processing stage"""
        self.current_stage = stage
        self.stage_total = total_items
        self.stage_progress = 0
        self.log_message(f"Stage: {self.stage_messages.get(stage, stage)}")
    
    def update_stage_progress(self, progress, current_item=""):
        """Update progress within current stage"""
        self.stage_progress = progress
        self.current_file = current_item
        if current_item:
            self.log_message(f"Processing: {current_item}")
        self.calculate_overall_progress()
    
    def increment_stage_progress(self, item_name=""):
        """Increment stage progress by 1"""
        self.stage_progress += 1
        self.current_file = item_name
        if item_name:
            self.log_message(f"Processing: {item_name}")
        self.calculate_overall_progress()
    
    def calculate_overall_progress(self):
        """Calculate overall progress percentage"""
        if self.current_stage == "idle":
            self.overall_progress = 0
            return
        
        # Calculate progress for completed stages
        completed_weight = 0
        stage_order = ["upload", "pdf_processing", "embedding_init", "embedding_creation"]
        
        try:
            current_stage_index = stage_order.index(self.current_stage)
            for i in range(current_stage_index):
                completed_weight += self.stage_weights[stage_order[i]]
        except ValueError:
            current_stage_index = 0
        
        # Calculate progress for current stage
        if self.stage_total > 0:
            current_stage_progress = (self.stage_progress / self.stage_total) * self.stage_weights[self.current_stage]
        else:
            current_stage_progress = 0
        
        self.overall_progress = completed_weight + current_stage_progress
        self.overall_progress = min(100, max(0, self.overall_progress))
    
    def log_message(self, message):
        """Add a timestamped message to the detailed log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.detailed_log.append(log_entry)
        print(log_entry)  # Also print to console/log file
    
    def get_progress_info(self):
        """Get current progress information for UI"""
        if self.current_stage == "idle":
            return {
                "progress": 0,
                "stage": "Ready",
                "current_file": "",
                "detailed_message": "Ready to process files"
            }
        
        if self.current_stage == "complete":
            return {
                "progress": 100,
                "stage": "Complete",
                "current_file": "",
                "detailed_message": "All documents processed successfully! Ready to chat."
            }
        
        stage_message = self.stage_messages.get(self.current_stage, self.current_stage)
        
        if self.current_file:
            if self.current_stage == "pdf_processing":
                detailed_message = f"{stage_message} - Converting document {self.stage_progress}/{self.stage_total}: {self.current_file}"
            elif self.current_stage == "embedding_creation":
                detailed_message = f"{stage_message} - Embedding document {self.stage_progress}/{self.stage_total}: {self.current_file}"
            else:
                detailed_message = f"{stage_message} - {self.current_file}"
        else:
            detailed_message = stage_message
        
        return {
            "progress": self.overall_progress,
            "stage": stage_message,
            "current_file": self.current_file,
            "detailed_message": detailed_message
        }
    
    def complete(self):
        """Mark processing as complete"""
        self.current_stage = "complete"
        self.overall_progress = 100
        self.current_file = ""  # Clear current file to avoid confusion
        self.log_message("Processing complete! Ready to chat.")

# Global progress tracker instance
progress_tracker = ProgressTracker()

# Check processed PDFs progress tracker
class CheckProgressTracker:
    """Progress tracker specifically for check processed PDFs operation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all progress tracking"""
        self.current_step = 0
        self.total_steps = 6  # Based on the log steps
        self.current_status = "Ready to check for processed PDFs"
        self.current_detail = ""
        self.progress_percentage = 0
        self.is_active = False
    
    def set_step(self, step_num, status, detail=""):
        """Set current step and status"""
        self.current_step = step_num
        self.current_status = status
        self.current_detail = detail
        self.progress_percentage = (step_num / self.total_steps) * 100
        self.is_active = True
    
    def complete(self):
        """Mark as complete"""
        self.current_step = self.total_steps
        self.progress_percentage = 100
        self.current_status = "Check completed successfully"
        self.current_detail = ""
        self.is_active = False
    
    def get_progress_info(self):
        """Get current progress information"""
        return {
            "progress": self.progress_percentage,
            "status": self.current_status,
            "detail": self.current_detail,
            "is_active": self.is_active
        }

# Global check progress tracker instance
check_progress_tracker = CheckProgressTracker()


# Personal Statistics Tracking System
class PersonalStatistics:
    """Tracks personal usage statistics for the RAG chatbot"""
    
    def __init__(self):
        self.reset_session_stats()
        self.total_stats = {
            # Query statistics
            'total_queries': 0,
            'total_rag_queries': 0,
            'total_non_rag_queries': 0,
            'total_query_chars': 0,
            'total_query_tokens': 0,
            
            # Response statistics
            'total_responses': 0,
            'total_response_chars': 0,
            'total_response_tokens': 0,
            'total_response_time': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            
            # Document statistics
            'total_documents_uploaded': 0,
            'total_documents_processed': 0,
            'total_processing_time': 0,
            'documents_retrieved_total': 0,
            'document_references': {},
            
            # Model usage statistics
            'model_usage': {},
            'prompt_type_usage': {},
            'conversation_mode_usage': {},
            'api_vs_local_usage': {'api': 0, 'local': 0},
            
            # UI interaction statistics
            'button_clicks': {
                'send': 0,
                'clear_chat': 0,
                'upload': 0,
                'check_processed': 0,
                'create_project': 0,
                'delete_project': 0,
                'view_log': 0,
                'view_config': 0,
                'view_prompt': 0,
                'log_stats': 0
            },
            
            # Project statistics
            'projects_created': 0,
            'projects_deleted': 0,
            'project_switches': 0,
            
            # Session statistics
            'total_sessions': 0,
            'total_session_time': 0,
            'rag_mode_toggles': 0,
            'config_changes': 0,
            
            # Error statistics
            'errors': {
                'upload_errors': 0,
                'processing_errors': 0,
                'query_errors': 0,
                'api_errors': 0,
                'general_errors': 0
            }
        }
        self.session_start_time = time.time()
    
    def reset_session_stats(self):
        """Reset session-specific statistics"""
        self.session_stats = {
            'session_queries': 0,
            'session_uploads': 0,
            'session_duration': 0,
            'session_errors': 0,
            'session_button_clicks': 0
        }
    
    def estimate_tokens(self, text):
        """Rough token estimation (1 token ≈ 4 characters for English)"""
        return len(text) // 4 if text else 0
    
    def track_query(self, query_text, is_rag_mode=True):
        """Track a user query"""
        if not query_text:
            return
            
        self.total_stats['total_queries'] += 1
        self.session_stats['session_queries'] += 1
        
        if is_rag_mode:
            self.total_stats['total_rag_queries'] += 1
        else:
            self.total_stats['total_non_rag_queries'] += 1
        
        char_count = len(query_text)
        token_count = self.estimate_tokens(query_text)
        
        self.total_stats['total_query_chars'] += char_count
        self.total_stats['total_query_tokens'] += token_count
    
    def track_response(self, response_text, response_time=0, success=True):
        """Track an AI response"""
        if not response_text:
            return
            
        self.total_stats['total_responses'] += 1
        
        if success:
            self.total_stats['successful_responses'] += 1
        else:
            self.total_stats['failed_responses'] += 1
        
        char_count = len(response_text)
        token_count = self.estimate_tokens(response_text)
        
        self.total_stats['total_response_chars'] += char_count
        self.total_stats['total_response_tokens'] += token_count
        self.total_stats['total_response_time'] += response_time
    
    def track_document_upload(self, num_documents, processing_time=0):
        """Track document upload and processing"""
        self.total_stats['total_documents_uploaded'] += num_documents
        self.total_stats['total_documents_processed'] += num_documents
        self.total_stats['total_processing_time'] += processing_time
        self.session_stats['session_uploads'] += 1
    
    def track_model_usage(self, model_name):
        """Track which models are used"""
        if model_name:
            self.total_stats['model_usage'][model_name] = self.total_stats['model_usage'].get(model_name, 0) + 1
            
            # Track API vs local usage
            openai_models = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o3', 'o4-mini']
            gemini_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']
            
            if model_name in openai_models or model_name in gemini_models:
                self.total_stats['api_vs_local_usage']['api'] += 1
            else:
                self.total_stats['api_vs_local_usage']['local'] += 1
    
    def track_prompt_type(self, prompt_type):
        """Track prompt type usage"""
        if prompt_type:
            self.total_stats['prompt_type_usage'][prompt_type] = self.total_stats['prompt_type_usage'].get(prompt_type, 0) + 1
    
    def track_conversation_mode(self, conversation_mode):
        """Track conversation mode usage"""
        if conversation_mode:
            self.total_stats['conversation_mode_usage'][conversation_mode] = self.total_stats['conversation_mode_usage'].get(conversation_mode, 0) + 1
    
    def track_button_click(self, button_name):
        """Track button clicks"""
        if button_name in self.total_stats['button_clicks']:
            self.total_stats['button_clicks'][button_name] += 1
            self.session_stats['session_button_clicks'] += 1
    
    def track_project_action(self, action):
        """Track project actions (create, delete, switch)"""
        if action == 'create':
            self.total_stats['projects_created'] += 1
        elif action == 'delete':
            self.total_stats['projects_deleted'] += 1
        elif action == 'switch':
            self.total_stats['project_switches'] += 1
    
    def track_error(self, error_type):
        """Track errors"""
        if error_type in self.total_stats['errors']:
            self.total_stats['errors'][error_type] += 1
            self.session_stats['session_errors'] += 1
    
    def track_config_change(self):
        """Track configuration changes"""
        self.total_stats['config_changes'] += 1
    
    def track_rag_mode_toggle(self):
        """Track RAG mode toggles"""
        self.total_stats['rag_mode_toggles'] += 1
    
    def track_document_references(self, doc_ids):
        """Track which documents are referenced in responses"""
        for doc_id in doc_ids:
            if doc_id:
                self.total_stats['document_references'][doc_id] = self.total_stats['document_references'].get(doc_id, 0) + 1
    
    def update_session_duration(self):
        """Update current session duration"""
        self.session_stats['session_duration'] = time.time() - self.session_start_time
    
    def get_statistics_report(self):
        """Generate a comprehensive statistics report"""
        self.update_session_duration()
        
        # Calculate averages
        avg_query_chars = self.total_stats['total_query_chars'] / max(1, self.total_stats['total_queries'])
        avg_query_tokens = self.total_stats['total_query_tokens'] / max(1, self.total_stats['total_queries'])
        avg_response_chars = self.total_stats['total_response_chars'] / max(1, self.total_stats['total_responses'])
        avg_response_tokens = self.total_stats['total_response_tokens'] / max(1, self.total_stats['total_responses'])
        avg_response_time = self.total_stats['total_response_time'] / max(1, self.total_stats['total_responses'])
        avg_processing_time = self.total_stats['total_processing_time'] / max(1, self.total_stats['total_documents_processed'])
        
        # Most used items
        most_used_model = max(self.total_stats['model_usage'].items(), key=lambda x: x[1]) if self.total_stats['model_usage'] else ("None", 0)
        most_used_prompt = max(self.total_stats['prompt_type_usage'].items(), key=lambda x: x[1]) if self.total_stats['prompt_type_usage'] else ("None", 0)
        most_referenced_doc = max(self.total_stats['document_references'].items(), key=lambda x: x[1]) if self.total_stats['document_references'] else ("None", 0)
        
        report = f"""
=== PERSONAL RAG CHATBOT USAGE STATISTICS ===
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== QUERY & RESPONSE STATISTICS ===
Total Queries: {self.total_stats['total_queries']}
  - RAG Mode Queries: {self.total_stats['total_rag_queries']}
  - Non-RAG Queries: {self.total_stats['total_non_rag_queries']}
  - RAG Usage Rate: {(self.total_stats['total_rag_queries'] / max(1, self.total_stats['total_queries']) * 100):.1f}%

Average Query Length: {avg_query_chars:.1f} characters ({avg_query_tokens:.1f} tokens)
Total Responses: {self.total_stats['total_responses']}
  - Successful: {self.total_stats['successful_responses']}
  - Failed: {self.total_stats['failed_responses']}
  - Success Rate: {(self.total_stats['successful_responses'] / max(1, self.total_stats['total_responses']) * 100):.1f}%

Average Response Length: {avg_response_chars:.1f} characters ({avg_response_tokens:.1f} tokens)
Average Response Time: {avg_response_time:.2f} seconds

=== DOCUMENT STATISTICS ===
Total Documents Uploaded: {self.total_stats['total_documents_uploaded']}
Total Documents Processed: {self.total_stats['total_documents_processed']}
Total Processing Time: {self.total_stats['total_processing_time']:.1f} seconds
Average Processing Time per Document: {avg_processing_time:.1f} seconds
Total Documents Retrieved: {self.total_stats['documents_retrieved_total']}
Most Referenced Document: {most_referenced_doc[0]} ({most_referenced_doc[1]} references)

=== MODEL & CONFIGURATION USAGE ===
Most Used Model: {most_used_model[0]} ({most_used_model[1]} uses)
API vs Local Usage: {self.total_stats['api_vs_local_usage']['api']} API, {self.total_stats['api_vs_local_usage']['local']} Local
Most Used Prompt Type: {most_used_prompt[0]} ({most_used_prompt[1]} uses)

Model Usage Breakdown:"""
        
        for model, count in sorted(self.total_stats['model_usage'].items(), key=lambda x: x[1], reverse=True):
            report += f"\n  - {model}: {count} uses"
        
        report += f"\n\nPrompt Type Usage:"
        for prompt_type, count in sorted(self.total_stats['prompt_type_usage'].items(), key=lambda x: x[1], reverse=True):
            report += f"\n  - {prompt_type}: {count} uses"
        
        report += f"\n\nConversation Mode Usage:"
        for mode, count in sorted(self.total_stats['conversation_mode_usage'].items(), key=lambda x: x[1], reverse=True):
            report += f"\n  - {mode}: {count} uses"
        
        report += f"""

=== UI INTERACTION STATISTICS ===
Total Button Clicks: {sum(self.total_stats['button_clicks'].values())}"""
        
        for button, count in sorted(self.total_stats['button_clicks'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                report += f"\n  - {button}: {count} clicks"
        
        report += f"""

=== PROJECT STATISTICS ===
Projects Created: {self.total_stats['projects_created']}
Projects Deleted: {self.total_stats['projects_deleted']}
Project Switches: {self.total_stats['project_switches']}

=== SESSION STATISTICS ===
Current Session Duration: {self.session_stats['session_duration'] / 3600:.1f} hours
Session Queries: {self.session_stats['session_queries']}
Session Uploads: {self.session_stats['session_uploads']}
Session Button Clicks: {self.session_stats['session_button_clicks']}
Session Errors: {self.session_stats['session_errors']}

RAG Mode Toggles: {self.total_stats['rag_mode_toggles']}
Configuration Changes: {self.total_stats['config_changes']}

=== ERROR STATISTICS ===
Total Errors: {sum(self.total_stats['errors'].values())}"""
        
        for error_type, count in self.total_stats['errors'].items():
            if count > 0:
                report += f"\n  - {error_type}: {count} errors"
        
        report += f"""

=== TOP DOCUMENT REFERENCES ==="""
        
        top_docs = sorted(self.total_stats['document_references'].items(), key=lambda x: x[1], reverse=True)[:10]
        for doc_id, count in top_docs:
            report += f"\n  - {doc_id}: {count} references"
        
        report += "\n\n=== END OF STATISTICS REPORT ===\n"
        
        return report

# Global statistics tracker instance
personal_stats = PersonalStatistics()

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

# Enhanced PDF processor with progress tracking
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
        """Process document using pre-loaded model with progress tracking"""
        # Check if path is to a single file or a directory
        # Process a single file or all files in a directory
        all_texts = []
        
        # Use the file directly if it's a single file, otherwise get all PDFs in directory
        files = [doc_path] if doc_path.is_file() else list(doc_path.glob("*.pdf"))
        
        if save_path is not None:
            if not os.path.exists(save_path):
                # Create save path if it doesn't exist
                progress_tracker.log_message(f"Creating save directory: {save_path}")
                os.makedirs(save_path)
        
        # Process each file
        for i, file in enumerate(files):
            # Skip if file already processed
            if save_path is not None and os.path.exists(save_path / f"{file.stem}.md"):
                progress_tracker.log_message(f"Skipping {file.stem} because .md already exists")
                # Load .md from disk
                with open(save_path / f"{file.stem}.md", "r") as f:
                    text = f.read()
                all_texts.append(text)
                continue
            
            progress_tracker.log_message(f"Processing {file.stem} ({i+1}/{len(files)})")
            
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

# Enhanced BiEncoder Pipeline with progress tracking
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
            
        progress_tracker.log_message(f"Initializing BiEncoderPipeline with model: {model_name}")
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

    def embed_documents(self, doc_texts, doc_ids=None, save_path=None, track_progress=True):
        """Embed documents using pre-loaded models with optional progress tracking"""
        # If string given (i.e., one document, big string), and not list (i.e., multiple documents or single document but list), make it a list
        if not isinstance(doc_texts, list):
            doc_texts = [doc_texts]
        
        if save_path is not None:
            actual_save_path = Path(save_path) / self.model_name.split("/")[-1] / f"chunk_size_{self.chunk_size}" / f"chunk_overlap_{self.chunk_overlap}"
            if track_progress:
                progress_tracker.log_message(f"Actual save path: {actual_save_path}")
                # Create save path if it doesn't exist
                progress_tracker.log_message(f"Creating save directory: {actual_save_path}")
            if not os.path.exists(actual_save_path):
                os.makedirs(actual_save_path)

        # Process each text in the list
        all_chunks = []
        all_vectors = []
        
        for i, doc in enumerate(doc_texts):
            doc_name = doc_ids[i] if doc_ids else f'doc_{i}'
            if track_progress:
                progress_tracker.log_message(f"Embedding document {i+1}/{len(doc_texts)}: {doc_name}")
                progress_tracker.increment_stage_progress(doc_name)
            
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
                    if track_progress:
                        progress_tracker.log_message(f"Skipping {doc_ids[i]} because all chunks already exist")
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

# Startup message
startup_message = f"""Naudojatės 8devices RAG Chatbot (v.0.4.1).
Turėkit omeny, kad ši aplikacija yra alfa versijoje, ir yra greitai atnaujinama. Dabartinis tikslas yra parodyti, kaip galima naudoti Retrieval-Augmented Generation (RAG) su PDF dokumentais, kad praturtinti LLM atsakymus. Visi modeliai yra lokalūs, ir dokumentai yra išsaugomi jūsų kompiuteryje, tad jūsų duomenys nėra perduodami jokiam serveriui. Dėl to, ši aplikacija veikia lėtai.

NAUJA:
- Kelių vartotojų palaikymas per automatinę prievadų paskirstymo sistemą (prievadai 8050-8100)
- Prievadų registravimas ir stebėjimas keliems vienu metu veikiantiems programų egzemplioriams
- Išsami asmeninio naudojimo statistika ir analitika
- Pažangi eigos stebėjimo sistema su kelių etapų apdorojimu
- Patobulinta klaidų stebėjimo ir kategorizuotų pranešimų sistema daugumai operacijų

ANKSČIAU: 
- Projektų Valdymo Sistema - Pilnas skirtingų projektų palaikymas su projektams specifinėmis dokumentų saugyklomis
- Google Gemini Integracija - Pilnas Gemini 2.5 Pro, Flash ir Flash-Lite modelių palaikymas
- Pokalbių Režimai - Signle-turn ir Multi-turn pokalbių palaikymas.
- Projektui Prisitaikantis Dokumentų Susiejimas - Spustelėjamos PDF nuorodos su projektui specifiniais URL
- Dinaminio HTML Turinio Atvaizdavimas - Pilnas paryškinto teksto ir formatuoto turinio palaikymas

Jei turite pastabų, galite jas pateikti adresu: konradas.m@8devices.com

GREITAI:
- Hyperlinks į konkrečius puslapius LLM atsakymuose naudotuose šaltiniuose
- Įkelti PDF dokumentus į saugią duomenų bazę, o ne į atmintį
- LLM atsakymų streaming
- Hibridinė dokumentų paieška su raktažodžiais
"""
startup_success = True

# Layout with enhanced progress bar
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("8devices RAG Chatbot (v.0.4.1)", className="text-center mb-2 mt-2"),
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
    
    # Hidden store for project data
    dcc.Store(id='project-data', data={'projects': [], 'current_project': None}),
    
    # Auto-refresh interval for log display
    dcc.Interval(id="log-interval", interval=1000, n_intervals=0),
    
    # Progress update interval (faster for smoother progress bar)
    dcc.Interval(id="progress-interval", interval=500, n_intervals=0),
    
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
                    html.Label("Bi-Encoder Model", className="mb-2"),
                    dbc.Select(
                        id='bi-encoder-model-select',
                        options=[
                            {'label': 'Snowflake Arctic Embed L v2.0', 'value': 'Snowflake/snowflake-arctic-embed-l-v2.0'},
                            {'label': 'BGE-M3', 'value': 'BAAI/bge-m3'}
                        ],
                        value=session_data['bi_encoder_config']['model_name'],
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
                        value=session_data['bi_encoder_config']['chunk_size'],
                        className="mb-3"
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Chunk Overlap", className="mb-2"),
                    dbc.Select(
                        id='chunk-overlap-select',
                        options=[
                            {'label': '0 tokens', 'value': 0},
                            {'label': '128 tokens', 'value': 128},
                            {'label': '256 tokens', 'value': 256}
                        ],
                        value=session_data['bi_encoder_config']['chunk_overlap'],
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Retrieval Count", className="mb-2"),
                    dbc.Select(
                        id='retrieval-count-select',
                        options=[
                            {'label': '50 documents', 'value': 50},
                            {'label': '100 documents', 'value': 100},
                            {'label': '200 documents', 'value': 200}
                        ],
                        value=session_data['bi_encoder_config']['retrieval_count'],
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
                            {'label': 'MxBai Rerank Base v2', 'value': 'mixedbread-ai/mxbai-rerank-base-v2'}
                        ],
                        value=session_data['cross_encoder_config']['model_name'],
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Reranking Count", className="mb-2"),
                    dbc.Select(
                        id='reranking-count-select',
                        options=[],  # Will be populated dynamically
                        value=session_data['cross_encoder_config']['top_n'],
                        className="mb-3"
                    )
                ], width=6)
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-rag-config-modal", color="secondary")
        ])
    ], id="rag-config-modal", size="lg", is_open=False),
    
], fluid=True)

# Enhanced callback for progress tracking
@app.callback(
    [Output('detailed-progress-bar', 'value'),
     Output('progress-status', 'children'),
     Output('current-file-status', 'children'),
     Output('progress-container', 'style')],
    [Input('progress-interval', 'n_intervals')]
)
def update_progress_display(n_intervals):
    """Update the progress bar display"""
    progress_info = progress_tracker.get_progress_info()
    
    # Show/hide progress container based on progress
    if progress_info['progress'] == 0:
        container_style = {'display': 'none'}
    else:
        container_style = {'display': 'block'}
    
    return (
        progress_info['progress'],
        progress_info['detailed_message'],
        f"Current: {progress_info['current_file']}" if progress_info['current_file'] else "",
        container_style
    )

# Callback for check processed PDFs progress tracking
@app.callback(
    [Output('check-processed-progress-bar', 'value'),
     Output('check-progress-status', 'children'),
     Output('check-current-status', 'children'),
     Output('check-progress-container', 'style')],
    [Input('progress-interval', 'n_intervals')]
)
def update_check_progress_display(n_intervals):
    """Update the check processed PDFs progress bar display"""
    progress_info = check_progress_tracker.get_progress_info()
    
    # Show/hide progress container based on activity
    if progress_info['is_active']:
        container_style = {'display': 'block'}
    else:
        container_style = {'display': 'none'}
    
    return (
        progress_info['progress'],
        progress_info['status'],
        progress_info['detail'],
        container_style
    )

# Basic callbacks for startup and project management


@app.callback(
    [Output('project-selector', 'options'),
     Output('project-data', 'data')],
    [Input('project-data', 'id')],
    prevent_initial_call=False  # Allow initial call to populate on startup
)
def initialize_projects(project_data_id):
    """Initialize project list on app startup"""
    if session_data['dir'] is None:
        session_data['dir'] = Path.cwd()
    
    # Get available projects
    projects = get_available_projects(session_data['dir'])
    session_data['projects'] = projects
    
    # Create options for dropdown
    options = [{'label': project, 'value': project} for project in projects]
    
    # Update project data
    project_data = {
        'projects': projects,
        'current_project': session_data['current_project']
    }
    
    return options, project_data

@app.callback(
    Output('project-selector', 'value'),
    [Input('project-data', 'data')]
)
def set_default_project(project_data):
    """Set default project selection"""
    if project_data['projects'] and not project_data['current_project']:
        return project_data['projects'][0] if project_data['projects'] else None
    return project_data['current_project']

@app.callback(
    [Output('project-status', 'children'),
     Output('new-project-input', 'value'),
     Output('project-selector', 'options', allow_duplicate=True),
     Output('project-selector', 'value', allow_duplicate=True)],
    [Input('create-project-button', 'n_clicks')],
    [State('new-project-input', 'value')],
    prevent_initial_call=True
)
def create_new_project(n_clicks, project_name):
    """Create a new project"""
    if n_clicks is None or not project_name:
        raise PreventUpdate
    
    try:
        if session_data['dir'] is None:
            session_data['dir'] = Path.cwd()
        
        # Check if project already exists
        existing_projects = get_available_projects(session_data['dir'])
        if project_name in existing_projects:
            return dbc.Alert(f"Project '{project_name}' already exists!", color="warning", dismissable=True), "", [{'label': project, 'value': project} for project in existing_projects], None
        
        # Create the project
        created_project = create_project(session_data['dir'], project_name)
        
        # Track project creation
        personal_stats.track_project_action('create')
        personal_stats.track_button_click('create_project')
        
        # Update project list
        updated_projects = get_available_projects(session_data['dir'])
        session_data['projects'] = updated_projects
        options = [{'label': project, 'value': project} for project in updated_projects]
        
        # Automatically select the newly created project
        return dbc.Alert(f"Project '{created_project}' created successfully!", color="success", dismissable=True), "", options, created_project
        
    except Exception as e:
        return dbc.Alert(f"Error creating project: {str(e)}", color="danger", dismissable=True), "", [{'label': project, 'value': project} for project in get_available_projects(session_data['dir'])], None

@app.callback(
    [Output('upload-pdf', 'disabled'),
     Output('check-processed-button', 'disabled'),
     Output('delete-project-button', 'disabled'),
     Output('project-data', 'data', allow_duplicate=True),
     Output('upload-status', 'children', allow_duplicate=True),
     Output('detailed-progress-bar', 'value', allow_duplicate=True),
     Output('progress-container', 'style', allow_duplicate=True)],
    [Input('project-selector', 'value')],
    prevent_initial_call=True
)
def select_project(selected_project):
    """Handle project selection"""
    if selected_project:
        session_data['current_project'] = selected_project
        
        # Track project switch
        personal_stats.track_project_action('switch')
        
        # Clear previous session data when switching projects
        session_data['embeddings'] = []
        session_data['docs_md'] = []
        session_data['doc_paths'] = []
        session_data['bi_encoder'] = None
        session_data['cross_encoder'] = None
        session_data['initialized'] = False
        
        # Reset progress trackers when switching projects
        progress_tracker.reset()
        check_progress_tracker.reset()
        
        print(f"Selected project: {selected_project}")
        
        # Update project data store
        project_data = {
            'projects': session_data['projects'],
            'current_project': selected_project
        }
        
        return False, False, False, project_data, "", 0, {'display': 'none'}  # Enable upload, check button, and delete button, clear status, hide progress bar
    else:
        session_data['current_project'] = None
        
        # Update project data store
        project_data = {
            'projects': session_data['projects'],
            'current_project': None
        }
        
        return True, True, True, project_data, "", 0, {'display': 'none'}  # Disable all buttons, clear status, hide progress bar

# Add a callback to clear project status when upload status changes (indicating activity)
@app.callback(
    Output('project-status', 'children', allow_duplicate=True),
    [Input('upload-status', 'children')],
    prevent_initial_call=True
)
def clear_project_status_on_activity(upload_status):
    """Clear project status message when there's upload activity"""
    if upload_status and upload_status != "":
        return ""  # Clear the project status message
    raise PreventUpdate

@app.callback(
    [Output('project-status', 'children', allow_duplicate=True),
     Output('project-selector', 'options', allow_duplicate=True),
     Output('project-selector', 'value', allow_duplicate=True)],
    [Input('delete-project-button', 'n_clicks')],
    [State('project-selector', 'value')],
    prevent_initial_call=True
)
def delete_selected_project(n_clicks, selected_project):
    """Delete the selected project"""
    if n_clicks is None or not selected_project:
        raise PreventUpdate
    
    try:
        if session_data['dir'] is None:
            session_data['dir'] = Path.cwd()
        
        # Delete the project
        delete_project(session_data['dir'], selected_project)
        
        # Track project deletion
        personal_stats.track_project_action('delete')
        personal_stats.track_button_click('delete_project')
        
        # Clear session data
        session_data['current_project'] = None
        session_data['embeddings'] = []
        session_data['docs_md'] = []
        session_data['doc_paths'] = []
        session_data['bi_encoder'] = None
        session_data['cross_encoder'] = None
        session_data['initialized'] = False
        
        # Update project list
        updated_projects = get_available_projects(session_data['dir'])
        session_data['projects'] = updated_projects
        options = [{'label': project, 'value': project} for project in updated_projects]
        
        return dbc.Alert(f"Project '{selected_project}' deleted successfully!", color="success", dismissable=True), options, None
        
    except Exception as e:
        return dbc.Alert(f"Error deleting project: {str(e)}", color="danger", dismissable=True), [{'label': project, 'value': project} for project in get_available_projects(session_data['dir'])], None

@app.callback(
    Output('log-display', 'children'),
    [Input('log-interval', 'n_intervals')]
)
def update_log_display(n_intervals):
    """Update log display with current log file contents"""
    try:
        with open(LOG_FILE, 'r') as f:
            log_content = f.read()
        return log_content
    except FileNotFoundError:
        return "No log file found."

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
    
    # Reset and start check progress tracking
    check_progress_tracker.reset()
    check_progress_tracker.set_step(1, "Checking for processed PDFs", "Initializing...")
    
    # Check if a project is selected
    current_project = project_data.get('current_project')
    if not current_project:
        check_progress_tracker.reset()
        return (
            dbc.Alert("Please select a project first.", color="warning", dismissable=True),
            'idle'
        )
    
    try:
        print(f"Checking for processed PDFs in project: {current_project}")
        check_progress_tracker.set_step(1, "Checking for processed PDFs", f"Project: {current_project}")
        
        # Track button click
        personal_stats.track_button_click('check_processed')
        
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
            check_progress_tracker.reset()
            return (
                dbc.Alert("No processed documents found. Please upload PDFs first.", color="warning", dismissable=True),
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
            check_progress_tracker.reset()
            return (
                dbc.Alert("No PDF files found. Please upload PDFs first.", color="warning", dismissable=True),
                'idle'
            )
        
        # Check if markdown files exist for all PDFs, and process missing ones
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
        
        # If there are missing markdown files, process them automatically
        if missing_md:
            print(f"Found PDFs without markdown files: {missing_md}. Processing them now...")
            
            # Initialize PDF processor if needed
            if session_data['pdf_processor'] is None:
                session_data['pdf_processor'] = PdfProcessor()
            
            # Process the missing PDFs
            for pdf_stem in missing_md:
                pdf_file = docs_dir / f"{pdf_stem}.pdf"
                if pdf_file.exists():
                    print(f"Processing {pdf_stem}...")
                    
                    try:
                        # Process the PDF
                        processed_text = session_data['pdf_processor'].process_document(
                            pdf_file, 
                            save_path=processed_dir
                        )
                        
                        # Add to processed docs
                        if processed_text:
                            processed_docs.extend(processed_text)
                            doc_ids.append(pdf_stem)
                            print(f"Successfully processed {pdf_stem}")
                        
                    except Exception as e:
                        print(f"Error processing {pdf_stem}: {str(e)}")
                        personal_stats.track_error('processing_errors')
                        continue
            
            print(f"Completed processing {len(missing_md)} PDFs")
        
        # Check if embeddings exist for current configuration
        embeddings_path = embeddings_dir / model_name / f"chunk_size_{chunk_size}" / f"chunk_overlap_{chunk_overlap}"
        
        if not embeddings_path.exists():
            print(f"No embeddings found for current configuration at {embeddings_path}")
            check_progress_tracker.reset()
            return (
                dbc.Alert("No embeddings found for current configuration. Please upload PDFs to process them.", color="warning", dismissable=True),
                'idle'
            )
        
        # Check if all documents have embeddings, and create missing ones
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
        
        # If there are missing embeddings, create them automatically
        if missing_embeddings:
            print(f"Found documents without embeddings: {missing_embeddings}. Creating embeddings now...")
            
            # Initialize models if needed
            if session_data['bi_encoder'] is None:
                print("Initializing bi-encoder...")
                session_data['bi_encoder'] = BiEncoderPipeline(
                    model_name=config['model_name'],
                    chunk_size=config['chunk_size'],
                    chunk_overlap=config['chunk_overlap']
                )
                print("Bi-encoder initialized")
            
            if session_data['cross_encoder'] is None:
                print("Initializing cross-encoder...")
                cross_encoder_config = session_data['cross_encoder_config']
                session_data['cross_encoder'] = CrossEncoderPipeline(
                    model_name=cross_encoder_config['model_name']
                )
                print("Cross-encoder initialized")
            
            # Prepare documents for embedding
            docs_to_embed = []
            doc_ids_to_embed = []
            
            for doc_id in missing_embeddings:
                md_file = processed_dir / f"{doc_id}.md"
                if md_file.exists():
                    with open(md_file, 'r') as f:
                        docs_to_embed.append(f.read())
                        doc_ids_to_embed.append(doc_id)
                        print(f"Prepared {doc_id} for embedding")
                else:
                    print(f"Warning: Markdown file not found for {doc_id}")
            
            # Create embeddings for missing documents
            if docs_to_embed:
                print(f"Creating embeddings for {len(docs_to_embed)} documents...")
                
                try:
                    # Create embeddings
                    new_embeddings = session_data['bi_encoder'].embed_documents(
                        docs_to_embed,
                        doc_ids=doc_ids_to_embed,
                        save_path=embeddings_dir,
                        track_progress=False
                    )
                    
                    print(f"Successfully created embeddings for {len(docs_to_embed)} documents")
                    
                except Exception as e:
                    print(f"Error creating embeddings: {str(e)}")
                    personal_stats.track_error('processing_errors')
                    check_progress_tracker.reset()
                    return (
                        dbc.Alert(f"Error creating embeddings: {str(e)}", color="danger", dismissable=True),
                        'error'
                    )
            
            print(f"Completed embedding creation for {len(missing_embeddings)} documents")
        
        print(f"Found processed documents: {doc_ids}")
        check_progress_tracker.set_step(2, "Found processed documents", f"Found {len(doc_ids)} documents")
        
        print("Loading documents and initializing models...")
        check_progress_tracker.set_step(3, "Loading documents and initializing models", "Preparing models...")
        
        # Initialize models if not already done
        if session_data['bi_encoder'] is None:
            print("Initializing bi-encoder...")
            check_progress_tracker.set_step(4, "Initializing bi-encoder", "Loading embedding model...")
            session_data['bi_encoder'] = BiEncoderPipeline(
                model_name=config['model_name'],
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap']
            )
        else:
            print("Bi-encoder model already loaded...")
            check_progress_tracker.set_step(4, "Bi-encoder ready", "Model already loaded")
        
        if session_data['cross_encoder'] is None:
            print("Initializing cross-encoder...")
            check_progress_tracker.set_step(5, "Initializing cross-encoder", "Loading reranking model...")
            cross_encoder_config = session_data['cross_encoder_config']
            session_data['cross_encoder'] = CrossEncoderPipeline(
                model_name=cross_encoder_config['model_name']
            )
        else:
            print("Cross-encoder model already loaded...")
            check_progress_tracker.set_step(5, "Cross-encoder ready", "Model already loaded")
        
        # Load embeddings
        print("Loading embeddings...")
        check_progress_tracker.set_step(6, "Loading embeddings", f"Loading embeddings for {len(doc_ids)} documents...")
        embeddings = session_data['bi_encoder'].embed_documents(
            processed_docs, 
            doc_ids=doc_ids,
            save_path=embeddings_dir,
            track_progress=False
        )
        session_data['embeddings'] = embeddings
        session_data['docs_md'] = processed_docs
        session_data['doc_paths'] = doc_ids
        session_data['initialized'] = True
        
        print("System ready for chatting!")
        check_progress_tracker.complete()
        
        # Create success message based on what happened
        message_parts = []
        
        if missing_md and missing_embeddings:
            # Both PDF processing and embedding creation happened
            pdf_processed = len([doc for doc in missing_md if doc in doc_ids])
            embeddings_created = len(missing_embeddings)
            existing_docs = len(doc_ids) - pdf_processed
            message_parts.append(f"Found {existing_docs} existing documents")
            message_parts.append(f"processed {pdf_processed} additional PDFs")
            message_parts.append(f"created embeddings for {embeddings_created} documents")
        elif missing_md:
            # Only PDF processing happened
            pdf_processed = len([doc for doc in missing_md if doc in doc_ids])
            existing_docs = len(doc_ids) - pdf_processed
            message_parts.append(f"Found {existing_docs} existing documents")
            message_parts.append(f"processed {pdf_processed} additional PDFs")
        elif missing_embeddings:
            # Only embedding creation happened
            embeddings_created = len(missing_embeddings)
            existing_docs = len(doc_ids) - embeddings_created
            message_parts.append(f"Found {existing_docs} existing documents")
            message_parts.append(f"created embeddings for {embeddings_created} documents")
        else:
            # No processing needed - just show progress briefly and complete
            message_parts.append(f"Found and loaded {len(doc_ids)} processed documents")
        
        success_msg = ", ".join(message_parts) + f". Loaded {len(doc_ids)} total documents. Ready to chat!"
        
        return (
            dbc.Alert(success_msg, color="success", dismissable=True),
            'ready'
        )
        
    except Exception as e:
        error_msg = f"Error checking processed PDFs: {str(e)}"
        print(error_msg)
        check_progress_tracker.reset()  # Reset progress on error
        return (
            dbc.Alert(error_msg, color="danger", dismissable=True),
            'error'
        )

# Enhanced file upload callback with detailed progress tracking
@app.callback(
    [Output('upload-status', 'children'),
     Output('processing-stage', 'children')],
    [Input('upload-pdf', 'contents')],
    [State('upload-pdf', 'filename'),
     State('project-data', 'data')]
)
def handle_enhanced_file_upload(contents, filenames, project_data):
    """Enhanced file upload handler with detailed progress tracking"""
    if contents is None:
        raise PreventUpdate
    
    # Reset progress tracker
    progress_tracker.reset()
    
    # Check if a project is selected
    current_project = project_data.get('current_project')
    if not current_project:
        return (
            dbc.Alert("Please select or create a project first.", color="warning", dismissable=True),
            'error'
        )
    
    try:
        # Stage 1: Upload files
        progress_tracker.set_stage("upload", len(contents))
        
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
        for i, (content, filename) in enumerate(zip(contents, filenames)):
            if filename.endswith('.pdf'):
                progress_tracker.update_stage_progress(i + 1, filename)
                
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                
                file_path = docs_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(decoded)
                uploaded_files.append(filename)
        
        if not uploaded_files:
            progress_tracker.log_message("No PDF files uploaded")
            return (
                dbc.Alert("Please upload PDF files only.", color="warning", dismissable=True),
                'error'
            )
        
        # Stage 2: Process PDFs
        pdf_files = list(docs_dir.glob("*.pdf"))
        progress_tracker.set_stage("pdf_processing", len(pdf_files))
        
        # Initialize PDF processor
        if session_data['pdf_processor'] is None:
            session_data['pdf_processor'] = PdfProcessor()
        
        # Process each PDF
        for i, pdf_file in enumerate(pdf_files):
            if not (processed_dir / f"{pdf_file.stem}.md").exists():
                progress_tracker.increment_stage_progress(pdf_file.stem)
                
                processed_docs = session_data['pdf_processor'].process_document(
                    pdf_file, 
                    save_path=processed_dir
                )
                session_data['docs_md'].extend(processed_docs)
            else:
                progress_tracker.increment_stage_progress(f"{pdf_file.stem} (skipped)")
        
        # Stage 3: Initialize embedder
        progress_tracker.set_stage("embedding_init", 1)
        
        if session_data['bi_encoder'] is None:
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
            
            session_data['initialized'] = True
        
        progress_tracker.update_stage_progress(1, "Models initialized")
        
        # Stage 4: Create embeddings
        processed_docs = []
        doc_ids = []
        for md_file in processed_dir.glob("*.md"):
            with open(md_file, 'r') as f:
                processed_docs.append(f.read())
            doc_ids.append(md_file.stem)
        
        progress_tracker.set_stage("embedding_creation", len(processed_docs))
        
        if processed_docs:
            # Use progress tracking in embedding
            embeddings = session_data['bi_encoder'].embed_documents(
                processed_docs, 
                doc_ids=doc_ids,
                save_path=embeddings_dir
            )
            session_data['embeddings'] = embeddings
        
        # Store doc paths for display
        session_data['doc_paths'] = [f.stem for f in pdf_files]
        
        # Complete processing
        progress_tracker.complete()
        
        # Track the upload statistics
        processing_time = time.time() - progress_tracker.session_start_time
        personal_stats.track_document_upload(len(uploaded_files), processing_time)
        personal_stats.track_button_click('upload')
        
        return (
            dbc.Alert("Processing complete! Ready to chat.", color="success", dismissable=True),
            'ready'
        )
            
    except Exception as e:
        progress_tracker.log_message(f"Error occurred: {str(e)}")
        personal_stats.track_error('upload_errors')
        return (
            dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True),
            'error'
        )

# Complete chat callback with full LLM integration
@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-messages', 'children'),
     Output('user-input', 'value')],
    [Input('send-button', 'n_clicks'),
     Input('user-input', 'n_submit'),
     Input('clear-chat-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('chat-messages', 'children'),
     State('processing-stage', 'children'),
     State('project-data', 'data'),
     State('rag-mode-checkbox', 'value'),
     State('prompt-type-select', 'value'),
     State('conversation-mode-select', 'value')]
)
def handle_chat(n_clicks, n_submit, clear_clicks, user_input, messages_json, processing_state, project_data, rag_mode, prompt_type, conversation_mode):
    """Complete chat handler with full LLM integration and RAG functionality"""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle clear chat
    if button_id == 'clear-chat-button':
        personal_stats.track_button_click('clear_chat')
        session_data['conversation_history'] = []
        return [], "[]", ""
    
    # Handle send message
    if button_id in ['send-button', 'user-input'] and user_input:
        print(f"Processing state: {processing_state}")
        
        # Update session data with current UI state
        session_data['rag_mode'] = rag_mode
        session_data['prompt_type'] = prompt_type
        session_data['conversation_mode'] = conversation_mode
        
        # Track the query
        personal_stats.track_query(user_input, rag_mode)
        personal_stats.track_prompt_type(prompt_type)
        personal_stats.track_conversation_mode(conversation_mode)
        personal_stats.track_button_click('send')
        
        # Check if we're in RAG mode and need a project
        if session_data['rag_mode']:
            current_project = project_data.get('current_project')
            if not current_project:
                # Return error message
                messages = json.loads(messages_json) if messages_json else []
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
            messages = json.loads(messages_json) if messages_json else []
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
            messages = json.loads(messages_json) if messages_json else []
            
            # Add user message
            messages.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # Track model usage for every query
            model_name = session_data['llm_config']['model_name']
            personal_stats.track_model_usage(model_name)
            
            if session_data['llm'] is None:
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
                
                # Track document references
                personal_stats.track_document_references(context_ids)
                personal_stats.total_stats['documents_retrieved_total'] += len(context_ids)
                
                # Get the selected prompt type and corresponding instructions
                prompt_type = session_data.get('prompt_type', 'strict')
                conversation_mode = session_data.get('conversation_mode', 'single')
                print(f"Using prompt type: {prompt_type}, conversation mode: {conversation_mode}")
                
                # Get the selected prompt instructions
                selected_instructions = PROMPT_INSTRUCTIONS.get(prompt_type, PROMPT_INSTRUCTIONS['strict'])
                
                # Build conversation history if in multi-turn mode
                conversation_history = ""
                if conversation_mode == 'multi':
                    conversation_history = build_conversation_history(json.dumps(messages[:-1]))  # Exclude current message
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
                
                # Start timing response generation
                response_start_time = time.time()
                
                # Handle different LLM types
                if session_data['llm'] == 'openai':
                    model_name = session_data['llm_config']['model_name']
                    client = session_data['llm_client']
                    
                    # Use proper conversation history format for OpenAI in multi-turn mode
                    if conversation_mode == 'multi':
                        # Build messages with conversation history
                        openai_messages = build_openai_conversation_history(json.dumps(messages[:-1]))
                        
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
                        openai_messages.insert(0, {"role": "system", "content": system_message})
                        
                        # Add current user query
                        openai_messages.append({"role": "user", "content": user_input})
                        
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=openai_messages,
                            max_tokens=1024,
                            n=1
                        )
                    else:
                        # Single-turn mode - use the current format
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": rag_prompt}],
                            max_tokens=1024,
                            n=1
                        )
                    response_content = response.choices[0].message.content.strip()
                    
                elif session_data['llm'] == 'gemini':
                    client = session_data['llm_client']
                    response = client.generate_content(rag_prompt)
                    response_content = response.text.strip()
                    
                else:
                    response = session_data['llm'].invoke(rag_prompt)
                    response_content = response.content.split("</think>")[1] if "</think>" in response.content else response.content
                
                # Add project-specific PDF links
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
                
                # Calculate response time
                response_time = time.time() - response_start_time
                
                # Track the response
                personal_stats.track_response(response_content, response_time, success=True)
                
            else:
                # Non-RAG mode: Direct chat without document context
                conversation_mode = session_data.get('conversation_mode', 'single')
                print(f"Generating direct response in {conversation_mode} mode...")
                
                # Build conversation history if in multi-turn mode
                conversation_history = ""
                if conversation_mode == 'multi':
                    conversation_history = build_conversation_history(json.dumps(messages[:-1]))
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
                
                # Start timing response generation
                response_start_time = time.time()
                
                # Handle different LLM types
                if session_data['llm'] == 'openai':
                    model_name = session_data['llm_config']['model_name']
                    client = session_data['llm_client']
                    
                    # Use proper conversation history format for OpenAI in multi-turn mode
                    if conversation_mode == 'multi':
                        # Build messages with conversation history
                        openai_messages = build_openai_conversation_history(json.dumps(messages[:-1]))
                        
                        # Create system message with instructions
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
                            max_tokens=1024,
                            n=1
                        )
                    else:
                        # Single-turn mode - use the current format
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": chat_prompt}],
                            max_tokens=1024,
                            n=1
                        )
                    response_content = response.choices[0].message.content.strip()
                    
                elif session_data['llm'] == 'gemini':
                    client = session_data['llm_client']
                    # Gemini uses the text-based conversation history format
                    response = client.generate_content(chat_prompt)
                    response_content = response.text.strip()
                    
                else:
                    response = session_data['llm'].invoke(chat_prompt)
                    response_content = response.content.split("</think>")[1] if "</think>" in response.content else response.content
                
                # Calculate response time
                response_time = time.time() - response_start_time
                
                # Track the response
                personal_stats.track_response(response_content, response_time, success=True)
            
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
            
            # Track the error
            personal_stats.track_error('query_errors')
            personal_stats.track_response(error_msg, 0, success=False)
            
            messages = json.loads(messages_json) if messages_json else []
            messages.append({
                'role': 'assistant',
                'content': error_msg,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            return [dbc.Alert(error_msg, color="danger", dismissable=True)], json.dumps(messages), ""
    
    raise PreventUpdate

# Configuration update callbacks
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
     Output('check-processed-button', 'disabled', allow_duplicate=True),
     Output('prompt-type-select', 'disabled'),
     Output('conversation-mode-select', 'disabled'),
     Output('clear-chat-button', 'disabled'),
     Output('llm-model-input', 'disabled'),
     Output('openai-api-key-input', 'disabled')],
    [Input('rag-mode-checkbox', 'value'),
     Input('project-data', 'data')],
    prevent_initial_call=True
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
    
    explainer_text = explainers.get(prompt_type, explainers['strict'])
    
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

# Port Management System
class PortManager:
    """Manages port allocation and tracking for multiple app instances"""
    
    def __init__(self, log_file="port_log.txt", start_port=8050, max_port=8100):
        self.log_file = log_file
        self.start_port = start_port
        self.max_port = max_port
        self.current_port = None
        self.pid = os.getpid()
        
    def is_port_available(self, port):
        """Check if a port is available for use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result != 0
        except:
            return False
    
    def read_port_log(self):
        """Read current port allocations from log file"""
        if not os.path.exists(self.log_file):
            return {}
        
        active_ports = {}
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        try:
                            pid, port, timestamp = line.split(':')
                            # Check if process is still running
                            if self.is_process_running(int(pid)):
                                active_ports[int(port)] = {'pid': int(pid), 'timestamp': timestamp}
                        except ValueError:
                            continue
        except:
            pass
        
        return active_ports
    
    def is_process_running(self, pid):
        """Check if a process is still running"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def find_available_port(self):
        """Find the next available port"""
        active_ports = self.read_port_log()
        
        for port in range(self.start_port, self.max_port + 1):
            if port not in active_ports and self.is_port_available(port):
                return port
        
        raise RuntimeError(f"No available ports found in range {self.start_port}-{self.max_port}")
    
    def register_port(self, port):
        """Register current port usage in log file"""
        self.current_port = port
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clean up stale entries first
        self.cleanup_stale_entries()
        
        # Add current entry
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{self.pid}:{port}:{timestamp}\n")
        except:
            pass
        
        # Register cleanup on exit
        atexit.register(self.cleanup_on_exit)
    
    def cleanup_stale_entries(self):
        """Remove entries for processes that are no longer running"""
        if not os.path.exists(self.log_file):
            return
        
        active_entries = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        try:
                            pid, port, timestamp = line.split(':')
                            if self.is_process_running(int(pid)):
                                active_entries.append(line)
                        except ValueError:
                            continue
            
            # Rewrite file with only active entries
            with open(self.log_file, 'w') as f:
                for entry in active_entries:
                    f.write(entry + '\n')
        except:
            pass
    
    def cleanup_on_exit(self):
        """Clean up current port entry when app exits"""
        if not self.current_port:
            return
        
        try:
            active_entries = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        try:
                            pid, port, timestamp = line.split(':')
                            if int(pid) != self.pid:
                                active_entries.append(line)
                        except ValueError:
                            active_entries.append(line)
            
            with open(self.log_file, 'w') as f:
                for entry in active_entries:
                    f.write(entry + '\n')
        except:
            pass
    
    def get_allocated_port(self):
        """Get an available port and register it"""
        port = self.find_available_port()
        self.register_port(port)
        return port

def start_app():
    """Start the app with automatic port allocation"""
    # Configure logging to reduce Werkzeug verbosity
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    port_manager = PortManager()
    
    try:
        port = port_manager.get_allocated_port()
        print(f"Starting RAG Chatbot on port {port}")
        print(f"Open your browser to: http://localhost:{port}")
        
        # Write port info to log for user reference
        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] App started on port {port} (PID: {os.getpid()})"
        print(log_message)
        
        # Also log to app_log.txt if it exists
        try:
            with open("app_log.txt", "a") as f:
                f.write(log_message + "\n")
        except:
            pass
        
        # Start the app with debug=False to prevent automatic reloading
        app.run(debug=False, port=port, host='localhost', use_reloader=False)
        
    except Exception as e:
        print(f"Error starting app: {e}")
        print("Please check if there are available ports in the range 8050-8100")
        sys.exit(1)

if __name__ == '__main__':
    start_app()