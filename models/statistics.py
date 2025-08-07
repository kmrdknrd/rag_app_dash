# Personal usage statistics tracking system
import time
from datetime import datetime

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