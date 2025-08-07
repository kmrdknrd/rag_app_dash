# Progress tracking callbacks for the RAG application
from dash import Input, Output, callback_context
from dash.exceptions import PreventUpdate

def register_progress_callbacks(app, progress_tracker, check_progress_tracker, query_progress_tracker):
    """Register all progress tracking callbacks"""
    
    @app.callback(
        [Output('detailed-progress-bar', 'value'),
         Output('progress-status', 'children'),
         Output('current-file-status', 'children'),
         Output('progress-container', 'style')],
        [Input('progress-interval', 'n_intervals')]
    )
    def update_progress_display(n_intervals):
        """Update the main progress bar display"""
        progress_info = progress_tracker.get_progress_info()
        
        # Only show progress container when processing is active
        container_style = {"display": "block"} if progress_info.get('is_active', False) or (progress_info['progress'] > 0 and progress_info['progress'] < 100) else {"display": "none"}
        
        return (
            progress_info['progress'],
            progress_info.get('detailed_message', 'Ready'),
            progress_info.get('current_file', ''),
            container_style
        )
    
    @app.callback(
        [Output('check-processed-progress-bar', 'value'),
         Output('check-progress-status', 'children'),
         Output('check-current-status', 'children'),
         Output('check-progress-container', 'style')],
        [Input('progress-interval', 'n_intervals')]
    )
    def update_check_progress_display(n_intervals):
        """Update the check processed PDFs progress bar"""
        progress_info = check_progress_tracker.get_progress_info()
        
        # Only show container when checking is active
        container_style = {"display": "block"} if progress_info.get('is_active', False) else {"display": "none"}
        
        return (
            progress_info.get('progress', 0),
            progress_info.get('status', 'Ready to check'),
            progress_info.get('detail', ''),
            container_style
        )
    
    @app.callback(
        [Output('query-progress-bar', 'value'),
         Output('query-progress-status', 'children'),
         Output('query-progress-detail', 'children'),
         Output('query-progress-container', 'style')],
        [Input('progress-interval', 'n_intervals')]
    )
    def update_query_progress_display(n_intervals):
        """Update the query processing progress bar"""
        progress_info = query_progress_tracker.get_progress_info()
        
        # Show container when query processing is active
        container_style = {"display": "block" if progress_info.get('is_active', False) else "none"}
        
        return (
            progress_info.get('progress', 0),
            progress_info.get('stage', 'Ready'),
            progress_info.get('detail', ''),
            container_style
        )
    
    @app.callback(
        Output('log-display', 'children'),
        [Input('log-interval', 'n_intervals')]
    )
    def update_log_display(n_intervals):
        """Update log display from file"""
        try:
            with open('app_log.txt', 'r') as f:
                return f.read()
        except:
            return "No log file available"