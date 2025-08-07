# Project management callbacks for the RAG application
import shutil
from pathlib import Path
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

def register_project_callbacks(app, session_data, personal_stats, progress_tracker, check_progress_tracker):
    """Register all project management callbacks"""
    
    from utils.project_management import get_available_projects, create_project, delete_project, get_project_directories
    from config.session_state import reset_models
    
    # Removed initialize_projects callback to fix race condition
    # Removed set_default_project callback to fix race condition
    
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
        prevent_initial_call='initial_duplicate'
    )
    def select_project(selected_project):
        """Handle project selection"""
        if selected_project:
            session_data['current_project'] = selected_project
            
            print("PROJECT SELECTED")
            
            # Track project switch
            personal_stats.track_project_action('switch')
            
            # Clear previous session data when switching projects
            session_data['embeddings'] = []
            session_data['doc_paths'] = []
            reset_models()
            session_data['initialized'] = False
            
            # Reset progress trackers when switching projects
            progress_tracker.reset()
            check_progress_tracker.reset()
            
            # Update project data store
            project_data = {
                'projects': session_data['projects'],
                'current_project': selected_project
            }
            
            return False, False, False, project_data, "", 0, {'display': 'none'}  # Enable upload, check button, and delete button, clear status, hide progress bar
        else:
            session_data['current_project'] = None
            
            print("NO PROJECT SELECTED")
            
            # Update project data store
            project_data = {
                'projects': session_data['projects'],
                'current_project': None
            }
            
            return True, True, True, project_data, "", 0, {'display': 'none'}  # Disable all buttons, clear status, hide progress bar

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
            session_data['doc_paths'] = []
            reset_models()
            session_data['initialized'] = False
            
            # Update project list
            updated_projects = get_available_projects(session_data['dir'])
            session_data['projects'] = updated_projects
            options = [{'label': project, 'value': project} for project in updated_projects]
            
            return dbc.Alert(f"Project '{selected_project}' deleted successfully!", color="success", dismissable=True), options, None
            
        except Exception as e:
            return dbc.Alert(f"Error deleting project: {str(e)}", color="danger", dismissable=True), [{'label': project, 'value': project} for project in get_available_projects(session_data['dir'])], None