# Reusable UI components for the RAG application
import dash_bootstrap_components as dbc
from dash import html

def create_error_alert(error_message, color="danger"):
    """Create a standardized error alert component"""
    return dbc.Alert(
        error_message,
        color=color,
        dismissable=True,
        className="mt-2"
    )

def create_progress_card(title, progress_id, status_id, additional_id=None, color="success", height="25px"):
    """Create a standardized progress card component"""
    card_body = [
        html.H6(title, className="card-title"),
        dbc.Progress(
            id=progress_id,
            value=0,
            striped=True,
            animated=True,
            color=color,
            style={"height": height, "margin-bottom": "10px"}
        ),
        html.Div(id=status_id, children=f"Ready for {title.lower()}", 
                style={"font-size": "14px", "color": "#666"}),
    ]
    
    if additional_id:
        card_body.append(
            html.Div(id=additional_id, children="", 
                    style={"font-size": "12px", "color": "#888", "margin-top": "5px"})
        )
    
    return dbc.Card([dbc.CardBody(card_body)])