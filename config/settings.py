# Application settings and constants
import dash_bootstrap_components as dbc
from dash import html

# Startup message
STARTUP_MESSAGE = """Naudojatės 8devices RAG Chatbot (v0.5.2).

NAUJA:
- Hyperlinks į LLM atsakymuose naudotų šaltinių konkrečius puslapius
- Hibridinė dokumentų paieška su raktažodžiais (BM25)
- UI atnaujinimas
- LLM atsakymų dalinimosi sistema - kiekvienas atsakymas turi po dalinimosi mygtuką, kurį paspaudus atsiveria langas, kur vartotojas gali pateikti atsiliepimą apie atsakymą

ANKSČIAU: 
- Išsami asmeninio naudojimo statistika ir analitika
- Projektų Valdymo Sistema - Pilnas skirtingų projektų palaikymas su projektams specifinėmis dokumentų saugyklomis
- Kelių vartotojų palaikymas per automatinę prievadų paskirstymo sistemą (prievadai 8050-8100)
- Prievadų registravimas ir stebėjimas keliems vienu metu veikiantiems programų egzemplioriams
- Pokalbių Režimai - Signle-turn ir Multi-turn pokalbių palaikymas.

Jei turite pastabų, galite jas pateikti adresu: konradas.m@8devices.com

GREITAI:
- LLM atsakymų streaming
- Test režimas
- Citation highlighting šaltiniuose
"""

STARTUP_SUCCESS = True

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