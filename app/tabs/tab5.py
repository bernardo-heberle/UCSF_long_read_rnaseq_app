# File: app/tabs/tab5.py
# Contains the layout for Tab 5.

from dash import html, Input, Output, callback
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        html.H2("This is Tab 5", 
            id="tab5-header",
            className="mt-3 dbc",
            style={
                "color": "#333333",
                "font-weight": "300",
                "letter-spacing": "0.5px"
            }
        ),
        dbc.Card([
            dbc.CardBody([
                html.P("Welcome to Tab 5 of this simple Dash app!",
                    id="tab5-content",
                    style={
                        "color": "#666666",
                        "font-size": "1.1rem"
                    }
                )
            ], id="tab5-card-body")
        ],
        id="tab5-card",
        style={
            "background-color": "#ffffff",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "6px",
            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
        })
    ], 
    fluid=True,  # Makes the container full-width
    id="tab5-container",
    style={
        "max-width": "98%",  # Use 98% of the viewport width
        "margin": "0 auto",  # Center the container
        "padding": "10px"    # Add some padding
    })

@callback(
    [Output("tab5-container", "style"),
     Output("tab5-header", "style"),
     Output("tab5-content", "style"),
     Output("tab5-card", "style")],
    [Input("window-dimensions", "data")]
)
def update_tab5_responsiveness(dimensions):
    if not dimensions:
        # Default styles if no dimensions available
        return (
            {"max-width": "98%", "margin": "0 auto", "padding": "10px"},
            {"color": "#333333", "font-weight": "300", "letter-spacing": "0.5px"},
            {"color": "#666666", "font-size": "1.1rem"},
            {"background-color": "#ffffff", "border": "1px solid rgba(0, 0, 0, 0.1)", 
             "border-radius": "6px", "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"}
        )
    
    width = dimensions.get('width', 1200)
    
    # Base styles
    container_style = {"max-width": "98%", "margin": "0 auto", "padding": "10px"}
    header_style = {
        "color": "#333333", 
        "font-weight": "300", 
        "letter-spacing": "0.5px"
    }
    content_style = {"color": "#666666", "font-size": "1.1rem"}
    card_style = {
        "background-color": "#ffffff", 
        "border": "1px solid rgba(0, 0, 0, 0.1)",
        "border-radius": "6px", 
        "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
    }
    
    # Responsive adjustments based on width
    if width < 576:  # Extra small devices
        container_style.update({"padding": "5px", "max-width": "100%"})
        header_style.update({"font-size": "1.4rem"})
        content_style.update({"font-size": "0.9rem"})
        card_style.update({"border-radius": "4px"})
    elif width < 768:  # Small devices
        container_style.update({"padding": "8px"})
        header_style.update({"font-size": "1.6rem"})
        content_style.update({"font-size": "1rem"})
    elif width < 992:  # Medium devices
        container_style.update({"padding": "10px"})
        header_style.update({"font-size": "1.8rem"})
        content_style.update({"font-size": "1.1rem"})
    else:  # Large devices
        container_style.update({"padding": "15px"})
        header_style.update({"font-size": "2rem"})
        content_style.update({"font-size": "1.2rem"})
        
    return container_style, header_style, content_style, card_style 