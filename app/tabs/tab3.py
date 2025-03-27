# File: app/tabs/tab3.py
# Contains the layout for Tab 3.

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("This is Tab 3", 
        className="mt-3 dbc",
        style={
            "color": "#333333",
            "font-weight": "300",
            "letter-spacing": "0.5px"
        }
    ),
    dbc.Card([
        dbc.CardBody([
            html.P("Welcome to Tab 3 of this simple Dash app!",
                style={
                    "color": "#666666",
                    "font-size": "1.1rem"
                }
            )
        ])
    ],
    style={
        "background-color": "#ffffff",
        "border": "1px solid rgba(0, 0, 0, 0.1)",
        "border-radius": "6px",
        "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
    })
], 
fluid=True,  # Makes the container full-width
style={
    "max-width": "98%",  # Use 98% of the viewport width
    "margin": "0 auto",  # Center the container
    "padding": "10px"    # Add some padding
}) 