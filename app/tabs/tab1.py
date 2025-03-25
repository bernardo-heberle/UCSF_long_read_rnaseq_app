# File: app/tabs/tab1.py
# Contains the layout for Tab 1.

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("This is Tab 1", className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.P("Welcome to Tab 1 of this simple Dash app!")
        ])
    ])
])
