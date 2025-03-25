# File: app/tabs/tab4.py
# Contains the layout for Tab 4.

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("This is Tab 4", className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.P("Welcome to Tab 4 of this simple Dash app!")
        ])
    ])
]) 