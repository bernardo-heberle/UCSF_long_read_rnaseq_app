# File: app/tabs/tab5.py
# Contains the layout for Tab 5.

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("This is Tab 5", className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.P("Welcome to Tab 5 of this simple Dash app!")
        ])
    ])
]) 