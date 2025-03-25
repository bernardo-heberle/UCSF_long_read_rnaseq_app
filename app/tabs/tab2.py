# File: app/tabs/tab2.py
# Contains the layout for Tab 2.

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("This is Tab 2", className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.P("Welcome to Tab 2 of this simple Dash app!")
        ])
    ])
]) 