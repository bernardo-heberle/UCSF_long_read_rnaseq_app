# File: app/tabs/tab3.py
# Contains the layout for Tab 3.

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("This is Tab 3", className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.P("Welcome to Tab 3 of this simple Dash app!")
        ])
    ])
]) 