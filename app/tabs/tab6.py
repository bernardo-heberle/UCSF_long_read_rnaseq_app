# File: app/tabs/tab6.py
# Contains the layout for Tab 6.

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("This is Tab 6", className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.P("Welcome to Tab 6 of this simple Dash app!")
        ])
    ])
]) 