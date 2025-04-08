# File: app/callbacks.py
# Contains the callback to update the content based on the selected tab.

from dash.dependencies import Input, Output, State
from dash import callback_context, html, dcc
from app import app
from app.layout import content_layout
import pandas as pd
import polars as pl
from polars import col, lit
import dash_bootstrap_components as dbc
from app.utils.db_utils import get_gene_data_with_metadata, duck_conn, POLARS_AVAILABLE
from app.utils.polars_utils import order_transcripts_by_expression
from app.utils.plotly_utils import get_n_colors
import RNApysoforms as RNApy
from dash import ClientsideFunction
import plotly.graph_objects as go

# Callback to update the active tab when a nav link is clicked
@app.callback(
    Output("active-tab", "data"),
    [
        Input("nav-1", "n_clicks"),
        Input("nav-2", "n_clicks"),
        Input("nav-3", "n_clicks"),
        Input("nav-4", "n_clicks"),
        Input("nav-5", "n_clicks"),
        Input("nav-6", "n_clicks")
    ],
    [State("active-tab", "data")]
)
def update_active_tab(n1, n2, n3, n4, n5, n6, current_tab):
    # Get the ID of the clicked nav item
    ctx = callback_context
    if not ctx.triggered:
        return current_tab
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Map nav ID to tab ID
    tab_mapping = {
        "nav-1": "tab-1",
        "nav-2": "tab-2",
        "nav-3": "tab-3",
        "nav-4": "tab-4",
        "nav-5": "tab-5",
        "nav-6": "tab-6"
    }
    
    return tab_mapping.get(button_id, current_tab)

# Callback to update nav link active states
@app.callback(
    [
        Output("nav-1", "active"),
        Output("nav-2", "active"),
        Output("nav-3", "active"),
        Output("nav-4", "active"),
        Output("nav-5", "active"),
        Output("nav-6", "active")
    ],
    [Input("active-tab", "data")]
)
def update_nav_active(active_tab):
    active_states = {
        "tab-1": [True, False, False, False, False, False],
        "tab-2": [False, True, False, False, False, False],
        "tab-3": [False, False, True, False, False, False],
        "tab-4": [False, False, False, True, False, False],
        "tab-5": [False, False, False, False, True, False],
        "tab-6": [False, False, False, False, False, True]
    }
    
    return active_states.get(active_tab, [False, False, False, False, False, False])

# Callback to update the content based on the active tab
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def render_content(tab):
    # Return the layout corresponding to the selected tab, or a default message if not found.
    return content_layout.get(tab, "Tab not found")
      
# Register a clientside callback to track window dimensions
# This callback uses JavaScript to measure the browser window size
# and stores the dimensions in the "window-dimensions" data store
# The callback is triggered by the interval component defined in layout.py
app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',  # This must match the window.clientside object
        function_name='updateWindowDimensions'  # This must match the function name
    ),
    Output("window-dimensions", "data"),
    Input("interval", "n_intervals"),
    prevent_initial_call=True
)
