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
    Input("interval", "n_intervals")
)
