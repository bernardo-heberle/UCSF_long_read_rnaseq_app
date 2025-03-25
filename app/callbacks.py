# File: app/callbacks.py
# Contains the callback to update the content based on the selected tab.

from dash.dependencies import Input, Output
from app import app
from app.layout import content_layout

@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def render_content(tab):
    # Return the layout corresponding to the selected tab, or a default message if not found.
    return content_layout.get(tab, "Tab not found")