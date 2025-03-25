# Defines the overall layout of the app including tabs and a content area.

# File: app/layout.py
# Defines the overall layout of the app including tabs and a content area.

from dash import html, dcc
import dash_bootstrap_components as dbc
from app.tabs import tab1, tab2, tab3, tab4, tab5, tab6

# Main layout with a header, a tab component, and a container for tab content
layout = dbc.Container([
    html.H1("My Simple Dash App", className="mt-4 mb-4"),
    dbc.Card([
        dbc.CardBody([
            dbc.Tabs(
                id="tabs",
                active_tab="tab-1",  # default selected tab
                children=[
                    dbc.Tab(label="Tab 1", tab_id="tab-1"),
                    dbc.Tab(label="Tab 2", tab_id="tab-2"),
                    dbc.Tab(label="Tab 3", tab_id="tab-3"),
                    dbc.Tab(label="Tab 4", tab_id="tab-4"),
                    dbc.Tab(label="Tab 5", tab_id="tab-5"),
                    dbc.Tab(label="Tab 6", tab_id="tab-6")
                ],
            ),
            html.Div(id="content", className="mt-3")  # Content area that will be updated based on the selected tab
        ])
    ], className="mb-4")
], fluid=True, className="dbc")

# Mapping of tab values to their corresponding layouts
content_layout = {
    "tab-1": tab1.layout,
    "tab-2": tab2.layout,
    "tab-3": tab3.layout,
    "tab-4": tab4.layout,
    "tab-5": tab5.layout,
    "tab-6": tab6.layout
}