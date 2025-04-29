# Defines the overall layout of the app including tabs and a content area.

# File: app/layout.py
# Defines the overall layout of the app including tabs and a content area.

from dash import html, dcc
import dash_bootstrap_components as dbc
from app.tabs import tab0, tab1, tab2, tab3, tab4, tab5

# Define some custom colors that complement COSMO theme
COLORS = {
    'text-primary': '#333333',        # Dark gray for main text
    'text-secondary': '#666666',      # Medium gray for secondary text
    'accent': '#2780E3',              # COSMO theme's primary blue
    'bg-card': '#fcfcfc',             # Very soft white background for cards
    'border': 'rgba(0, 0, 0, 0.1)',   # Light gray border
    'bg-primary': '#f8f8f8',          # Very light off-white background
    'bg-secondary': '#f5f5f5'         # Slightly darker off-white background
}

# Main layout with a header, a tab component, and a content area
layout = dbc.Container([
    # Window dimensions store - initialize with defaults
    dcc.Store(
        id='window-dimensions',
        storage_type='memory',
        data={'width': 1200, 'height': 800}
    ),

    # Active tab store - initialize with home tab
    dcc.Store(
        id='active-tab',
        data='tab-0',
        storage_type='memory'
    ),

    # Interval component to trigger the clientside callback (e.g. every second)
    dcc.Interval(id='interval', interval=500, n_intervals=0),    

    # Load JS code after page load
    html.Div(id='display-dimensions'),
    
    # Minimal, clean header with enhanced contrast and responsive font size
    html.Div([
        html.H1("Long-Read RNAseq Atlas of Aged Human Brain", 
            className="mt-5 mb-2 dbc", 
            style={
                "font-weight": "400",
                "letter-spacing": "1px",
                "color": COLORS['text-primary'],
                "border-bottom": f"3px solid {COLORS['accent']}",
                "padding-bottom": "0.5rem",
                "display": "inline-block",
                "font-size": "calc(1.5rem + 1vw)"  # Responsive font size
            }
        ),
        html.Div("Please cite us: XXXX", 
            className="mb-4",
            style={
                "font-size": "1.5rem",
                "font-style": "italic",
                "color": COLORS['text-secondary'],
                "margin-top": "-8px",
                "letter-spacing": "0.5px"
            }
        )
    ], className="d-inline-block"),
    
    # Tabs with enhanced styling and responsive font size
    dbc.Tabs(
        id="tabs",
        active_tab="tab-0",
        children=[
            dbc.Tab(label="Home", tab_id="tab-0", active_tab_style={"borderBottom": f"3px solid {COLORS['accent']}"}),
            dbc.Tab(label="Differential Expression", tab_id="tab-1", active_tab_style={"borderBottom": f"3px solid {COLORS['accent']}"}),
            dbc.Tab(label="Isoform Explorer", tab_id="tab-2", active_tab_style={"borderBottom": f"3px solid {COLORS['accent']}"}),
            dbc.Tab(label="Isoform Correlations", tab_id="tab-3", active_tab_style={"borderBottom": f"3px solid {COLORS['accent']}"}),
            dbc.Tab(label="QTL Explorer", tab_id="tab-4", active_tab_style={"borderBottom": f"3px solid {COLORS['accent']}"}),
            dbc.Tab(label="Download Data", tab_id="tab-5", active_tab_style={"borderBottom": f"3px solid {COLORS['accent']}"}),
        ],
        className="mb-4 nav-tabs-clean dbc",
        style={
            "border-bottom": f"1px solid {COLORS['border']}",
            "font-weight": "300",
            "font-size": "calc(0.85rem + 0.3vw)"  # Responsive font size for tabs
        }
    ),
    
    # Content area with enhanced contrast
    dbc.Card([
        dbc.CardBody(
            html.Div(id="content", 
                className="py-3 dbc",
                style={
                    "color": COLORS['text-primary'],
                    "font-size": "calc(0.9rem + 0.2vw)"  # Responsive base font size for content
                }
            )
        )
    ], 
    className="mb-5 dbc", 
    style={
        "border": f"1px solid {COLORS['border']}",
        "border-radius": "8px",
        "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
        "background-color": COLORS['bg-card']
    })
], 
fluid=True, 
className="dbc", 
style={
    "max-width": "98%",  # Increased from 1200px to use more screen space
    "margin": "0 auto",
    "padding": "10px",   # Added padding
    "color": COLORS['text-primary'],  # Ensure good contrast for all text
    "background-color": COLORS['bg-primary'],  # White background
    "font-size": "calc(0.9rem + 0.2vw)"  # Responsive base font size for container
})

# Mapping of tab values to their corresponding layouts
content_layout = {
    "tab-0": tab0.layout(),  # New Home tab
    "tab-1": tab1.layout(),
    "tab-2": tab2.layout(),
    "tab-3": tab3.layout(),
    "tab-4": tab4.layout(),
    "tab-5": tab5.layout()
}