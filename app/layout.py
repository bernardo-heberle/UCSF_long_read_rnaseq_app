# Defines the overall layout of the app including tabs and a content area.

# File: app/layout.py
# Defines the overall layout of the app including tabs and a content area.

from dash import html, dcc
import dash_bootstrap_components as dbc
from app.tabs import tab0, tab1, tab2, tab3, tab4, tab5, tab6

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
    # Minimal, clean header with enhanced contrast
    html.H1("Long-Read RNAseq Atlas of Aged Human Prefrontal Cortex and Alzheimer's Disease", 
        className="mt-5 mb-4 dbc", 
        style={
            "font-weight": "400",
            "letter-spacing": "1px",
            "color": COLORS['text-primary'],
            "border-bottom": f"3px solid {COLORS['accent']}",
            "padding-bottom": "0.5rem",
            "display": "inline-block"
        }
    ),
    
    # Tabs with enhanced styling
    dbc.Tabs(
        id="tabs",
        active_tab="tab-0",
        children=[
            dbc.Tab(label="Home", tab_id="tab-0"),
            dbc.Tab(label="Differential Expression", tab_id="tab-1"),
            dbc.Tab(label="Isoform Explorer", tab_id="tab-2"),
            dbc.Tab(label="Isoform Correlations", tab_id="tab-3"),
            dbc.Tab(label="eQTL Explorer", tab_id="tab-4"),
            dbc.Tab(label="Gene Coverage", tab_id="tab-5"),
            dbc.Tab(label="Download Data", tab_id="tab-6")
        ],
        className="mb-4 nav-tabs-clean dbc",
        style={
            "border-bottom": f"1px solid {COLORS['border']}",
            "font-weight": "300",
        }
    ),
    
    # Content area with enhanced contrast
    dbc.Card([
        dbc.CardBody(
            html.Div(id="content", 
                className="py-3 dbc",
                style={
                    "color": COLORS['text-primary']
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
    "background-color": COLORS['bg-primary']  # White background
})

# Mapping of tab values to their corresponding layouts
content_layout = {
    "tab-0": tab0.layout,  # New Home tab
    "tab-1": tab1.layout,
    "tab-2": tab2.layout,
    "tab-3": tab3.layout,
    "tab-4": tab4.layout,
    "tab-5": tab5.layout,
    "tab-6": tab6.layout
}