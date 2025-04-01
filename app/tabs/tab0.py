# File: app/tabs/tab0.py
# Defines the layout for the Home tab

from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

# Define the layout for the Home tab
def layout():
    return html.Div([
        html.H2("Welcome to the RNA-seq Analysis Dashboard", 
                id="tab0-header",
                className="mb-4"),
        html.P([
            "This dashboard provides interactive tools for analyzing RNA-seq data, including:",
            html.Ul([
                html.Li("Differential Expression Analysis", id="tab0-list-item-1"),
                html.Li("Isoform Exploration", id="tab0-list-item-2"),
                html.Li("Isoform Correlations", id="tab0-list-item-3"),
                html.Li("eQTL Analysis", id="tab0-list-item-4"),
                html.Li("Gene Coverage Visualization", id="tab0-list-item-5"),
                html.Li("Data Download Options", id="tab0-list-item-6")
            ], id="tab0-list")
        ], id="tab0-intro")
    ], id="tab0-container")

@callback(
    [Output("tab0-container", "style"),
     Output("tab0-header", "style"),
     Output("tab0-intro", "style"),
     Output("tab0-list", "style")],
    [Input("window-dimensions", "data")]
)
def update_tab0_responsiveness(dimensions):
    if not dimensions:
        # Default styles
        return ({}, {}, {}, {})
    
    width = dimensions.get('width', 1200)
    
    # Base styles
    container_style = {"padding": "20px"}
    header_style = {"font-weight": "400"}
    intro_style = {"font-size": "1.1rem"}
    list_style = {"margin-top": "10px"}
    
    # Responsive adjustments
    if width < 576:  # Extra small
        container_style.update({"padding": "10px"})
        header_style.update({"font-size": "1.5rem"})
        intro_style.update({"font-size": "0.9rem"})
    elif width < 768:  # Small
        container_style.update({"padding": "15px"})
        header_style.update({"font-size": "1.8rem"})
        intro_style.update({"font-size": "1rem"})
    elif width < 992:  # Medium
        container_style.update({"padding": "20px"})
        header_style.update({"font-size": "2rem"})
        intro_style.update({"font-size": "1.1rem"})
    else:  # Large
        container_style.update({"padding": "25px"})
        header_style.update({"font-size": "2.2rem"})
        intro_style.update({"font-size": "1.2rem"})
    
    return container_style, header_style, intro_style, list_style 