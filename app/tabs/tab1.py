# File: app/tabs/tab1.py
# Contains the layout for Tab 1.

from dash import html, dcc, Input, Output, callback, State, no_update
import dash_bootstrap_components as dbc
from app.utils.ui_components import (
    create_section_header,
    create_content_card,
    create_gene_search_dropdown
)
from app import app
import plotly.graph_objects as go
import dash_bio as dashbio
from app.utils.db_utils import search_genes_tab1, duck_conn
import pandas as pd
import json
from io import StringIO
import numpy as np

# Store the last valid search options to prevent them from disappearing
last_valid_options = []
last_search_value = None  # Store the last search value

# Store for loaded tables data
loaded_tables = {
    'deg': None,
    'dte': None,
    'dtu': None
}

def layout():
    return dbc.Container([
        # Add stores to hold the data
        dcc.Store(id='deg-data-store-tab1'),
        dcc.Store(id='dte-data-store-tab1'),
        dcc.Store(id='dtu-data-store-tab1'),
        
        # Add Download component for the SVG files
        dcc.Download(id="download-svg-tab1"),
        
        # Hidden div to hold form label styles
        html.Div(id="tab1-form-labels", style={"display": "none"}),
        
        # Hidden dummy graphs to prevent callback errors
        html.Div([
            dcc.Graph(id="dge-graph", figure={}, style={"display": "none"}),
            dcc.Graph(id="dte-graph", figure={}, style={"display": "none"}),
            dcc.Graph(id="dtu-graph", figure={}, style={"display": "none"})
        ], style={"display": "none"}),
        
        dbc.Card([
            dbc.CardBody([
                # First row - 2 columns for quadrants 1 and 2
                dbc.Row([
                    # Quadrant 1 (Top Left)
                    dbc.Col([
                        html.Div(id="tab1-controls-card", style={"height": "600px", "overflow-y": "auto", "padding": "15px 10px"}, children=[
                            html.H4("Analysis Controls", 
                                   style={"color": "#333333", "font-size": "1.5rem", "margin-bottom": "2rem", "font-weight": "700"}),
                            create_content_card(
                                # Analysis controls content
                                [
                                # Row 1: Group Comparison and Matrix Type in two columns
                                dbc.Row([
                                    # Left column - Group Comparison dropdown
                                    dbc.Col([
                                        html.Label("Group Comparison:", className="mb-1 tab1-form-label", 
                                                  id="tab1-comparison-label"),
                                        dcc.Dropdown(
                                            id="independent-var-dropdown-tab1",
                                            options=[],  # Will be populated by callback
                                            value=None,  # Will be set by callback
                                            clearable=False,
                                            className="mb-2 taller-dropdown",
                                            optionHeight=60,
                                            maxHeight=400
                                        )
                                    ], width=6),
                                    
                                    # Right column - Matrix Type dropdown
                                    dbc.Col([
                                        html.Label("Counts Matrix Type:", className="mb-1 tab1-form-label", 
                                                  id="tab1-matrix-label"),
                                        dcc.Dropdown(
                                            id="matrix-type-dropdown-tab1",
                                            options=[
                                                {"label": "Unique Counts", "value": "unique"},
                                                {"label": "Total Counts", "value": "total"}
                                            ],
                                            value="unique",
                                            clearable=False,
                                            className="mb-2 taller-dropdown",
                                            optionHeight=60,
                                            maxHeight=400
                                        )
                                    ], width=6)
                                ], className="mb-5"),
                                
                                # Spacer div with context information
                                html.Div([
                                    html.Hr(style={"margin": "0.5rem 0 1rem 0"}),
                                    html.P(
                                        "Adjust thresholds below to filter significant results. Genes or transcripts will be highlighted if they meet both FDR and Effect Size thresholds.",
                                        style={
                                            "color": "#666666", 
                                            "font-size": "1.0rem", 
                                            "margin-bottom": "1rem",
                                            "font-style": "italic"
                                        }
                                    ),
                                ], style={"margin-bottom": "1rem"}),
                                
                                # Row 2: FDR and Effect Size sliders in two columns
                                dbc.Row([
                                    # Left column - FDR p-value slider
                                    dbc.Col([
                                        html.Label("FDR p-value Threshold:", className="mb-1 tab1-form-label", 
                                                id="tab1-pvalue-label"),
                                        dcc.Slider(
                                            id="pvalue-slider-tab1",
                                            min=0,
                                            max=5,
                                            step=None,
                                            marks={
                                                0: {'label': '0.001', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                1: {'label': '0.01', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                2: {'label': '0.05', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                3: {'label': '0.1', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                4: {'label': '0.2', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                5: {'label': '0.3', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}}
                                            },
                                            value=2,  # Represents 0.05
                                            className="mb-1",
                                            tooltip=None,  # Disable the tooltip entirely
                                            updatemode='mouseup'
                                        ),
                                        html.Div(id="pvalue-slider-output-tab1", className="slider-output text-center text-muted", style={"font-size": "18px"})
                                    ], width=6),
                                    
                                    # Right column - Effect Size slider
                                    dbc.Col([
                                        html.Label("Effect Size Threshold:", className="mb-1 tab1-form-label", 
                                                  id="tab1-effect-label"),
                                        dcc.Slider(
                                            id="effect-size-slider-tab1",
                                            min=0,
                                            max=1.48,
                                            step=None,
                                            marks={
                                                0: {'label': 'None', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.08: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.18: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.28: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.38: {'label': '0.38', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.48: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.58: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.68: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.78: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.88: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                0.98: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                1.08: {'label': '1.08', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                1.18: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                1.28: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                1.38: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
                                                1.48: {'label': '1.48', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}}
                                            },
                                            value=0.38,
                                            className="mb-1",
                                            tooltip={"placement": "bottom", "always_visible": True, "style": {"font-size": "18px"}},
                                            updatemode='mouseup'
                                        ),
                                        html.Div(id="effect-size-slider-output-tab1", className="slider-output text-center text-muted", style={"font-size": "18px"})
                                    ], width=6)
                                ], className="mb-5"),
                                
                                # Row 3: Gene search and Download button in two columns
                                dbc.Row([
                                    # Left column - Gene search
                                    dbc.Col([
                                        html.Label("Highlight Gene:", className="mb-1 tab1-form-label", 
                                                  id="tab1-gene-label"),
                                        create_gene_search_dropdown(id="search-input-tab1"),
                                        html.Div("Genes/isoforms that didn't meet the inclusion threshold will not be highlighted.", 
                                                className="slider-output text-center text-muted", 
                                                style={"font-size": "18px", "margin-top": "4px"})
                                    ], width=6),
                                    
                                    # Right column - Download button
                                    dbc.Col([
                                        html.Label("Export Plots:", className="mb-1 tab1-form-label", 
                                                  id="tab1-export-label"),
                                        dbc.Button(
                                            [
                                                html.I(className="fas fa-download me-2"),
                                                "Download SVG"
                                            ],
                                            id="download-button-tab1",
                                            color="primary",
                                            className="w-100 mt-2", # Added margin-top to align with dropdown
                                            disabled=False
                                        ),
                                        html.Small(
                                            "Takes a while to generate plots",
                                            style={
                                                "color": "#666666",
                                                "display": "block",
                                                "marginTop": "4px",
                                                "textAlign": "center",
                                                "fontSize": "14px"
                                            }
                                        )
                                    ], width=6)
                                ], className="mb-5"),
                            ])
                        ]),
                    ], width=6, id="tab1-quadrant1-col"),
                    
                    # Quadrant 2 (Top Right)
                    dbc.Col([
                        create_content_card([
                            dbc.Spinner(
                                html.Div([], id="differential-gene-expression-plot"),
                                color="primary",
                                type="grow",
                                spinner_style={"width": "3rem", "height": "3rem"}
                            )
                        ])
                    ], width=6, id="tab1-quadrant2-col")
                ], className="mb-4", id="tab1-row1"),
                
                # Second row - 2 columns for quadrants 3 and 4
                dbc.Row([
                    # Quadrant 3 (Bottom Left)
                    dbc.Col([
                        create_content_card([
                            dbc.Spinner(
                                html.Div([], id="differential-transcript-expression-plot"),
                                color="primary",
                                type="grow",
                                spinner_style={"width": "3rem", "height": "3rem"}
                            )
                        ])
                    ], width=6, id="tab1-quadrant3-col"),
                    
                    # Quadrant 4 (Bottom Right)
                    dbc.Col([
                        create_content_card([
                            dbc.Spinner(
                                html.Div([], id="differential-transcript-usage-plot"),
                                color="primary",
                                type="grow",
                                spinner_style={"width": "3rem", "height": "3rem"}
                            )
                        ])
                    ], width=6, id="tab1-quadrant4-col")
                ], className="mb-4", id="tab1-row2")
            ], id="tab1-card-body")
        ],
        id="tab1-card",
        style={
            "background-color": "#ffffff",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "6px",
            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
        })
    ], 
    fluid=True,  # Makes the container full-width
    id="tab1-container",
    style={
        "max-width": "98%",  # Use 98% of the viewport width
        "margin": "0 auto",  # Center the container
        "padding": "10px"    # Add some padding
    })

# Callback to populate group comparison dropdown
@callback(
    [Output("independent-var-dropdown-tab1", "options"),
     Output("independent-var-dropdown-tab1", "value")],
    [Input("tab1-container", "id")]  # Trigger on page load
)
def populate_group_comparison_dropdown(_):
    """Get unique group_comparison values from degs table"""
    try:
        query = """
            SELECT DISTINCT group_comparison
            FROM degs
            ORDER BY group_comparison
        """
        result = duck_conn.execute(query).fetchall()
        options = [{"label": row[0], "value": row[0]} for row in result]
        # Set first option as default value
        default_value = options[0]["value"] if options else None
        return options, default_value
    except Exception as e:
        print(f"Error loading group comparisons: {e}")
        return [], None

# Callback for responsive design
@callback(
    [Output("tab1-container", "style"),
     Output("tab1-row1", "className"),
     Output("tab1-quadrant1-col", "width"),
     Output("tab1-quadrant2-col", "width"),
     Output("tab1-row2", "className"),
     Output("tab1-quadrant3-col", "width"),
     Output("tab1-quadrant4-col", "width"),
     # Add outputs for analysis controls font sizes and element styling
     Output("pvalue-slider-output-tab1", "style"),
     Output("effect-size-slider-output-tab1", "style"),
     Output("independent-var-dropdown-tab1", "className"),
     Output("matrix-type-dropdown-tab1", "className"),
     Output("search-input-tab1", "className"),
     # Add output for all label styles
     Output("tab1-form-labels", "style")],
    [Input("window-dimensions", "data")]
)
def update_tab1_responsiveness(dimensions):
    if not dimensions:
        # Default styles if no dimensions available
        return (
            {"max-width": "98%", "margin": "0 auto", "padding": "10px"},
            "mb-4", 6, 6, 
            "mb-4", 6, 6,
            {"font-size": "16px"},
            {"font-size": "16px"},
            "mb-3",
            "mb-3",
            "mb-3",
            {"font-weight": "500", "color": "#495057", "font-size": "16px"}
        )
    
    width = dimensions.get('width', 1200)
    height = dimensions.get('height', 800)
    
    # Calculate scaling factors for font sizes
    scaling_factor = max(0.5, min(1.2, width / 1920))
    font_size = max(16, int(18 * scaling_factor))
    slider_font_size = {"font-size": f"{font_size}px", "margin-top": "2px", "margin-bottom": "4px"}
    label_style = {"font-weight": "600", "color": "#495057", "font-size": f"{font_size}px"}
    
    # Calculate spacing classes based on screen size
    dropdown_class = "mb-3"
    
    # Base styles
    container_style = {"max-width": "98%", "margin": "0 auto", "padding": "10px"}
    row1_class = "mb-4"
    quadrant1_width = 6
    quadrant2_width = 6
    
    row2_class = "mb-4"
    quadrant3_width = 6
    quadrant4_width = 6
    
    # Responsive adjustments based on width
    if width < 576:  # Extra small devices
        container_style.update({"padding": "5px", "max-width": "100%"})
        row1_class = "mb-2 flex-column"
        quadrant1_width = 12
        quadrant2_width = 12
        
        row2_class = "mb-2 flex-column"
        quadrant3_width = 12
        quadrant4_width = 12
        
        dropdown_class = "mb-2"
        
    elif width < 768:  # Small devices
        container_style.update({"padding": "8px"})
        row1_class = "mb-3"
        quadrant1_width = 12
        quadrant2_width = 12
        
        row2_class = "mb-3"
        quadrant3_width = 12
        quadrant4_width = 12
        
    elif width < 992:  # Medium devices
        container_style.update({"padding": "10px"})
        row1_class = "mb-3"
        row2_class = "mb-3"
        
    return (
        container_style,
        row1_class, quadrant1_width, quadrant2_width,
        row2_class, quadrant3_width, quadrant4_width,
        slider_font_size,
        slider_font_size,
        dropdown_class,
        dropdown_class,
        dropdown_class,
        label_style
    )

# Add search functionality - now using the database search function
@app.callback(
    Output('search-input-tab1', 'options'),
    [Input('search-input-tab1', 'search_value'),
     Input('search-input-tab1', 'value')]
)
def update_search_options_tab1(search_value, selected_value):
    global last_valid_options, last_search_value
    
    # If we have a selected value but no search, return the last options
    # This keeps the dropdown populated after selection
    if selected_value and not search_value:
        # Make sure the selected value is in the options
        selected_in_options = any(opt.get('value') == selected_value for opt in last_valid_options)
        if not selected_in_options:
            # If we have a newly selected value, we need to add it to the options
            # First get the full gene details from the database
            try:
                gene_result = duck_conn.execute("""
                    SELECT gene_index, gene_name, gene_id
                    FROM gene_and_transcript_index_table 
                    WHERE gene_name = ?
                    GROUP BY gene_index, gene_name, gene_id
                    LIMIT 1
                """, [selected_value]).fetchone()
                
                if gene_result:
                    # Add this gene to the options
                    gene_index, gene_name, gene_id = gene_result
                    option = {
                        'label': f"{gene_name} ({gene_id})",
                        'value': gene_name
                    }
                    last_valid_options = [option]  # Just show the current selection
            except Exception as e:
                # If we can't get the details, just use the raw ID
                if selected_value:
                    last_valid_options = [{
                        'label': f"{selected_value}",
                        'value': selected_value
                    }]
        
        return last_valid_options
    
    # If no search value or too short, return latest options
    if not search_value or len(search_value) < 2:
        return last_valid_options
    
    # Process the search and return results
    results = search_genes_tab1(search_value, last_search_value)
    
    # Store the results and search value for future reference
    if results:
        last_valid_options = results
        last_search_value = search_value
    
    return results

# Function to determine table names based on selections
def get_table_names(count_type):
    """
    Determine the table names to load based on count type
    
    Args:
        count_type (str): Value from matrix type dropdown
    
    Returns:
        tuple: (deg_table, dte_table, dtu_table)
    """
    # DEG table is always "degs"
    deg_table = "degs"
    # DTE and DTU tables depend on count type
    dte_table = f"dte_{count_type}"
    dtu_table = f"dtu_{count_type}"
    
    return deg_table, dte_table, dtu_table

# Helper function to safely get column names
def get_table_columns(table_name):
    """Get column names from a table, safely handling case sensitivity"""
    try:
        # Query to get column names
        columns_query = f'SELECT * FROM "{table_name}" LIMIT 0'
        result = duck_conn.execute(columns_query)
        return [col[0] for col in result.description]
    except Exception as e:
        return []

# Callback to load data when parameters change
@app.callback(
    [Output('deg-data-store-tab1', 'data'),
     Output('dte-data-store-tab1', 'data'),
     Output('dtu-data-store-tab1', 'data')],
    [Input('independent-var-dropdown-tab1', 'value'),
     Input('matrix-type-dropdown-tab1', 'value')]
)
def load_table_data(group_comparison, count_type):
    
    # Default values if inputs are None
    if not group_comparison or not count_type:
        return None, None, None
    
    count_type = count_type if count_type else "unique"
    
    # Get the appropriate table names
    dge_table, dte_table, dtu_table = get_table_names(count_type)
    
    try:
        # Get DEG data filtered by group_comparison
        dge_query = f"""
            SELECT *
            FROM "{dge_table}"
            WHERE group_comparison = '{group_comparison}'
        """
  
        # Get DTE data filtered by group_comparison
        dte_query = f"""
            SELECT *
            FROM "{dte_table}"
            WHERE group_comparison = '{group_comparison}'
        """
        
        # Get DTU data filtered by group_comparison
        dtu_query = f"""
            SELECT *
            FROM "{dtu_table}"
            WHERE group_comparison = '{group_comparison}'
        """

        return dge_query, dte_query, dtu_query
        
    except Exception as e:
        # Return empty data on error
        print(f"Error loading data: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

# Add callbacks for displaying the current slider values
@app.callback(
    Output('pvalue-slider-output-tab1', 'children'),
    [Input('pvalue-slider-tab1', 'value')]
)
def update_pvalue_output(value):
    p_values = {0: 0.001, 1: 0.01, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.3}
    return f"Current threshold: {p_values[value]}"

@app.callback(
    Output('effect-size-slider-output-tab1', 'children'),
    [Input('effect-size-slider-tab1', 'value')]
)
def update_effect_size_output(value):
    if value == 0:
        return "No effect size threshold (use p-value only)"
    return f"Current threshold: {value}"

# Update the plots when data changes
@app.callback(
    [Output('differential-gene-expression-plot', 'children'),
     Output('differential-transcript-expression-plot', 'children'),
     Output('differential-transcript-usage-plot', 'children'),
     # Add outputs for the hidden dummy graphs
     Output('dge-graph', 'figure'),
     Output('dte-graph', 'figure'),
     Output('dtu-graph', 'figure')],
    [Input('deg-data-store-tab1', 'data'),
     Input('dte-data-store-tab1', 'data'),
     Input('dtu-data-store-tab1', 'data'),
     Input('search-input-tab1', 'value'),
     Input('pvalue-slider-tab1', 'value'),
     Input('effect-size-slider-tab1', 'value'),
     Input('matrix-type-dropdown-tab1', 'value'),
     Input('independent-var-dropdown-tab1', 'value'),
     Input('window-dimensions', 'data')]
)
def update_plots(dge_query, dte_query, dtu_query, selected_gene_name, pvalue_idx, effect_size, count_type, group_comparison, window_dimensions):
    # Convert p-value index to actual p-value
    p_values = {0: 0.001, 1: 0.01, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.3}
    pvalue_threshold = p_values[pvalue_idx]
    
    # Parse group_comparison to extract the numerator group (before " vs. ")
    # Format is "Group1 vs. Group2" where positive values mean upregulation in Group1
    if group_comparison and " vs. " in group_comparison:
        numerator_group = group_comparison.split(" vs. ")[0]
        comparison_text = f"({group_comparison})"
        interpretation_text = f"Positive log2 fold change indicates upregulation in {numerator_group}"
        interpretation_text_dtu = f"Positive effect size indicates increased transcript usage in {numerator_group}"
    else:
        # Fallback if group_comparison is not in expected format
        comparison_text = f"({group_comparison})" if group_comparison else ""
        interpretation_text = "Positive log2 fold change indicates upregulation in the first group"
        interpretation_text_dtu = "Positive effect size indicates increased transcript usage in the first group"
    
    # Default window dimensions if not available yet
    if not window_dimensions:
        window_dimensions = {'width': 1200, 'height': 800}

    # Create scaling factor and base font size
    scaling_factor = max(0.5, window_dimensions["width"] / 2540)
    base_font_size = 16 * scaling_factor
    title_size = base_font_size * 1.25  # 20px at normal size
    subtitle_size = base_font_size * 1.125  # 18px at normal size
    axis_label_size = base_font_size
    tick_label_size = base_font_size * 0.875
    legend_label_size = 16 * scaling_factor
    
    # Calculate responsive height for plots
    plot_height = int(window_dimensions['height'] * 0.785)  # Using 0.785x window height
    
    try:
        ## Load DEG data into a pandas dataframe
        dge_cursor = duck_conn.execute(dge_query)
        dge_columns = [desc[0] for desc in dge_cursor.cursor.description]
        dge_result = dge_cursor.fetchall()
        dge_data = pd.DataFrame(dge_result, columns=dge_columns)
        
        # Rename effectSize to log2FoldChange for volcano plot compatibility
        if 'effectSize' in dge_data.columns:
            dge_data.rename(columns={'effectSize': 'log2FoldChange'}, inplace=True)
        
        # Ensure numeric columns are properly typed
        dge_data['log2FoldChange'] = pd.to_numeric(dge_data['log2FoldChange'], errors='coerce')
        dge_data['PValue'] = pd.to_numeric(dge_data['PValue'], errors='coerce')
        dge_data['padj'] = pd.to_numeric(dge_data['padj'], errors='coerce')
        
        # Remove rows with NaN p-values to prevent VolcanoPlot errors
        dge_data = dge_data.dropna(subset=['PValue'])
        
        # Filter out extreme effect sizes (> 20 or < -20)
        dge_data = dge_data[(dge_data['log2FoldChange'] >= -20) & (dge_data['log2FoldChange'] <= 20)]
        
        ## Load DTE data into a pandas dataframe
        dte_cursor = duck_conn.execute(dte_query)
        dte_columns = [desc[0] for desc in dte_cursor.cursor.description]
        dte_result = dte_cursor.fetchall()
        dte_data = pd.DataFrame(dte_result, columns=dte_columns)
        
        # Rename effectSize to log2FoldChange for volcano plot compatibility
        if 'effectSize' in dte_data.columns:
            dte_data.rename(columns={'effectSize': 'log2FoldChange'}, inplace=True)
        
        # Ensure numeric columns are properly typed
        dte_data['log2FoldChange'] = pd.to_numeric(dte_data['log2FoldChange'], errors='coerce')
        dte_data['PValue'] = pd.to_numeric(dte_data['PValue'], errors='coerce')
        dte_data['padj'] = pd.to_numeric(dte_data['padj'], errors='coerce')
        
        # Remove rows with NaN p-values to prevent VolcanoPlot errors
        dte_data = dte_data.dropna(subset=['PValue'])
        
        # Filter out extreme effect sizes (> 20 or < -20)
        dte_data = dte_data[(dte_data['log2FoldChange'] >= -20) & (dte_data['log2FoldChange'] <= 20)]
        
        ## Load DTU data into a pandas dataframe
        dtu_cursor = duck_conn.execute(dtu_query)
        dtu_columns = [desc[0] for desc in dtu_cursor.cursor.description]
        dtu_result = dtu_cursor.fetchall()
        dtu_data = pd.DataFrame(dtu_result, columns=dtu_columns)
        
        # Rename effectSize to estimates and PValue to pval, padj to regular_FDR for volcano plot compatibility
        if 'effectSize' in dtu_data.columns:
            dtu_data.rename(columns={'effectSize': 'estimates', 'PValue': 'pval', 'padj': 'regular_FDR'}, inplace=True)
        
        # Ensure numeric columns are properly typed
        dtu_data['estimates'] = pd.to_numeric(dtu_data['estimates'], errors='coerce')
        dtu_data['pval'] = pd.to_numeric(dtu_data['pval'], errors='coerce')
        dtu_data['regular_FDR'] = pd.to_numeric(dtu_data['regular_FDR'], errors='coerce')
        
        # Remove rows with NaN p-values to prevent VolcanoPlot errors
        dtu_data = dtu_data.dropna(subset=['pval'])
        
        # Filter out extreme effect sizes (> 20 or < -20)
        dtu_data = dtu_data[(dtu_data['estimates'] >= -20) & (dtu_data['estimates'] <= 20)]
        
        # Check if any dataframes are empty
        if dge_data.empty or dte_data.empty or dtu_data.empty:
            empty_msg = []
            if dge_data.empty:
                empty_msg.append("DEG")
            if dte_data.empty:
                empty_msg.append("DTE")
            if dtu_data.empty:
                empty_msg.append("DTU")
            
            # Show error message in a styled div
            error_message = html.Div(
                html.P(f"No data available for the selected filters: {', '.join(empty_msg)} tables are empty. Try different parameters.",
                      style={"color": "#dc3545", "margin": 0, "text-align": "center", "padding": "20px", "font-weight": "500"}),
                style={
                    "height": "100%",
                    "width": "100%",
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                    "min-height": f"{plot_height}px",
                    "background-color": "#f8f9fa",
                    "border-radius": "6px",
                    "border": "2px solid #dc3545"
                }
            )
            empty_fig = {}
            return error_message, error_message, error_message, empty_fig, empty_fig, empty_fig

        # Calculate significance lines based on user-selected thresholds
        dge_sig_line = -np.log10(dge_data.loc[dge_data['padj'] < pvalue_threshold]['PValue'].max() + 0.000000001) \
            if not dge_data.loc[dge_data['padj'] < pvalue_threshold].empty else False
        dte_sig_line = -np.log10(dte_data.loc[dte_data['padj'] < pvalue_threshold]['PValue'].max() + 0.000000001) \
            if not dte_data.loc[dte_data['padj'] < pvalue_threshold].empty else False
        # DTU now uses 'pval' and 'regular_FDR' columns
        dtu_sig_line = -np.log10(dtu_data.loc[dtu_data['regular_FDR'] < pvalue_threshold]['pval'].max() + 0.000000001) \
            if not dtu_data.loc[dtu_data['regular_FDR'] < pvalue_threshold].empty else False
        # Create volcano plot with or without effect size lines
        # When effect_size is 0, [-effect_size, effect_size] = [0, 0]
        # This means all points (regardless of fold change) above p-value threshold are highlighted
        volcano_plot_dge = dashbio.VolcanoPlot(
            dataframe=dge_data,
            effect_size='log2FoldChange',
            p='PValue',
            xlabel='log2 Fold Change',
            ylabel='-log10(p)',
            genomewideline_value=dge_sig_line,
            genomewideline_color='black',
            effect_size_line=[-effect_size, effect_size],
            effect_size_line_color='rgba(0,0,0,0)' if effect_size == 0 else 'black',  # Invisible when no threshold
            highlight_color="#FF6692",
            col="#19D3F3",
            point_size=max(6, int(8 * scaling_factor)),
            effect_size_line_width=0 if effect_size == 0 else 2,  # Hide lines when no threshold
            genomewideline_width=2,
            highlight=False if dge_sig_line is False else True,
            gene='gene_name',
            snp=None
        )

        # Update trace names for proper legend
        for trace in volcano_plot_dge.data:
            if trace.marker and hasattr(trace.marker, 'color'):
                if isinstance(trace.marker.color, list):
                    # Check if this trace contains red points (DEGs)
                    if any(color == "#FF6692" for color in trace.marker.color):
                        trace.name = "DEGs"
                        trace.showlegend = True
                    elif any(color == "#19D3F3" for color in trace.marker.color):
                        trace.name = "Not DEGs"
                        trace.showlegend = True
                else:
                    # Single color trace
                    if trace.marker.color == "#FF6692":
                        trace.name = "DEGs"
                        trace.showlegend = True
                    elif trace.marker.color == "#19D3F3":
                        trace.name = "Not DEGs"
                        trace.showlegend = True

        if selected_gene_name is not None:
            show_legend = True
            gene_lookup = f"<br>GENE: {selected_gene_name}"
            for trace in volcano_plot_dge.data:
                texts = list(trace.text)
                xs = list(trace.x)
                ys = list(trace.y)
                # normalize color & size arrays
                colors = trace.marker.color
                sizes  = trace.marker.size

                # find matching indexes - exact match only
                idxs = [i for i, s in enumerate(texts) if gene_lookup == s]
                if not idxs:
                    continue

                # extract selected‐gene data
                xs_sel   = [xs[i] for i in idxs]
                ys_sel   = [ys[i] for i in idxs]
                texts_sel= [texts[i] for i in idxs]

                # remove them from the original trace (reverse order)
                for i in sorted(idxs, reverse=True):
                    xs.pop(i); ys.pop(i); texts.pop(i)

                # write back the pruned arrays
                trace.x = xs
                trace.y = ys
                trace.text = texts
                trace.marker.color = colors
                trace.marker.size  = sizes

                # add a new trace just for the selected gene
                volcano_plot_dge.add_trace(go.Scattergl(
                    x=xs_sel,
                    y=ys_sel,
                    text=texts_sel,
                    mode='markers',
                    marker=dict(color="#2CA02C", size=max(14, int(18 * scaling_factor))),  # Responsive highlighted point size
                    name=selected_gene_name,
                    showlegend=show_legend,
                    legendgroup='Highlighted Gene'
                ))

                show_legend = False

                # Move the selected gene trace to the front for better hoverability
                volcano_plot_dge.data = list(volcano_plot_dge.data[:-1]) + [volcano_plot_dge.data[-1]]

                

        dge_plot = dcc.Graph(
                    id='dge-graph', 
                    figure=volcano_plot_dge,
                    style={'height': f'{plot_height}px', 'width': '100%'}  # Responsive height
                )
        
        # Calculate axis ranges
        # X-axis: extend at least 0.05 beyond the effect size threshold
        if effect_size == 0:
            # No effect size threshold - use data range with padding
            if not dge_data.empty:
                data_x_min = dge_data['log2FoldChange'].min()
                data_x_max = dge_data['log2FoldChange'].max()
                dge_x_min = data_x_min - 0.1
                dge_x_max = data_x_max + 0.1
            else:
                dge_x_min = -2.0
                dge_x_max = 2.0
        else:
            # With effect size threshold
            dge_x_min = -(effect_size + 0.05)
            dge_x_max = effect_size + 0.05
            # Adjust if data extends beyond this range
            if not dge_data.empty:
                data_x_min = dge_data['log2FoldChange'].min()
                data_x_max = dge_data['log2FoldChange'].max()
                dge_x_min = min(dge_x_min, data_x_min - 0.1)
                dge_x_max = max(dge_x_max, data_x_max + 0.1)
        
        # Make x-axis symmetrical around 0 and add cushion
        dge_x_range = max(abs(dge_x_min), abs(dge_x_max)) + 0.1
        dge_x_min = -dge_x_range
        dge_x_max = dge_x_range
        
        # Y-axis: go from 0 to at least 0.5 above the significance line
        dge_y_max = (dge_sig_line + 0.5) if dge_sig_line is not False else 5.0
        # Adjust if data extends beyond this range
        if not dge_data.empty and 'PValue' in dge_data.columns:
            data_y_max = -np.log10(dge_data['PValue'].replace(0, 1e-300).min())
            dge_y_max = max(dge_y_max, data_y_max + 0.5)
        
        # Create subtitle based on thresholds
        if effect_size == 0:
            subtitle_text = f"{'No q-value threshold (no significant genes)' if dge_sig_line is False else f'Horizontal line shows significance cutoff (q-value < {pvalue_threshold})'}"
        else:
            subtitle_text = f"Vertical lines at |log2 fold change| = {effect_size} and {'q-value threshold not shown (no significant genes)' if dge_sig_line is False else f'horizontal line shows significance cutoff (q-value < {pvalue_threshold})'}"
        
        # Update DGE plot title and subtitle
        dge_plot.figure.update_layout(
            title={
                'text': f"<b>Differential Gene Expression Volcano Plot {comparison_text}</b><br><span style='font-size:{subtitle_size}px'>{subtitle_text}</span><br><span style='font-size:{base_font_size}px'>{interpretation_text}</span>",
                'y':0.96,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': title_size}
            },
            showlegend=True,
            legend=dict(
                itemsizing='constant',
                traceorder='normal',
                font=dict(
                    size=legend_label_size
                ),
                borderwidth=0
            ),
            template='ggplot2',
            xaxis=dict(
                range=[dge_x_min, dge_x_max],
                tickfont=dict(size=tick_label_size),
                title=dict(text='log2 Fold Change', font=dict(size=axis_label_size))
            ),
            yaxis=dict(
                range=[-0.2, dge_y_max],
                tickfont=dict(size=tick_label_size),
                title=dict(text='-log10(p)', font=dict(size=axis_label_size))
            )
        )
        
        # Add annotation if no entries meet FDR threshold
        if dge_sig_line is False:
            dge_plot.figure.add_annotation(
                text=f"No genes meet FDR threshold (q < {pvalue_threshold})",
                xref="paper", yref="paper",
                x=0.5, y=0.98,
                xanchor='center', yanchor='top',
                showarrow=False,
                font=dict(size=int(base_font_size * 0.9), color="red", family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red",
                borderwidth=2,
                borderpad=4
            )

        # Create DTE volcano plot
        # When effect_size is 0, [-effect_size, effect_size] = [0, 0]
        # This means all points (regardless of fold change) above p-value threshold are highlighted
        volcano_plot_dte = dashbio.VolcanoPlot(
            dataframe=dte_data,
            effect_size='log2FoldChange',
            p='PValue',
            xlabel='log2 Fold Change',
            ylabel='-log10(p)',
            genomewideline_value=dte_sig_line,
            genomewideline_color='black',
            effect_size_line=[-effect_size, effect_size],
            effect_size_line_color='rgba(0,0,0,0)' if effect_size == 0 else 'black',  # Invisible when no threshold
            highlight_color="#FF6692",
            col="#19D3F3",
            genomewideline_width=2,
            effect_size_line_width=0 if effect_size == 0 else 2,  # Hide lines when no threshold
            point_size=max(6, int(8 * scaling_factor)),
            highlight=False if dte_sig_line is False else True,
            gene='gene_name',
            annotation='transcript_id',
            snp=None
        )
        
        # Update trace names for proper legend
        for trace in volcano_plot_dte.data:
            if trace.marker and hasattr(trace.marker, 'color'):
                if isinstance(trace.marker.color, list):
                    # Check if this trace contains red points (DETs)
                    if any(color == "#FF6692" for color in trace.marker.color):
                        trace.name = "DETs"
                        trace.showlegend = True
                    elif any(color == "#19D3F3" for color in trace.marker.color):
                        trace.name = "Not DETs"
                        trace.showlegend = True
                else:
                    # Single color trace
                    if trace.marker.color == "#FF6692":
                        trace.name = "DETs"
                        trace.showlegend = True
                    elif trace.marker.color == "#19D3F3":
                        trace.name = "Not DETs"
                        trace.showlegend = True
        

        if selected_gene_name is not None:
            
            gene_lookup = f"<br>GENE: {selected_gene_name}<br>"
            show_legend = True
            for trace in volcano_plot_dte.data:
                texts = list(trace.text)
                xs = list(trace.x)
                ys = list(trace.y)
                # normalize color & size arrays
                colors = trace.marker.color
                sizes  = trace.marker.size

                # find matching indexes - exact match only
                idxs = [i for i, s in enumerate(texts) if gene_lookup in s]
                if not idxs:
                    continue

                # extract selected‐gene data
                xs_sel   = [xs[i] for i in idxs]
                ys_sel   = [ys[i] for i in idxs]
                texts_sel= [texts[i] for i in idxs]

                # remove them from the original trace (reverse order)
                for i in sorted(idxs, reverse=True):
                    xs.pop(i); ys.pop(i); texts.pop(i)

                # write back the pruned arrays
                trace.x = xs
                trace.y = ys
                trace.text = texts
                trace.marker.color = colors
                trace.marker.size  = sizes
                # add a new trace just for the selected gene
                volcano_plot_dte.add_trace(go.Scattergl(
                    x=xs_sel,
                    y=ys_sel,
                    text=texts_sel,
                    name=f"{selected_gene_name} isoforms",
                    mode='markers',
                    legendgroup='Highlighted Gene',
                    marker=dict(color="#2CA02C", size=max(14, int(18 * scaling_factor))),  # Responsive highlighted point size
                    showlegend=show_legend
                ))

                show_legend = False

                # Move the selected gene trace to the front for better hoverability
                volcano_plot_dte.data = list(volcano_plot_dte.data[:-1]) + [volcano_plot_dte.data[-1]]


        dte_plot = dcc.Graph(
            id='dte-graph',
            figure=volcano_plot_dte,
            style={'height': f'{plot_height}px', 'width': '100%'}  # Responsive height
        )
        

        # Calculate axis ranges for DTE
        # X-axis: extend at least 0.05 beyond the effect size threshold
        if effect_size == 0:
            # No effect size threshold - use data range with padding
            if not dte_data.empty:
                data_x_min = dte_data['log2FoldChange'].min()
                data_x_max = dte_data['log2FoldChange'].max()
                dte_x_min = data_x_min - 0.1
                dte_x_max = data_x_max + 0.1
            else:
                dte_x_min = -2.0
                dte_x_max = 2.0
        else:
            # With effect size threshold
            dte_x_min = -(effect_size + 0.05)
            dte_x_max = effect_size + 0.05
            # Adjust if data extends beyond this range
            if not dte_data.empty:
                data_x_min = dte_data['log2FoldChange'].min()
                data_x_max = dte_data['log2FoldChange'].max()
                dte_x_min = min(dte_x_min, data_x_min - 0.1)
                dte_x_max = max(dte_x_max, data_x_max + 0.1)
        
        # Make x-axis symmetrical around 0 and add cushion
        dte_x_range = max(abs(dte_x_min), abs(dte_x_max)) + 0.1
        dte_x_min = -dte_x_range
        dte_x_max = dte_x_range
        
        # Y-axis: go from 0 to at least 0.5 above the significance line
        dte_y_max = (dte_sig_line + 0.5) if dte_sig_line is not False else 5.0
        # Adjust if data extends beyond this range
        if not dte_data.empty and 'PValue' in dte_data.columns:
            data_y_max = -np.log10(dte_data['PValue'].replace(0, 1e-300).min())
            dte_y_max = max(dte_y_max, data_y_max + 0.5)
        
        # Create DTE subtitle based on thresholds
        if effect_size == 0:
            dte_subtitle = f"{'No q-value threshold (no significant transcripts)' if dte_sig_line is False else f'Horizontal line shows significance cutoff (q-value < {pvalue_threshold})'}"
        else:
            dte_subtitle = f"Vertical lines at |log2 fold change| = {effect_size} and {'q-value threshold not shown (no significant transcripts)' if dte_sig_line is False else f'horizontal line shows significance cutoff (q-value < {pvalue_threshold})'}"
        
        # Update DTE plot title and subtitle
        dte_plot.figure.update_layout(
            title={
                'text': f"<b>{count_type.capitalize()} Counts Differential Transcript Expression Volcano Plot {comparison_text}</b><br><span style='font-size:{subtitle_size}px'>{dte_subtitle}</span><br><span style='font-size:{base_font_size}px'>{interpretation_text}</span>",
                'y':0.96,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': title_size}
            },
            showlegend=True,
            legend=dict(
                itemsizing='constant',
                traceorder='normal',
                font=dict(
                    size=legend_label_size
                ),
                borderwidth=0
            ),
            template='ggplot2',
            xaxis=dict(
                range=[dte_x_min, dte_x_max],
                tickfont=dict(size=tick_label_size),
                title=dict(text='log2 Fold Change', font=dict(size=axis_label_size))
            ),
            yaxis=dict(
                range=[-0.2, dte_y_max],
                tickfont=dict(size=tick_label_size),
                title=dict(text='-log10(p)', font=dict(size=axis_label_size))
            )
        )
        
        # Add annotation if no entries meet FDR threshold
        if dte_sig_line is False:
            dte_plot.figure.add_annotation(
                text=f"No transcripts meet FDR threshold (q < {pvalue_threshold})",
                xref="paper", yref="paper",
                x=0.5, y=0.98,
                xanchor='center', yanchor='top',
                showarrow=False,
                font=dict(size=int(base_font_size * 0.9), color="red", family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red",
                borderwidth=2,
                borderpad=4
            )
        
        # Create DTU volcano plot
        # When effect_size is 0, [-effect_size, effect_size] = [0, 0]
        # This means all points (regardless of effect size) above p-value threshold are highlighted
        volcano_plot_dtu = dashbio.VolcanoPlot(
            dataframe=dtu_data,
            effect_size='estimates',
            p='pval',
            xlabel='Effect Size',
            ylabel='-log10(p)',
            genomewideline_value=dtu_sig_line,
            genomewideline_color='black',
            effect_size_line=[-effect_size, effect_size],
            effect_size_line_color='rgba(0,0,0,0)' if effect_size == 0 else 'black',  # Invisible when no threshold
            highlight_color="#FF6692",
            col="#19D3F3",
            genomewideline_width=2,
            effect_size_line_width=0 if effect_size == 0 else 2,  # Hide lines when no threshold
            point_size=max(6, int(8 * scaling_factor)),
            highlight=False if dtu_sig_line is False else True,
            gene='gene_name',
            annotation='transcript_id',
            snp=None
        )

        # Update trace names for proper legend
        for trace in volcano_plot_dtu.data:
            if trace.marker and hasattr(trace.marker, 'color'):
                if isinstance(trace.marker.color, list):
                    # Check if this trace contains red points (DUTs)
                    if any(color == "#FF6692" for color in trace.marker.color):
                        trace.name = "DUTs"
                        trace.showlegend = True
                    elif any(color == "#19D3F3" for color in trace.marker.color):
                        trace.name = "Not DUTs"
                        trace.showlegend = True
                else:
                    # Single color trace
                    if trace.marker.color == "#FF6692":
                        trace.name = "DUTs"
                        trace.showlegend = True
                    elif trace.marker.color == "#19D3F3":
                        trace.name = "Not DUTs"
                        trace.showlegend = True

        if selected_gene_name is not None:
        
            gene_lookup = f"<br>GENE: {selected_gene_name}<br>"
            show_legend = True

            for trace in volcano_plot_dtu.data:
                texts = list(trace.text)
                xs = list(trace.x)
                ys = list(trace.y)
                # normalize color & size arrays
                colors = trace.marker.color
                sizes  = trace.marker.size

                # find matching indexes - exact match only
                idxs = [i for i, s in enumerate(texts) if gene_lookup in s]
                if not idxs:
                    continue

                # extract selected‐gene data
                xs_sel   = [xs[i] for i in idxs]
                ys_sel   = [ys[i] for i in idxs]
                texts_sel= [texts[i] for i in idxs]


                # remove them from the original trace (reverse order)
                for i in sorted(idxs, reverse=True):
                    xs.pop(i); ys.pop(i); texts.pop(i)

                # write back the pruned arrays
                trace.x = xs
                trace.y = ys
                trace.text = texts
                trace.marker.color = colors
                trace.marker.size  = sizes
                # add a new trace just for the selected gene
                volcano_plot_dtu.add_trace(go.Scattergl(
                    x=xs_sel,
                    y=ys_sel,
                    text=texts_sel,
                    mode='markers',
                    marker=dict(color="#2CA02C", size=max(14, int(18 * scaling_factor))),  # Responsive highlighted point size
                    name=f"{selected_gene_name} isoforms",
                    showlegend=show_legend,
                    legendgroup='Highlighted Gene'
                ))

                show_legend = False

                # Move the selected gene trace to the front for better hoverability
                volcano_plot_dtu.data = list(volcano_plot_dtu.data[:-1]) + [volcano_plot_dtu.data[-1]]

        # Calculate axis ranges for DTU
        # X-axis: extend at least 0.05 beyond the effect size threshold
        if effect_size == 0:
            # No effect size threshold - use data range with padding
            if not dtu_data.empty:
                data_x_min = dtu_data['estimates'].min()
                data_x_max = dtu_data['estimates'].max()
                dtu_x_min = data_x_min - 0.1
                dtu_x_max = data_x_max + 0.1
            else:
                dtu_x_min = -2.0
                dtu_x_max = 2.0
        else:
            # With effect size threshold
            dtu_x_min = -(effect_size + 0.05)
            dtu_x_max = effect_size + 0.05
            # Adjust if data extends beyond this range
            if not dtu_data.empty:
                data_x_min = dtu_data['estimates'].min()
                data_x_max = dtu_data['estimates'].max()
                dtu_x_min = min(dtu_x_min, data_x_min - 0.1)
                dtu_x_max = max(dtu_x_max, data_x_max + 0.1)
        
        # Make x-axis symmetrical around 0 and add cushion
        dtu_x_range = max(abs(dtu_x_min), abs(dtu_x_max)) + 0.1
        dtu_x_min = -dtu_x_range
        dtu_x_max = dtu_x_range
        
        # Y-axis: go from 0 to at least 0.5 above the significance line
        dtu_y_max = (dtu_sig_line + 0.5) if dtu_sig_line is not False else 5.0
        # Adjust if data extends beyond this range
        if not dtu_data.empty and 'pval' in dtu_data.columns:
            data_y_max = -np.log10(dtu_data['pval'].replace(0, 1e-300).min())
            dtu_y_max = max(dtu_y_max, data_y_max + 0.5)
        
        # Now update the layout, keeping the horizontal line
        volcano_plot_dtu.update_layout(
            xaxis=dict(
                range=[dtu_x_min, dtu_x_max],
                tickfont=dict(size=tick_label_size),
                title=dict(text='Effect Size', font=dict(size=axis_label_size))
            ),
            yaxis=dict(
                range=[-0.2, dtu_y_max],
                tickfont=dict(size=tick_label_size),
                title=dict(text='-log10(p)', font=dict(size=axis_label_size))
            ),
            title={
                'text': f"<b>{count_type.capitalize()} Counts Differential Transcript Usage Volcano Plot {comparison_text}</b><br><span style='font-size:{subtitle_size}px'>{'No q-value threshold (no significant transcripts)' if dtu_sig_line is False else f'Horizontal line shows significance cutoff (q-value < {pvalue_threshold})'}{'' if effect_size == 0 else f' and vertical lines at |Effect Size| = {effect_size}'}</span><br><span style='font-size:{base_font_size}px'>{interpretation_text_dtu}</span>",
                'y':0.96,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': title_size}
            },
            showlegend=True,
            legend=dict(
                itemsizing='constant',
                traceorder='normal',
                font=dict(
                    size=legend_label_size
                ),
                borderwidth=0
            ),
            template='ggplot2'
        )

        # If we need to manually re-add the genome-wide significance line
        if dtu_sig_line is not False:
            # This adds a horizontal line that spans the full x-axis range
            volcano_plot_dtu.add_shape(
                type="line",
                x0=dtu_x_min,  # Start at the left edge of our dynamic range
                y0=dtu_sig_line,
                x1=dtu_x_max,   # End at the right edge of our dynamic range
                y1=dtu_sig_line,
                line=dict(
                    color="black",
                    width=2,
                    dash="dash",  # Make the line dashed
                ),
                layer="above"  # Place the line above the data points
            )
        
        # Add annotation if no entries meet FDR threshold
        if dtu_sig_line is False:
            volcano_plot_dtu.add_annotation(
                text=f"No transcripts meet FDR threshold (q < {pvalue_threshold})",
                xref="paper", yref="paper",
                x=0.5, y=0.98,
                xanchor='center', yanchor='top',
                showarrow=False,
                font=dict(size=int(base_font_size * 0.9), color="red", family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red",
                borderwidth=2,
                borderpad=4
            )

        # Now create the dcc.Graph with our modified figure
        dtu_plot = dcc.Graph(
            id='dtu-graph',
            figure=volcano_plot_dtu,
            style={'height': f'{plot_height}px', 'width': '100%'}  # Responsive height
        )
        
        return dge_plot, dte_plot, dtu_plot, volcano_plot_dge, volcano_plot_dte, volcano_plot_dtu
        
    except Exception as e:
        # Return empty div so spinner continues to show
        empty_div = html.Div()
        # Return empty figures for the dummy graphs when there's an error
        empty_fig = {}
        return empty_div, empty_div, empty_div, empty_fig, empty_fig, empty_fig

# Add download SVG callback
@app.callback(
    Output("download-svg-tab1", "data"),
    [Input("download-button-tab1", "n_clicks")],
    [State('dge-graph', 'figure'),
     State('dte-graph', 'figure'),
     State('dtu-graph', 'figure'),
     State('independent-var-dropdown-tab1', 'value'),
     State('matrix-type-dropdown-tab1', 'value')]
)
def download_plots_as_svg_tab1(n_clicks, dge_fig, dte_fig, dtu_fig, group_comparison, count_type):
    from dash import dcc, no_update
    import tempfile
    import zipfile
    import os
    import base64
    import shutil
    import gc
    import os    
    from dash.dependencies import Input, Output, State
    import plotly.graph_objs as go
    import plotly.io as pio
        
    if n_clicks is None or not n_clicks:
        return no_update
    
    # If no count type is selected, use unique counts by default
    count_type = count_type if count_type else 'unique'
    
    try:
        # Prepare filename based on group comparison
        # Replace spaces and special characters for filename
        comparison_text = group_comparison.replace(" ", "_").replace(".", "_") if group_comparison else "comparison"
        
        # Create a temporary directory for our files
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"Differential_analysis_plots_{comparison_text}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Create a zip file
        with zipfile.ZipFile(zip_path, 'w') as zipf:

            # Extract figures from the plot children if they exist
            if dge_fig:
                dge_svg_name = f"Differential_gene_expression_{comparison_text}.svg"
                
                # Update layout for larger size and wider ratio
                fig = go.Figure(dge_fig)  # Create a copy to avoid modifying the original
                fig.update_layout(
                    width=1200,  # Fixed width
                    height=800,  # Fixed height
                    margin=dict(l=80, r=40, t=120, b=60),  # Increased top margin for title
                    title=dict(
                        font=dict(size=24),  # Larger title
                        y=0.96,  # Move title up slightly
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    ),
                    xaxis=dict(
                        title_font=dict(size=20),
                        tickfont=dict(size=16)
                    ),
                    yaxis=dict(
                        title_font=dict(size=20),
                        tickfont=dict(size=16)
                    )
                )
                
                # 1) write SVG straight to disk
                tmp_svg = os.path.join(temp_dir, dge_svg_name)
                fig.write_image(tmp_svg, format="svg")
                zipf.write(tmp_svg, arcname=dge_svg_name)
                os.remove(tmp_svg)
                pio.kaleido.scope._shutdown_kaleido()
            
            # DTE Plot
            if dte_fig:
                dte_svg_name = f"Differential_transcript_expression_{comparison_text}_{count_type}.svg"
                
                # Update layout for larger size and wider ratio
                fig = go.Figure(dte_fig)  # Create a copy to avoid modifying the original
                fig.update_layout(
                    width=1200,  # Fixed width
                    height=800,  # Fixed height
                    margin=dict(l=80, r=40, t=120, b=60),  # Increased top margin for title
                    title=dict(
                        font=dict(size=24),  # Larger title
                        y=0.96,  # Move title up slightly
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    ),
                    xaxis=dict(
                        title_font=dict(size=20),
                        tickfont=dict(size=16)
                    ),
                    yaxis=dict(
                        title_font=dict(size=20),
                        tickfont=dict(size=16)
                    )
                )
                
                # 1) write SVG straight to disk
                tmp_svg = os.path.join(temp_dir, dte_svg_name)
                fig.write_image(tmp_svg, format="svg")
                zipf.write(tmp_svg, arcname=dte_svg_name)
                os.remove(tmp_svg)
                pio.kaleido.scope._shutdown_kaleido()
            
            # DTU Plot
            if dtu_fig:
                dtu_svg_name = f"Differential_transcript_usage_{comparison_text}_{count_type}.svg"
                
                # Update layout for larger size and wider ratio
                fig = go.Figure(dtu_fig)  # Create a copy to avoid modifying the original
                fig.update_layout(
                    width=1200,  # Fixed width
                    height=800,  # Fixed height
                    margin=dict(l=80, r=40, t=120, b=60),  # Increased top margin for title
                    title=dict(
                        font=dict(size=24),  # Larger title
                        y=0.96,  # Move title up slightly
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    ),
                    xaxis=dict(
                        title_font=dict(size=20),
                        tickfont=dict(size=16)
                    ),
                    yaxis=dict(
                        title_font=dict(size=20),
                        tickfont=dict(size=16)
                    )
                )
                
                # 1) write SVG straight to disk
                tmp_svg = os.path.join(temp_dir, dtu_svg_name)
                fig.write_image(tmp_svg, format="svg")
                zipf.write(tmp_svg, arcname=dtu_svg_name)
                os.remove(tmp_svg)
                pio.kaleido.scope._shutdown_kaleido()
        
        ## Write file
        out = dcc.send_file(zip_path)
            
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        # Return the zip file
        return out
            
    except Exception as e:
        return no_update

# Update the p-value slider to have responsive font size
@app.callback(
    Output('pvalue-slider-tab1', 'marks'),
    [Input('window-dimensions', 'data')]
)
def update_pvalue_slider_marks(dimensions):
    if not dimensions:
        # Default mark style
        return {
            0: {'label': '0.001', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            1: {'label': '0.01', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            2: {'label': '0.05', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            3: {'label': '0.1', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            4: {'label': '0.2', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            5: {'label': '0.3', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}}
        }
    
    # Calculate scaling factors for font sizes
    width = dimensions.get('width', 1200)
    scaling_factor = max(0.5, min(1.2, width / 1920))
    font_size = max(16, int(18 * scaling_factor))
    
    return {
        0: {'label': '0.001', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        1: {'label': '0.01', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        2: {'label': '0.05', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        3: {'label': '0.1', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        4: {'label': '0.2', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        5: {'label': '0.3', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}}
    }

# Update the effect size slider to have responsive font size
@app.callback(
    Output('effect-size-slider-tab1', 'marks'),
    [Input('window-dimensions', 'data')]
)
def update_effect_size_slider_marks(dimensions):
    if not dimensions:
        # Default mark style
        return {
            0: {'label': 'None', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.08: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.18: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.28: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.38: {'label': '0.38', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.48: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.58: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.68: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.78: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.88: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            0.98: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            1.08: {'label': '1.08', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            1.18: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            1.28: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            1.38: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}},
            1.48: {'label': '1.48', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': '18px'}}
        }
    
    # Calculate scaling factors for font sizes
    width = dimensions.get('width', 1200)
    scaling_factor = max(0.5, min(1.2, width / 1920))
    font_size = max(16, int(18 * scaling_factor))
    
    return {
        0: {'label': 'None', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.08: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.18: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.28: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.38: {'label': '0.38', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.48: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.58: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.68: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.78: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.88: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        0.98: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        1.08: {'label': '1.08', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        1.18: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        1.28: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        1.38: {'label': '', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}},
        1.48: {'label': '1.48', 'style': {'color': '#495057', 'font-weight': '500', 'font-size': f'{font_size}px'}}
    }

# Update the effect size slider tooltip to have responsive font size
@app.callback(
    Output('effect-size-slider-tab1', 'tooltip'),
    [Input('window-dimensions', 'data')]
)
def update_effect_size_tooltip(dimensions):
    if not dimensions:
        # Default tooltip style
        return {"placement": "bottom", "always_visible": True, "style": {"font-size": "18px"}}
    
    # Calculate scaling factors for font sizes - make it a bit smaller for the compact layout
    width = dimensions.get('width', 1200)
    scaling_factor = max(0.5, min(1.1, width / 1920))
    font_size = max(14, int(18 * scaling_factor))
    
    return {"placement": "bottom", "always_visible": True, "style": {"font-size": f"{font_size}px"}}

# Update all form labels with the same style
@app.callback(
    [Output("tab1-comparison-label", "style"),
     Output("tab1-matrix-label", "style"),
     Output("tab1-pvalue-label", "style"),
     Output("tab1-effect-label", "style"),
     Output("tab1-gene-label", "style"),
     Output("tab1-export-label", "style")],
    [Input("tab1-form-labels", "style")]
)
def update_form_labels(form_style):
    if not form_style or form_style.get("display") == "none":
        # Default style - font-weight and color are set in className 
        default_style = {"font-weight": "600", "color": "#495057", "font-size": "18px", "margin-bottom": "2px"}
        return default_style, default_style, default_style, default_style, default_style, default_style
    
    # Add margin-bottom to make more compact
    compact_style = dict(form_style)
    compact_style["margin-bottom"] = "2px"
    
    # Return the same style for all labels
    return compact_style, compact_style, compact_style, compact_style, compact_style, compact_style

# Add a callback to adjust the height of the Analysis Controls content card
@app.callback(
    [Output("tab1-controls-card", "style"),
     Output("tab1-controls-card", "className")],
    [Input("window-dimensions", "data")]
)
def update_controls_card_height(dimensions):
    if not dimensions:
        # Default height
        return (
            {"height": "600px", "min-height": "600px", "overflow-y": "auto", "padding": "15px 10px"},
            "d-flex flex-column justify-content-start"  # Start from the top
        )
    
    # Scale the height with the same factor as the plots
    height = int(dimensions['height'] * 0.785)  # Exact same calculation as volcano plots
    
    # Return style with the calculated height and maintain compact layout
    return (
        {
            "height": f"{height}px",        # Same height as volcano plots
            "min-height": "600px",          # Minimum height for small screens
            "overflow-y": "auto",           # Add scrolling if content is too large
            "padding": "15px 10px"          # More padding to distribute content
        },
        "d-flex flex-column justify-content-start"  # Start content from the top
    )
