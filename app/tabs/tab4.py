# File: app/tabs/tab4.py
# Contains the layout for Tab 4 - eQTL Explorer

from dash import html, dcc, Input, Output, callback, no_update, State, MATCH, ALL
import dash_bootstrap_components as dbc
from app import app
from app.utils.db_utils import search_rsids, duck_conn, get_rsid_data, get_matrix_dropdown_options
from app.utils.ui_components import (
    create_gene_search_dropdown,
    create_rsid_search_dropdown,
    create_section_header,
    create_content_card,
    create_radio_items,
    create_checklist,
    create_matrix_dropdown
)
import plotly.graph_objects as go
import polars as pl
from dash.dependencies import Input, Output, State
from dash import callback_context, html, dcc
from app import app
from polars import col, lit
import dash_bootstrap_components as dbc
from app.utils.db_utils import get_gene_data_with_metadata, duck_conn, POLARS_AVAILABLE
from app.utils.polars_utils import order_transcripts_by_expression
from app.utils.plotly_utils import get_n_colors
import RNApysoforms as RNApy
from dash import ClientsideFunction
import plotly.graph_objects as go

# Get the dropdown options
dropdown_options = get_matrix_dropdown_options()
default_table = dropdown_options[0]['value'] if dropdown_options else None

# Store the last valid search options to prevent them from disappearing
last_valid_rsid_options = []
last_rsid_search_value = None  # Store the last search value

@app.callback(
    Output('rsid-search-input', 'options'),
    [Input('rsid-search-input', 'search_value'),
     Input('rsid-search-input', 'value')]
)
def update_rsid_search_options(search_value, selected_value):
    global last_valid_rsid_options, last_rsid_search_value
    
    # If we have a selected value but no search, return the last options
    # This keeps the dropdown populated after selection
    if selected_value and not search_value:
        # Make sure the selected value is in the options
        selected_in_options = any(opt.get('value') == selected_value for opt in last_valid_rsid_options)
        if not selected_in_options:
            # If we have a newly selected value, we need to add it to the options
            # First check if the RSID exists in the database
            try:
                rsid_result = duck_conn.execute("""
                    SELECT rsid 
                    FROM genotyping 
                    WHERE rsid = ?
                    LIMIT 1
                """, [selected_value]).fetchone()
                
                if rsid_result:
                    # Add this RSID to the options
                    rsid = rsid_result[0]
                    option = {
                        'label': rsid,
                        'value': rsid
                    }
                    last_valid_rsid_options = [option]  # Just show the current selection
            except Exception as e:
                # If we can't get the details, just use the raw ID
                if selected_value:
                    last_valid_rsid_options = [{
                        'label': selected_value,
                        'value': selected_value
                    }]
        
        return last_valid_rsid_options
        
    # If no search value or too short, return latest options
    if not search_value or len(search_value) < 2:
        return last_valid_rsid_options
        
    # Process the search and return results
    results = search_rsids(search_value, last_rsid_search_value)
    
    # Store the results and search value for future reference
    if results:
        last_valid_rsid_options = results
        last_rsid_search_value = search_value
        
    return results

@app.callback(
    [Output('rsid-genotype-plot', 'figure'),
     Output('genotype-plot-store', 'data')],
    [Input('rsid-search-input', 'value'),
     Input('matrix-table-dropdown', 'value'),
     Input('search-input', 'value'),
     Input('metadata-checklist-rsid', 'value'),
     Input('log-transform-option', 'value'),
     Input('plot-style-option-rsid', 'value'),
     Input('window-dimensions', 'data'),
     Input('isoform-range-slider', 'value')]
)
def update_rsid_genotype_plot(selected_rsid, selected_table, selected_gene, selected_metadata, log_transform, plot_style, window_dimensions, isoform_range):
    if not selected_rsid or not selected_gene:
        # Return an empty figure instead of HTML div when no gene is selected
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Please select a gene to display data",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="#666666", size=14)
        )
        empty_fig.update_layout(
            paper_bgcolor="#f8f9fa",
            plot_bgcolor="#f8f9fa",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return empty_fig, empty_fig
    
    ## Create scaling factor
    scaling_factor = window_dimensions["width"]/2540
    
    # Ensure minimum dimensions for usability
    if window_dimensions["width"] > window_dimensions["height"]:
        plot_width = window_dimensions['width'] * 0.8
        plot_height = window_dimensions['height'] * 0.8
    else:
        plot_width = window_dimensions['width'] * 0.7
        plot_height = window_dimensions['height'] * 0.8
    
    try:
        # Get the RSID data
        df_rsid = get_rsid_data(selected_rsid, with_polars=True)
        df_rsid = df_rsid.rename({"sample_and_flowcell_id": "sample_id"})
        
        # Set default genotype column name
        genotype_column = "genotype"

        # Get gene info
        gene_info = duck_conn.execute("""
            SELECT gene_id, gene_name 
            FROM transcript_annotation 
            WHERE gene_id = ?
            LIMIT 1
        """, [selected_gene]).fetchone()
        
        if not gene_info:
            # Return an empty figure with error message
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Gene not found in the annotation table",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#666666", size=14)
            )
            empty_fig.update_layout(
                paper_bgcolor="#f8f9fa",
                plot_bgcolor="#f8f9fa",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            return empty_fig, empty_fig
        
        actual_gene_id, gene_name = gene_info
        
        # Get data with metadata - remove the limit to get all samples
        expression = get_gene_data_with_metadata(actual_gene_id, selected_table, with_polars=True, limit=None)

        ## Join expression data with RSID data
        df = df_rsid.join(expression, on="sample_id", how="left")

        # Get annotation data for this gene
        annotation_query = """
            SELECT * 
            FROM transcript_annotation 
            WHERE gene_id = ?
        """
        annotation = duck_conn.execute(annotation_query, [actual_gene_id]).pl()

        # Extract gene annotation details for plot title and subtitle        # Extract gene annotation details for plot title and subtitle
        chromosome = annotation["seqnames"][0] if "seqnames" in annotation.columns else "Unknown"
        strand = annotation["strand"][0] if "strand" in annotation.columns else "?"
        min_start = annotation["start"].min() if "start" in annotation.columns else "Unknown"
        max_end = annotation["end"].max() if "end" in annotation.columns else "Unknown"
        
        #############################################################
        # Process the annotation data
        annotation = RNApy.shorten_gaps(annotation)


        # Convert 1-based slider values to 0-based indices for the function
        # Add 1 to the upper bound to make it inclusive
        zero_based_range = [isoform_range[0] - 1, isoform_range[1]]
        # Use df (the joined dataframe) instead of expression
        df, annotation = order_transcripts_by_expression(
            annotation_df=annotation, 
            expression_df=df, 
            expression_column="cpm_normalized_tmm",
            top_n=zero_based_range  # Use the converted range
        )
        
        # Handle metadata selection for expression_hue
        if selected_metadata is None or len(selected_metadata) == 0:
            # No metadata selected, use genotype directly
            expression_hue = "genotype"
        else:
            # If multiple columns are selected, create a combined column using polars methods
            combined_col_name = "combined_metadata"
            
            # Start with genotype column
            combined_expr = pl.col("genotype").cast(pl.Utf8)
            
            # Add the rest of the columns with separator
            for col_name in selected_metadata:
                combined_expr = combined_expr + pl.lit(" | ") + pl.col(col_name).cast(pl.Utf8)

            # Create the new column
            df = df.with_columns([
                combined_expr.alias(combined_col_name)
            ])
            
            # Filter out rows with missing data in any of the selected metadata columns
            for col_name in (selected_metadata + ["genotype"]):
                df = df.filter(~pl.col(col_name).is_null())

            # Get unique values from combined metadata to create custom color map
            unique_hue_values = df[combined_col_name].unique().sort().to_list()

            # If we have any unique values, ensure they all have colors
            if len(unique_hue_values) > 0:
                # We'll add this parameter to make_traces later
                custom_colors = True
                expression_hue = combined_col_name
            else:
                # No valid data points with this combination
                expression_hue = None

        # Apply log transformation if selected
        if log_transform:
            # Create copies of the data columns with log transform
            # We need to add 1 to avoid log(0) issues
            for col in ["counts", "cpm_normalized_tmm"]:
                if col in df.columns:
                    import numpy as np
                    # Create a new column with log transform
                    log_col = f"log_{col}"
                    df = df.with_columns([
                        (pl.col(col).add(1).log10()).alias(log_col)
                    ])
            
            # Use the log-transformed columns in the plot
            expression_columns = ["log_counts", "log_cpm_normalized_tmm", "relative_abundance"]
        else:
            # Use original columns
            expression_columns = ["counts", "cpm_normalized_tmm", "relative_abundance"]
        

        # Update the trace creation to use the correct columns
        trace_params = {
            "annotation": annotation,
            "expression_matrix": df,  # Use df instead of expression
            "x_start": "rescaled_start",
            "x_end": "rescaled_end",
            "y": "transcript_id",
            "annotation_hue": "transcript_biotype",
            "hover_start": "start",
            "hover_end": "end",
            "expression_columns": expression_columns,
            "marker_size": 5*scaling_factor,
            "arrow_size": 12*scaling_factor,
            "expression_plot_style": plot_style  # Add the plot style parameter
        }
        
        ## Create appropriate color palette for the expression_hue
        if expression_hue is not None:
            unique_hue_values = df[expression_hue].unique().sort().to_list()
            if len(unique_hue_values) > 0:
                # Use a continuous pastel rainbow colorscale for softer visual progression
                # This provides a smoother transition between pastel colors while maintaining differentiation
                custom_colors_list = get_n_colors(len(unique_hue_values), 'Plotly_r')
                # Other bright options could be 'Rainbow', 'Jet', or 'HSV' for maximum visibility
                color_map = {val: color for val, color in zip(unique_hue_values, custom_colors_list)}
                trace_params["expression_color_map"] = color_map
        # Only add expression_hue if it's not None
        if expression_hue is not None:
            trace_params["expression_hue"] = expression_hue
            df = df.sort(by=expression_hue, descending=True)
            trace_params["expression_matrix"] = df

        traces = RNApy.make_traces(**trace_params)
        
        # Create appropriate subplot titles based on transformation
        if log_transform:
            subplot_titles = ["Transcript Structure", "Log Counts", "Log TMM(per million)", "Relative Abundance(%)"]
        else:
            subplot_titles = ["Transcript Structure", "Counts", "TMM(per million)", "Relative Abundance(%)"]
                
        # Use the dynamic dimensions for your plot
        fig = RNApy.make_plot(traces=traces, 
                    subplot_titles=subplot_titles,
                    boxgap=0.1,
                    boxgroupgap=0.1, 
                    width=plot_width, 
                    height=plot_height,
                    legend_font_size=20*scaling_factor,
                    yaxis_font_size=20*scaling_factor, 
                    xaxis_font_size=20*scaling_factor,
                    subplot_title_font_size=24*scaling_factor, 
                    template="ggplot2", 
                    hover_font_size=14*scaling_factor,
                    legend_title_font_size=20*scaling_factor,
                    column_widths=[0.4, 0.2, 0.2, 0.2])
        
        # Calculate the expanded x-axis range for the first subplot
        # Get the min and max x values from the annotation data
        x_min = annotation["rescaled_start"].min() if "rescaled_start" in annotation.columns else 0
        x_max = annotation["rescaled_end"].max() if "rescaled_end" in annotation.columns else 1
        
        # Add padding to both sides (10% of the total range)
        padding = (x_max - x_min) * 0.05
        new_x_min = x_min - padding
        new_x_max = x_max + padding
        
        # Update the x-axis limits for the first subplot and remove ticks
        fig.update_xaxes(
            range=[new_x_min, new_x_max],
            showticklabels=False,  # Hide tick labels
            showline=False,        # Hide axis line
            zeroline=False,        # Hide zero line
            ticks="",             # Remove tick marks
            row=1, col=1
        )
        
        # Set the relative abundance subplot (row 2, col 2) y-axis range and ticks
        fig.update_xaxes(
            range=[-0.5, 100.5],  # Fixed range from -0.5 to 100.5
            tickvals=[0, 20, 40, 60, 80, 100],  # Tick marks at intervals of 20
            row=1, col=4
        )
        
        # Update layout with gene info title and subtitle
        fig.update_layout(
            autosize=True,
            title={
                'text': f"{gene_name} ({actual_gene_id})<br><sub>Region: chr{chromosome}({strand}):{min_start}-{max_end}<sub>",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 26*scaling_factor}
            })
            
        # Update subplot titles separately - the subplot_titles parameter should be a list, not a dict
        for i, annotation in enumerate(fig['layout']['annotations']):
            if i < len(subplot_titles):  # Only modify the subplot title annotations
                if i == 0:
                    annotation["x"] = 0
                elif i == 1:
                    annotation["x"] = 0.395
                elif i ==2:
                    annotation["x"] = 0.602
                else:
                    annotation["x"] = 0.812
                annotation["xanchor"] = "left"
            
        # Create a copy of the figure for the dcc.Store
        # Directly store the figure for download access
        
        return dcc.Graph(
            figure=fig,
            style={
                "height": "100%",  # Take full height of parent
                "width": "100%",   # Take full width of parent
                "min-height": "0"  # Allow container to shrink
            },
            config={
                "responsive": True,
                "displayModeBar": True,
                "scrollZoom": False,
                "modeBarButtonsToRemove": ["autoScale2d"],
                "displaylogo": False
            }
        ), fig
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        
        # Try to get column names to help with debugging
        column_info = ""
        try:
            if 'expression' in locals() and expression is not None:
                column_info = f"Available columns: {', '.join(expression.columns)}"
        except:
            column_info = "Could not retrieve column names"
            
        # Create an error figure instead of HTML div
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            x=0.5, y=0.8,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="#dc3545", size=14)
        )
        
        if column_info:
            error_fig.add_annotation(
                text=column_info,
                x=0.5, y=0.6,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#dc3545", size=12)
            )
            
        error_fig.add_annotation(
            text=trace,
            x=0.5, y=0.2,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="#dc3545", size=10)
        )
        
        error_fig.update_layout(
            paper_bgcolor="#f8f9fa",
            plot_bgcolor="#f8f9fa",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        return error_fig, error_fig

def layout():
    return dbc.Container([
        # Add a Store to hold the genotype plot figure
        dcc.Store(id="genotype-plot-store"),
        
        # Add a Store for the isoform plot
        dcc.Store(id="isoform-plot-store"),
        
        # Add a hidden gene-plot-container for the callback output
        html.Div(id="gene-plot-container", style={"display": "none"}),
        
        # Add these components that are required by the callback but were missing
        html.Div([
            dcc.RangeSlider(
                id='isoform-range-slider',
                min=1,
                max=10,
                step=1,
                value=[1, 5],
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            dcc.RadioItems(
                id='log-transform-option',
                options=[
                    {"label": "Original Values", "value": False},
                    {"label": "Log Transform (log10(x+1))", "value": True}
                ],
                value=False
            ),
            # Add this hidden component to satisfy the callback in callbacks.py
            dcc.Checklist(
                id='metadata-checklist',
                options=[],
                value=[]
            ),
            # Add plot-style-option component
            dcc.RadioItems(
                id='plot-style-option',
                options=[
                    {"label": "Box Plot", "value": "boxplot"},
                    {"label": "Violin Plot", "value": "violin"}
                ],
                value="boxplot"
            )
        ], style={"display": "none"}),  # Hide these components since they're just for the callback
        
        dbc.Card([
            dbc.CardBody([
                # First row - columns for search, options
                dbc.Row([
                    dbc.Col([
                        create_section_header("Search Gene:"),
                        create_gene_search_dropdown()
                    ], width=3, id="tab4-gene-search-col"),
                    dbc.Col([
                        create_section_header("Search RSID:"),
                        create_rsid_search_dropdown()
                    ], width=3, id="tab4-rsid-search-col"),
                    dbc.Col([
                        create_section_header("Data Matrix:"),
                        create_matrix_dropdown(dropdown_options, default_table)
                    ], width=3, id="tab4-matrix-col"),
                    dbc.Col([
                        create_section_header("Show data separated by:"),
                        create_content_card([
                            html.Div([
                                create_checklist(
                                    id="metadata-checklist-rsid",
                                    options=[
                                        {"label": "Braak Stage", "value": "braak_tangle_score"},
                                        {"label": "Sex", "value": "sex"},
                                        {"label": "AD Status", "value": "ebbert_ad_status"},
                                        {"label": "APOE Genotype", "value": "apoe"}
                                    ],
                                    value=[]
                                )
                            ])
                        ])
                    ], width=3, id="tab4-metadata-col"),
                ], className="mb-4 dbc", id="tab4-row1"),

                # Option row
                dbc.Row([
                    dbc.Col([
                        create_section_header("Plot Style:"),
                        create_content_card([
                            html.Div([
                                create_radio_items(
                                    id="plot-style-option-rsid",
                                    options=[
                                        {"label": "Box Plot", "value": "boxplot"},
                                        {"label": "Violin Plot", "value": "violin"}
                                    ],
                                    value="boxplot",
                                    inline=True
                                )
                            ], className="radio-group-container dbc")
                        ])
                    ], width=12, id="tab4-plot-style-col")
                ], className="mb-4 dbc", id="tab4-options-row"),

                # Second row - RSID plot
                dbc.Row([
                    dbc.Col([
                        create_section_header("Genotype Distribution"),
                        create_content_card([
                            dcc.Graph(
                                id='rsid-genotype-plot',
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': False,
                                    'modeBarButtonsToRemove': ['autoScale2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}
                            )
                        ])
                    ], width=12, id="tab4-col2-1"),
                ], className="mb-4 dbc", id="tab4-row2"),

                # Third row - two columns
                dbc.Row([
                    dbc.Col([
                        create_section_header("eQTL Effect"),
                        create_content_card([
                            dcc.Graph(
                                id='eqtl-plot',
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': False,
                                    'modeBarButtonsToRemove': ['autoScale2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}
                            )
                        ])
                    ], width=6, id="tab4-col3-1"),
                    dbc.Col([
                        create_section_header("Genomic Context"),
                        create_content_card([
                            dcc.Graph(
                                id='genomic-context-plot',
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': False,
                                    'modeBarButtonsToRemove': ['autoScale2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}
                            )
                        ])
                    ], width=6, id="tab4-col3-2"),
                ], className="mb-4 dbc", id="tab4-row3"),
            ], id="tab4-card-body")
        ],
        id="tab4-card",
        style={
            "background-color": "#ffffff",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "6px",
            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
        })
    ], 
    fluid=True,  # Makes the container full-width
    id="tab4-container",
    style={
        "max-width": "98%",  # Use 98% of the viewport width
        "margin": "0 auto",  # Center the container
        "padding": "10px"    # Add some padding
    })

@callback(
    [Output("tab4-container", "style"),
     Output("tab4-row1", "className"),
     Output("tab4-gene-search-col", "width"),
     Output("tab4-rsid-search-col", "width"),
     Output("tab4-metadata-col", "width"),
     Output("tab4-matrix-col", "width"),
     Output("tab4-options-row", "className"),
     Output("tab4-plot-style-col", "width"),
     Output("tab4-row2", "className"),
     Output("tab4-col2-1", "width"),
     Output("tab4-row3", "className"),
     Output("tab4-col3-1", "width"),
     Output("tab4-col3-2", "width")],
    [Input("window-dimensions", "data")]
)
def update_tab4_responsiveness(dimensions):
    if not dimensions:
        # Default styles if no dimensions available
        return (
            {"max-width": "98%", "margin": "0 auto", "padding": "10px"},
            "mb-4 dbc", 3, 3, 3, 3,
            "mb-4 dbc", 12,
            "mb-4 dbc", 12,
            "mb-4 dbc", 6, 6
        )
    
    width = dimensions.get('width', 1200)
    
    # Base styles
    container_style = {"max-width": "98%", "margin": "0 auto", "padding": "10px"}
    row1_class = "mb-4 dbc"
    gene_search_col_width = 3
    rsid_search_col_width = 3
    metadata_col_width = 3
    matrix_col_width = 3
    
    options_row_class = "mb-4 dbc"
    plot_style_col_width = 12
    
    row2_class = "mb-4 dbc"
    col2_1_width = 12
    
    row3_class = "mb-4 dbc"
    col3_1_width = 6
    col3_2_width = 6
    
    # Responsive adjustments based on width
    if width < 576:  # Extra small devices
        container_style.update({"padding": "5px", "max-width": "100%"})
        row1_class = "mb-2 dbc flex-column"
        gene_search_col_width = 12
        rsid_search_col_width = 12
        metadata_col_width = 12
        matrix_col_width = 12
        
        options_row_class = "mb-2 dbc"
        
        row2_class = "mb-2 dbc"
        
        row3_class = "mb-2 dbc flex-column"
        col3_1_width = 12
        col3_2_width = 12
        
    elif width < 768:  # Small devices
        container_style.update({"padding": "8px"})
        row1_class = "mb-3 dbc"
        gene_search_col_width = 6
        rsid_search_col_width = 6
        metadata_col_width = 6
        matrix_col_width = 6
        
        options_row_class = "mb-3 dbc"
        
        row3_class = "mb-3 dbc flex-column"
        col3_1_width = 12
        col3_2_width = 12
        
    elif width < 992:  # Medium devices
        container_style.update({"padding": "10px"})
        
    return (
        container_style,
        row1_class, gene_search_col_width, rsid_search_col_width, metadata_col_width, matrix_col_width,
        options_row_class, plot_style_col_width,
        row2_class, col2_1_width,
        row3_class, col3_1_width, col3_2_width
    ) 