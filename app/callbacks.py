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

# Add this new callback
@app.callback(
    Output('matrix-content', 'children'),
    [Input('matrix-table-dropdown', 'value'),
     Input('search-input', 'value')]
)
def update_matrix_content(selected_table, selected_gene):
    if not selected_table:
        return html.Div(
            html.P("Please select a matrix table", 
                  style={"color": "#666666", "margin": 0}),
            style={
                "height": "100%",
                "width": "100%",
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "min-height": "500px",  # Set a minimum height to ensure good visibility
                "background-color": "#f8f9fa",
                "border-radius": "6px"
            }
        )
    
    if not selected_gene:
        return html.Div()  # Return empty div when no gene is selected
    
    try:
        # Verify the gene exists using our compatibility wrapper
        gene_info = duck_conn.execute("""
            SELECT gene_id, gene_name 
            FROM transcript_annotation 
            WHERE gene_id = ?
            LIMIT 1
        """, [selected_gene]).fetchone()
        
        if not gene_info:
            return html.P(f"Gene ID '{selected_gene}' not found in the annotation table", 
                        style={"color": "#666666"})
        
        actual_gene_id, gene_name = gene_info
        
        # First get a row count - this helps assess the data size 
        # and provides immediate feedback to the user
        row_count = duck_conn.execute("""
            SELECT COUNT(*) 
            FROM {}
            WHERE gene_id = ?
        """.format(selected_table), [actual_gene_id]).fetchone()[0]
        
        if row_count == 0:
            return html.P(f"No data found for gene {gene_name} ({actual_gene_id})", 
                        style={"color": "#666666"})
        
        # Get the matrix data - limit to first 50 rows for better performance
        max_rows = 50  # Limit the number of rows for display
        data_loaded = False
        using_polars = False
        has_metadata = False
        err_msg = None
        
        # Always try with Polars first - it's faster and more memory efficient
        try:
            df = get_gene_data_with_metadata(actual_gene_id, selected_table, with_polars=True, limit=max_rows)
            using_polars = True
            data_loaded = True
        except Exception as e:
            err_msg = str(e)
            try:
                # This will use a direct query, still returning Polars
                fallback_df = duck_conn.execute("""
                    SELECT * 
                    FROM {}
                    WHERE gene_id = ?
                    LIMIT {}
                """.format(selected_table, max_rows), [actual_gene_id]).pl()
                df = fallback_df
                data_loaded = True
                using_polars = True
                err_msg = "Used direct query fallback (no metadata join)"
            except Exception as final_e:
                raise Exception(f"All data retrieval methods failed. Last error: {str(final_e)}")
        
        # Return an empty div to let the gene-plot-container update with the data
        return html.Div()
            
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return html.Div([
            html.P(f"Error loading table: {str(e)}", 
                 style={"color": "#dc3545"}),  # Error in red
            html.Pre(trace, style={"color": "#dc3545", "font-size": "0.8rem"})
        ])
      
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

# Add a new callback for the plot
@app.callback(
    [Output('gene-plot-container', 'children'),
     Output('isoform-plot-store', 'data')],
    [Input('matrix-table-dropdown', 'value'),
     Input('search-input', 'value'),
     Input('metadata-checklist', 'value'),
     Input('log-transform-option', 'value'),
     Input('plot-style-option', 'value'),
     Input('window-dimensions', 'data'),
     Input('isoform-range-slider', 'value')]
)
def update_gene_plot(selected_table, selected_gene, selected_metadata, log_transform, plot_style, window_dimensions, isoform_range):
    
    # Return message if no gene is selected
    if not selected_gene:
        return html.Div(
            html.P("Please select a gene to display data", 
                  style={"color": "#666666", "margin": 0}),
            style={
                "height": "100%",
                "width": "100%",
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "min-height": "500px",
                "background-color": "#f8f9fa",
                "border-radius": "6px"
            }
        ), None
    
   #Create scaling factor
    scaling_factor = window_dimensions["width"]/2540

    # Ensure minimum dimensions for usability
    if window_dimensions["width"] > window_dimensions["height"]:
        plot_width = window_dimensions['width'] * 0.8
        plot_height = window_dimensions['height'] * 0.8
    else:
        plot_width = window_dimensions['width'] * 0.7
        plot_height = window_dimensions['height'] * 0.8
    

    try:
        # Get gene info
        gene_info = duck_conn.execute("""
            SELECT gene_id, gene_name 
            FROM transcript_annotation 
            WHERE gene_id = ?
            LIMIT 1
        """, [selected_gene]).fetchone()
        
        if not gene_info:
            return html.Div(
                html.P("Gene not found in the annotation table", 
                      style={"color": "#666666", "margin": 0}),
                style={
                    "height": "100%",
                    "width": "100%",
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                    "min-height": "500px",
                    "background-color": "#f8f9fa",
                    "border-radius": "6px"
                }
            ), None
        
        actual_gene_id, gene_name = gene_info
        
        # Get data with metadata - remove the limit to get all samples
        expression = get_gene_data_with_metadata(actual_gene_id, selected_table, with_polars=True, limit=None)
        
        # Get annotation data for this gene
        annotation_query = """
            SELECT * 
            FROM transcript_annotation 
            WHERE gene_id = ?
        """
        annotation = duck_conn.execute(annotation_query, [actual_gene_id]).pl()
        
        # Extract gene annotation details for plot title and subtitle
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
        expression, annotation = order_transcripts_by_expression(
            annotation_df=annotation, 
            expression_df=expression, 
            expression_column="cpm_normalized_tmm",
            top_n=zero_based_range  # Use the converted range
        )

        # Handle metadata selection for expression_hue
        if selected_metadata is None or len(selected_metadata) == 0:
            # No metadata selected, don't use any expression_hue
            expression_hue = None
        elif len(selected_metadata) == 1:
            # If only one metadata column is selected, use it directly
            expression_hue = selected_metadata[0]
        else:
            # If multiple columns are selected, create a combined column using polars methods
            combined_col_name = "combined_metadata"
            
            # Convert selected columns to strings and combine them
            # First create an expression that converts each column to string
            
            
            # Start with the first column
            combined_expr = pl.col(selected_metadata[0]).cast(pl.Utf8)
            
            # Add the rest of the columns with separator
            for col_name in selected_metadata[1:]:
                combined_expr = combined_expr + pl.lit(" | ") + pl.col(col_name).cast(pl.Utf8)
            
            # Create the new column
            expression = expression.with_columns([
                combined_expr.alias(combined_col_name)
            ])
            
            # Filter out rows with missing data in any of the selected metadata columns
            for col_name in selected_metadata:
                expression = expression.filter(~pl.col(col_name).is_null())
                
            expression_hue = combined_col_name
            
            # Get unique values from combined metadata to create custom color map
            unique_hue_values = expression[combined_col_name].unique().sort().to_list()
            
            # If we have any unique values, ensure they all have colors
            if len(unique_hue_values) > 0:
                # We'll add this parameter to make_traces later
                custom_colors = True
            else:
                # No valid data points with this combination
                expression_hue = None

        # Apply log transformation if selected
        if log_transform:
            # Create copies of the data columns with log transform
            # We need to add 1 to avoid log(0) issues
            for col in ["counts", "cpm_normalized_tmm"]:
                if col in expression.columns:
                    import numpy as np
                    # Create a new column with log transform
                    log_col = f"log_{col}"
                    expression = expression.with_columns([
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
            "expression_matrix": expression,
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
            unique_hue_values = expression[expression_hue].unique().sort().to_list()
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
            expression = expression.sort(by=expression_hue, descending=True)
            trace_params["expression_matrix"] = expression

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
            
        return html.Div([
            html.P(f"Error creating visualization: {str(e)}", 
                  style={"color": "#dc3545"}),
            html.P(column_info, 
                  style={"color": "#dc3545", "font-size": "0.9rem"}),
            html.Pre(trace, style={"color": "#dc3545", "font-size": "0.8rem"})
        ]), None

@app.callback(
    [Output('isoform-range-slider', 'max'),
     Output('isoform-range-slider', 'marks'),
     Output('isoform-range-slider', 'value')],
    [Input('search-input', 'value')]
)
def update_slider_range(selected_gene):
    if not selected_gene:
        # Default range when no gene is selected
        marks = {i: str(i) for i in range(1, 11)}
        return 10, marks, [1, 5]
    
    try:
        # Query to get the number of transcripts for this gene
        transcript_count = duck_conn.execute("""
            SELECT COUNT(DISTINCT transcript_id) 
            FROM transcript_annotation 
            WHERE gene_id = ?
        """, [selected_gene]).fetchone()[0]
        
        if not transcript_count:
            # Fallback if no transcripts found
            marks = {i: str(i) for i in range(1, 11)}
            return 10, marks, [1, 5]
        
        # Create marks for the actual number of transcripts
        marks = {i: str(i) for i in range(1, transcript_count + 1)}
        
        # Ensure the range is at least 1 isoform
        # If current range is [1,1], keep it as is
        # Otherwise, ensure the range is at least 1 isoform
        new_value = [1, min(5, transcript_count)]
        
        return transcript_count, marks, new_value
        
    except Exception as e:
        # Fallback to default values
        marks = {i: str(i) for i in range(1, 11)}
        return 10, marks, [1, 5]
