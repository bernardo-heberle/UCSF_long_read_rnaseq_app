# File: app/tabs/tab4.py
# Contains the layout for Tab 4 - eQTL Explorer

from dash import html, dcc, Input, Output, callback, no_update, State, MATCH, ALL
import dash_bootstrap_components as dbc
from app import app
from app.utils.db_utils import search_rsids, duck_conn, get_rsid_data, get_matrix_dropdown_options, get_gene_density_data, get_total_gene_data_with_metadata
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
import plotly.io as pio
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
import os
import tempfile
import zipfile
import base64
import shutil

# Get the dropdown options
dropdown_options = get_matrix_dropdown_options()
default_table = dropdown_options[0]['value'] if dropdown_options else None

# Load the density plot figure
density_plot_path = os.path.join('app', 'assets', 'figures', 'gene_level_density_plot.json')
with open(density_plot_path, 'r') as f:
    density_fig = pio.from_json(f.read())

# Update figure layout for consistency
density_fig.update_layout(
    template="plotly_white",
    margin=dict(l=50, r=20, t=50, b=50),
    title={
        'text': "Gene Expression Distribution (All expressed genes, mean counts > 0)",
        'y':0.95,
        'x':0.02,
        'xanchor': 'left',
        'yanchor': 'middle',
        'font': {'size': 18, 'weight': 'bold'}
    },
    xaxis_title="Log10 Mean TMM(per million)",
    xaxis=dict(
        title_font=dict(size=16),
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        showticklabels=False,  # Hide y-axis labels
        ticks="",              # Remove tick marks
        title_font=dict(size=16)  # Make y-axis title larger
    )
)

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
    Output('density-plot-tab4', 'figure'),
    [Input('search-input', 'value'),
     Input('search-input', 'options')]
)
def update_density_plot(selected_gene, options):
    if not selected_gene:
        return density_fig
        
    try:
        # Get the gene name from the options
        gene_name = None
        for option in options:
            if option['value'] == selected_gene:
                # Extract gene name from the label (format: "gene_name (gene_id)")
                gene_name = option['label'].split(' (')[0]
                break
        
        if not gene_name:
            gene_name = selected_gene  # Fallback to using gene_id if name not found
            
        # Get the gene's density plot data
        log10_mean_tmm, expression_percentile = get_gene_density_data(selected_gene)
        
        if log10_mean_tmm is not None and expression_percentile is not None:
            # Create a new figure and copy the data from the original
            fig = go.Figure()
            
            # Copy all traces from the original figure
            for trace in density_fig.data:
                fig.add_trace(trace)
            
            # Copy the layout from the original figure
            fig.update_layout(density_fig.layout)
            
            # Add vertical line
            fig.add_vline(
                x=log10_mean_tmm,
                line_dash="dash",
                line_color="black",
                line_width=2,
                y0=0,
                y1=1
            )
            
            # Add percentile label at the top of the line
            percentile = int(round(expression_percentile * 100, 0))
            suffix = "th"
            if percentile % 10 == 1 and percentile != 11:
                suffix = "st"
            elif percentile % 10 == 2 and percentile != 12:
                suffix = "nd" 
            elif percentile % 10 == 3 and percentile != 13:
                suffix = "rd"
            
            fig.add_annotation(
                x=log10_mean_tmm,
                y=1,
                text=f"{gene_name} ({percentile}{suffix} percentile)",
                showarrow=False,
                font=dict(size=14, color="black", weight="bold"),
                xref="x",
                yref="paper",
                xanchor="right" if log10_mean_tmm > 2.5 else "left",  # Anchor text based on x position
                align="right" if log10_mean_tmm > 2.5 else "left",    # Align text based on x position
                yanchor="middle"
            )
            return fig
            
    except Exception as e:
        print(f"Error updating density plot: {e}")
        
    return density_fig

@app.callback(
    [Output('rsid-genotype-plot-container', 'children'),
     Output('genotype-plot-store', 'data')],
    [Input('rsid-search-input', 'value'),
     Input('matrix-table-dropdown', 'value'),
     Input('search-input', 'value'),
     Input('metadata-checklist', 'value'),
     Input('log-transform-option', 'value'),
     Input('plot-style-option', 'value'),
     Input('window-dimensions', 'data'),
     Input('isoform-range-slider', 'value')]
)
def update_rsid_genotype_plot(selected_rsid, selected_table, selected_gene, selected_metadata, log_transform, plot_style, window_dimensions, isoform_range):
    if not selected_rsid or not selected_gene:
        # Return an empty figure wrapped in dcc.Graph component with tab2-like styling
        return html.Div(
            html.P("Please select a gene and RSID to display data", 
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
            # Return an error message with clean styling
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
            # No metadata selected, use genotype as default
            expression_hue = "genotype"
            group_col = "genotype"
            # Filter out nulls
            df = df.filter(~pl.col(group_col).is_null())
        else:
            # Create a combined column that includes both genotype and selected metadata
            combined_col_name = "combined_metadata"
            
            # Start with genotype
            combined_expr = pl.col("genotype").cast(pl.Utf8)
            
            # Add the metadata columns with separator
            for col_name in selected_metadata:
                combined_expr = combined_expr + pl.lit(" | ") + pl.col(col_name).cast(pl.Utf8)
            
            # Create the new column
            df = df.with_columns([
                combined_expr.alias(combined_col_name)
            ])
            
            # Filter out rows with missing data in any of the selected metadata columns
            for col_name in selected_metadata:
                df = df.filter(~pl.col(col_name).is_null())
                
            expression_hue = combined_col_name
            group_col = combined_col_name

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
                    boxgap=0.15,
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
        
        # Main title annotation:
        fig.add_annotation(
            x=0.5,
            y=1.12,
            xref='paper',
            yref='paper',
            text=f"{gene_name} ({actual_gene_id})",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(size=26 * scaling_factor)
        )

        # Subtitle annotation:
        fig.add_annotation(
            x=0.5,
            y=1.08,  # Slightly lower than the main title
            xref='paper',
            yref='paper',
            text=f"Region: chr{chromosome}({strand}):{min_start}-{max_end}",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(size=18 * scaling_factor)
        )

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
        
        # Return an error message with clean styling
        return html.Div(
            [
                html.P(f"Error creating visualization: {str(e)}", 
                      style={"color": "#dc3545", "margin": "0 0 10px 0"}),
                html.Pre(trace, 
                        style={"color": "#dc3545", "font-size": "0.8rem", "margin": 0})
            ],
            style={
                "height": "100%",
                "width": "100%",
                "display": "flex",
                "flex-direction": "column",
                "justify-content": "center",
                "align-items": "center",
                "min-height": "500px",
                "background-color": "#f8f9fa",
                "border-radius": "6px",
                "padding": "20px",
                "text-align": "center"
            }
        ), None

@app.callback(
    Output('gene-level-plot-tab4', 'figure'),
    [Input('search-input', 'value'),
     Input('search-input', 'options'),
     Input('metadata-checklist', 'value'),
     Input('log-transform-option', 'value'),
     Input('plot-style-option', 'value'),
     Input('rsid-search-input', 'value')]
)
def update_gene_level_plot(selected_gene, options, selected_metadata, log_transform, plot_style, selected_rsid):
    if not selected_gene or not selected_rsid:
        return go.Figure()
        
    try:
        # Get the gene name from the options
        gene_name = None
        for option in options:
            if option['value'] == selected_gene:
                # Extract gene name from the label (format: "gene_name (gene_id)")
                gene_name = option['label'].split(' (')[0]
                break
        
        if not gene_name:
            gene_name = selected_gene  # Fallback to using gene_id if name not found
            
        # Get RSID data and rename the sample ID column for joining
        df_rsid = get_rsid_data(selected_rsid, with_polars=True)
        if df_rsid.is_empty():
            return go.Figure()
            
        df_rsid = df_rsid.rename({"sample_and_flowcell_id": "sample_id"})
        
        # Get the gene's data
        df = get_total_gene_data_with_metadata(selected_gene, with_polars=True)
        df = df.join(df_rsid, on="sample_id", how="left")
        
        if df is None or len(df) == 0:
            return go.Figure()
            
        # Handle metadata selection for expression_hue
        if selected_metadata is None or len(selected_metadata) == 0:
            # No metadata selected, use genotype as default
            expression_hue = "genotype"
            group_col = "genotype"
            # Filter out nulls
            df = df.filter(~pl.col(group_col).is_null())
        else:
            # Create a combined column that includes both genotype and selected metadata
            combined_col_name = "combined_metadata"
            
            # Start with genotype
            combined_expr = pl.col("genotype").cast(pl.Utf8)
            
            # Add the metadata columns with separator
            for col_name in selected_metadata:
                combined_expr = combined_expr + pl.lit(" | ") + pl.col(col_name).cast(pl.Utf8)
            
            # Create the new column
            df = df.with_columns([
                combined_expr.alias(combined_col_name)
            ])
            
            # Filter out rows with missing data in any of the selected metadata columns
            for col_name in selected_metadata:
                df = df.filter(~pl.col(col_name).is_null())
                
            expression_hue = combined_col_name
            group_col = combined_col_name

        # Apply log transformation if selected
        if log_transform:
            # Create a new column with log transform
            df = df.with_columns([
                (pl.col("cpm_normalized_tmm").add(1).log10()).alias("log_cpm_normalized_tmm")
            ])
            value_col = "log_cpm_normalized_tmm"
            axis_title = "Log TMM(per million)"
        else:
            value_col = "cpm_normalized_tmm"
            axis_title = "TMM(per million)"

        
        # Get unique values from the group column, sorted for consistent order
        unique_hue_values = df[group_col].unique().sort(descending=False).to_list()

        
        # Generate colors for each group - use the same approach as the transcript plot
        from app.utils.plotly_utils import get_n_colors
        custom_colors_list = get_n_colors(len(unique_hue_values), 'Plotly_r')
        color_map = {val: color for val, color in zip(unique_hue_values, custom_colors_list)}
        
        # Convert to pandas for easier manipulation with plotly
        pdf = df.to_pandas()
        
        # Create a new figure
        fig = go.Figure()
        
        # Add traces based on plot style
        for group in unique_hue_values[::-1]:
            group_data = pdf[pdf[group_col] == group]
            
            if plot_style == "boxplot":
                # Add box plot
                fig.add_trace(go.Box(
                    x=group_data[value_col] if group_data[value_col].count() > 0 else [0],
                    name=str(group),
                    boxpoints='all',  # Show all points
                    jitter=0.3,       # Add jitter to points
                    pointpos=0,       # Position points at the center of box
                    orientation='h',  # Horizontal orientation
                    marker=dict(
                        color='black',  # Black points
                        size=4          # Smaller points
                    ),
                    line=dict(
                        color='black',  # Black outline
                        width=1         # Thin line
                    ),
                    fillcolor=color_map[group],         # Box fill color
                    opacity=1,                        # Semi-transparent
                    boxmean=True                        # Show mean line
                ))
            else:  # violin plot
                # Add violin plot
                fig.add_trace(go.Violin(
                    x=group_data[value_col] if group_data[value_col].count() > 0 else [0],
                    name=str(group),
                    points='all',     # Show all points
                    pointpos=0,       # Position points at the center
                    orientation='h',  # Horizontal orientation
                    jitter=0.3,       # Add jitter to points
                    marker=dict(
                        color='black',  # Black points
                        size=4          # Smaller points
                    ),
                    line=dict(
                        color='black',  # Black outline
                        width=1         # Thin line
                    ),
                    fillcolor=color_map[group],         # Violin fill color
                    opacity=1,                        # Semi-transparent
                    box_visible=False,                   # Show box plot inside violin
                    spanmode='hard'                     # Hard boundaries for violin
                ))

        # Update layout for consistency
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=50, r=20, t=50, b=50),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,  # Position legend outside the plot area on the right
                font=dict(size=14),
                traceorder="reversed"  # Standard order for legend, already reversed by our list order
            ),
            title={
                'text': f"Total Gene Expression: {gene_name}",
                'y':0.95,
                'x':0.02,
                'xanchor': 'left',
                'yanchor': 'middle',
                'font': {'size': 18, 'weight': 'bold'}
            },
            xaxis_title=axis_title,
            xaxis=dict(
                title_font=dict(size=16),
                tickfont=dict(size=16)  # Increased tick font size
            ),
            yaxis=dict(
                showticklabels=False,  # Hide y-axis labels
                title=None  # Remove y-axis title
            )
        )
        
        return fig
            
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Error updating gene level plot: {e}")
        print(trace)
        return go.Figure()

@app.callback(
    Output("download-svg-tab4", "data"),
    [Input("download-button-tab4", "n_clicks")],
    [State('density-plot-tab4', 'figure'),
     State('gene-level-plot-tab4', 'figure'),
     State('genotype-plot-store', 'data'),
     State('search-input', 'value'),
     State('rsid-search-input', 'value')]
)
def download_plots_as_svg_tab4(n_clicks, density_fig, gene_level_fig, genotype_fig, selected_gene, selected_rsid):
    if n_clicks is None or not n_clicks or selected_gene is None or selected_rsid is None:
        return no_update
    
    try:
        # Get the gene name for the filename
        gene_info = duck_conn.execute("""
            SELECT gene_id, gene_name 
            FROM transcript_annotation 
            WHERE gene_id = ?
            LIMIT 1
        """, [selected_gene]).fetchone()
        
        gene_name = gene_info[1] if gene_info else selected_gene
        
        # Create a temporary directory for our files
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"{gene_name}_{selected_rsid}_RNA_isoform_expression_plots.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Create a zip file
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Export the density plot
            if density_fig:
                print("Creating density plot SVG")
                density_svg_name = f"{gene_name}_{selected_rsid}_density_distribution_plot.svg"
                real_fig = go.Figure(density_fig)
                # Update layout for larger size and wider ratio
                real_fig.update_layout(
                    width=1200,  # Increased width
                    height=800,  # Increased height
                    margin=dict(l=80, r=40, t=80, b=60),  # Adjusted margins
                    title=dict(
                        font=dict(size=24),  # Larger title
                        y=0.95,
                        x=0.02
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
                density_svg = real_fig.to_image(format="svg").decode('utf-8')
                zipf.writestr(density_svg_name, density_svg)
                print(f"Added density plot to zip: {density_svg_name}")
                
            # Export the gene expression plot    
            if gene_level_fig:
                print("Creating gene expression plot SVG")
                gene_expr_svg_name = f"{gene_name}_{selected_rsid}_gene_expression_plot.svg"
                real_fig = go.Figure(gene_level_fig)
                # Update layout for larger size and wider ratio
                real_fig.update_layout(
                    width=1200,  # Increased width
                    height=800,  # Increased height
                    margin=dict(l=80, r=40, t=80, b=60),  # Adjusted margins
                    title=dict(
                        font=dict(size=24),  # Larger title
                        y=0.95,
                        x=0.02
                    ),
                    xaxis=dict(
                        title_font=dict(size=20),
                        tickfont=dict(size=16)
                    ),
                    yaxis=dict(
                        title_font=dict(size=20),
                        tickfont=dict(size=16)
                    ),
                    legend=dict(
                        font=dict(size=16),
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                gene_expr_svg = real_fig.to_image(format="svg").decode('utf-8')
                zipf.writestr(gene_expr_svg_name, gene_expr_svg)
                print(f"Added gene expression plot to zip: {gene_expr_svg_name}")
                
            # Export the genotype plot
            if genotype_fig:
                print("Creating genotype plot SVG")
                genotype_svg_name = f"{gene_name}_{selected_rsid}_genotype_plot.svg"
                try:
                    real_fig = go.Figure(genotype_fig)
                    genotype_svg = real_fig.to_image(format="svg").decode('utf-8')
                    zipf.writestr(genotype_svg_name, genotype_svg)
                    print(f"Successfully added genotype plot to zip: {genotype_svg_name}")
                except Exception as genotype_error:
                    print(f"Error creating genotype SVG: {genotype_error}")
                    # Create placeholder instead
                    placeholder_fig = go.Figure()
                    placeholder_fig.add_annotation(
                        text=f"Genotype Plot for {gene_name} and {selected_rsid} (could not render)",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=20)
                    )
                    placeholder_svg = placeholder_fig.to_image(format="svg").decode('utf-8')
                    zipf.writestr(genotype_svg_name, placeholder_svg)
            else:
                # Create a placeholder if genotype fig is not available
                print("No genotype figure found, creating placeholder")
                genotype_svg_name = f"{gene_name}_{selected_rsid}_RNA_isoform_plot.svg"
                placeholder_fig = go.Figure()
                placeholder_fig.add_annotation(
                    text=f"Genotype Plot for {gene_name} and {selected_rsid}",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                placeholder_svg = placeholder_fig.to_image(format="svg").decode('utf-8')
                zipf.writestr(genotype_svg_name, placeholder_svg)
            
        # Read the zip file
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
            
        # Clean up temp directory
        shutil.rmtree(temp_dir)
            
        # Return the zip file
        return dict(
            content=base64.b64encode(zip_data).decode(),
            filename=zip_filename,
            type="application/zip",
            base64=True
        )
            
    except Exception as e:
        import traceback
        print(f"Error creating zip archive: {e}")
        print(traceback.format_exc())
        return no_update

def layout():
    return dbc.Container([
        # Add Download component for the SVG files
        dcc.Download(id="download-svg-tab4"),
        
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
                    ], width=2, id="tab4-gene-search-col"),
                    dbc.Col([
                        create_section_header("Search RSID:"),
                        create_rsid_search_dropdown()
                    ], width=2, id="tab4-rsid-search-col"),
                    dbc.Col([
                        create_section_header("Data Matrix:"),
                        create_matrix_dropdown(dropdown_options, default_table)
                    ], width=2, id="tab4-matrix-col"),
                    dbc.Col([
                        create_section_header("Data Transformation:"),
                        create_content_card([
                            html.Div([
                                create_radio_items(
                                    id="log-transform-option",
                                    options=[
                                        {"label": "Original Values", "value": False},
                                        {"label": "Log Transform (log10(x+1))", "value": True}
                                    ],
                                    value=False,
                                    inline=True
                                )
                            ], className="radio-group-container dbc")
                        ])
                    ], width=3, id="tab4-transform-col"),
                    dbc.Col([
                        create_section_header("Plot Style:"),
                        create_content_card([
                            html.Div([
                                create_radio_items(
                                    id="plot-style-option",
                                    options=[
                                        {"label": "Box Plot", "value": "boxplot"},
                                        {"label": "Violin Plot", "value": "violin"}
                                    ],
                                    value="boxplot",
                                    inline=True
                                )
                            ], className="radio-group-container dbc")
                        ])
                    ], width=3, id="tab4-plot-style-col"),
                ], className="mb-4 dbc", id="tab4-row1"),

                # Second row - RSID plot
                dbc.Row([
                    dbc.Col([
                        create_content_card(
                            dbc.Spinner(
                                html.Div([
                                    # Matrix content div
                                    html.Div(
                                        id='rsid-genotype-plot-container',
                                        style={
                                            "background-color": "#ffffff",
                                            "padding": "10px",
                                            "border-radius": "5px",
                                            "border": "1px solid rgba(0, 0, 0, 0.1)",
                                            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                                            "width": "100%",
                                            "height": "100%",
                                            "min-height": "400px"
                                        }
                                    )
                                ]),
                                color="primary",
                                type="grow",
                                spinner_style={"width": "3rem", "height": "3rem"}
                            )
                        )
                    ], width=12, id="tab4-col2-1"),
                ], 
                className="mb-4 dbc",
                id="tab4-row2",
                style={"height": "90vh"}  # Make the row take up 90% of viewport height
                ),

                # Third row - three columns
                dbc.Row([
                    dbc.Col([
                        create_section_header("Show data separated by:"),
                        create_content_card([
                            html.Div([
                                create_checklist(
                                    id="metadata-checklist",
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
                    ], width=4, id="tab4-col3-1"),
                    dbc.Col([
                        create_section_header("Isoform Range"),
                        create_content_card([
                            html.Div([
                                html.H6(
                                    "Isoform Rank Selection",
                                    style={
                                        "marginBottom": "15px",
                                        "color": "#495057"
                                    }
                                ),
                                dcc.RangeSlider(
                                    id='isoform-range-slider',
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=[1, 5],
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True,
                                             "style": {"color": "black", "font-weight": "bold", "font-size": "14px"}},
                                    className="custom-range-slider",
                                    allowCross=True
                                ),
                                html.Small(
                                    "Select range of top expressed isoforms by rank",
                                    style={
                                        "color": "#666666",
                                        "display": "block",
                                        "marginTop": "8px",
                                        "textAlign": "center"
                                    }
                                )
                            ], style={"padding": "10px"})
                        ])
                    ], width=4, id="tab4-col3-2"),
                    dbc.Col([
                        create_section_header("Download Plots"),
                        create_content_card([
                            html.Div([
                                html.P(
                                    "Export current plots as SVG vector graphics.",
                                    style={
                                        "marginBottom": "15px",
                                        "color": "#495057"
                                    }
                                ),
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-download me-2"),
                                        "Download SVG"
                                    ],
                                    id="download-button-tab4",
                                    color="primary",
                                    className="w-100 mb-3",
                                    disabled=False
                                ),
                                html.Small(
                                    "Generates high-quality vector graphics for publications",
                                    style={
                                        "color": "#666666",
                                        "display": "block",
                                        "marginTop": "8px",
                                        "textAlign": "center"
                                    }
                                )
                            ], style={"padding": "10px"})
                        ])
                    ], width=4, id="tab4-col3-3"),
                ], className="mb-4 dbc", id="tab4-row3"),

                # Fourth row - two columns
                dbc.Row([
                    dbc.Col([
                        create_section_header(""),
                        create_content_card([
                            dcc.Graph(
                                id='density-plot-tab4',
                                figure=density_fig,
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': False,
                                    'modeBarButtonsToRemove': ['autoScale2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}
                            )
                        ])
                    ], width=6, id="tab4-col4-1"),
                    dbc.Col([
                        create_section_header(""),
                        create_content_card([
                            dcc.Graph(
                                id='gene-level-plot-tab4',
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': False,
                                    'modeBarButtonsToRemove': ['autoScale2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}
                            )
                        ])
                    ], width=6, id="tab4-col4-2"),
                ], className="mb-4 dbc", id="tab4-row4"),
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
     Output("tab4-matrix-col", "width"),
     Output("tab4-transform-col", "width"),
     Output("tab4-plot-style-col", "width"),
     Output("tab4-row2", "className"),
     Output("tab4-col2-1", "width"),
     Output("tab4-row3", "className"),
     Output("tab4-col3-1", "width"),
     Output("tab4-col3-2", "width"),
     Output("tab4-col3-3", "width"),
     Output("tab4-row4", "className"),
     Output("tab4-col4-1", "width"),
     Output("tab4-col4-2", "width")],
    [Input("window-dimensions", "data")]
)
def update_tab4_responsiveness(dimensions):
    if not dimensions:
        # Default styles if no dimensions available
        return (
            {"max-width": "98%", "margin": "0 auto", "padding": "10px"},
            "mb-4 dbc", 2, 2, 2, 3, 3,
            "mb-4 dbc", 12,
            "mb-4 dbc", 4, 4, 4,
            "mb-4 dbc", 6, 6
        )
    
    width = dimensions.get('width', 1200)
    
    # Base styles
    container_style = {"max-width": "98%", "margin": "0 auto", "padding": "10px"}
    row1_class = "mb-4 dbc"
    gene_search_col_width = 2
    rsid_search_col_width = 2
    matrix_col_width = 2
    transform_col_width = 3
    plot_style_col_width = 3
    
    row2_class = "mb-4 dbc"
    col2_1_width = 12
    
    row3_class = "mb-4 dbc"
    col3_1_width = 4
    col3_2_width = 4
    col3_3_width = 4
    
    row4_class = "mb-4 dbc"
    col4_1_width = 6
    col4_2_width = 6
    
    # Responsive adjustments based on width
    if width < 576:  # Extra small devices
        container_style.update({"padding": "5px", "max-width": "100%"})
        row1_class = "mb-2 dbc flex-column"
        gene_search_col_width = 12
        rsid_search_col_width = 12
        matrix_col_width = 12
        transform_col_width = 12
        plot_style_col_width = 12
        
        row2_class = "mb-2 dbc"
        
        row3_class = "mb-2 dbc flex-column"
        col3_1_width = 12
        col3_2_width = 12
        col3_3_width = 12
        
        row4_class = "mb-2 dbc flex-column"
        col4_1_width = 12
        col4_2_width = 12
        
    elif width < 768:  # Small devices
        container_style.update({"padding": "8px"})
        row1_class = "mb-3 dbc"
        gene_search_col_width = 6
        rsid_search_col_width = 6
        matrix_col_width = 6
        transform_col_width = 6
        plot_style_col_width = 6
        
        row3_class = "mb-3 dbc flex-column"
        col3_1_width = 12
        col3_2_width = 12
        col3_3_width = 12
        
        row4_class = "mb-3 dbc flex-column"
        col4_1_width = 12
        col4_2_width = 12
        
    elif width < 992:  # Medium devices
        container_style.update({"padding": "10px"})
        
    return (
        container_style,
        row1_class, gene_search_col_width, rsid_search_col_width, matrix_col_width, transform_col_width, plot_style_col_width,
        row2_class, col2_1_width,
        row3_class, col3_1_width, col3_2_width, col3_3_width,
        row4_class, col4_1_width, col4_2_width
    ) 