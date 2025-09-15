# File: app/tabs/tab3.py
# Contains the layout for Tab 3 - Gene Expression Explorer

from dash import html, dcc, Input, Output, callback, no_update, State, MATCH, ALL
import dash_bootstrap_components as dbc
from app import app
from app.utils.db_utils import search_genes, duck_conn, get_rsid_data, get_matrix_dropdown_options, get_gene_density_data, get_total_gene_data_with_metadata, get_gene_data_with_metadata
from app.utils.ui_components import (
    create_gene_search_dropdown,
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
from app.utils.db_utils import duck_conn, POLARS_AVAILABLE
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
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import spearmanr, pearsonr # Import correlation functions
import numpy as np # For handling potential NaN/Inf

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
last_valid_options = []
last_search_value = None  # Store the last search value

# Default APP gene information
DEFAULT_APP_GENE_INDEX = 9758
DEFAULT_APP_GENE_NAME = "APP"
DEFAULT_APP_GENE_ID = "ENSG00000142192"

@app.callback(
    Output('density-plot-tab3', 'figure'),
    [Input('search-input-tab3', 'value'),
     Input('search-input-tab3', 'options'),
     Input('window-dimensions', 'data')]
)
def update_density_plot(selected_gene, options, window_dimensions):
    # Default window dimensions if not available yet
    if not window_dimensions:
        window_dimensions = {'width': 1200, 'height': 800}

    # Create scaling factor and base font size
    scaling_factor = max(0.5, window_dimensions["width"] / 2540) # Ensure minimum scaling
    base_font_size = 20 * scaling_factor # Base size for scaling - Increased from 16
    title_size = base_font_size * 1.125
    axis_label_size = base_font_size
    tick_label_size = base_font_size * 0.875
    annotation_size = base_font_size * 0.875

    # Always create a figure object to apply scaling
    fig = go.Figure()
    for trace in density_fig.data:
        fig.add_trace(trace)
    fig.update_layout(density_fig.layout)

    # Apply scaled fonts to the base layout
    fig.update_layout(
        title={
            'font': {'size': title_size, 'weight': 'bold'}
        },
        xaxis_title_font_size=axis_label_size,
        xaxis_tickfont_size=tick_label_size,
        yaxis_title_font_size=axis_label_size,
        # yaxis ticks are hidden, no need to scale
        margin=dict(l=50, r=20, t=60, b=50) # Slightly larger top margin for title
    )

    if not selected_gene:
        # Set default to APP gene
        selected_gene = DEFAULT_APP_GENE_INDEX

    try:
        gene_name = None
        for option in options:
            if option['value'] == selected_gene:
                gene_name = option['label'].split(' (')[0]
                break
        if not gene_name:
            gene_name = selected_gene

        log10_mean_tmm, expression_percentile = get_gene_density_data(selected_gene)

        if log10_mean_tmm is not None and expression_percentile is not None:
            # Add vertical line (styling is fixed size)
            fig.add_vline(
                x=log10_mean_tmm,
                line_dash="dash",
                line_color="black",
                line_width=2,
                y0=0,
                y1=1
            )

            # Calculate percentile suffix
            percentile = int(round(expression_percentile * 100, 0))
            suffix = "th"
            if percentile % 10 == 1 and percentile != 11: suffix = "st"
            elif percentile % 10 == 2 and percentile != 12: suffix = "nd"
            elif percentile % 10 == 3 and percentile != 13: suffix = "rd"

            # Add annotation with scaled font
            fig.add_annotation(
                x=log10_mean_tmm,
                y=1,
                text=f"{gene_name} ({percentile}{suffix} percentile)",
                showarrow=False,
                font=dict(size=annotation_size, color="black", weight="bold"), # Use scaled annotation_size
                xref="x",
                yref="paper",
                xanchor="right" if log10_mean_tmm > 2.5 else "left",
                align="right" if log10_mean_tmm > 2.5 else "left",
                yanchor="middle"
            )
            return fig

    except Exception as e:
        print(f"Error updating density plot: {e}")

    # Return the base scaled figure if error or no data for gene
    return fig

@app.callback(
    Output('search-input-tab3', 'options'),
    [Input('search-input-tab3', 'search_value'),
     Input('search-input-tab3', 'value')]
)
def update_search_options(search_value, selected_value):
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
                    SELECT gene_index, gene_id, gene_name 
                    FROM gene_and_transcript_index_table 
                    WHERE gene_index = ?
                    GROUP BY gene_index, gene_id, gene_name
                    LIMIT 1
                """, [selected_value]).fetchone()
                
                if gene_result:
                    # Add this gene to the options
                    gene_index, gene_id, gene_name = gene_result
                    option = {
                        'label': f"{gene_name} ({gene_id})",
                        'value': gene_index
                    }
                    last_valid_options = [option]  # Just show the current selection
            except Exception as e:
                print(f"Error getting gene details: {e}")
                # If we can't get the details, just use the raw ID
                if selected_value:
                    last_valid_options = [{
                        'label': f"{selected_value}",
                        'value': selected_value
                    }]
        
        return last_valid_options
    
    # If no search value and no selected value, return APP gene as default
    if not search_value and not selected_value:
        # Don't override if there's already an initial value set
        return last_valid_options
        
    # If no search value or too short, return latest options
    if not search_value or len(search_value) < 2:
        return last_valid_options
        
    # Process the search and return results
    results = search_genes(search_value, last_search_value)
    
    # Store the results and search value for future reference
    if results:
        last_valid_options = results
        last_search_value = search_value
        
    return results

@app.callback(
    Output('gene-level-plot-tab3', 'figure'),
    [Input('search-input-tab3', 'value'),
     Input('search-input-tab3', 'options'),
     Input('metadata-checklist-tab3', 'value'),
     Input('trendline-type-option-tab3', 'value'),
     Input('correlation-var-tab3', 'value'),
     Input('window-dimensions', 'data'),
     Input('correlation-type-tab3', 'value')]
)
def update_gene_level_plot(selected_gene, options, selected_metadata, trendline_type, correlation_var, window_dimensions, correlation_type):
    if selected_gene is None:
        # Set default to APP gene
        selected_gene = DEFAULT_APP_GENE_INDEX
        
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
            
        # Get the gene's data
        df = get_total_gene_data_with_metadata(selected_gene, with_polars=True)
        
        if df is None or len(df) == 0:
            return go.Figure()
            
        # Handle metadata selection for gene-level plot
        if selected_metadata is None or len(selected_metadata) == 0:
            # No metadata selected, use gray for all points
            df = df.with_columns([
                pl.lit("All Samples").alias("_group")
            ])
            group_col = "_group"
            color_map = {"All Samples": "#808080"}  # Set to gray
        elif len(selected_metadata) == 1:
            # If only one metadata column is selected, use it directly
            group_col = selected_metadata[0]
            # Filter out nulls
            df = df.filter(~pl.col(group_col).is_null())
            # Generate colors for each group
            unique_hue_values = df[group_col].unique().sort(descending=False).to_list()
            custom_colors_list = get_n_colors(len(unique_hue_values), 'Plotly_r')
            color_map = {val: color for val, color in zip(unique_hue_values, custom_colors_list)}
        else:
            # If multiple columns are selected, create a combined column
            combined_col_name = "combined_metadata"
            
            # Start with the first column
            combined_expr = pl.col(selected_metadata[0]).cast(pl.Utf8)
            
            # Add the rest of the columns with separator
            for col_name in selected_metadata[1:]:
                combined_expr = combined_expr + pl.lit(" | ") + pl.col(col_name).cast(pl.Utf8)
            
            # Create the new column
            df = df.with_columns([
                combined_expr.alias(combined_col_name)
            ])
            
            # Filter out rows with missing data in any of the selected metadata columns
            for col_name in selected_metadata:
                df = df.filter(~pl.col(col_name).is_null())
                
            group_col = combined_col_name
            
            # Generate colors for combined groups
            unique_hue_values = df[group_col].unique().sort(descending=False).to_list()
            custom_colors_list = get_n_colors(len(unique_hue_values), 'Plotly_r')
            color_map = {val: color for val, color in zip(unique_hue_values, custom_colors_list)}

        # Default window dimensions if not available yet
        if not window_dimensions:
            window_dimensions = {'width': 1200, 'height': 800}

        # Create scaling factor
        scaling_factor = window_dimensions["width"]/2540
        base_font_size = 20 * scaling_factor

        # Get the label for the correlation variable
        var_labels = {
            "expired_age": "Age (Years)",
            "pmi": "Post-Mortem Interval (Hours)",
            "brain_weight_grams": "Brain Weight (Grams)",
            "rin": "RNA Integrity (RIN)",
            "tin_median": "TIN Median",
            "all_reads": "All Reads (Count)",
            "trimmed_&_processed_pass_reads": "Trimmed & Filtered Reads (Count)",
            "filtered_primary_alignments_(mapq)": "Primary Alignments MAPQ ≥ 10 (Count)",
            "mapping_rate_-_mapq_>=_10_(%)": "Mapping Rate MAPQ ≥ 10 (%)",
            "mapping_rate_(%)": "Mapping Rate (%)",
            "n50_bam": "N50 (Nucleotides)",
            "median_read_length_bam": "Median Read Length (Nucleotides)",
            "astrocyte_proportion": "Astrocyte Proportion",
            "oligodendrocyte_proportion": "Oligodendrocyte Proportion",
            "neuronal_proportion": "Neuronal Proportion",
            "microglia-pvm_proportion": "Microglia-PVM Proportion",
            "plaquef": "Plaque Density - Frontal Cortex",
            "plaquetotal": "Total Plaque Density",
            "tanglef": "Tangle Density - Frontal Cortex",
            "tangletotal": "Total Tangle Density",
            "demential_age": "Dementia Onset Age (Years)",
            "dementia_years": "Years Living with Dementia (Years)"
        }
        x_label = var_labels.get(correlation_var, correlation_var)

        # Convert to pandas for plotly and correlation
        pdf = df.to_pandas()

        # --- Calculate Correlation --- (Updated section)
        correlation_texts = []
        min_points_for_corr = 3

        # Check if metadata is used for grouping
        if group_col != "_group" and group_col in pdf.columns:
            # Calculate correlation per subgroup
            unique_subgroups = sorted(list(pdf[group_col].unique()))
            for subgroup_name in unique_subgroups:
                subgroup_data = pdf[pdf[group_col] == subgroup_name]
                x_data_sub = subgroup_data[correlation_var].replace([np.inf, -np.inf], np.nan).dropna()
                y_data_sub = subgroup_data["cpm_normalized_tmm"].replace([np.inf, -np.inf], np.nan).dropna()
                common_indices_sub = x_data_sub.index.intersection(y_data_sub.index)
                x_data_sub = x_data_sub.loc[common_indices_sub]
                y_data_sub = y_data_sub.loc[common_indices_sub]
                abbreviated_name = str(subgroup_name) # Abbreviate name
                abbreviated_name = abbreviated_name.replace(" | ", "+")
                abbreviated_name = abbreviated_name.replace("Male", "M")
                abbreviated_name = abbreviated_name.replace("Female", "F")
                abbreviated_name = abbreviated_name.replace("Control", "CT")
                abbreviated_name = abbreviated_name.replace("Stage", "")
                abbreviated_name = abbreviated_name.replace("-", "/")
                abbreviated_name = abbreviated_name.replace("E", "")

                # Set correlation symbol based on type
                if correlation_type == 'spearman':
                    corr_symbol = "ρ"
                else: # Pearson or default
                    corr_symbol = "r"
                
                # Initialize correlation text with NA
                subgroup_corr_text = f"{abbreviated_name}: {corr_symbol}=NA" 

                if len(x_data_sub) >= min_points_for_corr:
                    try:
                        # Calculate correlation
                        if correlation_type == 'spearman':
                            corr, _ = spearmanr(x_data_sub, y_data_sub)
                        else: # Pearson or default
                            corr, _ = pearsonr(x_data_sub, y_data_sub)
                            
                        # Only update if correlation is a valid number
                        if not np.isnan(corr):
                            subgroup_corr_text = f"{abbreviated_name}: {corr_symbol}={corr:.2f}"
                    except Exception as corr_e_sub:
                        print(f"Could not calculate correlation for subgroup {subgroup_name}: {corr_e_sub}")
                
                # Always add the correlation text (either with value or NA)
                correlation_texts.append(subgroup_corr_text)
                
            correlation_text = " | ".join(correlation_texts)
        else:
            # Calculate overall correlation if no subgroups
            x_data = pdf[correlation_var].replace([np.inf, -np.inf], np.nan).dropna()
            y_data = pdf["cpm_normalized_tmm"].replace([np.inf, -np.inf], np.nan).dropna()
            common_indices = x_data.index.intersection(y_data.index)
            x_data = x_data.loc[common_indices]
            y_data = y_data.loc[common_indices]

            # Set correlation symbol based on type
            if correlation_type == 'spearman':
                corr_symbol = "ρ"
            else: # Pearson or default
                corr_symbol = "r"
            
            # Initialize with NA
            correlation_text = f"{corr_symbol}=NA"

            if len(x_data) >= min_points_for_corr:
                try:
                    # Calculate correlation
                    if correlation_type == 'spearman':
                        corr, _ = spearmanr(x_data, y_data)
                    else: # Pearson or default
                        corr, _ = pearsonr(x_data, y_data)
                        
                    # Only update if correlation is a valid number
                    if not np.isnan(corr):
                        correlation_text = f"{corr_symbol}={corr:.2f}"
                except Exception as corr_e:
                    print(f"Could not calculate overall correlation: {corr_e}")
        # --- End Calculate Correlation ---

        # Prepare category orders for legend matching correlation annotations
        category_orders_dict = {}
        if group_col != "_group":
            # Use the same sorted list used for correlation calculation
            if 'unique_subgroups' in locals():
                 category_orders_dict[group_col] = unique_subgroups
            else: # Fallback just in case
                 category_orders_dict[group_col] = sorted(list(pdf[group_col].unique()))

        # Create scatter plot
        gene_scatter = px.scatter(pdf,
                                x=correlation_var,
                                y="cpm_normalized_tmm",
                                color=group_col,
                                color_discrete_map=color_map,
                                trendline="ols" if trendline_type == "linear" else "lowess",
                                # Add category_orders to control legend order
                                category_orders=category_orders_dict if category_orders_dict else None,
                                template="ggplot2")

        # Update marker opacity to make points more transparent
        gene_scatter.update_traces(marker=dict(opacity=0.5))

        # Clip LOWESS trendline at y=0 if necessary
        if trendline_type == 'lowess':
            for trace in gene_scatter.data:
                if trace.mode == 'lines':
                    if trace.y is not None:
                        y_data = list(trace.y)
                        clipped_y = [max(0, y_val) for y_val in y_data if y_val is not None]
                        if len(clipped_y) == len(y_data):
                             trace.y = tuple(clipped_y)

        # Define Y-axis label text
        y_label_text = "TMM (per million)"

        # Update layout - Align with isoform scatter plot styling
        gene_scatter.update_layout(
            template="ggplot2",
            showlegend=len(selected_metadata) > 0 if selected_metadata else False,  # Show legend only if metadata selected
            legend=dict(
                title=None,  # Remove legend title
                font=dict(size=base_font_size * 0.8), # Match isoform plot legend font size
                x=1.05, # Match isoform plot legend position
                xanchor='left',
                y=1, # Match isoform plot legend position
                yanchor='top' # Match isoform plot legend position
            ),
            margin=dict(l=80, r=20, t=60, b=60), # Adjusted margin similar to isoform plot
            font=dict(size=base_font_size),
            title={
                'text': f"Gene Level Scatter Plot: {gene_name}",
                'x': 0.01, # Slightly offset from left edge
                'y': 0.98, # Position near top
                'xanchor': 'left',
                'yanchor': 'top', # Anchor to top
                'font': {'size': base_font_size * 1.2, 'weight': 'bold'}
            },
            # Add correlation annotation
            annotations=list(gene_scatter.layout.annotations) + [
                go.layout.Annotation(
                    text=correlation_text, # Use combined/overall text
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0, # Position top-left (as per user edits)
                    y=1.05, # Position top-left (as per user edits)
                    font=dict(size=base_font_size * 0.6, color="black") # Smaller font (as per user edits)
                )
            ]
        )

        # Update axes - Ensure titles are blank as labels are handled by annotations
        gene_scatter.update_xaxes(
            title_text=x_label,
            title_font=dict(size=base_font_size, color="black"),
            tickfont=dict(size=base_font_size * 0.9)
        )
        gene_scatter.update_yaxes(
            title_text=y_label_text,
            title_font=dict(size=base_font_size, color="black"),
            tickfont=dict(size=base_font_size * 0.9)
        )

        return gene_scatter
            
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Error updating gene level plot: {e}")
        print(trace)
        return go.Figure()

@app.callback(
    Output("download-svg-tab3", "data"),
    [Input("download-button-tab3", "n_clicks")],
    [State('density-plot-tab3', 'figure'),
     State('gene-level-plot-tab3', 'figure'),
     State('isoform-plot-store-tab3', 'data'),
     State('search-input-tab3', 'value'),
     State('scatter-plot-tab3', 'figure'),
     State('metadata-checklist-tab3', 'value'),
     State('matrix-table-dropdown-tab3', 'value')],
)
def download_plots_as_svg_tab3(n_clicks, density_fig, gene_level_fig, isoform_fig, selected_gene, scatter_fig, selected_metadata, count_type):
 
    import plotly.io as pio

    if n_clicks is None or not n_clicks or selected_gene is None:
        return no_update
    
    # If no count type is selected, use total counts by default
    count_type = count_type if count_type else 'total'
    
    try:
        # Get the gene name for the filename
        gene_info = duck_conn.execute("""
            SELECT g.gene_index, g.gene_name, g.gene_id
            FROM gene_and_transcript_index_table g
            WHERE g.gene_index = ?
            GROUP BY g.gene_index, g.gene_name, g.gene_id
            LIMIT 1
        """, [selected_gene]).fetchone()
        
        gene_name = gene_info[1] if gene_info else selected_gene
        
        # Create a temporary directory for our files
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"{gene_name}_RNA_isoform_correlation_plots.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Create a zip file
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Export the density plot
            if density_fig:
                density_svg_name = f"{gene_name}_density_distribution_plot.svg"
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
                
            # Export the gene expression plot    
            if gene_level_fig:
                gene_expr_svg_name = f"{gene_name}_gene_scatter_plot.svg"
                # Extract axis labels from annotations before modifying for SVG
                original_fig = go.Figure(gene_level_fig)
                svg_x_label = None
                svg_y_label = None
                if original_fig.layout.annotations:
                    for annotation in original_fig.layout.annotations:
                        if annotation.textangle == -90:
                            svg_y_label = annotation.text
                        elif annotation.y < 0:
                            svg_x_label = annotation.text
                
                real_fig = go.Figure(gene_level_fig)
                # Update layout for larger size and wider ratio
                real_fig.update_layout(
                    width=1200,  # Increased width
                    height=800,  # Increased height
                    font=dict(size=16),  # Base font size for SVG
                    margin=dict(l=110, r=40, t=80, b=80),  # Adjusted margins, larger bottom for x-axis title
                    title=dict(
                        font=dict(size=24),  # Larger title
                    ),
                    # Set standard axis titles for SVG
                    xaxis_title=svg_x_label if svg_x_label else "",
                    yaxis_title=svg_y_label if svg_y_label else "",
                    xaxis=dict(
                        title_font=dict(size=20), # Ensure title font size is set
                        tickfont=dict(size=16)
                    ),
                    yaxis=dict(
                        title_font=dict(size=20), # Ensure title font size is set
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
                # Remove axis label annotations and increase correlation annotation font size for SVG
                new_annotations = []
                for annotation in real_fig.layout.annotations:
                    # Increase font size for correlation annotation
                    if annotation.xref == 'paper' and annotation.yref == 'paper' and annotation.x == 0 and annotation.y > 1:
                        annotation.font.size = 18 # Increase font size for SVG
                        new_annotations.append(annotation)
                    # Keep other annotations if necessary (e.g., trendline info), but skip axis labels
                    elif not (annotation.textangle == -90 or annotation.y < 0):
                        new_annotations.append(annotation)
                
                real_fig.layout.annotations = new_annotations
                
                gene_expr_svg = real_fig.to_image(format="svg").decode('utf-8')
                zipf.writestr(gene_expr_svg_name, gene_expr_svg)
                
            # Export the Isoform Scatter plot if available
            if scatter_fig:
                scatter_svg_name = f"{gene_name}_isoform_scatter_plot_{count_type}.svg"
                try:
                    real_fig = go.Figure(scatter_fig)
                    # Calculate number of selected metadata items for positioning adjustments
                    number_of_selected_metadata = len(selected_metadata) if selected_metadata else 0

                    # Apply SVG formatting similar to gene level plot
                    real_fig.update_layout(
                        width=1200,
                        height=800,
                        font=dict(size=16),
                        margin=dict(l=100, r=40, t=80, b=80), # Adjusted margins
                        xaxis=dict( # Apply to all x-axes
                            title_font=dict(size=20),
                            tickfont=dict(size=16)
                        ),
                        yaxis=dict( # Apply to all y-axes
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
                    # Increase font sizes and adjust axis label positions for SVG
                    for annotation in real_fig.layout.annotations:
                        # Identify correlation annotations (heuristically, by position)
                        if annotation.xref == 'paper' and annotation.yref == 'paper' and annotation.x == 0:
                            annotation.font.size = 14 # Increase font size
                            annotation.y += 0.005 # Adjust y-position upward
                        # Also update facet labels (transcript IDs)
                        elif annotation.text.startswith('transcript_id='):
                             annotation.font.size = 18 # Increase facet label size

                        # Adjust Y-axis label position and size for SVG (independent check)
                        if annotation.textangle == -90:
                            annotation.font.size = 20 # Increase axis label size
                            annotation.x = annotation.x + 0.12

                            # Add additional offset if there is any selected metadata
                            if number_of_selected_metadata > 0:
                                annotation.x = annotation.x + (0.04 * number_of_selected_metadata)

                        # Adjust X-axis label position and size for SVG (independent check)
                        if annotation.y < 0:
                            annotation.font.size = 20 # Increase axis label size
                            annotation.y -= 0.03

                    scatter_svg = real_fig.to_image(format="svg").decode('utf-8')
                    zipf.writestr(scatter_svg_name, scatter_svg)
                except Exception as scatter_error:
                    print(f"Error creating isoform scatter SVG: {scatter_error}")
                    # Create placeholder instead
                    placeholder_fig = go.Figure()
                    placeholder_fig.add_annotation(
                        text=f"Isoform Scatter Plot for {gene_name} (could not render)",
                        x=0.5, y=0.5, showarrow=False, font=dict(size=20)
                    )
                    placeholder_svg = placeholder_fig.to_image(format="svg").decode('utf-8')
                    zipf.writestr(scatter_svg_name, placeholder_svg)

            # Export the RNA isoform plot if available
            if isoform_fig:
                isoform_svg_name = f"{gene_name}_RNA_isoform_structure_plot_{count_type}.svg"
                try:
                    real_fig = go.Figure(isoform_fig)
                    isoform_svg = real_fig.to_image(format="svg").decode('utf-8')
                    zipf.writestr(isoform_svg_name, isoform_svg)
                except Exception as isoform_error:
                    # Create placeholder instead
                    placeholder_fig = go.Figure()
                    placeholder_fig.add_annotation(
                        text=f"RNA Isoform Plot for {gene_name} (could not render)",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=20)
                    )
                    placeholder_svg = placeholder_fig.to_image(format="svg").decode('utf-8')
                    zipf.writestr(isoform_svg_name, placeholder_svg)
            else:
                # Create a placeholder if isoform fig is not available
                isoform_svg_name = f"{gene_name}_RNA_isoform_plot_{count_type}.svg"
                placeholder_fig = go.Figure()
                placeholder_fig.add_annotation(
                    text=f"RNA Isoform Plot for {gene_name}",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                placeholder_svg = placeholder_fig.to_image(format="svg").decode('utf-8')
                zipf.writestr(isoform_svg_name, placeholder_svg)
            
        # Read the zip file
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
            
        # Clean up temp directory
        shutil.rmtree(temp_dir)

        # Shutdown kaleido scope to avoid memory waste
        pio.kaleido.scope._shutdown_kaleido()
            
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
        dcc.Download(id="download-svg-tab3"),
        
        # Add a Store to hold the isoform plot figure
        dcc.Store(id="isoform-plot-store-tab3"),
        
        # Add a hidden gene-plot-container for the callback output
        html.Div(id="gene-plot-container-tab3", style={"display": "none"}),
        
        # Add these components that are required by the callback but were missing
        html.Div([
            dcc.RangeSlider(
                id='isoform-range-slider-tab3',
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
            dcc.RadioItems(
                id='log-transform-option-tab3',
                options=[
                    {"label": "Original Values", "value": False},
                    {"label": "Log Transform (log10(x+1))", "value": True}
                ],
                value=False
            ),
            # Add this hidden component to satisfy the callback in callbacks.py
            dcc.Checklist(
                id='metadata-checklist-tab3',
                options=[
                    {"label": "Braak Stage", "value": "braak_score"},
                    {"label": "Sex", "value": "sex"},
                    {"label": "AD Status", "value": "ebbert_ad_status"},
                    {"label": "APOE Genotype", "value": "apoe"}
                ],
                value=[]
            ),
            dbc.Col([
                create_section_header("Correlation Type:"),
                create_content_card([
                    html.Div([
                        create_radio_items(
                            id="correlation-type-tab3",
                            options=[
                                {"label": "Spearman", "value": "spearman"},
                                {"label": "Pearson", "value": "pearson"}
                            ],
                            value="spearman",
                            inline=True
                        )
                    ], className="radio-group-container dbc")
                ])
            ], id="tab3-correlation-col", style={"width": "16.66%", "padding": "0 2px"}),
        ], style={"display": "none"}),  # Hide these components since they're just for the callback
        
        dbc.Card([
            dbc.CardBody([
                # First row - columns for search, options
                html.Table([
                    html.Tr([
                        html.Td([
                            create_section_header("Search Gene:"),
                            create_gene_search_dropdown(
                            id="search-input-tab3",
                            initial_value=DEFAULT_APP_GENE_INDEX,
                            initial_options=[{
                                'label': f"{DEFAULT_APP_GENE_NAME} ({DEFAULT_APP_GENE_ID})",
                                'value': DEFAULT_APP_GENE_INDEX
                            }]
                        )
                        ], id="tab3-gene-search-col", style={"width": "33.33%", "padding": "0 2px"}),
                        html.Td([
                            create_section_header("Data Matrix:"),
                            create_matrix_dropdown(dropdown_options, default_table, id="matrix-table-dropdown-tab3")
                        ], id="tab3-matrix-col", style={"width": "33.33%", "padding": "0 2px"}),
                        html.Td([
                            create_section_header("Trendline Type:"),
                            create_content_card([
                                html.Div([
                                    create_radio_items(
                                        id="trendline-type-option-tab3",
                                        options=[
                                            {"label": "LOWESS", "value": "lowess"},
                                            {"label": "Linear", "value": "linear"}
                                        ],
                                        value="lowess",
                                        inline=True
                                    )
                                ], className="radio-group-container dbc")
                            ])
                        ], id="tab3-transform-col", style={"width": "33.33%", "padding": "0 2px"}),
                    ])
                ], style={"width": "100%", "tableLayout": "fixed"}, className="mb-2", id="tab3-row1"),

                # New second row with remaining controls
                html.Table([
                    html.Tr([
                        html.Td([
                            create_section_header("Correlation Type:"),
                            create_content_card([
                                html.Div([
                                    create_radio_items(
                                        id="correlation-type-tab3",
                                        options=[
                                            {"label": "Spearman", "value": "spearman"},
                                            {"label": "Pearson", "value": "pearson"}
                                        ],
                                        value="spearman",
                                        inline=True
                                    )
                                ], className="radio-group-container dbc")
                            ])
                        ], id="tab3-correlation-col", style={"width": "33.33%", "padding": "0 2px"}),
                        html.Td([
                            create_section_header("Scatter Plot X-axis:"),
                            create_content_card([
                                html.Div([
                                    dcc.Dropdown(
                                        id="correlation-var-tab3",
                                        options=[
                                            {"label": "Age at Death", "value": "expired_age"},
                                            {"label": "All Reads", "value": "all_reads"},
                                            {"label": "Astrocyte Proportion", "value": "astrocyte_proportion"},
                                            {"label": "Brain Weight", "value": "brain_weight_grams"},
                                            {"label": "Dementia Onset Age (Years)", "value": "demential_age"},
                                            {"label": "Mapping Rate", "value": "mapping_rate_(%)"},
                                            {"label": "Mapping Rate (MAPQ ≥ 10)", "value": "mapping_rate_-_mapq_>=_10_(%)"},
                                            {"label": "Median Read Length", "value": "median_read_length_bam"},
                                            {"label": "Microglia-PVM Proportion", "value": "microglia-pvm_proportion"},
                                            {"label": "N50", "value": "n50_bam"},
                                            {"label": "Neuronal Proportion", "value": "neuronal_proportion"},
                                            {"label": "Oligodendrocyte Proportion", "value": "oligodendrocyte_proportion"},
                                            {"label": "Plaque Density Frontal Cortex", "value": "plaquef"},
                                            {"label": "Post-Mortem Interval", "value": "pmi"},
                                            {"label": "Primary Alignments (MAPQ ≥ 10)", "value": "filtered_primary_alignments_(mapq)"},
                                            {"label": "RNA Integrity (RIN)", "value": "rin"},
                                            {"label": "Tangle Density Frontal Cortex", "value": "tanglef"},
                                            {"label": "TIN Median", "value": "tin_median"},
                                            {"label": "Total Plaque Density", "value": "plaquetotal"},
                                            {"label": "Total Tangle Density", "value": "tangletotal"},
                                            {"label": "Trimmed & Filtered Reads", "value": "trimmed_&_processed_pass_reads"},
                                            {"label": "Years Living with Dementia (Years)", "value": "dementia_years"}
                                        ],
                                        value="expired_age",
                                        clearable=False,
                                        optionHeight=60,
                                        className="axis-dropdown",
                                        maxHeight=400
                                    )
                                ], style={
                                    "position": "relative",
                                    "zIndex": 1000,
                                    "width": "100%"
                                })
                            ])
                        ], id="tab3-x-axis-col", style={"width": "33.33%", "padding": "0 2px"}),
                        html.Td([
                            create_section_header("Scatter Plot Y-axis:"),
                            create_content_card([
                                html.Div([
                                    create_radio_items(
                                        id="y-axis-metric-tab3",
                                        options=[
                                            {"label": "TMM (per million)", "value": "tmm"},
                                            {"label": "Relative Abundance", "value": "relative"}
                                        ],
                                        value="tmm",
                                        inline=True
                                    )
                                ], className="radio-group-container dbc")
                            ])
                        ], id="tab3-y-axis-col", style={"width": "33.33%", "padding": "0 2px"}),
                    ])
                ], style={"width": "100%", "tableLayout": "fixed"}, className="mb-4", id="tab3-row2"),

                # Third row - Two separate graphs
                dbc.Row([
                    dbc.Col([
                        create_content_card(
                            dbc.Spinner(
                                html.Div([
                                    dcc.Graph(
                                        id='rnapy-plot-tab3',
                                        style={
                                            "background-color": "#ffffff",
                                            "padding": "10px",
                                            "border-radius": "5px",
                                            "border": "1px solid rgba(0, 0, 0, 0.1)",
                                            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                                            "width": "100%",
                                            "height": "100%",
                                            "min-height": "400px"
                                        },
                                        config={
                                            'displayModeBar': True,
                                            'scrollZoom': False,
                                            'modeBarButtonsToRemove': ['autoScale2d'],
                                            'displaylogo': False
                                        }
                                    )
                                ]),
                                color="primary",
                                type="grow",
                                spinner_style={"width": "3rem", "height": "3rem"}
                            )
                        )
                    ], width=8, id="tab3-col3-1"),
                    dbc.Col([
                        create_content_card(
                            dbc.Spinner(
                                html.Div([
                                    dcc.Graph(
                                        id='scatter-plot-tab3',
                                        style={
                                            "background-color": "#ffffff",
                                            "padding": "10px",
                                            "border-radius": "5px",
                                            "border": "1px solid rgba(0, 0, 0, 0.1)",
                                            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                                            "width": "100%",
                                            "height": "100%",
                                            "min-height": "400px"
                                        },
                                        config={
                                            'displayModeBar': True,
                                            'scrollZoom': False,
                                            'modeBarButtonsToRemove': ['autoScale2d'],
                                            'displaylogo': False
                                        }
                                    )
                                ]),
                                color="primary",
                                type="grow",
                                spinner_style={"width": "3rem", "height": "3rem"}
                            )
                        )
                    ], width=4, id="tab3-col3-2"),
                ], 
                className="mb-4 dbc",
                id="tab3-row3",
                style={"height": "90vh"}  # Make the row take up 90% of viewport height
                ),

                # Fourth row - three columns
                dbc.Row([
                    dbc.Col([
                        create_section_header("Show data separated by:"),
                        create_content_card([
                            html.Div([
                                create_checklist(
                                    id="metadata-checklist-tab3",
                                    options=[
                                        {"label": "Braak Stage", "value": "braak_score"},
                                        {"label": "Sex", "value": "sex"},
                                        {"label": "AD Status", "value": "ebbert_ad_status"},
                                        {"label": "APOE Genotype", "value": "apoe"}
                                    ],
                                    value=[]
                                )
                            ])
                        ])
                    ], width=4, id="tab3-col4-1"),
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
                                    id='isoform-range-slider-tab3',
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
                                    "Select range of top expressed isoforms by rank (maximum 3 isoforms)",
                                    style={
                                        "color": "#666666",
                                        "display": "block",
                                        "marginTop": "8px",
                                        "textAlign": "center"
                                    }
                                )
                            ], style={"padding": "10px"})
                        ])
                    ], width=4, id="tab3-col4-2"),
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
                                    id="download-button-tab3",
                                    color="primary",
                                    className="w-100 mb-3",
                                    disabled=False
                                ),
                                html.Small(
                                    "Takes a while to generate plots",
                                    style={
                                        "color": "#666666",
                                        "display": "block",
                                        "marginTop": "8px",
                                        "textAlign": "center"
                                    }
                                )
                            ], style={"padding": "10px"})
                        ])
                    ], width=4, id="tab3-col4-3"),
                ], className="mb-4 dbc", id="tab3-row4"),

                # Fifth row - two columns (Density and Gene Level Plots)
                dbc.Row([
                    dbc.Col([
                        create_section_header(""),
                        create_content_card([
                            dcc.Graph(
                                id='density-plot-tab3',
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
                    ], width=8, id="tab3-col5-1"),
                    dbc.Col([
                        create_section_header(""),
                        create_content_card([
                            dcc.Graph(
                                id='gene-level-plot-tab3',
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': False,
                                    'modeBarButtonsToRemove': ['autoScale2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}
                            )
                        ])
                    ], width=4, id="tab3-col5-2"),
                ], className="mb-4 dbc", id="tab3-row5"),
            ], id="tab3-card-body")
        ],
        id="tab3-card",
        style={
            "background-color": "#ffffff",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "6px",
            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
        })
    ], 
    fluid=True,  # Makes the container full-width
    id="tab3-container",
    style={
        "max-width": "98%",  # Use 98% of the viewport width
        "margin": "0 auto",  # Center the container
        "padding": "10px"    # Add some padding
    })

@callback(
    [Output("tab3-container", "style"),
     Output("tab3-row1", "className"),
     Output("tab3-gene-search-col", "width"),
     Output("tab3-matrix-col", "width"),
     Output("tab3-transform-col", "width"),
     Output("tab3-row2", "className"),
     Output("tab3-correlation-col", "width"),
     Output("tab3-x-axis-col", "width"),
     Output("tab3-y-axis-col", "width"),
     Output("tab3-row3", "className"),
     Output("tab3-col3-1", "width"),
     Output("tab3-col3-2", "width"),
     Output("tab3-row4", "className"),
     Output("tab3-col4-1", "width"),
     Output("tab3-col4-2", "width"),
     Output("tab3-col4-3", "width"),
     Output("tab3-row5", "className"),
     Output("tab3-col5-1", "width"),
     Output("tab3-col5-2", "width")],
    [Input("window-dimensions", "data")]
)
def update_tab3_responsiveness(dimensions):
    if not dimensions:
        # Default styles if no dimensions available
        return (
            {"max-width": "98%", "margin": "0 auto", "padding": "10px"},
            "mb-2 dbc", 33, 33, 33,
            "mb-4 dbc", 33, 33, 33,
            "mb-4 dbc", 8, 4,
            "mb-4 dbc", 4, 4, 4,
            "mb-4 dbc", 8, 4
        )
    
    width = dimensions.get('width', 1200)
    
    # Base styles
    container_style = {"max-width": "98%", "margin": "0 auto", "padding": "10px"}
    
    # Row 1 - Search, matrix, trendline
    row1_class = "mb-2 dbc"
    gene_search_col_width = 33
    matrix_col_width = 33
    transform_col_width = 33
    
    # Row 2 - Correlation, x-axis, y-axis
    row2_class = "mb-4 dbc"
    correlation_col_width = 33
    x_axis_col_width = 33
    y_axis_col_width = 33
    
    # Row 3 - Plots
    row3_class = "mb-4 dbc"
    col3_1_width = 8
    col3_2_width = 4
    
    # Row 4 - Controls
    row4_class = "mb-4 dbc"
    col4_1_width = 4
    col4_2_width = 4
    col4_3_width = 4
    
    # Row 5 - Density and gene level plots
    row5_class = "mb-4 dbc"
    col5_1_width = 8
    col5_2_width = 4
    
    # Responsive adjustments based on width
    if width < 576:  # Extra small devices
        container_style.update({"padding": "5px", "max-width": "100%"})
        
        # Row 1 - full width columns
        row1_class = "mb-2 dbc flex-column"
        gene_search_col_width = 100
        matrix_col_width = 100
        transform_col_width = 100
        
        # Row 2 - full width columns
        row2_class = "mb-2 dbc flex-column"
        correlation_col_width = 100
        x_axis_col_width = 100
        y_axis_col_width = 100
        
        # Row 3 - full width columns
        row3_class = "mb-2 dbc flex-column"
        col3_1_width = 12
        col3_2_width = 12
        
        # Row 4 - full width columns
        row4_class = "mb-2 dbc flex-column"
        col4_1_width = 12
        col4_2_width = 12
        col4_3_width = 12
        
        # Row 5 - full width columns
        row5_class = "mb-2 dbc flex-column"
        col5_1_width = 12
        col5_2_width = 12
        
    elif width < 768:  # Small devices
        container_style.update({"padding": "8px"})
        
        # Row 1 - two columns per row
        row1_class = "mb-3 dbc"
        gene_search_col_width = 50
        matrix_col_width = 50
        transform_col_width = 100
        
        # Row 2 - two columns per row
        row2_class = "mb-3 dbc"
        correlation_col_width = 50
        x_axis_col_width = 50
        y_axis_col_width = 100
        
        # Row 3 - stacked columns
        row3_class = "mb-3 dbc flex-column"
        col3_1_width = 12
        col3_2_width = 12
        
        # Row 4 - two columns per row (split over two rows)
        row4_class = "mb-3 dbc"
        col4_1_width = 6
        col4_2_width = 6
        col4_3_width = 12
        
        # Row 5 - full width columns
        row5_class = "mb-3 dbc flex-column"
        col5_1_width = 12
        col5_2_width = 12
        
    elif width < 992:  # Medium devices
        container_style.update({"padding": "10px"})
        # No changes to default for medium devices
        
    return (
        container_style,
        row1_class, gene_search_col_width, matrix_col_width, transform_col_width,
        row2_class, correlation_col_width, x_axis_col_width, y_axis_col_width,
        row3_class, col3_1_width, col3_2_width,
        row4_class, col4_1_width, col4_2_width, col4_3_width,
        row5_class, col5_1_width, col5_2_width
    )

# Global variable to track previous state of the range slider
previous_range = [1, 3]  # Changed initial range to 1-3

@app.callback(
    Output('isoform-range-slider-tab3', 'max'),
    Output('isoform-range-slider-tab3', 'marks'),
    Output('isoform-range-slider-tab3', 'value'),
    [Input('search-input-tab3', 'value'),
     Input('isoform-range-slider-tab3', 'value')]
)
def update_slider_range_tab3(selected_gene, current_range):
    global previous_range # Need to access and modify the global variable

    # --- Initial Setup and Max Value Calculation ---
    if not selected_gene:
        # Use APP gene as default
        selected_gene = DEFAULT_APP_GENE_INDEX

    try:
        # Query to get the number of transcripts for this gene
        transcript_count = duck_conn.execute("""
            SELECT COUNT(DISTINCT transcript_id)
            FROM gene_and_transcript_index_table
            WHERE gene_index = ?
        """, [selected_gene]).fetchone()[0]

        if not transcript_count or transcript_count == 0:  # Handle case where gene has 0 transcripts
            marks = {1: '1'}
            previous_range[:] = [1, 1] # Reset previous range
            return 1, marks, [1, 1]

        # Max value for slider
        max_val = max(1, transcript_count)

        # --- Create Marks ---
        if transcript_count <= 20:
            marks = {i: str(i) for i in range(1, max_val + 1)} # Use max_val here
        else:  # Reduce marks for large numbers
            step = max(1, max_val // 10)  # Use max_val here
            marks = {i: str(i) for i in range(1, max_val + 1, step)}
            if max_val not in marks:  # Ensure last value is a mark
                marks[max_val] = str(max_val)

        # --- Handle Initial/Invalid Range ---
        if current_range is None or len(current_range) != 2:
             default_range = [1, min(3, max_val)]
             previous_range[:] = default_range # Reset previous range
             return max_val, marks, default_range

        # --- Core Logic for Range Adjustment ---
        start, end = current_range
        prev_start, prev_end = previous_range # Get previous state

        new_start, new_end = start, end # Initialize new range with current values

        # Check if range size constraint is violated (now 3 instead of 5)
        if end - start + 1 > 3:
            start_changed = start != prev_start
            end_changed = end != prev_end

            # Determine adjustment based on which knob moved
            if start_changed and not end_changed: # Left knob moved (expanding left)
                new_start = start
                new_end = start + 2  # Changed from +4 to +2 for range of 3
            elif end_changed and not start_changed: # Right knob moved (expanding right)
                new_end = end
                new_start = end - 2  # Changed from -4 to -2 for range of 3
            else: # Both/neither changed (or initial state where range > 3): prioritize end
                 new_end = end
                 new_start = end - 2  # Changed from -4 to -2 for range of 3

        # --- Clamping and Boundary Adjustments ---
        # First, clamp the potentially adjusted values
        new_start = max(1, new_start)
        new_end = min(max_val, new_end)

        # Second, ensure the range size is no more than 3 after clamping.
        # Prioritize the clamped boundary.
        if new_end - new_start + 1 > 3:
             if new_end == max_val: # If end was clamped to max_val, adjust start
                 new_start = max(1, new_end - 2)  # Changed from -4 to -2 for range of 3
             elif new_start == 1: # If start was clamped to 1, adjust end
                 new_end = min(max_val, new_start + 2)  # Changed from +4 to +2 for range of 3
             else: # Should not happen if max_val >= 3, but handles max_val < 3 case
                  new_start = max(1, new_end - 2)  # Changed from -4 to -2 for range of 3

        # --- Update Previous State and Return ---
        # The final calculated range
        final_range = [new_start, new_end]

        # Update the global previous_range *after* all calculations and clamping
        previous_range[:] = final_range

        return max_val, marks, final_range

    except Exception as e:
        import traceback
        print(f"Error updating slider range for gene {selected_gene}: {e}")
        print(traceback.format_exc())
        # Fallback to default values in case of error
        marks = {i: str(i) for i in range(1, 11)}
        previous_range[:] = [1, 3] # Reset previous range to 1-3
        return 10, marks, [1, 3]

# New callback to limit metadata checklist selections
@app.callback(
    Output('metadata-checklist-tab3', 'value'),
    Input('metadata-checklist-tab3', 'value'),
    # State('metadata-checklist-tab3', 'options'), # Removed State as it's not needed
    prevent_initial_call=True
)
def limit_metadata_selections(selected_values):
    if not selected_values:
        return []

    last_selected = selected_values[-1]
    restricted_options = {'braak_score', 'apoe'}
    # allowed_pair_options = {'sex', 'ebbert_ad_status'} # Only non-restricted

    # Case 1: Last selected item is restricted
    if last_selected in restricted_options:
        # Enforce restricted item is the only selection
        return [last_selected]

    # Case 2: Last selected item is NOT restricted
    else:
        # Check if ANY restricted item exists in the selections BEFORE the last one was added
        previous_selection_set = set(selected_values[:-1])
        restricted_previously_present = previous_selection_set.intersection(restricted_options)

        if restricted_previously_present:
            # A restricted item was selected previously, and now a non-restricted item was added.
            # Rule: Only the newly selected non-restricted item should remain.
            return [last_selected]
        else:
            # No restricted items were present before adding the last (non-restricted) item.
            # Limit to a maximum of 2 non-restricted items.
            current_selection_set = set(selected_values)
            if len(current_selection_set) > 2:
                # This happens if the user had ['sex', 'ebbert_ad_status'] and selected one of them again,
                # or if there were hypothetically more than 2 non-restricted options.
                # Keep the last two distinct selections.
                # Since the only non-restricted are sex and ebbert_ad_status, this keeps them both.
                return list(current_selection_set)[-2:] # Or simply return the allowed pair
                # return list(allowed_pair_options) # Safer if selected_values could have duplicates temporarily
            else:
                # 1 or 2 non-restricted items selected, which is allowed.
                return selected_values # Return the current list

@app.callback(
    [Output('rnapy-plot-tab3', 'figure'),
     Output('scatter-plot-tab3', 'figure'),
     Output('isoform-plot-store-tab3', 'data')],
    [Input('matrix-table-dropdown-tab3', 'value'),
     Input('search-input-tab3', 'value'),
     Input('metadata-checklist-tab3', 'value'),
     Input('trendline-type-option-tab3', 'value'),
     Input('window-dimensions', 'data'),
     Input('isoform-range-slider-tab3', 'value'),
     Input('correlation-var-tab3', 'value'),
     Input('y-axis-metric-tab3', 'value'),
     Input('correlation-type-tab3', 'value')]
)
def update_gene_plot_tab3(count_type, selected_gene, selected_metadata, trendline_type, window_dimensions, isoform_range, correlation_var, y_axis_metric, correlation_type):
    
    # Set default to APP gene if no gene is selected
    if selected_gene is None:
        selected_gene = DEFAULT_APP_GENE_INDEX

    try:
        # If no count type is selected, use total counts by default
        count_type = count_type if count_type else 'total'

        # Default window dimensions if not available yet
        if not window_dimensions:
            window_dimensions = {'width': 1200, 'height': 800}

        # Create scaling factor and base font size
        scaling_factor = window_dimensions["width"]/2540
        base_font_size = 20 * scaling_factor

        # Ensure minimum dimensions for usability
        if window_dimensions["width"] > window_dimensions["height"]:
            plot_width_1 = (window_dimensions['width'] * 0.8) * 0.7
            plot_width_2 = (window_dimensions['width'] * 0.8) * 0.33
            plot_height = window_dimensions['height'] * 0.8
        else:
            plot_width_1 = (window_dimensions['width'] * 0.7) * 0.7
            plot_width_2 = (window_dimensions['width'] * 0.7) * 0.33
            plot_height = window_dimensions['height'] * 0.8

        # Get the gene name for the filename
        gene_info = duck_conn.execute("""
            SELECT g.gene_index, g.gene_name, g.gene_id
            FROM gene_and_transcript_index_table g
            WHERE g.gene_index = ?
            GROUP BY g.gene_index, g.gene_name, g.gene_id
            LIMIT 1
        """, [selected_gene]).fetchone()

        if not gene_info:
            return go.Figure(), go.Figure(), None

        gene_index, gene_name, actual_gene_id = gene_info

        # Get data with metadata
        expression = get_gene_data_with_metadata(gene_index, with_polars=True, limit=None)
        if expression is None or len(expression) == 0:
            return go.Figure(), go.Figure(), None

        # Select the correct columns based on count_type
        tmm_col = f"{count_type}_cpm_normalized_tmm"
        abundance_col = f"{count_type}_relative_abundance"

        # Make sure these columns exist
        if tmm_col not in expression.columns or abundance_col not in expression.columns:
            error_cols = [col for col in [tmm_col, abundance_col] if col not in expression.columns]
            error_message = f"Missing columns in expression data: {', '.join(error_cols)}"
            return go.Figure(), go.Figure(), None

        # Get annotation data
        annotation_query = """
        SELECT a.*, g.gene_id, g.gene_name, g.transcript_id, g.seqnames
        FROM gene_and_transcript_index_table g
        JOIN transcript_annotation a ON a.gene_index = g.gene_index AND a.transcript_index = g.transcript_index
        WHERE g.gene_index = ?
        """
        annotation = duck_conn.execute(annotation_query, [gene_index]).pl()

        expression = expression.join(annotation.select(["transcript_index", "transcript_id", "gene_id", "gene_name"]).unique(), 
                                on="transcript_index", how="inner")
        
        ## Drop indexes
        expression = expression.drop(["transcript_index", "gene_index"])
        annotation = annotation.drop(["transcript_index", "gene_index"])

        # First convert strand to integer type to ensure consistent comparisons
        annotation = annotation.with_columns(pl.col("strand").cast(pl.Utf8).alias("strand"))
        
        # Then convert the integer values to string symbols
        annotation = annotation.with_columns(
            pl.when(pl.col("strand") == "-1").then(pl.lit("-"))
              .when(pl.col("strand") == "1").then(pl.lit("+"))
              .otherwise(pl.col("strand").cast(pl.Utf8))
              .alias("strand")
        )

        

        if annotation is None or len(annotation) == 0:
            return go.Figure(), go.Figure(), None

        chromosome = annotation["seqnames"][0] if "seqnames" in annotation.columns else "Unknown"
        strand = annotation["strand"][0] if "strand" in annotation.columns else "?"
        min_start = annotation["start"].min() if "start" in annotation.columns else "Unknown"
        max_end = annotation["end"].max() if "end" in annotation.columns else "Unknown"

        annotation = RNApy.shorten_gaps(annotation)
        zero_based_range = [isoform_range[0] - 1, isoform_range[1]]
        expression, annotation = order_transcripts_by_expression(
            annotation_df=annotation,
            expression_df=expression,
            expression_column=tmm_col,
            top_n=zero_based_range
        )

        # Handle metadata hue for RNA isoform plot
        expression_hue = None
        if selected_metadata:
            if len(selected_metadata) == 1:
                expression_hue = selected_metadata[0]
                # Filter out nulls for single metadata column
                expression = expression.filter(~pl.col(expression_hue).is_null())
                # Generate colors for each group
                unique_hue_values = expression[expression_hue].unique().sort().to_list()
                if unique_hue_values:
                    custom_colors_list = get_n_colors(len(unique_hue_values), 'Plotly_r')
                    color_map = {val: color for val, color in zip(unique_hue_values, custom_colors_list)}
            else:
                combined_col_name = "combined_metadata"
                combined_expr = pl.col(selected_metadata[0]).cast(pl.Utf8)
                for col_name in selected_metadata[1:]:
                    combined_expr = combined_expr + pl.lit(" | ") + pl.col(col_name).cast(pl.Utf8)
                expression = expression.with_columns([combined_expr.alias(combined_col_name)])
                for col_name in selected_metadata: 
                    expression = expression.filter(~pl.col(col_name).is_null())
                expression_hue = combined_col_name
                # Generate colors for combined groups
                unique_hue_values = expression[expression_hue].unique().sort().to_list()
                if unique_hue_values:
                    custom_colors_list = get_n_colors(len(unique_hue_values), 'Plotly_r')
                    color_map = {val: color for val, color in zip(unique_hue_values, custom_colors_list)}
        else:
            # No metadata selected, use gray for all points
            expression = expression.with_columns([
                pl.lit("All Samples").alias("_group")
            ])
            expression_hue = "_group"
            color_map = {"All Samples": "#808080"}  # Set to gray

        # Define annotation colormap (consistent with tab2)
        annotation_hue_values = ["protein_coding", "retained_intron", "protein_coding_CDS_not_defined", "nonsense_mediated_decay",
                                 "new_low_confidence", "new_high_confidence", "lncRNA", "other"]
        annotation_colormap = {val: color for val, color in zip(annotation_hue_values, get_n_colors(len(annotation_hue_values), 'Plotly'))}

        # Build trace parameters
        trace_params = {
            "annotation": annotation,
            "x_start": "rescaled_start",
            "x_end": "rescaled_end",
            "y": "transcript_id",
            "annotation_hue": "transcript_biotype",
            "hover_start": "start",
            "hover_end": "end",
            "marker_size": 5*scaling_factor,
            "arrow_size": 12*scaling_factor,
            "annotation_color_map": annotation_colormap  # Add the annotation colormap here
        }

        traces = RNApy.make_traces(**trace_params)

        subplot_titles = ["Transcript Structure"]

        rnapy_fig = RNApy.make_plot(traces=traces, subplot_titles=subplot_titles,
                    boxgap=0.15, boxgroupgap=0.1, width=plot_width_1, height=plot_height,
                    legend_font_size=20*scaling_factor, yaxis_font_size=20*scaling_factor,
                    xaxis_font_size=20*scaling_factor, subplot_title_font_size=24*scaling_factor,
                    template="ggplot2", hover_font_size=14*scaling_factor,
                    legend_title_font_size=20*scaling_factor)

        x_min = annotation["rescaled_start"].min() if "rescaled_start" in annotation.columns else 0
        x_max = annotation["rescaled_end"].max() if "rescaled_end" in annotation.columns else 1
        padding = (x_max - x_min) * 0.05
        rnapy_fig.update_xaxes(range=[x_min - padding, x_max + padding], showticklabels=False, showline=False, zeroline=False, ticks="", row=1, col=1)

        rnapy_fig.add_annotation(x=0.5, y=1.12, xref='paper', yref='paper', text=f"{gene_name} ({actual_gene_id})", showarrow=False, xanchor="center", yanchor="top", font=dict(size=26 * scaling_factor))
        rnapy_fig.add_annotation(x=0.5, y=1.08, xref='paper', yref='paper', text=f"Region: chr{chromosome}({strand}):{min_start}-{max_end}", showarrow=False, xanchor="center", yanchor="top", font=dict(size=18 * scaling_factor))

        for i, annot in enumerate(rnapy_fig['layout']['annotations']):
            if i < len(subplot_titles):
                if i == 0: annot["x"] = 0
                annot["xanchor"] = "left"

        # Create scatter plot for RNA isoform level
        y_col = tmm_col if y_axis_metric == "tmm" else abundance_col
        y_label = "TMM (per million)" if y_axis_metric == "tmm" else "Relative Abundance (%)"
        
        # Convert expression data to pandas for plotting
        pdf = expression.to_pandas()
        
        # Prepare category orders
        # Get sorted list of transcript IDs for facet order (reversed for top-to-bottom display)
        transcript_ids_in_order = expression["transcript_id"].unique(maintain_order=True).to_list()[::-1]
        category_orders_dict = {"transcript_id": transcript_ids_in_order}
        # Get sorted list of hue values for legend order
        if expression_hue and expression_hue != "_group": # Only add if hue is used and not the default _group
            # Ensure unique_hue_values is defined and sorted (should be from color map generation)
            if 'unique_hue_values' in locals() and unique_hue_values:
                 category_orders_dict[expression_hue] = unique_hue_values # Add the sorted hue values
            else: # Fallback if unique_hue_values isn't available for some reason
                 sorted_hue_fallback = sorted(list(pdf[expression_hue].unique()))
                 category_orders_dict[expression_hue] = sorted_hue_fallback
        
        scatter_fig = px.scatter(pdf, 
                               x=correlation_var, 
                               y=y_col,
                               color=expression_hue,
                               color_discrete_map=color_map,
                               facet_row="transcript_id",
                               trendline="ols" if trendline_type == "linear" else "lowess",
                               # Use the explicit category orders for both facets and legend
                               category_orders=category_orders_dict,
                               template="ggplot2")
        
        # Update marker opacity to make points more transparent
        scatter_fig.update_traces(marker=dict(opacity=0.5))
        
        # Clip LOWESS trendline at y=0 if necessary
        if trendline_type == 'lowess':
            for trace in scatter_fig.data:
                # Identify trendline traces (typically mode='lines')
                # We might need a more robust check if other line types exist
                if trace.mode == 'lines':
                    # Add check to ensure trace.y is not None before processing
                    if trace.y is not None:
                        y_data = list(trace.y)
                        # Clip negative values to zero
                        clipped_y = [max(0, y_val) for y_val in y_data if y_val is not None] # Ensure None values are handled
                        # Update the trace y-data if clipping occurred
                        if len(clipped_y) == len(y_data): # Check length to avoid issues if filtering Nones changed size
                             trace.y = tuple(clipped_y)

        scatter_fig.update_yaxes(matches=None)
        scatter_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        # Calculate responsive font sizes based on number of subplots and window size
        n_subplots = len(expression["transcript_id"].unique())
        
        base_font_size = 20 * scaling_factor
        # Get the label for the correlation variable
        var_labels = {
            "expired_age": "Age (Years)",
            "pmi": "Post-Mortem Interval (Hours)",
            "brain_weight_grams": "Brain Weight (Grams)",
            "rin": "RNA Integrity (RIN)",
            "tin_median": "TIN Median",
            "all_reads": "All Reads (Count)",
            "trimmed_&_processed_pass_reads": "Trimmed & Filtered Reads (Count)",
            "filtered_primary_alignments_(mapq)": "Primary Alignments MAPQ ≥ 10 (Count)",
            "mapping_rate_-_mapq_>=_10_(%)": "Mapping Rate MAPQ ≥ 10 (%)",
            "mapping_rate_(%)": "Mapping Rate (%)",
            "n50_bam": "N50 (Nucleotides)",
            "median_read_length_bam": "Median Read Length (Nucleotides)",
            "astrocyte_proportion": "Astrocyte Proportion",
            "oligodendrocyte_proportion": "Oligodendrocyte Proportion",
            "neuronal_proportion": "Neuronal Proportion",
            "microglia-pvm_proportion": "Microglia-PVM Proportion",
            "plaquef": "Plaque Density - Frontal Cortex",
            "plaquetotal": "Total Plaque Density",
            "tanglef": "Tangle Density - Frontal Cortex",
            "tangletotal": "Total Tangle Density",
            "demential_age": "Dementia Onset Age (Years)",
            "dementia_years": "Years Living with Dementia (Years)"
        }
        x_label = var_labels.get(correlation_var, correlation_var)
        
        # Update layout with no legend title
        scatter_fig.update_layout(
            template="ggplot2",
            width=plot_width_2,
            height=plot_height,
            margin=dict(l=80, r=20, t=60, b=60),
            showlegend=len(selected_metadata) > 0,  # Only show legend if metadata is selected
            legend=dict(
                title=None,  # Remove legend title
                font=dict(size=base_font_size * 0.8),
                x=1.05, # Keep legend slightly further to the right
                xanchor='left',
                y=1, # Align to top
                yanchor='top'
            ),
            font=dict(size=base_font_size)
        )
        
        # Add vertical y-axis label with more stable positioning
        scatter_fig.update_layout(
            yaxis_title=y_label,
            yaxis_title_standoff=10,  # Adjust standoff for better spacing
            yaxis_title_font=dict(size=base_font_size, color="black"),
        )
        # For multi-facet plots, ensure the title appears properly
        scatter_fig.add_annotation(
            x=-0.18 + (0.0007 * n_subplots) - (len(selected_metadata) * 0.05),  # Fixed position relative to paper
            y=0.5,    # Center vertically
            xref="paper",
            yref="paper",
            text=y_label,
            showarrow=False,
            textangle=-90,  # Rotate text 90 degrees counterclockwise
            font=dict(size=base_font_size, color="black"),
            align="left"
        )
        
        # Add horizontal x-axis label at the bottom
        scatter_fig.add_annotation(
            x=0.5,    # Center horizontally
            y=-0.08,   # Position below the plot
            xref="paper",
            yref="paper",
            text=x_label,  # Use x_label for x-axis
            showarrow=False,
            font=dict(size=base_font_size, color="black"),
            align="center"
        )
        
        # Update axes and facet labels with responsive fonts
        for i in range(1, n_subplots + 1):
            scatter_fig.update_xaxes(
                title_text="",
                row=i,
                col=1,
                tickfont=dict(size=base_font_size * 0.9),  # Slightly smaller for tick labels
                ticks="outside" if i == 1 else "",  # Show ticks only on bottom plot
                showticklabels=i == 1  # Show tick labels only on bottom plot
            )
            scatter_fig.update_yaxes(
                title_text="",
                row=i,
                col=1,
                tickfont=dict(size=base_font_size * 0.9)  # Slightly smaller for tick labels
            )
            
        # Update facet labels (transcript IDs)
        scatter_fig.for_each_annotation(lambda a: a.update(
            font=dict(size=base_font_size * 0.9)  # Slightly smaller for facet labels
        ))

        # --- Add Correlation Annotations per Facet --- # (Updated section)
        min_points_for_corr = 3
        facet_annotations = []
        transcript_ids_in_order = expression["transcript_id"].unique(maintain_order=True).to_list()[::-1]
        n_subplots = len(transcript_ids_in_order)

        for i, transcript_id in enumerate(transcript_ids_in_order):
            facet_data = pdf[pdf["transcript_id"] == transcript_id]
            correlation_texts_facet = []

            # Check if metadata is used for grouping within this facet
            if expression_hue != "_group" and expression_hue in facet_data.columns:
                # Calculate correlation per subgroup within the facet
                # Sort unique subgroup names for consistent order
                unique_subgroups_facet = sorted(list(facet_data[expression_hue].unique()))
                for subgroup_name in unique_subgroups_facet:
                    subgroup_data = facet_data[facet_data[expression_hue] == subgroup_name]
                    x_data_sub = subgroup_data[correlation_var].replace([np.inf, -np.inf], np.nan).dropna()
                    y_data_sub = subgroup_data[y_col].replace([np.inf, -np.inf], np.nan).dropna()
                    common_indices_sub = x_data_sub.index.intersection(y_data_sub.index)
                    x_data_sub = x_data_sub.loc[common_indices_sub]
                    y_data_sub = y_data_sub.loc[common_indices_sub]

                    # Abbreviate subgroup name (same logic as gene-level plot)
                    abbreviated_name = str(subgroup_name) # Abbreviate name
                    abbreviated_name = abbreviated_name.replace(" | ", "+")
                    abbreviated_name = abbreviated_name.replace("Male", "M")
                    abbreviated_name = abbreviated_name.replace("Female", "F")
                    abbreviated_name = abbreviated_name.replace("Control", "CT")
                    abbreviated_name = abbreviated_name.replace("Stage", "")
                    abbreviated_name = abbreviated_name.replace("-", "/")
                    abbreviated_name = abbreviated_name.replace("E", "")

                    # Set correlation symbol based on type
                    if correlation_type == 'spearman':
                        corr_symbol = "ρ"
                    else: # Pearson or default
                        corr_symbol = "r"
                        
                    # Initialize with NA for correlation
                    subgroup_corr_text = f"{abbreviated_name}: {corr_symbol}=NA"
                    
                    if len(x_data_sub) >= min_points_for_corr:
                        try:
                            # Calculate correlation
                            if correlation_type == 'spearman':
                                corr, _ = spearmanr(x_data_sub, y_data_sub)
                            else: # Pearson or default
                                corr, _ = pearsonr(x_data_sub, y_data_sub)
                                
                            # Only update text if correlation is a valid number
                            if not np.isnan(corr):
                                subgroup_corr_text = f"{abbreviated_name}: {corr_symbol}={corr:.2f}"
                        except Exception as corr_e_sub:
                            print(f"Could not calculate correlation for {transcript_id} subgroup {subgroup_name}: {corr_e_sub}")
                    
                    # Always add correlation text (either with value or NA)
                    correlation_texts_facet.append(subgroup_corr_text)
                
                correlation_text_facet = " | ".join(correlation_texts_facet)
            else:
                # Calculate overall correlation for the facet if no subgroups
                x_data_facet = facet_data[correlation_var].replace([np.inf, -np.inf], np.nan).dropna()
                y_data_facet = facet_data[y_col].replace([np.inf, -np.inf], np.nan).dropna()
                common_indices_facet = x_data_facet.index.intersection(y_data_facet.index)
                x_data_facet = x_data_facet.loc[common_indices_facet]
                y_data_facet = y_data_facet.loc[common_indices_facet]

                # Set correlation symbol based on type
                if correlation_type == 'spearman':
                    corr_symbol = "ρ"
                else: # Pearson or default
                    corr_symbol = "r"
                
                # Initialize with NA
                correlation_text_facet = f"{corr_symbol}=NA"
                
                if len(x_data_facet) >= min_points_for_corr:
                    try:
                        # Calculate correlation
                        if correlation_type == 'spearman':
                            corr, _ = spearmanr(x_data_facet, y_data_facet)
                        else: # Pearson or default
                            corr, _ = pearsonr(x_data_facet, y_data_facet)
                            
                        # Only update text if correlation is a valid number
                        if not np.isnan(corr):
                            correlation_text_facet = f"{corr_symbol}={corr:.2f}"
                    except Exception as corr_e_facet:
                        print(f"Could not calculate overall correlation for {transcript_id}: {corr_e_facet}")

            # Create annotation for this facet
            facet_annotation = go.layout.Annotation(
                text=correlation_text_facet, # Use combined/overall text for facet
                align='left',
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(size=base_font_size * 0.6, color="black"),
                xanchor='left',
                yanchor='top',
                y = (1.025 * ((n_subplots - i)/n_subplots)), # Position as per user edits
                x=0 # Position as per user edits
            )
            facet_annotations.append(facet_annotation)

        for annotation in facet_annotations:
            scatter_fig.add_annotation(annotation)

        return rnapy_fig, scatter_fig, rnapy_fig

    except Exception as e:
        print(f"Error in update_gene_plot_tab3: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return go.Figure(), go.Figure(), None