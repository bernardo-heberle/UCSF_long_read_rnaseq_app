# File: app/tabs/tab2.py
# Contains the layout for Tab 2.

from dash import html, dcc, Input, Output, callback, no_update, State, MATCH, ALL, ClientsideFunction
import dash_bootstrap_components as dbc
from app import app
from app.utils.db_utils import get_matrix_dropdown_options, search_genes, duck_conn, get_gene_density_data, get_total_gene_data_with_metadata, get_gene_data_with_metadata
from app.utils.ui_components import (
    create_gene_search_dropdown,
    create_matrix_dropdown,
    create_section_header,
    create_content_card,
    create_radio_items,
    create_checklist
)
from app.utils.polars_utils import order_transcripts_by_expression
from app.utils.plotly_utils import get_n_colors
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import os
import polars as pl
import io
import base64
import kaleido
import json
import RNApysoforms as RNApy


# Configure polars to not truncate output in print statements and head() calls
pl.Config.set_tbl_rows(100)  # Show up to 100 rows
pl.Config.set_tbl_cols(100)  # Show up to 100 columns
pl.Config.set_tbl_width_chars(2000)  # Increase width for better display
pl.Config.set_fmt_str_lengths(100)  # Don't truncate string values


# Get the dropdown options
dropdown_options = get_matrix_dropdown_options()
default_table = 'total'  # Default to total counts

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
        'y':0.97,
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

# Default MAPT gene information
DEFAULT_MAPT_GENE_INDEX = 17074
DEFAULT_MAPT_GENE_NAME = "MAPT"
DEFAULT_MAPT_GENE_ID = "ENSG00000186868"

@app.callback(
    Output('density-plot-tab2', 'figure'),
    [Input('search-input-tab2', 'value'),
     Input('search-input-tab2', 'options'),
     Input('window-dimensions', 'data')]
)
def update_density_plot(selected_gene, options, window_dimensions):
    # Default window dimensions if not available yet
    if not window_dimensions:
        window_dimensions = {'width': 1200, 'height': 800}

    # Create scaling factor and base font size
    scaling_factor = max(0.5, window_dimensions["width"] / 2540)
    base_font_size = 20 * scaling_factor
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
        margin=dict(l=50, r=20, t=60, b=50)
    )

    if selected_gene is None:
        # Set default to MAPT gene
        selected_gene = DEFAULT_MAPT_GENE_INDEX

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
            fig.add_vline(
                x=log10_mean_tmm,
                line_dash="dash",
                line_color="black",
                line_width=2,
                y0=0,
                y1=1
            )

            percentile = int(round(expression_percentile * 100, 0))
            suffix = "th"
            if percentile % 10 == 1 and percentile != 11: suffix = "st"
            elif percentile % 10 == 2 and percentile != 12: suffix = "nd"
            elif percentile % 10 == 3 and percentile != 13: suffix = "rd"

            fig.add_annotation(
                x=log10_mean_tmm,
                y=1,
                text=f"{gene_name} ({percentile}{suffix} percentile)",
                showarrow=False,
                font=dict(size=annotation_size, color="black", weight="bold"), # Scaled font
                xref="x",
                yref="paper",
                xanchor="right" if log10_mean_tmm > 2.5 else "left",
                align="right" if log10_mean_tmm > 2.5 else "left",
                yanchor="middle"
            )
            return fig

    except Exception as e:
        pass  # Silently fail for density plot updates

    return fig

@app.callback(
    Output('search-input-tab2', 'options'),
    [Input('search-input-tab2', 'search_value'),
     Input('search-input-tab2', 'value')]
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
                    SELECT gene_index, gene_name, gene_id
                    FROM gene_and_transcript_index_table 
                    WHERE gene_index = ?
                    GROUP BY gene_index, gene_name, gene_id
                    LIMIT 1
                """, [selected_value]).fetchone()
                
                if gene_result:
                    # Add this gene to the options
                    gene_index, gene_name, gene_id = gene_result
                    option = {
                        'label': f"{gene_name} ({gene_id})",
                        'value': gene_index
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
    
    # If no search value and no selected value, return MAPT gene as default
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
    Output('gene-level-plot-tab2', 'figure'),
    [Input('search-input-tab2', 'value'),
     Input('search-input-tab2', 'options'),
     Input('metadata-checklist-tab2', 'value'),
     Input('log-transform-option-tab2', 'value'),
     Input('plot-style-option-tab2', 'value'),
     Input('matrix-table-dropdown-tab2', 'value'),  # Add matrix table dropdown
     Input('window-dimensions', 'data')]
)
def update_gene_level_plot(selected_gene, options, selected_metadata, log_transform, plot_style, count_type, window_dimensions):
    if selected_gene is None:
        # Set default to MAPT gene
        selected_gene = DEFAULT_MAPT_GENE_INDEX

    # If no count type is selected, use total counts by default
    count_type = count_type if count_type else 'total'

    # Default window dimensions if not available yet
    if not window_dimensions:
        window_dimensions = {'width': 1200, 'height': 800}

    # Create scaling factor and base font size
    scaling_factor = max(0.5, window_dimensions["width"] / 2540)
    base_font_size = 20 * scaling_factor
    title_size = base_font_size * 1.125
    axis_label_size = base_font_size
    tick_label_size = base_font_size * 0.875
    legend_label_size = base_font_size * 0.875

    try:
        # Get the gene name from the options
        gene_name = None
        for option in options:
            if option['value'] == selected_gene:
                gene_name = option['label'].split(' (')[0]
                break
        if not gene_name:
            gene_name = selected_gene

        # Get the gene's data
        df = get_total_gene_data_with_metadata(selected_gene, with_polars=True)


        if df is None or len(df) == 0:
            return go.Figure()

        # Use the selected count type's columns
        tmm_col = f"cpm_normalized_tmm"
        # Check if the column exists
        if tmm_col not in df.columns:
            # Return an empty figure with an error message
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"Column '{tmm_col}' not found in the data",
                showarrow=False,
                font=dict(color="red", size=16)
            )
            return fig

        # Handle metadata selection
        if selected_metadata is None or len(selected_metadata) == 0:
            expression_hue = None
            df = df.with_columns([pl.lit("All Samples").alias("_group")])
            group_col = "_group"
        elif len(selected_metadata) == 1:
            expression_hue = selected_metadata[0]
            group_col = expression_hue
            df = df.filter(~pl.col(group_col).is_null())
        else:
            combined_col_name = "combined_metadata"
            combined_expr = pl.col(selected_metadata[0]).cast(pl.Utf8)
            for col_name in selected_metadata[1:]:
                combined_expr = combined_expr + pl.lit(" | ") + pl.col(col_name).cast(pl.Utf8)
            df = df.with_columns([combined_expr.alias(combined_col_name)])
            for col_name in selected_metadata:
                df = df.filter(~pl.col(col_name).is_null())
            expression_hue = combined_col_name
            group_col = combined_col_name

        # Apply log transformation
        if log_transform:
            log_tmm_col = f"log_{tmm_col}"
            df = df.with_columns([(pl.col(tmm_col).add(1).log10()).alias(log_tmm_col)])
            value_col = log_tmm_col
            axis_title = "Log TMM(per million)"
        else:
            value_col = tmm_col
            axis_title = "TMM(per million)"

        # Get unique values and colors
        unique_hue_values = df[group_col].unique().sort(descending=False).to_list()
        if expression_hue != None:
            custom_colors_list = get_n_colors(len(unique_hue_values), 'Plotly_r')
            color_map = {val: color for val, color in zip(unique_hue_values, custom_colors_list)}
        else:
            color_map = {val: 'grey' for val in unique_hue_values}

        # Convert to pandas
        pdf = df.to_pandas()

        # Create figure
        fig = go.Figure()

        # Add traces
        for group in unique_hue_values[::-1]:
            group_data = pdf[pdf[group_col] == group]
            if plot_style == "boxplot":
                fig.add_trace(go.Box(
                    x=group_data[value_col] if group_data[value_col].count() > 0 else [0],
                    name=str(group),
                    boxpoints='all', jitter=0.3, pointpos=0, orientation='h',
                    marker=dict(color='black', size=4),
                    line=dict(color='black', width=1),
                    fillcolor=color_map[group], opacity=1, boxmean=True,
                    customdata=group_data[['sample_id']] if 'sample_id' in group_data.columns else None,
                    hovertemplate='%{x}, %{customdata[0]}'
                ))
            else: # violin plot
                fig.add_trace(go.Violin(
                    x=group_data[value_col] if group_data[value_col].count() > 0 else [0],
                    name=str(group),
                    points='all', pointpos=0, orientation='h', jitter=0.3,
                    marker=dict(color='black', size=4),
                    line=dict(color='black', width=1),
                    fillcolor=color_map[group], opacity=1,
                    box_visible=False, spanmode='hard',
                    customdata=group_data[['sample_id']] if 'sample_id' in group_data.columns else None,
                    hovertemplate='%{x}, %{customdata[0]}'
                ))

        # Get proper display name for the count type
        count_type_display = {"total": "Total", "unique": "Unique", "fullLength": "Full Length"}
        count_type_name = count_type_display.get(count_type, count_type.capitalize())

        # Update layout with responsive fonts
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=50, r=20, t=60, b=50), # Adjusted top margin
            showlegend=expression_hue is not None, # Show legend only if hue is used
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=legend_label_size), # Scaled legend font
                traceorder="reversed"
            ),
            title={
                'text': f"Total Gene Expression: {gene_name}",
                'y':0.97, # Adjusted y position
                'x':0.02,
                'xanchor': 'left',
                'yanchor': 'top', # Anchor to top
                'font': {'size': title_size, 'weight': 'bold'} # Scaled title font
            },
            xaxis_title=axis_title,
            xaxis=dict(
                title_font=dict(size=axis_label_size), # Scaled axis label
                tickfont=dict(size=tick_label_size)  # Scaled tick label
            ),
            yaxis=dict(
                showticklabels=False,
                title=None
            )
        )

        return fig

    except Exception as e:
        return go.Figure()

@app.callback(
    Output("download-svg-tab2", "data"),
    [Input("download-button-tab2", "n_clicks")],
    [State('density-plot-tab2', 'figure'),
     State('gene-level-plot-tab2', 'figure'),
     State('isoform-plot-store-tab2', 'data'),
     State('search-input-tab2', 'value'),
     State('matrix-table-dropdown-tab2', 'value')]
)
def download_plots_as_svg(n_clicks, density_fig, gene_level_fig, isoform_fig, selected_gene, count_type):

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
        import tempfile
        import zipfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"{gene_name}_RNA_isoform_expression_plots.zip"
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
                gene_expr_svg_name = f"{gene_name}_gene_expression_plot.svg"
                real_fig = go.Figure(gene_level_fig)
                # Update layout for larger size and wider ratio
                real_fig.update_layout(
                    width=1200,  # Increased width
                    height=800,  # Increased height
                    margin=dict(l=80, r=40, t=80, b=60),  # Adjusted margins
                    title=dict(
                        font=dict(size=24),  # Larger title
                        y=0.975,
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
                
            # Export the RNA isoform plot if available
            if isoform_fig:
                isoform_svg_name = f"{gene_name}_RNA_isoform_plot_{count_type}.svg"
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
        import shutil
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
        dcc.Download(id="download-svg-tab2"),
        
        # Add a Store to hold the isoform plot figure
        dcc.Store(id="isoform-plot-store-tab2"),
        
        # Add a hidden gene-plot-container for the callback output
        html.Div(id="gene-plot-container-tab2", style={"display": "none"}),
        
        # Add these components that are required by the callback but were missing
        html.Div([
            dcc.RangeSlider(
                id='isoform-range-slider-tab2',
                min=1,
                max=10,
                step=1,
                value=[1, 5],
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            dcc.RadioItems(
                id='log-transform-option-tab2',
                options=[
                    {"label": "Original Values", "value": False},
                    {"label": "Log Transform (log10(x+1))", "value": True}
                ],
                value=False
            ),
            # Add this hidden component to satisfy the callback in callbacks.py
            dcc.Checklist(
                id='metadata-checklist-tab2',
                options=[
                    {"label": "Condition", "value": "condition"}
                ],
                value=['condition']  # Checked by default for dodging
            ),
            # Add plot-style-option component
            dcc.RadioItems(
                id='plot-style-option-tab2',
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
                        create_gene_search_dropdown(
                            id="search-input-tab2",
                            initial_value=DEFAULT_MAPT_GENE_INDEX,
                            initial_options=[{
                                'label': f"{DEFAULT_MAPT_GENE_NAME} ({DEFAULT_MAPT_GENE_ID})",
                                'value': DEFAULT_MAPT_GENE_INDEX
                            }]
                        )
                    ], width=3, id="tab2-search-col"),
                    dbc.Col([
                        create_section_header("Data Matrix:"),
                        create_matrix_dropdown(dropdown_options, default_table, id="matrix-table-dropdown-tab2")
                    ], width=3, id="tab2-matrix-col"),
                    dbc.Col([
                        create_section_header("Data Transformation:"),
                        create_content_card([
                            html.Div([
                                create_radio_items(
                                    id="log-transform-option-tab2",
                                    options=[
                                        {"label": "Original Values", "value": False},
                                        {"label": "Log Transform (log10(x+1))", "value": True}
                                    ],
                                    value=False,
                                    inline=True
                                )
                            ], className="radio-group-container dbc")
                        ])
                    ], width=3, id="tab2-transform-col"),
                    dbc.Col([
                        create_section_header("Plot Style:"),
                        create_content_card([
                            html.Div([
                                create_radio_items(
                                    id="plot-style-option-tab2",
                                    options=[
                                        {"label": "Box Plot", "value": "boxplot"},
                                        {"label": "Violin Plot", "value": "violin"}
                                    ],
                                    value="boxplot",
                                    inline=True
                                )
                            ], className="radio-group-container dbc")
                        ])
                    ], width=3, id="tab2-plot-style-col"),
                ], className="mb-4 dbc", id="tab2-row1"),

                # Second row - RSID plot
                dbc.Row([
                    dbc.Col([
                        create_content_card(
                            dbc.Spinner(
                                html.Div([
                                    # Matrix content div
                                    html.Div(
                                        id='matrix-content-tab2',
                                        style={
                                            "background-color": "#ffffff",
                                            "padding": "10px",
                                            "border-radius": "5px", 
                                            "border": "1px solid rgba(0, 0, 0, 0.1)",
                                            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                                            "width": "100%",
                                            "height": "100%",
                                            "min-height": "500px", # Increased from 400px to match the graph's min-height
                                            "display": "flex",
                                            "justify-content": "center",
                                            "align-items": "center"
                                        }
                                    )
                                ]),
                                color="primary",
                                type="grow",
                                spinner_style={"width": "3rem", "height": "3rem"}
                            )
                        )
                    ], width=12),
                ], 
                className="mb-4 dbc",
                id="tab2-row2",
                style={"height": "90vh", "min-height": "600px"}  # Add min-height to ensure initial rendering
                ),

                # Third row - three columns
                dbc.Row([
                    dbc.Col([
                        create_section_header("Show data separated by:"),
                        create_content_card([
                            html.Div([
                                create_checklist(
                                    id="metadata-checklist-tab2",
                                    options=[
                                        {"label": "Condition", "value": "condition"}
                                    ],
                                    value=['condition']  # Checked by default for dodging
                                )
                            ])
                        ])
                    ], width=3, id="tab2-metadata-col"),
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
                                    id='isoform-range-slider-tab2',
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
                    ], width=4, id="tab2-range-col"),
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
                                    id="download-button-tab2",
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
                    ], width=5, id="tab2-col3-3"),
                ], className="mb-4 dbc", id="tab2-row3"),

                # Fourth row - two columns
                dbc.Row([
                    dbc.Col([
                        create_section_header(""),
                        create_content_card([
                            dcc.Graph(
                                id='density-plot-tab2',
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
                    ], width=6, id="tab2-col4-1"),
                    dbc.Col([
                        create_section_header(""),
                        create_content_card([
                            dcc.Graph(
                                id='gene-level-plot-tab2',
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': False,
                                    'modeBarButtonsToRemove': ['autoScale2d'],
                                    'displaylogo': False
                                },
                                style={'height': '400px'}
                            )
                        ])
                    ], width=6, id="tab2-col4-2"),
                ], className="mb-4 dbc", id="tab2-row4"),
            ], id="tab2-card-body")
        ],
        id="tab2-card",
        style={
            "background-color": "#ffffff",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "6px",
            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
        })
    ], 
    fluid=True,  # Makes the container full-width
    id="tab2-container",
    style={
        "max-width": "98%",  # Use 98% of the viewport width
        "margin": "0 auto",  # Center the container
        "padding": "10px"    # Add some padding
    })

@callback(
    [Output("tab2-container", "style"),
     Output("tab2-row1", "className"),
     Output("tab2-search-col", "width"),
     Output("tab2-matrix-col", "width"),
     Output("tab2-transform-col", "width"),
     Output("tab2-plot-style-col", "width"),
     Output("tab2-row3", "className"),
     Output("tab2-metadata-col", "width"),
     Output("tab2-range-col", "width"),
     Output("tab2-col3-3", "width"),
     Output("tab2-row4", "className"),
     Output("tab2-col4-1", "width"),
     Output("tab2-col4-2", "width")],
    [Input("window-dimensions", "data")]
)
def update_tab2_responsiveness(dimensions):
    if not dimensions:
        # Default styles if no dimensions available
        return (
            {"max-width": "98%", "margin": "0 auto", "padding": "10px"},
            "mb-4 dbc", 3, 3, 3, 3,
            "mb-4 dbc", 4, 4, 4,
            "mb-4 dbc", 6, 6
        )
    
    width = dimensions.get('width', 1200)
    
    # Base styles
    container_style = {"max-width": "98%", "margin": "0 auto", "padding": "10px"}
    row1_class = "mb-4 dbc"
    search_col_width = 3
    matrix_col_width = 3
    transform_col_width = 3
    plot_style_col_width = 3
    
    row3_class = "mb-4 dbc"
    metadata_col_width = 3
    range_col_width = 4
    col3_3_width = 5
    
    row4_class = "mb-4 dbc"
    col4_1_width = 6
    col4_2_width = 6
    
    # Responsive adjustments based on width
    if width < 576:  # Extra small devices
        container_style.update({"padding": "5px", "max-width": "100%"})
        row1_class = "mb-2 dbc flex-column"
        search_col_width = 12
        matrix_col_width = 12
        transform_col_width = 12
        plot_style_col_width = 12
        
        row3_class = "mb-2 dbc flex-column"
        metadata_col_width = 12
        range_col_width = 12
        col3_3_width = 12
        
        row4_class = "mb-2 dbc flex-column"
        col4_1_width = 12
        col4_2_width = 12
        
    elif width < 768:  # Small devices
        container_style.update({"padding": "8px"})
        row1_class = "mb-3 dbc"
        search_col_width = 6
        matrix_col_width = 6
        transform_col_width = 6
        plot_style_col_width = 6
        
        row3_class = "mb-3 dbc"
        metadata_col_width = 12
        range_col_width = 6
        col3_3_width = 6
        
        row4_class = "mb-3 dbc"
        col4_1_width = 12
        col4_2_width = 12
        
    elif width < 992:  # Medium devices
        container_style.update({"padding": "10px"})
        transform_col_width = 3
        plot_style_col_width = 3
        metadata_col_width = 3
        range_col_width = 4
        col3_3_width = 5
        
    return (
        container_style,
        row1_class, search_col_width, matrix_col_width, transform_col_width, 
        plot_style_col_width,
        row3_class, metadata_col_width, range_col_width, col3_3_width,
        row4_class, col4_1_width, col4_2_width
    ) 

# Add the main RNApysoforms plot callback
@app.callback(
    [Output('matrix-content-tab2', 'children'),
     Output('isoform-plot-store-tab2', 'data')],
    [Input('matrix-table-dropdown-tab2', 'value'),
     Input('search-input-tab2', 'value'),
     Input('metadata-checklist-tab2', 'value'),
     Input('log-transform-option-tab2', 'value'),
     Input('plot-style-option-tab2', 'value'),
     Input('window-dimensions', 'data'), # Assuming window-dimensions is app-wide
     Input('isoform-range-slider-tab2', 'value')]
)
def update_gene_plot_tab2(selected_table, selected_gene, selected_metadata, log_transform, plot_style, window_dimensions, isoform_range):
    # Set default to MAPT gene if no gene is selected
    if selected_gene is None:
        selected_gene = DEFAULT_MAPT_GENE_INDEX
        
    # If no count type is selected, use total counts by default
    count_type = selected_table if selected_table else 'total'

    # Default window dimensions if not available yet
    if not window_dimensions:
        window_dimensions = {'width': 1200, 'height': 800} # Provide default dimensions

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
            SELECT g.gene_index, g.gene_name, g.gene_id
            FROM gene_and_transcript_index_table g
            WHERE g.gene_index = ?
            GROUP BY g.gene_index, g.gene_name, g.gene_id
            LIMIT 1
        """, [selected_gene]).fetchone()

        if not gene_info:
            return html.Div(
                html.P("Gene not found in the annotation table",
                      style={"color": "#666666", "margin": 0}),
                style={
                    "height": "100%", "width": "100%", "display": "flex",
                    "justify-content": "center", "align-items": "center",
                    "min-height": "500px", "background-color": "#f8f9fa", "border-radius": "6px"
                }
            ), None

        # Unpack gene_info - gene_index, gene_name, gene_id
        gene_index, gene_name, gene_id = gene_info

        # Get data with metadata - use all_transcript_data table
        expression = get_gene_data_with_metadata(gene_index, with_polars=True, limit=None)

        # Get annotation data for this gene - join with gene_and_transcript_index_table to get gene_id and transcript_id
        annotation_query = """
            SELECT a.*, g.gene_id, g.gene_name, g.transcript_id, g.seqnames
            FROM gene_and_transcript_index_table g
            JOIN transcript_annotation a
                ON a.gene_index = g.gene_index AND a.transcript_index = g.transcript_index
            WHERE a.gene_index = ?
        """
        annotation = duck_conn.execute(annotation_query, [gene_index]).pl()

        ## Get the transcript_id, gene_id, gene_name from the annotation table
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


        # Extract gene annotation details for plot title and subtitle
        chromosome = annotation["seqnames"][0] if "seqnames" in annotation.columns else "Unknown"
        strand = annotation["strand"][0] if "strand" in annotation.columns else "?"
        min_start = annotation["start"].min() if "start" in annotation.columns else "Unknown"
        max_end = annotation["end"].max() if "end" in annotation.columns else "Unknown"

        #############################################################
        # Process the annotation data
        annotation = RNApy.shorten_gaps(annotation)

        # Select the correct columns based on count_type
        count_col = f"{count_type}_counts"
        tmm_col = f"{count_type}_cpm_normalized_tmm"
        abundance_col = f"{count_type}_relative_abundance"

        # Make sure these columns exist
        if count_col not in expression.columns or tmm_col not in expression.columns or abundance_col not in expression.columns:
            error_cols = [col for col in [count_col, tmm_col, abundance_col] if col not in expression.columns]
            error_message = f"Missing columns in expression data: {', '.join(error_cols)}"
            return html.Div(
                html.P(error_message, style={"color": "#dc3545", "margin": 0}),
                style={"height": "100%", "width": "100%", "display": "flex", "justify-content": "center", 
                      "align-items": "center", "min-height": "500px", "background-color": "#f8f9fa", "border-radius": "6px"}
            ), None

        # Convert 1-based slider values to 0-based indices for the function
        # Add 1 to the upper bound to make it inclusive
        zero_based_range = [isoform_range[0] - 1, isoform_range[1]]
        expression, annotation = order_transcripts_by_expression(
            annotation_df=annotation,
            expression_df=expression,
            expression_column=tmm_col,
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
            combined_expr = pl.col(selected_metadata[0]).cast(pl.Utf8)
            for col_name in selected_metadata[1:]:
                combined_expr = combined_expr + pl.lit(" | ") + pl.col(col_name).cast(pl.Utf8)
            expression = expression.with_columns([combined_expr.alias(combined_col_name)])
            for col_name in selected_metadata: # Filter out nulls AFTER combining
                 expression = expression.filter(~pl.col(col_name).is_null())
            expression_hue = combined_col_name

        # Apply log transformation if selected
        if log_transform:
            log_count_col = f"log_{count_col}"
            log_tmm_col = f"log_{tmm_col}"
            expression = expression.with_columns([
                (pl.col(count_col).add(1).log10()).alias(log_count_col),
                (pl.col(tmm_col).add(1).log10()).alias(log_tmm_col)
            ])
            expression_columns = [log_count_col, log_tmm_col, abundance_col]
        else:
            expression_columns = [count_col, tmm_col, abundance_col]

        hue_values = ["protein_coding", "retained_intron", "protein_coding_CDS_not_defined", "nonsense_mediated_decay",
                              "new_low_confidence", "new_high_confidence", "lncRNA", "other"]
        
        colormap = {val: color for val, color in zip(hue_values, get_n_colors(len(hue_values), 'Plotly'))}

        # Update the trace creation to use the correct columns
        trace_params = {
            "annotation": annotation, "expression_matrix": expression,
            "x_start": "rescaled_start", "x_end": "rescaled_end", "y": "transcript_id",
            "annotation_hue": "transcript_biotype", "hover_start": "start", "hover_end": "end",
            "expression_columns": expression_columns, "marker_size": 5*scaling_factor,
            "arrow_size": 12*scaling_factor, "expression_plot_style": plot_style,
            "annotation_color_map": colormap
        }

        ## Create appropriate color palette for the expression_hue
        if expression_hue is not None:
            # Ensure expression_hue column exists before trying to access it
            if expression_hue in expression.columns:
                 unique_hue_values = expression[expression_hue].unique().sort().to_list()
                 if len(unique_hue_values) > 0:
                     custom_colors_list = get_n_colors(len(unique_hue_values), 'Plotly_r')
                     color_map = {val: color for val, color in zip(unique_hue_values, custom_colors_list)}
                     trace_params["expression_color_map"] = color_map
                 trace_params["expression_hue"] = expression_hue
                 expression = expression.sort(by=expression_hue, descending=True) # Sort AFTER potential filtering
                 trace_params["expression_matrix"] = expression # Update matrix after sorting
            else:
                 # Handle case where combined metadata resulted in no rows or column doesn't exist
                 expression_hue = None # Reset hue if column isn't valid
                 trace_params.pop("expression_hue", None) # Remove from params if set

        traces = RNApy.make_traces(**trace_params)

        # Get proper display names for the count type
        count_type_display = {"total": "Total", "unique": "Unique", "fullLength": "Full Length"}
        count_type_name = count_type_display.get(count_type, count_type.capitalize())

        # Create appropriate subplot titles based on transformation
        if log_transform:
            subplot_titles = ["Transcript Structure", f"Log {count_type_name} Counts", f"Log TMM(per million)", f"Relative Abundance(%)"]
        else:
            subplot_titles = ["Transcript Structure", f"{count_type_name} Counts", f"TMM(per million)", f"Relative Abundance(%)"]

        # Use the dynamic dimensions for your plot
        fig = RNApy.make_plot(traces=traces,
                    subplot_titles=subplot_titles,
                    boxgap=0.15, boxgroupgap=0.1, width=plot_width, height=plot_height,
                    legend_font_size=20*scaling_factor, yaxis_font_size=20*scaling_factor,
                    xaxis_font_size=20*scaling_factor, subplot_title_font_size=24*scaling_factor,
                    template="ggplot2", hover_font_size=14*scaling_factor,
                    legend_title_font_size=20*scaling_factor, column_widths=[0.4, 0.2, 0.2, 0.2])

        # Calculate the expanded x-axis range for the first subplot
        x_min = annotation["rescaled_start"].min() if "rescaled_start" in annotation.columns else 0
        x_max = annotation["rescaled_end"].max() if "rescaled_end" in annotation.columns else 1
        padding = (x_max - x_min) * 0.05
        new_x_min = x_min - padding
        new_x_max = x_max + padding
        fig.update_xaxes(range=[new_x_min, new_x_max], showticklabels=False, showline=False, zeroline=False, ticks="", row=1, col=1)
        fig.update_xaxes(range=[-0.5, 100.5], tickvals=[0, 20, 40, 60, 80, 100], row=1, col=4)

        # Main title annotation:
        fig.add_annotation(x=0.5, y=1.12, xref='paper', yref='paper', text=f"{gene_name} ({gene_id})", showarrow=False, xanchor="center", yanchor="top", font=dict(size=26 * scaling_factor))

        # Subtitle annotation:
        fig.add_annotation(x=0.5, y=1.08, xref='paper', yref='paper', text=f"Region: chr{chromosome}({strand}):{min_start}-{max_end}", showarrow=False, xanchor="center", yanchor="top", font=dict(size=18 * scaling_factor))

        # Update subplot titles separately
        for i, annot in enumerate(fig['layout']['annotations']):
             if i < len(subplot_titles): # Only update the subplot titles
                if i == 0: annot["x"] = 0
                elif i == 1: annot["x"] = 0.395
                elif i == 2: annot["x"] = 0.602
                else: annot["x"] = 0.812
                annot["xanchor"] = "left"


        return dcc.Graph(
            figure=fig,
            style={
                "height": "100%",
                "width": "100%",
                "min-height": "500px"  # Set explicit minimum height for initial rendering
            },
            config={
                "responsive": True,
                "displayModeBar": True,
                "scrollZoom": False,
                "modeBarButtonsToRemove": ["autoScale2d"],
                "displaylogo": False
            }
        ), fig # Return figure for store

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        column_info = ""
        try:
            if 'expression' in locals() and expression is not None: column_info = f"Available columns: {', '.join(expression.columns)}"
        except: column_info = "Could not retrieve column names"
        return html.Div([
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
        }), None


# Add the slider update callback
@app.callback(
    [Output('isoform-range-slider-tab2', 'max'),
     Output('isoform-range-slider-tab2', 'marks'),
     Output('isoform-range-slider-tab2', 'value')],
    [Input('search-input-tab2', 'value')]
)
def update_slider_range_tab2(selected_gene):
    if selected_gene is None:
        # Use MAPT gene as default
        selected_gene = DEFAULT_MAPT_GENE_INDEX

    try:
        # Query to get the number of transcripts for this gene
        transcript_count = duck_conn.execute("""
            SELECT COUNT(DISTINCT transcript_index)
            FROM gene_and_transcript_index_table
            WHERE gene_index = ?
        """, [selected_gene]).fetchone()[0]

        if not transcript_count or transcript_count == 0: # Handle case where gene has 0 transcripts
            # Fallback if no transcripts found
            marks = {1: '1'} # Min 1 mark
            return 1, marks, [1, 1]

        # Max value for slider
        max_val = max(1, transcript_count) # Ensure max is at least 1

        # Create marks for the actual number of transcripts, ensuring reasonable spacing if count is high
        if transcript_count <= 20:
             marks = {i: str(i) for i in range(1, transcript_count + 1)}
        else: # Reduce marks for large numbers
             step = max(1, transcript_count // 10) # Aim for ~10 marks
             marks = {i: str(i) for i in range(1, transcript_count + 1, step)}
             if transcript_count not in marks: # Ensure last value is a mark
                 marks[transcript_count] = str(transcript_count)

        # Default slider value
        new_value = [1, min(5, max_val)]

        return max_val, marks, new_value

    except Exception as e:
        # Fallback to default values in case of error
        marks = {i: str(i) for i in range(1, 11)}
        return 10, marks, [1, 5] 