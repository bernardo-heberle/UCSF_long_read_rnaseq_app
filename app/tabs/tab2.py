# File: app/tabs/tab2.py
# Contains the layout for Tab 2.

from dash import html, dcc, Input, Output, callback, no_update, State, MATCH, ALL, ClientsideFunction
import dash_bootstrap_components as dbc
from app import app
from app.utils.db_utils import get_matrix_dropdown_options, search_genes, duck_conn, get_gene_density_data, get_total_gene_data_with_metadata
from app.utils.ui_components import (
    create_gene_search_dropdown,
    create_matrix_dropdown,
    create_section_header,
    create_content_card,
    create_radio_items,
    create_checklist
)
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import os
import polars as pl
import io
import base64
import kaleido
import json

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

@app.callback(
    Output('density-plot', 'figure'),
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
    Output('search-input', 'options'),
    [Input('search-input', 'search_value'),
     Input('search-input', 'value')]
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
                    SELECT gene_id, gene_name 
                    FROM transcript_annotation 
                    WHERE gene_id = ?
                    LIMIT 1
                """, [selected_value]).fetchone()
                
                if gene_result:
                    # Add this gene to the options
                    gene_id, gene_name = gene_result
                    option = {
                        'label': f"{gene_name} ({gene_id})",
                        'value': gene_id
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
    Output('gene-level-plot', 'figure'),
    [Input('search-input', 'value'),
     Input('search-input', 'options'),
     Input('metadata-checklist', 'value'),
     Input('log-transform-option', 'value'),
     Input('plot-style-option', 'value')]
)
def update_gene_level_plot(selected_gene, options, selected_metadata, log_transform, plot_style):
    if not selected_gene:
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
            
        # Get the gene's data
        df = get_total_gene_data_with_metadata(selected_gene, with_polars=True)
        
        if df is None or len(df) == 0:
            return go.Figure()
            
        # Handle metadata selection for expression_hue
        if selected_metadata is None or len(selected_metadata) == 0:
            # No metadata selected, don't use any expression_hue
            expression_hue = None
            # Create a basic placeholder category
            df = df.with_columns([
                pl.lit("All Samples").alias("_group")
            ])
            group_col = "_group"
        elif len(selected_metadata) == 1:
            # If only one metadata column is selected, use it directly
            expression_hue = selected_metadata[0]
            group_col = expression_hue
            # Filter out nulls
            df = df.filter(~pl.col(group_col).is_null())
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
    Output("download-svg", "data"),
    [Input("download-button", "n_clicks")],
    [State('density-plot', 'figure'),
     State('gene-level-plot', 'figure'),
     State('isoform-plot-store', 'data'),
     State('search-input', 'value')]
)
def download_plots_as_svg(n_clicks, density_fig, gene_level_fig, isoform_fig, selected_gene):
    if n_clicks is None or not n_clicks or selected_gene is None:
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
        import tempfile
        import zipfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"{gene_name}_plots.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Create a zip file
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Export the density plot
            if density_fig:
                print("Creating density plot SVG")
                density_svg_name = f"{gene_name}_density_distribution_plot.svg"
                real_fig = go.Figure(density_fig)
                density_svg = real_fig.to_image(format="svg").decode('utf-8')
                zipf.writestr(density_svg_name, density_svg)
                print(f"Added density plot to zip: {density_svg_name}")
                
            # Export the gene expression plot    
            if gene_level_fig:
                print("Creating gene expression plot SVG")
                gene_expr_svg_name = f"{gene_name}_gene_expression_plot.svg"
                real_fig = go.Figure(gene_level_fig)
                gene_expr_svg = real_fig.to_image(format="svg").decode('utf-8')
                zipf.writestr(gene_expr_svg_name, gene_expr_svg)
                print(f"Added gene expression plot to zip: {gene_expr_svg_name}")
                
            # Export the RNA isoform plot if available
            if isoform_fig:
                print("Creating isoform plot SVG")
                isoform_svg_name = f"{gene_name}_RNA_isoform_plot.svg"
                try:
                    real_fig = go.Figure(isoform_fig)
                    isoform_svg = real_fig.to_image(format="svg").decode('utf-8')
                    zipf.writestr(isoform_svg_name, isoform_svg)
                    print(f"Successfully added isoform plot to zip: {isoform_svg_name}")
                except Exception as isoform_error:
                    print(f"Error creating isoform SVG: {isoform_error}")
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
                print("No isoform figure found, creating placeholder")
                isoform_svg_name = f"{gene_name}_RNA_isoform_plot.svg"
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
        # Download component for the SVG files
        dcc.Download(id="download-svg"),
        
        # Store component for the isoform plot
        dcc.Store(id="isoform-plot-store"),
        
        # Add a hidden div to load the client-side JavaScript
        html.Div(id="clientside-script-loader", style={"display": "none"}),
        
        # JavaScript to capture isoform plot figure
        html.Script(
            id="clientside-script",
            children="""
            if (!window.clientside) {
                window.clientside = {};
            }
            
            window.clientside.captureIsoformPlot = function(matrixContent) {
                if (!matrixContent) return null;
                
                // Function to find figure in a nested structure
                function findFigure(obj) {
                    if (!obj) return null;
                    
                    // Check if this is a Dash graph component
                    if (obj.type === 'Graph' && obj.props && obj.props.figure) {
                        return obj.props.figure;
                    }
                    
                    // Check if it's a component with children
                    if (obj.props && obj.props.children) {
                        let children = obj.props.children;
                        
                        // Handle array of children
                        if (Array.isArray(children)) {
                            for (let child of children) {
                                let result = findFigure(child);
                                if (result) return result;
                            }
                        } 
                        // Handle single child
                        else {
                            return findFigure(children);
                        }
                    }
                    
                    return null;
                }
                
                // Try to find the figure
                const figure = findFigure(matrixContent);
                console.log("Clientside result:", figure ? "Found figure" : "No figure found");
                return figure;
            };
            """
        ),
        
        dbc.Card([
            dbc.CardBody([
                # First row - four columns, with search, dropdown, and visualization options
                dbc.Row([
                    dbc.Col([
                        create_section_header("Search Gene:"),
                        create_gene_search_dropdown()
                    ], width=3, id="tab2-search-col"),
                    dbc.Col([
                        create_section_header("Select a data matrix to analyze:"),
                        create_matrix_dropdown(dropdown_options, default_table)
                    ], width=3, id="tab2-matrix-col"),
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
                    ], width=3, id="tab2-transform-col"),
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
                    ], width=3, id="tab2-plot-style-col"),
                ], className="mb-4 dbc", id="tab2-row1"),

                # Second row - one column for matrix content
                dbc.Row([
                    dbc.Col([
                        create_content_card(
                            dbc.Spinner(
                                html.Div([
                                    # Matrix content div
                                    html.Div(
                                        id='matrix-content',
                                        style={
                                            "width": "100%"
                                        }
                                    ),
                                    # Add gene-plot-container here to ensure it exists in the layout
                                    html.Div(
                                        id='gene-plot-container',
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
                    ], width=12, id="tab2-matrix-content-col")
                ], 
                className="mb-4 dbc",
                id="tab2-row2",
                style={"height": "90hv"}
                ),

                # Third row - three columns
                dbc.Row([
                    dbc.Col([
                        create_section_header("Show expression separated by:"),
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
                                    value=["ebbert_ad_status"]
                                )
                            ])
                        ])
                    ], width=4, id="tab2-metadata-col"),
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
                                    id="download-button",
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
                    ], width=4, id="tab2-col3-3"),
                ], className="mb-4 dbc", id="tab2-row3"),

                # Fourth row - two columns
                dbc.Row([
                    dbc.Col([
                        create_section_header(""),
                        create_content_card([
                            dcc.Graph(
                                id='density-plot',
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
                                id='gene-level-plot',
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
    metadata_col_width = 4
    range_col_width = 4
    col3_3_width = 4
    
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
        metadata_col_width = 4
        range_col_width = 4
        col3_3_width = 4
        
    return (
        container_style,
        row1_class, search_col_width, matrix_col_width, transform_col_width, 
        plot_style_col_width,
        row3_class, metadata_col_width, range_col_width, col3_3_width,
        row4_class, col4_1_width, col4_2_width
    ) 