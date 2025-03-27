# File: app/tabs/tab2.py
# Contains the layout for Tab 2.

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from app import app
from app.utils.db_utils import get_matrix_dropdown_options, search_genes, duck_conn
from app.utils.ui_components import (
    create_gene_search_dropdown,
    create_matrix_dropdown,
    create_section_header,
    create_content_card,
    create_radio_items,
    create_checklist
)

# Get the dropdown options
dropdown_options = get_matrix_dropdown_options()
default_table = dropdown_options[0]['value'] if dropdown_options else None

# Store the last valid search options to prevent them from disappearing
last_valid_options = []
last_search_value = None  # Store the last search value

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

layout = dbc.Container([
    dbc.Card([
        dbc.CardBody([
            # First row - four columns, with search, dropdown, and visualization options
            dbc.Row([
                dbc.Col([
                    create_section_header("Search Gene:"),
                    create_gene_search_dropdown()
                ], width=3),
                dbc.Col([
                    create_section_header("Select a data matrix to analyze:"),
                    create_matrix_dropdown(dropdown_options, default_table)
                ], width=3),
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
                        ], className="radio-group-container")
                    ])
                ], width=3),
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
                        ], className="radio-group-container")
                    ])
                ], width=3)
            ], className="mb-4 dbc"),

            # Second row - one column for matrix content
            dbc.Row([
                dbc.Col([
                    create_content_card(
                        dbc.Spinner(
                            html.Div(
                                id='matrix-content',
                                style={
                                    "height": "90hv",
                                    "width": "100%"
                                }
                            ),
                            color="primary",
                            type="grow",
                            spinner_style={"width": "3rem", "height": "3rem"}
                        )
                    )
                ], width=12)
            ], 
            className="mb-4 dbc", 
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
                ], width=4),
                dbc.Col([
                    create_section_header("Column 3-2"),
                    create_content_card("Content 3-2")
                ], width=4),
                dbc.Col([
                    create_section_header("Column 3-3"),
                    create_content_card("Content 3-3")
                ], width=4),
            ], className="mb-4 dbc"),

            # Fourth row - two columns
            dbc.Row([
                dbc.Col([
                    create_section_header("Column 4-1"),
                    create_content_card("Content 4-1")
                ], width=6),
                dbc.Col([
                    create_section_header("Column 4-2"),
                    create_content_card("Content 4-2")
                ], width=6),
            ], className="mb-4 dbc")
        ])
    ],
    style={
        "background-color": "#ffffff",
        "border": "1px solid rgba(0, 0, 0, 0.1)",
        "border-radius": "6px",
        "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
    })
], 
fluid=True,  # Makes the container full-width
style={
    "max-width": "98%",  # Use 98% of the viewport width
    "margin": "0 auto",  # Center the container
    "padding": "10px"    # Add some padding
}) 