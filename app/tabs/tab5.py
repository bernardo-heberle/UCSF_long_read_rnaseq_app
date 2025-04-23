# File: app/tabs/tab5.py
# Contains the layout for Tab 5 (formerly Tab 6).

from dash import html, Input, Output, callback, dcc
import dash_bootstrap_components as dbc

def layout():
    return html.Div([
        html.H2("Download Data", 
            id="tab5-header",
            className="mb-4 text-center",
            style={"font-weight": "300", "letter-spacing": "0.5px"}
        ),
        
        # Subtle divider matching tab0 style
        html.Hr(style={"margin": "1rem 15%", "opacity": "0.4"}),
        
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.A(
                            dbc.Button(
                                "Sample Metadata",
                                id="tab5-button-1",
                                color="primary",
                                outline=True,
                                className="w-100 py-3 mb-3",
                                style={"border-radius": "4px", "letter-spacing": "0.5px", "font-weight": "600", "font-size": "1.1rem"}
                            ),
                            href="https://zenodo.org/records/15238190/files/metadata.tsv?download=1",
                            target="_blank"
                        ),
                    ], className="mb-3"),
                    
                    html.Div([
                        html.A(
                            dbc.Button(
                                "Differential Expression/Usage Analysis Results",
                                id="tab5-button-2",
                                color="primary",
                                outline=True,
                                className="w-100 py-3 mb-3",
                                style={"border-radius": "4px", "letter-spacing": "0.5px", "font-weight": "600", "font-size": "1.1rem"}
                            ),
                            href="https://zenodo.org/records/15238190/files/differential_expression_results.zip?download=1",
                            target="_blank"
                        ),
                    ], className="mb-3"),
                    
                    html.Div([
                        html.A(
                            dbc.Button(
                                "Counts Matrices and Annotations",
                                id="tab5-button-3",
                                color="primary",
                                outline=True,
                                className="w-100 py-3 mb-3",
                                style={"border-radius": "4px", "letter-spacing": "0.5px", "font-weight": "600", "font-size": "1.1rem"}
                            ),
                            href="https://zenodo.org/records/15238190/files/counts_matrices_and_annotations.zip?download=1",
                            target="_blank"
                        ),
                    ], className="mb-3"),
                    
                    html.Div([
                        html.A(
                            dbc.Button([
                                html.Span("Quantitative Trait Loci Analysis Results and Genotyping Data"),
                                html.Br(),
                                html.Span(
                                    "(Larger Files: 9.1 GB Zipped Directory, 71.1 GB Unzipped)",
                                    style={"font-size": "0.85em", "opacity": "0.85"}
                                )
                            ],
                                id="tab5-button-4",
                                color="primary",
                                outline=True,
                                className="w-100 py-3 mb-3",
                                style={"border-radius": "4px", "letter-spacing": "0.5px", "font-weight": "600", "font-size": "1.1rem", "white-space": "normal", "text-align": "center"}
                            ),
                            href="https://zenodo.org/records/15238190/files/QTL_results.zip?download=1",
                            target="_blank"
                        ),
                    ], className="mb-3"),
                    
                    # New Raw Data (Synapse) button
                    html.Div([
                        dbc.Button(
                            "Raw Data (FASTQ files, sequencing summary files, BAM files, PLINK files, and more)",
                            id="tab5-button-5",
                            color="primary",
                            outline=True,
                            className="w-100 py-3 mb-3",
                            style={"border-radius": "4px", "letter-spacing": "0.5px", "font-weight": "600", "font-size": "1.1rem"},
                            disabled=True
                        ),
                    ], className="mb-3"),
                    
                    # New Processed Data (Zenodo) button
                    html.Div([
                        dbc.Button(
                            "Processed Data (Counts Matrices, Annotations, Processed genotype files, QTL results, reference files, and more)",
                            id="tab5-button-6",
                            color="primary",
                            outline=True,
                            className="w-100 py-3 mb-3",
                            style={"border-radius": "4px", "letter-spacing": "0.5px", "font-weight": "600", "font-size": "1.1rem"},
                            disabled=True
                        ),
                    ], className="mb-3"),
                ], width=12, md=8, lg=6, className="mx-auto")
            ], className="justify-content-center")
        ], 
        id="tab5-container",
        fluid=True,
        className="px-4 py-3")
    ], id="tab5-content")

@callback(
    [Output("tab5-content", "style"),
     Output("tab5-header", "style"),
     Output("tab5-button-1", "style"),
     Output("tab5-button-2", "style"),
     Output("tab5-button-3", "style"),
     Output("tab5-button-4", "style"),
     Output("tab5-button-5", "style"),
     Output("tab5-button-6", "style")],
    [Input("window-dimensions", "data")]
)
def update_tab5_responsiveness(dimensions):
    if not dimensions:
        # Default button style
        button_style = {
            "border-radius": "4px", 
            "letter-spacing": "0.5px", 
            "font-weight": "600", 
            "font-size": "1.1rem"
        }
        
        large_button_style = {
            "border-radius": "4px", 
            "letter-spacing": "0.5px", 
            "font-weight": "600", 
            "font-size": "1.1rem",
            "white-space": "normal",
            "text-align": "center"
        }
        
        # Default styles
        return (
            {},  # container style
            {"font-weight": "300", "letter-spacing": "0.5px"},  # header style
            button_style,
            button_style,
            button_style,
            large_button_style,
            button_style,
            button_style
        )
    
    width = dimensions.get('width', 1200)
    
    # Base styles
    container_style = {}
    header_style = {"font-weight": "300", "letter-spacing": "0.5px"}
    
    # Base button style
    button_style = {
        "border-radius": "4px", 
        "letter-spacing": "0.5px", 
        "font-weight": "600"
    }
    
    # Special style for the large button
    large_button_style = {
        "border-radius": "4px", 
        "letter-spacing": "0.5px", 
        "font-weight": "600",
        "white-space": "normal",
        "text-align": "center"
    }
    
    # Responsive adjustments
    if width < 576:  # Extra small
        header_style.update({"font-size": "1.6rem"})
        button_style.update({"font-size": "0.9rem", "padding": "8px 5px"})
        large_button_style.update({"font-size": "0.9rem", "padding": "8px 5px"})
    elif width < 768:  # Small
        header_style.update({"font-size": "1.8rem"})
        button_style.update({"font-size": "1.0rem", "padding": "10px 8px"})
        large_button_style.update({"font-size": "1.0rem", "padding": "10px 8px"})
    elif width < 992:  # Medium
        header_style.update({"font-size": "1.9rem"})
        button_style.update({"font-size": "1.1rem"})
        large_button_style.update({"font-size": "1.1rem"})
    else:  # Large
        header_style.update({"font-size": "2rem"})
        button_style.update({"font-size": "1.2rem"})
        large_button_style.update({"font-size": "1.2rem"})
    
    return (
        container_style, 
        header_style,
        button_style,
        button_style,
        button_style,
        large_button_style,
        button_style,
        button_style
    ) 