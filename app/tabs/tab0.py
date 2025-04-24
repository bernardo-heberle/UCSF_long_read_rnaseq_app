# File: app/tabs/tab0.py
# Defines the layout for the Home tab

from dash import html, dcc, Input, Output, callback, State, callback_context, no_update
import dash
import dash_bootstrap_components as dbc
import base64
import os
from app import app  # Import the app instance



# Define the layout for the Home tab
def layout():

    ## Try multiple approaches to locate the image
    # 1. Direct path relative to current directory
    image_filename = './assets/study_design.png'
    # 2. Path relative to the app directory
    if not os.path.exists(image_filename):
        image_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'study_design.png')
    # 3. Path relative to current working directory
    if not os.path.exists(image_filename):
        image_filename = os.path.join(os.getcwd(), 'assets', 'study_design.png')
    
    # Try to load the image file
    try:
        with open(image_filename, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('ascii')
        image_src = f'data:image/png;base64,{encoded_image}'
    except Exception as e:
        print(f"Error loading image: {e}")
        # Fallback to direct URL if file not found
        image_src = '/assets/study_design.png'

    return html.Div([
        # Tools section with sleek styling
        html.Div([
            html.H2([
                "Navigate through the tabs above to access our interactive tools:",
            ], id="tab0-transition", 
               className="mb-4 text-center",
               style={"font-weight": "300", "letter-spacing": "0.5px"}),
            
            # Info cards replacing buttons
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Differential Expression", className="text-center"),
                        html.P("Explore differential expression/usage between conditions", 
                               className="text-center text-muted small")
                    ], className="p-3 border rounded shadow-sm h-100")
                ], width=12, md=4, lg=2, className="mb-3 mx-1"),
                dbc.Col([
                    html.Div([
                        html.H4("Isoform Explorer", className="text-center"),
                        html.P("Visualize RNA isoforms and their expression patterns.", 
                               className="text-center text-muted small")
                    ], className="p-3 border rounded shadow-sm h-100")
                ], width=12, md=4, lg=2, className="mb-3 mx-1"),
                dbc.Col([
                    html.Div([
                        html.H4("Isoform Correlations", className="text-center"),
                        html.P("Visualize correlations between RNA isoforms and continous variables.", 
                               className="text-center text-muted small")
                    ], className="p-3 border rounded shadow-sm h-100")
                ], width=12, md=4, lg=2, className="mb-3 mx-1"),
                dbc.Col([
                    html.Div([
                        html.H4("QTL Explorer", className="text-center"),
                        html.P("Examine quantitative trait loci at the gene and RNA isoform level.", 
                               className="text-center text-muted small")
                    ], className="p-3 border rounded shadow-sm h-100")
                ], width=12, md=4, lg=2, className="mb-3 mx-1"),
                dbc.Col([
                    html.Div([
                        html.H4("Download Data", className="text-center"),
                        html.P("Access and download datasets from our research.", 
                               className="text-center text-muted small")
                    ], className="p-3 border rounded shadow-sm h-100")
                ], width=12, md=4, lg=2, className="mb-3 mx-1"),
            ], id="tab0-info-row", className="mb-5 px-4 justify-content-center"),
            
            # YouTube Tutorial button
            html.Div([
                html.A(
                    dbc.Button(
                        "Video Tutorial for Website Usage",
                        id="tab0-video-button",
                        color="primary",
                        outline=True,
                        className="px-5 py-4",
                        style={
                            "font-weight": "600",
                            "font-size": "1.3rem",
                            "letter-spacing": "0.5px",
                            "border-radius": "4px"
                        }
                    ),
                    href="https://www.youtube.com/watch?v=DZKYjN3lY-g",
                    target="_blank"
                )
            ], className="text-center mb-4"),
        ], className="py-2 mt-3"),
        
        # Subtle divider
        html.Hr(style={"margin": "2rem 15%", "opacity": "0.4"}),
        
        # Overview section with minimalist design
        html.Div([
            html.H2("Data Overview", 
                    id="tab0-overview-header",
                    className="mb-4 text-center",
                    style={"font-weight": "300", "letter-spacing": "0.5px"}),
            
            # Centered image container with shadow
            html.Div([
                html.Img(
                    src=image_src,
                    id="tab0-png-figure",
                    style={
                        "width": "70%", 
                        "height": "auto", 
                        "box-shadow": "0 3px 10px rgba(0,0,0,0.1)",
                        "background-color": "transparent"
                    }
                )
            ], id="tab0-figure-container", className="text-center mb-5"),
            
            # Styled manuscript reference
            html.H3("For more information checkout our manuscript: XXX",
                id="tab0-manuscript-ref",
                className="mt-4 text-center",
                style={"font-weight": "300", "font-style": "italic", "letter-spacing": "0.3px", "opacity": "0.85"})
        ], className="px-4 py-3")
    ], id="tab0-container", className="container-fluid")

# Callback for responsive design
@callback(
    [Output("tab0-container", "style"),
     Output("tab0-info-row", "style"),
     Output("tab0-overview-header", "style"),
     Output("tab0-png-figure", "style"),
     Output("tab0-manuscript-ref", "style")],
    [Input("window-dimensions", "data")]
)
def update_tab0_responsiveness(dimensions):
    if not dimensions:
        # Default styles
        return (
            {}, 
            {}, 
            {"font-weight": "300", "letter-spacing": "0.5px"}, 
            {
                "width": "70%", 
                "height": "auto", 
                "box-shadow": "0 3px 10px rgba(0,0,0,0.1)",
                "background-color": "transparent"
            }, 
            {"font-weight": "300", "font-style": "italic", "letter-spacing": "0.3px", "opacity": "0.85"}
        )
    
    width = dimensions.get('width', 1200)
    
    # Base styles
    container_style = {}
    info_row_style = {}
    overview_header_style = {"font-weight": "300", "letter-spacing": "0.5px"}
    img_style = {
        "width": "70%", 
        "height": "auto", 
        "box-shadow": "0 3px 10px rgba(0,0,0,0.1)", 
        "max-width": "100%",
        "background-color": "transparent"
    }
    manuscript_style = {"font-weight": "300", "font-style": "italic", "letter-spacing": "0.3px", "opacity": "0.85"}
    
    # Responsive adjustments
    if width < 576:  # Extra small
        overview_header_style.update({"font-size": "1.6rem"})
        img_style.update({"width": "95%"})
        manuscript_style.update({"font-size": "1.1rem"})
    elif width < 768:  # Small
        overview_header_style.update({"font-size": "1.8rem"})
        img_style.update({"width": "90%"})
        manuscript_style.update({"font-size": "1.2rem"})
    elif width < 992:  # Medium
        overview_header_style.update({"font-size": "1.9rem"})
        img_style.update({"width": "80%"})
        manuscript_style.update({"font-size": "1.3rem"})
    else:  # Large
        overview_header_style.update({"font-size": "2rem"})
        img_style.update({"width": "70%"})
        manuscript_style.update({"font-size": "1.4rem"})
    
    return (
        container_style, 
        info_row_style, 
        overview_header_style, 
        img_style, 
        manuscript_style
    ) 