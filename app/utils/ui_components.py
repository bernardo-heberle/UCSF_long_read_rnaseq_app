from dash import html, dcc
import dash_bootstrap_components as dbc

def create_gene_search_dropdown(initial_value=None, initial_options=None, id='search-input'):
    """Create a dropdown for gene search with specific styling."""
    return dcc.Dropdown(
        id=id,
        placeholder="Enter gene name or ID...",
        clearable=True,
        searchable=True,
        options=initial_options or [],
        value=initial_value,
        style={
            "background-color": "#ffffff",
            "color": "#333333",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "4px"
        }
    )

def create_rsid_search_dropdown(initial_value=None, initial_options=None):
    """Create a dropdown for RSID search with specific styling."""
    return dcc.Dropdown(
        id='rsid-search-input',
        placeholder="Enter RSID (e.g., rs429358)...",
        clearable=True,
        searchable=True,
        options=initial_options or [],
        value=initial_value,
        style={
            "background-color": "#ffffff",
            "color": "#333333",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "4px"
        }
    )

def create_matrix_dropdown(options, default_value=None, id='matrix-table-dropdown'):
    """Create a standardized matrix selection dropdown component."""
    return dcc.Dropdown(
        id=id,
        options=options,
        value=default_value,
        placeholder="Select a data table...",
        clearable=False,
        style={
            "background-color": "#ffffff",
            "color": "#333333",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "4px"
        },
        optionHeight=35
    )

def create_section_header(title):
    """Create a standardized section header."""
    return html.P(title,
        style={
            "color": "#333333",
            "font-size": "1.1rem",
            "margin-bottom": "1rem",
            "font-weight": "500"
        }
    )

def create_content_card(content, id=None):
    """Create a standardized content card with consistent styling."""
    card_props = {
        "style": {
            "background-color": "#ffffff",
            "border": "1px solid rgba(0, 0, 0, 0.1)",
            "border-radius": "6px",
            "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
        }
    }
    
    if id:
        card_props["id"] = id
        
    return dbc.Card(
        dbc.CardBody(content),
        **card_props
    )

def create_radio_items(id, options, value=None, inline=True):
    """Create a standardized radio button group with COSMO styling."""
    return dbc.RadioItems(
        id=id,
        options=options,
        value=value,
        inline=inline,
        style={
            "color": "#333333",
            "font-size": "0.9rem"
        },
        inputStyle={
            "cursor": "pointer"
        },
        labelStyle={
            "cursor": "pointer",
            "margin-right": "1rem",
            "padding": "0.25rem 0"
        },
        className="dbc"
    )

def create_checklist(id, options, value=None, inline=True):
    """Create a standardized checklist with COSMO styling."""
    return dbc.Checklist(
        id=id,
        options=options,
        value=value,
        inline=inline,
        style={
            "color": "#333333",
            "font-size": "0.9rem"
        },
        inputStyle={
            "cursor": "pointer"
        },
        labelStyle={
            "cursor": "pointer",
            "margin-right": "1rem",
            "padding": "0.25rem 0"
        },
        className="dbc"
    ) 