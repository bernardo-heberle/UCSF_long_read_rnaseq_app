print("[DEBUG] app/__init__.py: Starting execution")

from dash import Dash
import dash_bootstrap_components as dbc

# --- Logging Setup ---
# Memory tracking removed

# --- Scheduler Setup ---
# Removed scheduler setup

# --- Graceful Shutdown ---
# Memory tracker shutdown function removed

# --- End Scheduler ---


dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
# Create Dash application
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO, dbc_css],
    suppress_callback_exceptions=True,
    assets_folder='assets'
)
server = app.server

# Memory tracking removed
