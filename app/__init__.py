from dash import Dash
import dash_bootstrap_components as dbc
import atexit
from app.utils.db_utils import cleanup
import os

# --- Logging Setup ---
# Memory tracking removed

# --- Scheduler Setup ---
# Removed scheduler setup

# --- Graceful Shutdown ---
# Memory tracker shutdown function removed

# --- End Scheduler ---

# Get absolute path for assets folder
assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
# Create Dash application
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO, dbc_css],
    suppress_callback_exceptions=True,
    assets_folder=assets_path,
    serve_locally=True,  # Ensure assets are served locally
    eager_loading=True   # Pre-load all assets at startup
)
server = app.server

# Memory tracking removed

# Register cleanup functions to be called when the app exits
atexit.register(cleanup)
# Memory tracker shutdown removed
