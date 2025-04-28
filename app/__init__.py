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

# Get absolute path to assets folder
current_dir = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(current_dir, 'assets')

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
# Create Dash application
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO, dbc_css],
    suppress_callback_exceptions=True,
    assets_folder=assets_path
)
server = app.server

# Memory tracking removed

# Register cleanup functions to be called when the app exits
atexit.register(cleanup)
# Memory tracker shutdown removed
