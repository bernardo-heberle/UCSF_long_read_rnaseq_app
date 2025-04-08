from dash import Dash
import dash_bootstrap_components as dbc
import atexit
from app.utils.db_utils import cleanup

# --- Logging Setup ---
# Removed memory logging setup

# Variable to track the peak memory usage
# Removed max_memory_mb variable

# --- Scheduler Setup ---
# Removed scheduler setup

# --- Graceful Shutdown ---
# Removed scheduler shutdown function

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

# Register cleanup function to be called when the app exits
# Note: The original db cleanup is still registered
atexit.register(cleanup)
