from dash import Dash
import dash_bootstrap_components as dbc
import atexit
from app.utils.db_utils import cleanup


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
atexit.register(cleanup)
