# Simple entry point for local development
# Run this with: python run.py

# Ensure proper dependency initialization
import dash
import dash_bootstrap_components as dbc

# Import the app with a different name to avoid naming conflicts
from app import app as dash_app

# We must import these AFTER importing app to ensure callbacks are registered
from app.layout import layout
dash_app.layout = layout

# Import callbacks to register them
import app.callbacks

if __name__ == "__main__":
    dash_app.run_server(debug=True) 