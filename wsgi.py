# File: wsgi.py
# The main entry point for Heroku deployment.

# Ensure proper dependency initialization
import dash
import dash_bootstrap_components as dbc

# Import what we need from the app package
from app import app, server  # This is the key: explicitly import both app and server

# We must import these AFTER importing app to ensure callbacks are registered
from app.layout import layout
import app.callbacks  # This registers all the callbacks

# Set the app layout
app.layout = layout

# This is only needed for local development
if __name__ == "__main__":
    app.run_server(debug=True)