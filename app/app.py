# File: app/app.py
# This file re-exports the Dash app instance to support both local development and deployment.

# Re-export the app and server from __init__.py
from app import app as dash_app, server

if __name__ == "__main__":
    # For local development only
    dash_app.run_server(debug=False)