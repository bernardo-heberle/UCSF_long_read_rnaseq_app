print("[DEBUG] wsgi.py: Starting execution")
# File: wsgi.py
# The main entry point for both local development and Heroku deployment.

# Import what we need from the app package
from app import app as dash_app, server
from app.layout import layout
import app.callbacks  # This registers all the callbacks

# Set the app layout
dash_app.layout = layout

# This is only needed for local development
if __name__ == "__main__":
    dash_app.run_server(debug=True)