# File: app.py
# The main entry point for both local development and Heroku deployment.

# Import what we need from the app package
from app import app, server  # This is the key: explicitly import both app and server
from app.layout import layout
import app.callbacks  # This registers all the callbacks

# Set the app layout
app.layout = layout

# This is only needed for local development
if __name__ == "__main__":
    app.run_server(debug=True)