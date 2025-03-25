# File: app/app.py
# The main entry point that sets the layout, registers callbacks, and runs the server.

# Import the dash app instance with a different name to avoid namespace conflicts
from app import app as dash_app
from app.layout import layout
import app.callbacks  # This registers the callbacks

# Set the app layout to our defined layout
dash_app.layout = layout

if __name__ == "__main__":
    dash_app.run_server(debug=True)