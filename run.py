# Simple entry point for local development
# Run this with: python run.py

# Import the app with a different name to avoid naming conflicts
from app import app as dash_app

# Import and set the layout
from app.layout import layout
dash_app.layout = layout

# Import callbacks to register them
import app.callbacks

if __name__ == "__main__":
    dash_app.run_server(debug=True) 