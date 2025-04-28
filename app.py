# app.py - Entry point for Heroku
# Import the dash app instance first to ensure proper initialization

# Import the dash app instance itself (not just the server)
from app import app

# Then get the server for Gunicorn
from app import server

# This module also imports the app's callbacks, layout, etc.
# through the app/__init__.py, app/app.py import chain
# which ensures that everything is properly initialized
# before Gunicorn starts the server

if __name__ == "__main__":
    # Allow running the app directly (for development)
    from app import app as dash_app
    dash_app.run_server(debug=True)

# This file serves as a bridge between Heroku's expected entry point 
# (as specified in Procfile: "web: gunicorn app:server")
# and our actual app structure where the server lives in app/__init__.py 