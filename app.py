# app.py - Entry point for Heroku
# Import the server from the app package for Heroku deployment

from app import server

# This file serves as a bridge between Heroku's expected entry point 
# (as specified in Procfile: "web: gunicorn app:server")
# and our actual app structure where the server lives in app/__init__.py 