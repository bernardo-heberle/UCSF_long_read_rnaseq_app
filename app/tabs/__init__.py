# File: app/tabs/__init__.py
# This file initializes the tabs for the application.

from app.tabs.tab1 import layout as tab1_layout
from app.tabs.tab2 import layout as tab2_layout

# Define the tabs we want to include
tabs = {
    "tab1": tab1_layout,
    "tab2": tab2_layout,
}
