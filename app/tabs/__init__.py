# File: app/tabs/__init__.py
# This file initializes the tabs for the application.

from app.tabs.tab0 import layout as tab0_layout
from app.tabs.tab1 import layout as tab1_layout
from app.tabs.tab2 import layout as tab2_layout
from app.tabs.tab3 import layout as tab3_layout
from app.tabs.tab4 import layout as tab4_layout
from app.tabs.tab5 import layout as tab5_layout

# Define the tabs we want to include
tabs = {
    "tab0": tab0_layout,
    "tab1": tab1_layout,
    "tab2": tab2_layout,
    "tab3": tab3_layout,
    "tab4": tab4_layout,
    "tab5": tab5_layout,
}

# This file can be empty, or you can directly import tab modules if needed
# You could import tab1 here if you want to make it directly available when importing app.tabs
