# File: app/tabs/__init__.py
# This file initializes the tabs for the application.

from app.tabs.tab0 import layout as tab0_layout
from app.tabs.tab1 import layout as tab1_layout
from app.tabs.tab2 import layout as tab2_layout
from app.tabs.tab3 import layout as tab3_layout
from app.tabs.tab4 import layout as tab4_layout
from app.tabs.tab5 import layout as tab5_layout
from app.tabs.tab6 import layout as tab6_layout

# Define the content for each tab
tab_content = {
    "tab-0": tab0_layout(),
    "tab-1": tab1_layout(),
    "tab-2": tab2_layout(),
    "tab-3": tab3_layout(),
    "tab-4": tab4_layout(),
    "tab-5": tab5_layout(),
    "tab-6": tab6_layout()
}

# This file can be empty, or you can directly import tab modules if needed
# You could import tab1 here if you want to make it directly available when importing app.tabs