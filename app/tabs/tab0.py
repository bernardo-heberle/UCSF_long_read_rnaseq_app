# File: app/tabs/tab0.py
# Defines the layout for the Home tab

from dash import html, dcc
import dash_bootstrap_components as dbc

# Define the layout for the Home tab
layout = html.Div([
    html.H2("Welcome to the RNA-seq Analysis Dashboard", className="mb-4"),
    html.P([
        "This dashboard provides interactive tools for analyzing RNA-seq data, including:",
        html.Ul([
            html.Li("Differential Expression Analysis"),
            html.Li("Isoform Exploration"),
            html.Li("Isoform Correlations"),
            html.Li("eQTL Analysis"),
            html.Li("Gene Coverage Visualization"),
            html.Li("Data Download Options")
        ])
    ])
]) 