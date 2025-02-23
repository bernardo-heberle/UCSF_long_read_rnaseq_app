# My Dash App for Alzheimer's Disease Data Visualization

## Overview
This project is a Python Dash web application designed to visualize, download, and explore gene and RNA isoform expression data, genotyping data, technical sequencing/genotyping variables, demographics, and pathology data for a cohort of Alzheimer's disease patients and non-demented controls. The app features multiple tabs to organize different data visualizations and exploratory tools.

## Directory Structure

AD_RNAseq_dash_app/ 
├── app/
│ ├── init.py
│ ├── app.py
│ ├── layout.py
│ ├── callbacks.py
│ └── tabs/
│ ├── tab1.py
│ ├── tab2.py
│ ├── tab3.py
│ └── ... ├── assets/
│ ├── custom.css
│ ├── custom.js
│ └── images/
├── data/
│ └── migrations/
├── tests/
│ └── test_app.py
├── config.py
├── Procfile
├── runtime.txt
├── requirements.txt
├── README.md
└── .gitignore