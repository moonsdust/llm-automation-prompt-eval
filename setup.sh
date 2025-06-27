#!/bin/bash
# Run this script if this is your first time running this repository. It will create a new virtual environment,
# activate the virtual environment, and install any necessary packages. 

python3.11 -m venv venv # Create the virtual environment 
source venv/bin/activate # Activate virtual environment 
pip install -r requirements.txt # Install required libraries 
