#!/bin/bash

# PREREQUISITE: Have ran setup.sh in the past to create the virtual environment and install necessary packages. 
# This script runs main.py to start the program. 
source venv/bin/activate # Activate virtual environment 
python3.11 scripts/main.py # Run the program