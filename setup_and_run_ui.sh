#!/bin/bash -l
  
# Define the name of the virtual environment  
VENV_DIR="streamlit-env"  
  
# Check if the virtual environment already exists  
if [ ! -d "$VENV_DIR" ]; then  
    echo "Creating virtual environment..."  
    python -m venv "$VENV_DIR"  
else  
    echo "Virtual environment already exists."  
fi  
  
# Activate the virtual environment  
source "$VENV_DIR/Scripts/activate"  
  
# Check if Streamlit is already installed  
if ! pip show streamlit > /dev/null 2>&1; then  
    echo "Installing Streamlit..."  
    pip install streamlit  
else  
    echo "Streamlit is already installed."  
fi  
  
# Run the Streamlit application  
echo "Running Streamlit application..."  
streamlit run streamlit.py  