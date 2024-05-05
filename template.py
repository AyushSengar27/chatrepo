import os
from pathlib import Path
import logging

# Configure logging to output date, time, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of files and directories to be created
list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.html"
]

# Loop through each file path in the list
for filepath in list_of_files:
    filepath = Path(filepath)  # Create a Path object for file path operations
    filedir, filename = os.path.split(filepath)  # Split the filepath into directory and filename

    # Check if the directory exists and create it if not
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)  # Ensure the directory exists
        logging.info(f"Creating directory; {filedir} for the file {filename}")  # Log directory creation

    # Check if the file exists and is not empty, create or overwrite if necessary
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:  # Open the file in write mode
            pass  # Create an empty file, could add default content here
        logging.info(f"Creating empty file: {filepath}")  # Log file creation
    else:
        logging.info(f"{filename} is already created")  # Log if the file already exists
