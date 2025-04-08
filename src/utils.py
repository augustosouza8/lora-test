"""
utils.py
========
Utility functions for the LoRA Fine-Tuning Project.
This script includes a basic function to list data files in the designated data folder,
helping to verify that the project directory structure is correct.
"""

import os


def list_data_files(data_dir: str) -> None:
    """
    Lists all files in the specified data directory.

    Args:
        data_dir (str): Path to the data folder.
    """
    try:
        files = os.listdir(data_dir)
        if not files:
            print("No files found in the data directory.")
        else:
            print("Files in the data directory:")
            for file in files:
                print(f" - {file}")
    except Exception as e:
        print(f"Error listing files: {e}")


if __name__ == "__main__":
    # Adjust the path if necessary based on your directory structure.
    list_data_files("../data")
