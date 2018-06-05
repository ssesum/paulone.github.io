"""
Open the folder at the specified location.
"""
import os


def open_folder(directory):
    """
    Open folder at a specified location.
    Python Script Folder: C:/Users/plee/AppData/Local/Programs/Python/Python36-32/Scripts
    """
    os.startfile(directory)
    return


if __name__ == "__main__":
    open_folder("Main")
