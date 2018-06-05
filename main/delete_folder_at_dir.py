"""
Delete the folder in the given directory.
"""
import shutil
import os


def delete_folder_at_dir(directory):
    """Delete all contents of the given directory."""
    shutil.rmtree(directory)
    os.rmdir(directory)
    os.mkdir(directory)
    return

if __name__ == "__main__":
    delete_folder_at_dir(".")
