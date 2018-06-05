"""
Rename files in the given directory.
"""
import os


def rename_files(directory, prefix):
    """Rename files in a certain directory with a given prefix."""
    for file in os.listdir(directory):
        os.rename(file, prefix + "_" + file)
    return

if __name__ == "__main__":
    rename_files(".", "YES")
