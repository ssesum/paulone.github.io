"""
Change the case of the file/folder names to either lower or upper case.
"""
import os


def change_case_of_file_names(directory):
    """Change the default case of the file names to upper/lower case."""
    case = input("Type L for lower case, and U for upper case.")
    if case == 'L':
        for file in os.listdir(directory):
            os.rename(file, file.lower())
    elif case == 'U':
        for file in os.listdir(directory):
            os.rename(file, file.upper())
    else:
        print("Invalid character.")
    return


if __name__ == "__main__":
    change_case_of_file_names(".")
