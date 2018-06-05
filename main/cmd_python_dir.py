"""
This program will open a command prompt in the window of the main
python script folder.
"""
import os
import pyperclip


def cmd_python_dir():
    """Open command prompt on the main python folder."""
    copy_content = "pylint C:/Users/plee/Desktop/Main/"
    os.chdir("C:\\Users\\plee\\AppData\\Local\\Programs\\Python\\Python36-32\\Scripts")
    os.system("start cmd")
    pyperclip.copy(copy_content)
    return


if __name__ == "__main__":
    cmd_python_dir()
