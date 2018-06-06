"""
Open the main python folder at work.
"""
import open_folder
import os


if __name__ == "__main__":
	work_dir = "C:/Users/plee/AppData/Local/Programs/Python/Python36-32/Scripts"
	os.chdir(work_dir)
	os.system("start cmd")
