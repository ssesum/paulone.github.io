"""
Remove any character that you want from a file.
"""
import os


def remove_char_from_file(directory, file, character):
    """Remove a character from a text file."""
    os.chdir(directory)
    with open("NEW_" + file, 'w') as file_out:
        with open(file, 'r') as file_in:
            for line in file_in:
                file_out.write(line.replace(character, ""))
            file_in.close()
        file_out.close()
    return


if __name__ == "__main__":
    remove_char_from_file(".", "test.txt", "b")
