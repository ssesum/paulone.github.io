"""
Convert text file to a csv file.
"""
import csv
import os


def convert_text_to_csv_file(directory):
    """Convert a text file to a csv file."""
    in_delimiter = ","
    lineterminator = "\n"
    out_delimiter = ","

    for file in os.listdir(directory):
        if ".txt" in file:
            file_name = file.replace(".txt", ".csv")
            in_text = csv.reader(open(file, "r"), delimiter=in_delimiter)
            out_text = csv.writer(open(file_name, "w"),
                                  delimiter=out_delimiter,
                                  lineterminator=lineterminator)
            out_text.writerows(in_text)
    return


if __name__ == "__main__":
    convert_text_to_csv_file(".")
