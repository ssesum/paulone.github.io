"""
Convert the CSV file to a text file.
"""
import csv
import os

def convert_csv_to_txt_file(directory):
    """Convert a csv file to a text file."""
    in_delimiter = ","
    out_delimiter = ","
    lineterminator = "\n"

    for file in os.listdir(directory):
        if ".csv" in file:
            file_name = file.replace(".csv", ".txt")
            in_text = csv.reader(open(file, "r"),
                                 delimiter=in_delimiter)
            out_text = csv.writer(open(file_name, "w"),
                                  delimiter=out_delimiter,
                                  lineterminator=lineterminator)
            out_text.writerows(in_text)
    return

if __name__ == "__main__":
    convert_csv_to_txt_file(".")
