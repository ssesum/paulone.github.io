"""
Convert a CSV file into a TSV file.
"""
import csv
import os

def convert_csv_to_tsv_file(directory):
    """Convert a csv file to a tsv file."""
    for file in os.listdir(directory):
        if ".csv" in file:
            file_name = file.replace(".csv", ".tsv")
            with open(file, 'r') as in_f, open(file_name, 'w') as out_f:
                reader = csv.reader(in_f)
                writer = csv.writer(out_f, delimiter='\t')
                for line in reader:
                    writer.writerow(line)
                out_f.close()
                in_f.close()
    return

if __name__ == "__main__":
    convert_csv_to_tsv_file(".")
