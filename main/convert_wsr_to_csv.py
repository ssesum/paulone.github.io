"""
This program takes my WSR, strips out the Task Description column,
then it converts the dataframe into a CSV.
"""
import pandas as pd


def convert_wsr_to_csv():
    """
    Convert my WSR to CSV for easier upload to Kibana.
    """
    wsr_paul = read_data_excel("C:/Users/plee/Desktop/Main/", "PAUL-WSR.xlsx", "Paul")
    wsr_paul = drop_columns(wsr_paul, ['TASK DESCRIPTION'], 1)
    wsr_paul.to_csv("PAUL-WSR.csv", index=False, header=None)
    return


def read_data_excel(directory, data_set_name, sheetname):
    """Reads data into a dataframe using pandas."""
    return pd.read_excel(directory + data_set_name, sheetname=sheetname)


def drop_columns(dataframe, columns, axis):
    """Drop set of column's from the given dataframe."""
    return dataframe.drop(columns, axis=axis)


if __name__ == "__main__":
    convert_wsr_to_csv()
