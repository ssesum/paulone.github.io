"""
This module will return a dataframe that is parsed for linear regression.
"""
import pandas as pd


def get_numeric_dataframe(dataframe):
    """This function grabs all the columns from a dataframe that are numeric."""
    numeric_dtypes = ['int16',
                      'int32',
                      'int64',
                      'float16',
                      'float32',
                      'float64']
    return dataframe.select_dtypes(include=numeric_dtypes)


def get_unique_value_counts(dataframe, column):
    """Return the number of unique value counts."""
    return len(list((dataframe[column].value_counts())))


def get_parsed_linear_regression_df(csv_file_name):
    """Return a dataframe parsed for linear regression."""
    dataframe = pd.read_csv(csv_file_name)
    df_numeric = get_numeric_dataframe(dataframe)
    df_parsed = df_numeric[df_numeric['AGE'] == 2]
    thresh = len(df_parsed) * 0.20
    df_parsed.dropna(thresh=thresh, axis=1, inplace=True)
    cols = df_parsed.columns
    for col in cols:
        if get_unique_value_counts(df_parsed, col) == 1:
            df_parsed.drop(col, axis=1, inplace=True)
    return df_parsed
