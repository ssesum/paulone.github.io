"""
Print metadata about the data.
"""

def unique_value_counts(dataframe, column):
    """Return the number of unique value counts of a column."""
    return str(len(list((dataframe[column].value_counts()))))


def print_stats(subject, left_justified_number, count):
    """Print formatted statistics with a left justification."""
    print(subject.ljust(left_justified_number), count)
    return


def column_count(dataframe, column):
    """Return the length of the specified dataframe column."""
    return dataframe[column].count()


def column_data_type(dataframe, column):
    """Return the datatype of the column specified."""
    return dataframe.dtypes[column]


def get_percentage_missing(dataframe, column):
    """Caluculate the percentage of null values in a dataframe column."""
    num = dataframe[column].isnull().sum()
    den = len(dataframe)
    return str(round(num/den, 4) * 100) + "%"


def print_df_metrics(dataframe, mode):
    """Print metadata about the data."""
    ljust = 23
    index_ljust = 7
    col_ljust = 35
    unique_ljust = 10
    count_ljust = 9
    type_ljust = 8
    pm_ljust = 5
    dataframe_rows, dataframe_cols = dataframe.shape
    total_null_values = dataframe.isnull().sum().sum()
    column_names = dataframe.columns
    print("X" * 90)
    print_stats("Number of elements:", ljust, str(dataframe.size))
    print_stats("Number of rows:", ljust, dataframe_rows)
    print_stats("Number of columns:", ljust, dataframe_cols)
    print_stats("Number of NULL values:", ljust, total_null_values)
    print("X" * 90)
    dataframe.info()
    print("X" * 90)
    print("Index".ljust(index_ljust),
          "Column Name".ljust(col_ljust),
          "Unique".ljust(unique_ljust),
          "Count".ljust(count_ljust),
          "Type".ljust(type_ljust),
          "Percentage Missing".ljust(pm_ljust))
    if mode == 0:
        for index, col in enumerate(column_names):
            print(str(index+1).ljust(index_ljust),
                  col.ljust(col_ljust),
                  unique_value_counts(dataframe, col).ljust(unique_ljust),
                  str(column_count(dataframe, col)).ljust(count_ljust),
                  str(column_data_type(dataframe, col)).ljust(type_ljust),
                  get_percentage_missing(dataframe, col))
    elif mode == 1:
        for index, col in enumerate(column_names):
            if column_count(dataframe, col) == 0:
                print(str(index+1).ljust(index_ljust),
                      col.ljust(col_ljust),
                      unique_value_counts(dataframe, col).ljust(unique_ljust),
                      str(column_count(dataframe, col)).ljust(count_ljust),
                      column_data_type(dataframe, col))
    print("X" * 90)
    return
