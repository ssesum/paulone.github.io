r"""
Module contains low-level and medium level functions related to Dataframe
manipulation. It's good to import this library when working with data science,
data manipulation, data analysis, etc.

Sayings:
    Be on the look-out for things you do on repeat, automate it.
    Donald Knuth - We should forget about small efficiencies,
                   say about 97% of the time: premature optimization
                   is the root of all evil.
    Make it usable by other people than yourself.
    
Magic:
    %%timeit
    %time
    %prun
    %lprun
    %memit
    %mprun

Bayes Theorem:
    #%%latex
    #$$P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}$$


Functions:
    def print_random_new_lines(counter=1):
    def clear_screen():
    def print_calendar():
    def random_hex_color():
    def read_csv(directory, data_set_name):
    def read_table(directory, data_set):
    def df_cols(dataframe):
    def df_size(dataframe):
    def df_shape(dataframe):
    def df_null_counts(dataframe):
    def print_bar(count=1):
    def unique_value_counts(dataframe, column):
    def column_count(dataframe, column):
    def column_data_type(dataframe, column):
    def lowercase_all_columns(dataframe):
    def uppercase_all_columns(dataframe):
    def change_column_character(dataframe, change_from, change_to):
    def value_counts(dataframe, column):
    def print_stats(subject, left_justified_number, count):
    def df_corr(dataframe, method):
    def convert_column_to_categorical(dataframe, column):
    def dtype_columns(dataframe, dtype):
    def convert_column_value_to_null(dataframe, column, value):
    def column_data_type_conversion(dataframe, column, type_change):
    def pair_plot(dataframe, hue):
    def pr_err(text):
    def split_data(dataframe, target_col, learning_cols):
    def split_data_test_train(dataframe, target_col, learning_cols, split_size):
    def cat_to_num_conversion(dataframe, category):
    def df_metrics(dataframe, mode):
    def convert_column_to_numerical(dataframe, column):
    def convert_new_column_to_numerical(dataframe, column):
    def print_metrics(y_test, pred):
    def pipeline_linear_regression(dataframe, target_col, learning_cols, split, rep):
    def pipeline_random_forest(dataframe, target_col, learning_cols, split_size):
    def join_3_data_sets(directory, set_1, set_2, set_3):
    def combine_3_similar_datasets(directory, set_1, set_2, set_3, save_name):
    def split_data_by_month(dataframe, month_1, month_2, month_3, save_name):
    def quarter_difference_calculator(directory, set_1, set_2, set_3):
    def group_and_sort_by(dataframe, groupby, sortby):
    def death_summation(directory, data):
    def value_counts_to_dataframe(dataframe, column):
    def seaborn_heatmap(dataframe, column):
    def subset_check(set_in, set_check):
    def value_count_list(dataframe, column):
    def merg_all_ltch_quarter_data():
    def cwd():
    def convert_dataframe_to_csv(dataframe, filename):
    def remove_empty_columns(dataframe):
    def target_rest_split(dataframe, target_col):
    def numeric_dtype_selection(dataframe):
    def column_to_list(dataframe):
    def rfe_data_pipeline_numeric(dataframe, target_col):
    def trim_list_whitespace(list_to_trim):
    def view_value_counts(dataframe):
    def get_percentage_missing(dataframe, column):
    def pdf_shape(dataframe):
"""
#%matplotlib inline
#%load_ext memory_profiler
import calendar
import random as rd
import datetime
import csv
import time
import math
import shutil
import os
import operator as op
import sys
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pyperclip
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
sb.set()


#Global Variables
if("plee" in os.getcwd()):
    DESKTOP = "C:/Users/plee/Desktop/"
else:
    DESKTOP = "C:/Users/Paul/Desktop/"

DATADIR = DESKTOP + "data/"
SANDBOX = DESKTOP + "Main/sandbox/"
RUNMODE = 1


def print_random_new_lines(counter=1):
    """Print random new lines."""
    symbols = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+"]
    length_of_symbols_list = len(symbols)-1
    counter = counter
    while counter > 0:
        random_symbol_list = ""
        ticker = 0
        while ticker < 50:
            random_symbol_list += symbols[rd.randint(0, length_of_symbols_list)]
            ticker += 1
        print(random_symbol_list)
        counter -= 1
    return


def clear_screen():
    """Clears the screen with a specified count"""
    print("\n" * 200)
    return


def print_calendar():
    """Prints the calendar."""
    now = datetime.datetime.now()
    cal = calendar.TextCalendar()
    cal.pryear(2018)
    print("Current date and time using strftime:")
    print(now.strftime("%m-%d-%Y %H:%M"))
    return


def random_hex_color():
    """
        Return a random hex color
    """
    random_integer = lambda: rd.randint(0, 255)
    return '#%02X%02X%02X' % (random_integer(), random_integer(), random_integer())


def read_csv(directory, data_set_name):
    """Read data from a csv into a pandas dataframe."""
    return pd.read_csv(directory + data_set_name)


def read_table(directory, data_set):
    """Read data from any format into a pandas dataframe."""
    return pd.read_table(directory + data_set, delimiter=',', na_values=['NA'])


def df_cols(dataframe):
    """Return the columns of the given dataframe."""
    return dataframe.columns


def df_size(dataframe):
    """Return the size of the dataframe, which is just rows * columns."""
    return dataframe.size


def df_shape(dataframe):
    """Return the shape of the dataframe(rows, columns)."""
    return dataframe.shape


def df_null_counts(dataframe):
    """This function returns the number of null values in the dataframe."""
    return dataframe.isnull().sum().sum()


def print_bar(count=1):
    """Print a horizontal bar of siize count"""
    print("-" * count)


def unique_value_counts(dataframe, column):
    """Return the number of unique value counts of a column."""
    return str(len(list((dataframe[column].value_counts()))))


def column_count(dataframe, column):
    """Return the length of the specified dataframe column."""
    return dataframe[column].count()


def column_data_type(dataframe, column):
    """Return the datatype of the column specified."""
    return dataframe.dtypes[column]


def lowercase_all_columns(dataframe):
    """Lowercase all column names in a dataframe."""
    dataframe.columns = dataframe.columns.str.lower()
    return


def uppercase_all_columns(dataframe):
    """Uppercase all column names in a dataframe."""
    dataframe.columns = dataframe.columns.str.upper()
    return


def change_column_character(dataframe, change_from, change_to):
    """Change a column character in the dataframe columns for consistency."""
    dataframe.columns = dataframe.columns.str.replace(change_from, change_to)
    return


def value_counts(dataframe, column):
    """Print the value counts of a column."""
    print(dataframe[column].value_counts())
    return


def print_stats(subject, left_justified_number, count):
    """Print formatted statistics with a left justification."""
    print(subject.ljust(left_justified_number), count)
    return


def df_corr(dataframe, method):
    """Using pandas corr method to do a full numerical correlation analysis."""
    dataframe.\
        corr(method=method).\
        style.\
        format("{:.2}").\
        background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
    return


def convert_column_to_categorical(dataframe, column):
    """Convert dataframe column to a categorical column."""
    dataframe[column] = dataframe[column].astype('category')
    return


def dtype_columns(dataframe, dtype):
    """Returns all column names with the specified data type."""
    return dataframe.select_dtypes([dtype]).columns


def convert_column_value_to_null(dataframe, column, value):
    """Convert a indicated value in column to null."""
    dataframe[column] = dataframe[column].replace(value, None)
    return


def column_data_type_conversion(dataframe, column, type_change):
    """Convert a dataframe column to the specified type."""
    dataframe[column] = dataframe[column].astype(type_change)
    return


def pair_plot(dataframe, hue):
    """Seaborn's pairplot function."""
    sb.pairplot(dataframe, hue=hue)
    return


def pr_err(text):
    """Print to sys.stderr."""
    print(text, file=sys.stderr)
    return


def split_data(dataframe, target_col, learning_cols):
    """Parse the data into the target column and the learning columns."""
    combined_columns = learning_cols + target_col
    parsed_dataframe = dataframe[combined_columns]
    parsed_dataframe = parsed_dataframe.dropna(axis=0)
    x_all = parsed_dataframe[learning_cols]
    y_all = parsed_dataframe[target_col]
    return x_all, y_all


def split_data_test_train(dataframe, target_col, learning_cols, split_size):
    """
    This function uses the sklearn train_test_split function to
    split the given parsed data into 4 different parts. The 4 parts are
    x_train, x_test, y_train, and y_test.
    """
    x_all, y_all = split_data(dataframe, target_col, learning_cols)
    return train_test_split(x_all, y_all, test_size=split_size)


def cat_to_num_conversion(dataframe, category):
    """
    This function will find all columns with a certain category,
    then it will convert all the found columns to categorical.
    """
    columns = dtype_columns(dataframe, category)
    dataframe[columns] = dataframe[columns].apply(lambda x: x.cat.codes)
    return


def df_metrics(dataframe, mode):
    """
    This function will print many beginner analytics for the first
    analytics of a data set. This function is also expected to grow a decent
    size because I'll keep learning more and more.
    """
    ljust = 23
    index_ljust = 7
    col_ljust = 35
    unique_ljust = 10
    count_ljust = 9
    type_ljust = 8
    pm_ljust = 5

    dataframe_rows, dataframe_cols = df_shape(dataframe)
    total_null_values = df_null_counts(dataframe)
    print_bar(90)
    print_stats("Number of elements:", ljust, df_size(dataframe))
    print_stats("Number of rows:", ljust, dataframe_rows)
    print_stats("Number of columns:", ljust, dataframe_cols)
    print_stats("Number of NULL values:", ljust, total_null_values)
    print_bar(90)


    dataframe.info()
    print_bar(90)


    column_names = dataframe.columns
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
    print_bar(90)
    return


def convert_column_to_numerical(dataframe, column):
    """
    Two step function that converts a dataframe column to categorical,
    then converts to numerical.
    """
    convert_column_to_categorical(dataframe, column)
    cat_to_num_conversion(dataframe, 'category')
    return


def convert_new_column_to_numerical(dataframe, column):
    """
    The purpose of this function was to work with dataframe columns without
    modifying the existing column. It converts a non-numerical column to
    numerical, but then it's a duplicated column with a NEW_ prefix.
    """
    new_column_name = "NEW_" + column
    dataframe[new_column_name] = dataframe[column]
    convert_column_to_categorical(dataframe, new_column_name)
    cat_to_num_conversion(dataframe, 'category')
    convert_column_value_to_null(dataframe, new_column_name, -1)
    column_data_type_conversion(dataframe, new_column_name, int)
    return


def print_metrics(y_test, pred):
    """
    This function will predict 6 different final statistics of a machine
    learning results. It's good to know how each of these metrics are built
    for they have seperate uses for different machine learning libraries.
    """
    explained_variance_score = metrics.explained_variance_score(y_test, pred)
    mean_absolute_error = metrics.mean_absolute_error(y_test, pred)
    mean_squared_error = metrics.mean_squared_error(y_test, pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_test, pred)
    median_absolute_error = metrics.median_absolute_error(y_test, pred)
    r2_score = metrics.r2_score(y_test, pred)
    print_stats("Explained Variance Score:", 25, explained_variance_score)
    print_stats("Mean Absolute Error:", 25, mean_absolute_error)
    print_stats("Mean Squared Error:", 25, mean_squared_error)
    print_stats("Mean Squared Log Error:", 25, mean_squared_log_error)
    print_stats("Median Absolute Error:", 25, median_absolute_error)
    print_stats("R2 Score:", 25, r2_score)
    return


def pipeline_linear_regression(dataframe, target_col, learning_cols, split, rep):
    """
    This function builds a linear regression model for you with the initial step
    of simply giving the columns you want to analyze on the data.
    """
    for count in range(rep):
        print("Current repetition: ", count)
        x_train, x_test, y_train, y_test = split_data_test_train(dataframe,
                                                                 target_col,
                                                                 learning_cols,
                                                                 split)
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        pred = regr.predict(x_test)
        print_metrics(y_test, pred)

    return


def pipeline_random_forest(dataframe, target_col, learning_cols, split_size):
    """
    This function will build a random forest pipeline for you from the very
    start. The only thing that it does not do yet is figure out which columns
    it's going to use for you. The algorithm also optimizes itself using the
    GridSearchCV function.

    TODO:
        Use the other grid function to find the necessary parameters for
        GridSearchCV.
    """
    x_train, x_test, y_train, y_test = split_data_test_train(dataframe,
                                                             target_col,
                                                             learning_cols,
                                                             split_size)
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    random_forest = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=random_forest,
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(x_train, y_train)
    best_grid = grid_search.best_estimator_
    pred = best_grid.predict(x_test)
    print_metrics(y_test, pred)
    return


def join_3_data_sets(directory, set_1, set_2, set_3):
    """
    This function takes in three different sets and
    combines them on a single key.
    """
    df1 = read_csv(directory, set_1)
    df2 = read_csv(directory, set_2)
    df3 = read_csv(directory, set_3)

    df1 = df1.rename(index=str,
                     columns={"PRVDR_NUM":"key"})
    df2 = df2.rename(index=str,
                     columns={"Reporting Group Code":"key"})
    df3 = df3.rename(index=str,
                     columns={"A0100B_CMS_CRTFCTN_NUM":"key",
                              "STATE_CD":"STATE_CD2",
                              "PRVDR_INTRNL_NUM":"PRVDR_INTRNL_NUM2"})
    df1_df2_combination = df1.join(df2.set_index('key'), on='key')
    return df1_df2_combination.join(df3.set_index('key'), on='key')


def combine_3_similar_datasets(directory, set_1, set_2, set_3, save_name):
    """Combines 3 quarters worth of data into a single csv file."""
    df1 = read_csv(directory, set_1)
    df2 = read_csv(directory, set_2)
    df3 = read_csv(directory, set_3)

    df1['QRTR'] = '2017Q1'
    df2['QRTR'] = '2017Q2'
    df3['QRTR'] = '2017Q3'

    df_combined = [df1, df2, df3]
    final = pd.concat(df_combined)
    final.to_csv(save_name, index=False)
    return


def split_data_by_month(dataframe, month_1, month_2, month_3, save_name):
    """
    This function was used for the splitting of data for different months,
    useful when you need to parse the data by a certain date.
    """
    parsed_dataframe = dataframe[(dataframe.month == month_1)|\
                                 (dataframe.month == month_2)|\
                                 (dataframe.month == month_3)]
    parsed_dataframe.to_csv(save_name, index=False)
    return


def quarter_difference_calculator(directory, set_1, set_2, set_3):
    """
    This function is for when you want to see the value differences between the
    unique counts. The results will be formatted in CSV format.
    """
    df1 = read_csv(directory, set_1)
    df2 = read_csv(directory, set_2)
    df3 = read_csv(directory, set_3)

    column_names = df1.columns
    list1 = []
    list2 = []
    list3 = []

    for col in column_names:
        list1.append(len(list((df1[col].value_counts()))))
        list2.append(len(list((df2[col].value_counts()))))
        list3.append(len(list((df3[col].value_counts()))))

    list3_list2_difference = list(map(op.sub, list3, list2))
    list2_list1_difference = list(map(op.sub, list2, list1))
    print(list3_list2_difference)
    print(list2_list1_difference)
    return


def group_and_sort_by(dataframe, groupby, sortby):
    """
    This function is used when you want to groupby certain values,
    then you want to sort by the values.
    """
    return dataframe.groupby(groupby).sum().sort_values(sortby, ascending=False)


def death_summation(directory, data):
    """
    This function is used when you want to create the death summation data set,
    it will grab all the death counts from the death data set, then it will
    take a sum and place it into a nice csv for you to view.
    """
    dataframe = read_csv(directory, data)
    final_df = pd.DataFrame()

    for year in range(1999, 2016):
        df_year = dataframe[dataframe.Year == year]
        df_parsed = df_year[['Cause Name', 'Deaths']]
        new_cn = "Cause Name " + str(year)
        new_de = "Deaths " + str(year)

        df_parsed.columns = [new_cn, new_de]
        df_sum = group_and_sort_by(df_parsed, new_cn, new_de)
        if year == 1999:
            final_df.append(df_sum)
        else:
            final_df.join(df_sum)

    final_df.to_csv("all_death_counts.csv")
    return


def value_counts_to_dataframe(dataframe, column):
    """Returns a dataframe consisting of the value counts of a column."""
    return dataframe[column].\
           value_counts.\
           rename_axis('unique_values').\
           reset_index(name='counts')


def seaborn_heatmap(dataframe, column):
    """Draw a seaborn headmap of a dataframe column."""
    value_cnts = value_counts_to_dataframe(dataframe, column)
    pal = sb.dark_palette("palegreen", as_cmap=True)
    plt.subplots(figsize=(20, 15))
    sb.heatmap(value_cnts, annot=True, fmt='d', cmap=pal)
    return


def subset_check(set_in, set_check):
    """Checks to see if set_in is a subset of set_check"""
    return set(set_in).issubset(set_check)


def value_count_list(dataframe, column):
    """Grab a list of value counts from a dataframe column."""
    return dataframe[column].value_counts().index.tolist()


def merg_all_ltch_quarter_data():
    """Merge 3 quarters of LTCH data."""
    directory = "./"

    set_1 = "cleaned_ltch_providers_2017Q3.csv"
    set_2 = "cleaned_ltch_cdc_2017Q3.csv"
    set_3 = "ltch_assessments_2017Q3.csv"

    df1 = read_csv(directory, set_1)
    df2 = read_csv(directory, set_2)
    df3 = read_csv(directory, set_3)

    df1 = df1.rename(index=str,
                     columns={"PRVDR_NUM":"key"})
    df2 = df2.rename(index=str,
                     columns={"Reporting Group Code":"key"})
    df3 = df3.rename(index=str,
                     columns={"A0100B_CMS_CRTFCTN_NUM":"key",
                              "STATE_CD":"STATE_CD2",
                              "PRVDR_INTRNL_NUM":"PRVDR_INTRNL_NUM2"})
    df1_df2_combination = df1.join(df2.set_index('key'), on='key')
    df1_df2_combination = df1_df2_combination.drop_duplicates()
    final_dataframe = df3.merge(df1_df2_combination, how='left', on='key')
    final_dataframe.to_csv("final.csv", index=False)
    return


def cwd():
    """Prints the current working directory"""
    print(os.getcwd())
    return


def convert_dataframe_to_csv(dataframe, filename):
    """Convert the given dataframe to csv."""
    dataframe.to_csv(filename, index=False)
    return


def remove_empty_columns(dataframe):
    """Delete all empty columns"""
    columns = dataframe.columns
    for col in columns:
        if column_count(dataframe, col) == 0:
            dataframe.drop(col, axis=1, inplace=True)
    return


def target_rest_split(dataframe, target_col):
    """
    This function splits the data into a single target column,
    and the rest are predictors
    """
    data_vars = dataframe.columns.values.tolist()
    target = [target_col]
    predicted = [col for col in data_vars if col not in target]
    return dataframe[target], dataframe[predicted]


def numeric_dtype_selection(dataframe):
    """This function grabs all the columns from a dataframe that are numeric"""
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return dataframe.select_dtypes(include=numeric_dtypes)


def column_to_list(dataframe):
    """Convert dataframe columns to a list."""
    return dataframe.columns.values.tolist()


def rfe_data_pipeline_numeric(dataframe, target_col):
    """Complete numeric data to rfe feature selection pipeline."""
    df_numeric = numeric_dtype_selection(dataframe)
    target, predictors = target_rest_split(df_numeric, target_col)
    x_train, x_test, y_train, y_test = split_data_test_train(df_numeric,
                                                             column_to_list(target),
                                                             column_to_list(predictors),
                                                             0.2)
    linreg = LinearRegression()
    rfe = RFE(linreg)
    rfe = rfe.fit(x_train, y_train)
    print(rfe.support_)
    print(rfe.ranking_)
    logreg = LogisticRegression()
    rfe = RFE(logreg)
    rfe = rfe.fit(x_train, y_train)
    print(rfe.support_)
    print(rfe.ranking_)
    return x_test, y_test


def trim_list_whitespace(list_to_trim):
    """Removes all whitespace from the elements of a list."""
    return_list = []
    for index, value in enumerate(list_to_trim):
        return_list.append(list_to_trim[index].replace(' ', ''))
    return return_list


def view_value_counts(dataframe):
    """
    View the value counts of columns and put the results inside of
    a text file.
    """
    df_cols = dataframe.columns
    for col in df_cols:
        print(col, value_count_list(dataframe, col))
    return


def get_percentage_missing(dataframe, column):
    """ 
    Caluculate the percentage of null values in a dataframe column.
    """
    num = dataframe[column].isnull().sum()
    den = len(dataframe)
    return str(round(num/den, 4) * 100) + "%"


def pdf_shape(dataframe):
    """
    Pretty print the shape of a dataframe.
    """
    rows, cols = df_shape(df_age_less_than_25)
    print("Rows: " + str(rows))
    print("Cols: " + str(cols))
    return


if __name__ == "__main__":
    if "ubuntu" in str(os.getcwd()):
        pass
    else:
        import win32api
        put_quotations_around_columns()
    print_random_new_lines(5)