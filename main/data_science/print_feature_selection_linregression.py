"""
Feature selection for linear regression on the given column.
"""
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def get_target_predictors(dataframe, target_col):
    """
    Return the dataframe parsed into a target column
    and predictors.
    """
    dataframe = dataframe.dropna(axis=0)
    data_vars = convert_column_to_list(dataframe)
    target = [target_col]
    predictors = [col for col in data_vars if col not in target]
    return dataframe[target], dataframe[predictors]


def convert_column_to_list(dataframe):
    """Convert dataframe columns to a list."""
    return dataframe.columns.values.tolist()


def print_feature_selection_linreg(dataframe, target_col):
    """Print the feature selection for linear regression."""

    #Split data into target and predictors.
    target, predictors = get_target_predictors(dataframe, target_col)

    linreg = LinearRegression()
    rfe = RFE(linreg)
    rfe = rfe.fit(predictors, target.values.ravel())
    print(rfe.support_)
    print(rfe.ranking_)
