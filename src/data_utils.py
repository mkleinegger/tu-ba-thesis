import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def split_data(df, target, drop_na=False):
    """
    Splits a DataFrame into features (X) and target (y).
    """
    _df = df.copy()
    if drop_na:
        _df = _df.dropna()
    X = _df.drop(columns=[target])
    y = _df[target]
    return X, y

def custom_one_hot_encoding(X_train, X_test, nominal_features):
    """
    Performs one-hot encoding on categorical features and aligns columns between train and test sets.
    """
    X_train = pd.get_dummies(X_train, columns=nominal_features)
    X_test = pd.get_dummies(X_test, columns=nominal_features)
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)
    X_train = X_train.astype(int)
    X_test = X_test.astype(int)
    return X_train, X_test

def apply_binning(X):
    """
    Applies binning to numerical features for better categorization.
    """
    X = X.reset_index(drop=True)
    cols = list(X.columns)
    X[cols] = X[cols].replace([" ?"], np.nan)
    X = X.dropna()
    
    def strip_str(x):
        return x.strip() if isinstance(x, str) else x
    
    X = X.applymap(strip_str)
    X["hours-per-week"] = pd.cut(
        x=X["hours-per-week"],
        bins=[0.9, 25, 39, 40, 55, 100],
        labels=["PartTime", "MidTime", "FullTime", "OverTime", "BrainDrain"],
    )
    X["age"] = pd.qcut(X["age"], q=5)
    X["capital-gain"] = pd.cut(
        x=X["capital-gain"],
        bins=[-1, 0, 5000, 10000, 50000, max(100000, X["capital-gain"].max() + 1)],
        labels=["NoGain", "LowGain", "MediumGain", "HighGain", "VeryHighGain"],
    )
    X["capital-loss"] = pd.cut(
        x=X["capital-loss"],
        bins=[-1, 0, 1000, 2000, 5000, 100000],
        labels=["NoLoss", "LowLoss", "MediumLoss", "HighLoss", "VeryHighLoss"],
    )
    return X

def apply_pdf(X, target):
    """
    Transforms numerical attributes into their probability density function (PDF) values.
    """
    def transform_to_pdf(data, numerical_columns, target):
        transformed_data = data.copy()
        for column in numerical_columns:
            if column not in ["sex", target]:
                kde = gaussian_kde(data[column].dropna())
                transformed_data[column] = kde(data[column])
        return transformed_data
    
    cols = list(X.columns)
    X[cols] = X[cols].replace([" ?"], np.nan)
    X = X.dropna()
    X = X.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    X = transform_to_pdf(X, ["age", "hours-per-week", "capital-gain", "capital-loss"], target)
    return X

def convert_intervals(obj):
    """
    Converts pandas Interval objects to strings for JSON serialization.
    """
    if isinstance(obj, pd.Interval):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def prepare_data_fair_learning(df_train, df_test, nominal_features, target):
    """
    Prepares datasets for fair learning by performing data cleaning and one-hot encoding.
    """
    _df_train = df_train.copy()
    _df_test = df_test.copy()
    X_train, y_train = split_data(_df_train, target, drop_na=True)
    X_test, y_test = split_data(_df_test, target, drop_na=True)
    y_train = pd.Series(y_train.factorize(sort=True)[0], index=y_train.index, name=y_train.name)
    y_test = pd.Series(y_test.factorize(sort=True)[0], index=y_test.index, name=y_test.name)
    X_train, X_test = custom_one_hot_encoding(X_train, X_test, nominal_features)
    return X_train, y_train, X_test, y_test
