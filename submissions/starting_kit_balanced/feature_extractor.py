import pandas as pd


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        return _transform(X_df_new)


def _transform(X_df):
    """
    Cached version of the transform method.
    """

    # Some features have all their values set to NaN in the correlation matrix, we simply drop them
    X_df.drop(['s18', 's19', 'op_set_3'], axis=1, inplace=True)

    # Some feature have 0 correlation with target and the other features, we drop them too
    X_df.drop(['op_set_1', 'op_set_2', 's1', 's5', 's16'], axis=1, inplace=True)

    # Some features are highly correlated (>0.9), we drop one of them(s9,s14) (s8,s13)
    X_df.drop(['s12', 's13', 's14'], axis=1, inplace=True)

    X_df = rolling_std(X_df, 's11', 10)
    X_df = rolling_mean(X_df, 's11', 10)


    return X_df


def rolling_std(data, feature, cycle_window, center=True):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (number of cycles) of a defined feature

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    cycle_window : str
        string that defines the length of the cycle window passed to rolling
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """

    df_to_return = pd.DataFrame()
    ids = data.ID.unique()
    name = '_'.join([feature, str(cycle_window), 'std'])
    for i in ids:
        sub_eng = data.loc[lambda df: df.ID == i, :].copy()
        sub_eng.loc[:, name] = sub_eng[feature].rolling(cycle_window, center=center).std()
        sub_eng.loc[:, name] = sub_eng[name].ffill().bfill()
        sub_eng.loc[:, name] = sub_eng[name].astype(sub_eng[feature].dtype)
        df_to_return = pd.concat([df_to_return, sub_eng], axis=0)
    return df_to_return


def rolling_mean(data, feature, cycle_window, center=False):
    """
    For a given dataframe, compute the mean over
    a defined period of time (number of cycles) of a defined feature

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    cycle_window : str
        string that defines the length of the cycle window passed to rolling
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """

    df_to_return = pd.DataFrame()
    ids = data.ID.unique()
    name = '_'.join([feature, str(cycle_window), 'mean'])
    for i in ids:
        sub_eng = data.loc[lambda df: df.ID == i, :].copy()
        sub_eng.loc[:, name] = sub_eng[feature].rolling(cycle_window, win_type='hamming', center=center).mean()
        sub_eng.loc[:, name] = sub_eng[name].ffill().bfill()
        sub_eng.loc[:, name] = sub_eng[name].astype(sub_eng[feature].dtype)
        df_to_return = pd.concat([df_to_return, sub_eng], axis=0)
    return df_to_return