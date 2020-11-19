import pandas as pd  # data manipulation
from sklearn.model_selection import train_test_split


def impute_categorical(df, categorical_column1, categorical_column2):
    """Imputes categorical_column2 using the mode of it corresponding to each value in categorical_column1"""
    cat_frames = []
    for column_value in list(set(df[categorical_column1])):
        df_category = df[df[categorical_column1] == column_value]
        if len(df_category) > 1:
            df_category[categorical_column2].fillna(
                df_category[categorical_column2].mode()[0], inplace=True
            )
        else:
            df_category[categorical_column2].fillna(
                df[categorical_column2].mode()[0], inplace=True
            )
        cat_frames.append(df_category)
        cat_df = pd.concat(cat_frames)
    return cat_df


def impute_numerical(df, categorical_column, numerical_column):
    """Imputes numerical_column using the mean of its values corresponding to each value in categorical_column1"""
    cat_frames = []
    for i in list(set(df[categorical_column])):
        df_category = df[df[categorical_column] == i]
        if len(df_category) > 1:
            df_category[numerical_column].fillna(
                df_category[numerical_column].mean(), inplace=True
            )
        else:
            df_category[numerical_column].fillna(
                df[numerical_column].mode(), inplace=True
            )
        cat_frames.append(df_category)
        cat_df = pd.concat(cat_frames)
    return cat_df


# TODO: before dropping correlated features, find which of each pair is more important in the prediction
def drop_correlated_features(df, target_variable, corr_threshold=0.5):
    """Calculates the correlation between columns and drops all columns that correlate more than
    corr_threshold with each other (only drops the first column in the pair)
    You shouldn't run this until you understand your data.
    """
    correlations = (
        df.corr().drop([target_variable], axis=1).drop(target_variable, axis=0)
    )
    correlation_ranking = []
    for col in correlations.columns:
        # correlation needs to be below 1.0 since a column always correlates perfectly with itself
        correlation_ranking.append(
            [
                col,
                correlations.loc[:, col][correlations.loc[:, col] < 1].abs().idxmax(),
                correlations.loc[:, col][correlations.loc[:, col] < 1].abs().max(),
            ]
        )

    correlation_ranking.sort(key=lambda x: x[2], reverse=True)

    return [
        ranking[0]
        for ranking in correlation_ranking
        if float(ranking[2]) > corr_threshold
    ]


def preprocess_data(
    src_url=None,
    src_type="csv",
    test_size=0.2,
    target_variable=None,
    drop_correlated=False,
):
    """Pre-processes the dataset with common data cleaning methods."""
    if src_url is None:
        raise Exception("The data source URL is empty. Cannot import data.")

    if src_type == "csv":
        df = pd.read_csv(src_url)
    elif src_type == "json":
        df = pd.read_json(src_url)

    df.columns = [col.lower() for col in df.columns]

    if drop_correlated:
        correlated_features = drop_correlated_features(df, target_variable)
        correlated_features.append(target_variable)
        X = df.drop(correlated_features, axis=1)
    else:
        X = df.drop([target_variable], axis=1)

    y = df[target_variable]

    return train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True)
