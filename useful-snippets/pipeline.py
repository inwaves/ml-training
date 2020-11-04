import pandas as pd # data manipulation
from sklearn.model_selection import train_test_split

# TODO: build a proper pipeline

def preprocess_data():
    """ Pre-processes the dataset with common data cleaning methods.
    """

    diabetes_df = pd.read_csv('../data/diabetes-classification.csv')

    diabetes_df.columns = [col.lower() for col in diabetes_df.columns]

    X = diabetes_df.drop(['outcome', 'age', 'skinthickness'], axis=1) # dropping correlated variables
    y = diabetes_df['outcome']

    return train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
