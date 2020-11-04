import pandas as pd # data manipulation
from sklearn.model_selection import train_test_split

# TODO: build a proper pipeline

def impute_categorical(df, categorical_column1, categorical_column2):
    """ Imputes categorical_column2 using the mode of it corresponding to each value in categorical_column1
    """
    cat_frames = []
    for column_value in list(set(df[categorical_column1])):
        df_category = df[df[categorical_column1] == column_value]
        if len(df_category) > 1:    
            df_category[categorical_column2].fillna(df_category[categorical_column2].mode()[0],inplace = True)        
        else:
            df_category[categorical_column2].fillna(df[categorical_column2].mode()[0],inplace = True)
        cat_frames.append(df_category)    
        cat_df = pd.concat(cat_frames)
    return cat_df

def impute_numerical(df, categorical_column, numerical_column):
    """ Imputes numerical_column using the mean of its values corresponding to each value in categorical_column1
    """
    cat_frames = []
    for i in list(set(df[categorical_column])):
        df_category = df[df[categorical_column]== i]
        if len(df_category) > 1:    
            df_category[numerical_column].fillna(df_category[numerical_column].mean(),inplace = True)        
        else:
            df_category[numerical_column].fillna(df[numerical_column].mode(),inplace = True)
        cat_frames.append(df_category)    
        cat_df = pd.concat(cat_frames)
    return cat_df

def preprocess_data():
    """ Pre-processes the dataset with common data cleaning methods.
    """

    diabetes_df = pd.read_csv('../data/diabetes-classification.csv')

    diabetes_df.columns = [col.lower() for col in diabetes_df.columns]

    X = diabetes_df.drop(['outcome', 'age', 'skinthickness'], axis=1) # dropping correlated variables
    y = diabetes_df['outcome']

    return train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
