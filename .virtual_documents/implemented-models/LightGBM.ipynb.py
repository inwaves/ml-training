from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


diabetes_df = pd.read_csv('../data/diabetes.csv')

# I want the column names to be a bit more descriptive
diabetes_df.rename(columns={'S1':'t_cells', 'S2':'ld_lipo', 'S3':'hd_lipo',
                            'S4':'thyroid_sh', 'S5':'lamotrigine', 'S6':'blood_sugar'}, inplace=True)

diabetes_df.columns = [col.lower() for col in diabetes_df]


diabetes_df.info()


diabetes_df.corr()


# diabetes_df = diabetes_df.drop('ld_lipo', axis=1)
diabetes_df = diabetes_df.drop('thyroid_sh', axis=1)


# let's eliminate the predicted column, then split the data
X = diabetes_df.drop('y', axis=1)
y = diabetes_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, shuffle=True)


def run_booster(learning_rate):
    bst = LGBMRegressor(n_estimators=500, learning_rate=learning_rate) # initialising using scikit API
    bst.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False)

    # predicting the test data
    return bst.predict(X_test)


# testing 7 values for the learning rate, equally spaced between 0.001 and 1

results = []
alpha_range = np.linspace(0.001, 1, num=20)
for alpha in alpha_range:
    predictions_bst = run_booster(alpha)
    results.append([alpha, 
                    mean_absolute_error(y_test, predictions_bst), 
                    round(r2_score(y_test, predictions_bst),2)])
    
column_names = ['learning_rate', 'mean_absolute_error', 'r2_score']
res_df = pd.DataFrame(results, columns=column_names).set_index('learning_rate')
res_df.sort_values(by='r2_score', ascending=False)


# just playing around with different visualisations
# what would be useful to visualise?
def plot_altair(column):
    return alt.Chart(diabetes_df).mark_point(filled=True).encode(
        x = alt.X(column, scale=alt.Scale(zero=False)),
        y = alt.Y('y:Q', scale=alt.Scale(zero=False)))
        # color = alt.Color('SEX:N'),
        # size = alt.Size('blood_sugar:Q', title='Blood sugar'),
        # opacity = alt.OpacityValue(0.5))

# a regression line for each variable against the target variable
# but this is the /actual/ target variable, not the model's prediction of it
charts = []
for col in list(X.columns):
    chart = plot_altair(col + ':Q')
    charts.append(chart + chart.transform_regression(str(col), 'Y').mark_line())

alt.vconcat(*charts[2:])



