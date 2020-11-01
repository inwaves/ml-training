from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import graphviz


diabetes_df = pd.read_csv('../data/diabetes.csv')

# I want the column names to be a bit more descriptive
diabetes_df.rename(columns={'S1':'t_cells', 'S2':'ld_lipo', 'S3':'hd_lipo',
                            'S4':'thyroid_sh', 'S5':'lamotrigine', 'S6':'blood_sugar'}, inplace=True)

diabetes_df.columns = [col.lower() for col in diabetes_df]


diabetes_df.info()


diabetes_df.corr()


# diabetes_df = diabetes_df.drop('ld_lipo', axis=1)
diabetes_df = diabetes_df.drop('thyroid_sh', axis=1)
# diabetes_df = diabetes_df.drop('t_cells', axis=1)
# diabetes_df = diabetes_df.drop('sex', axis=1)


# let's eliminate the predicted column, then split the data
X = diabetes_df.drop('y', axis=1)
y = diabetes_df['y']

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, shuffle=True)
scaler.fit(X_train) # fit the scaler on the training data
X_train = scaler.transform(X_train) # scale training data
X_test = scaler.transform(X_test) # scale test data


def run_perceptron(X_train, y_train, X_test, solver, hidden_layer_sizes, alpha, max_iter):
    """ This function parameterises the creation of a multilayer perceptron
        classifier so that multiple tests can be run for parameter optimisation.
        returns: predictions
    """
    clf = MLPClassifier(solver=solver, alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        random_state=1,
                        max_iter=max_iter)

    clf.fit(X_train, y_train) # train the model
    return clf.predict(X_test) # then predict our test set FIXME: shape of the Series



results = []
alpha_range = np.linspace(0.001, 0.1, num=7)
for alpha in alpha_range:
    predictions_clf = run_perceptron(X_train, y_train, X_test,
                                     solver='adam', 
                                     hidden_layer_sizes=(5,2),
                                     alpha=alpha, 
                                     max_iter=500)
    results.append([alpha, 
                    mean_absolute_error(y_test, predictions_clf), 
                    round(r2_score(y_test, predictions_clf),2)])
    
column_names = ['l2_regularisation', 'mean_absolute_error', 'r2_score']
res_df = pd.DataFrame(results, columns=column_names).set_index('l2_regularisation')
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



