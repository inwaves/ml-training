from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns


diabetes_df = pd.read_csv('../data/diabetes.csv')

# I want the column names to be a bit more descriptive
diabetes_df.rename(columns={'S1':'t_cells', 'S2':'ld_lipo', 'S3':'hd_lipo',
                            'S4':'thyroid_sh', 'S5':'lamotrigine', 'S6':'blood_sugar'}, inplace=True)

diabetes_df.columns = [col.lower() for col in diabetes_df]


diabetes_df.info()


diabetes_df.describe()['y']


diabetes_df = diabetes_df.drop('ld_lipo', axis=1)
diabetes_df = diabetes_df.drop('thyroid_sh', axis=1)


# let's eliminate the predicted column, then split the data
X = diabetes_df.drop('y', axis=1)
y = diabetes_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, shuffle=True)

# training the linear regressor on the training data
models = {}
predictions = {}
models['linear'] = linear_model.LinearRegression()
models['ridge'] = linear_model.Ridge(random_state=0)
models['lasso'] = linear_model.Lasso(random_state=0)
models['elasticnet'] = linear_model.ElasticNet(random_state=0)

# print([k, v for (k,v) in models.items()])
for (model_name, model) in models.items():
    model.fit(X_train, y_train)
    predictions[model_name] = model.predict(X_test)


# evaluating model
scores = []
for (model, prediction) in predictions.items():
    scores.append([model, 
                   mean_absolute_error(y_test, prediction),
                   r2_score(y_test, prediction)])

column_names = ['model_type', 'mean_absolute_error', 'r2_score']
res_df = pd.DataFrame(scores, columns=column_names).set_index('model_type')
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



