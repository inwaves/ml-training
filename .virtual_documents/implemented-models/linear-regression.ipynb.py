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
                            'S4':'thyroid-sh', 'S5':'lamotrigine', 'S6':'blood_sugar'}, inplace=True)

diabetes_df.columns = [col.lower() for col in diabetes_df]


diabetes_df.info()


diabetes_df.describe()['y']


# let's eliminate the predicted column, then split the data
X = diabetes_df.drop('y', axis=1)
y = diabetes_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, shuffle=True)

# training the linear regressor on the training data
model_lr = linear_model.LinearRegression()
model_lr.fit(X_train, y_train)

# predicting the test data
predictions_lr = model_lr.predict(X_test)


# evaluating model
print('MAE:', mean_absolute_error(y_test, predictions_lr))
print('R2:', round(r2_score(y_test, predictions_lr),2))


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



