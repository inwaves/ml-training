import numpy as np # linear algebra
import pandas as pd # data manipulation
import matplotlib.pyplot as plt # visualisation
import altair as alt
import seaborn as sns
import eli5
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, plot_confusion_matrix


diabetes_df = pd.read_csv('../data/diabetes-classification.csv')

diabetes_df.columns = [col.lower() for col in diabetes_df.columns]


diabetes_df.info()


diabetes_df.corr()


# plt.figure(figsize=(12, 7))
# sns.boxplot(data=diabetes_df, x='outcome', y='bmi', palette='BuPu')

alt.Chart(diabetes_df).mark_bar().encode(
    x = 'outcome:O',
    y = 'count()'
).properties(width=150)


alt.Chart(diabetes_df).mark_bar(width=10).encode(
    alt.X('pregnancies:Q'),
    alt.Y('count():Q'))


alt.Chart(diabetes_df[diabetes_df['outcome']==1]).mark_bar().encode(
    x = alt.X('age:Q'),
    y = alt.Y('count():Q'),
    tooltip = [alt.Tooltip('age:Q'),
              alt.Tooltip('count()')]
). properties(
                                width=600,
                                height=300)


X = diabetes_df.drop(['outcome', 'age', 'skinthickness'], axis=1)
y = diabetes_df['outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)


def print_results(lin_model, predictions_lin_model, solver):
    print(solver)
    print("-----------------")
    disp = plot_confusion_matrix(lin_model, X_test, y_test,
                                display_labels=['positive', 'negative'],
                                cmap=plt.cm.Blues,
                                values_format= '.0f')
    disp.ax_.set_title("Confusion matrix")
    plt.show()
    print("Accuracy: {}".format(accuracy_score(y_test, predictions_lin_model)))
    print("Precision: {}".format(precision_score(y_test, predictions_lin_model)))
    print("Recall: {}".format(recall_score(y_test, predictions_lin_model)))
    print("ROC AUC: {} \n".format(roc_auc_score(y_test, predictions_lin_model)))

def parameterise_model(solver, C=1.0, max_iter=100):
    lin_model = LogisticRegression(solver=solver, random_state=0,
                              C=C, max_iter=max_iter)
    lin_model.fit(X_train, y_train)
    return lin_model

weights = []
for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
    lin_model = parameterise_model(solver, C=0.8,
                                    max_iter=1000)
    
    # Looking at permutation importance for each label
    perm_importance = eli5.sklearn.PermutationImportance(lin_model,
                                                        random_state=1).fit(X_test, y_test)
    weights.append(eli5.show_weights(perm_importance, feature_names = X_test.columns.tolist()))

    predictions_lin_model = lin_model.predict(X_test)
    print_results(lin_model, predictions_lin_model, solver)
    
for display_weights in weights:
    display(display_weights)
    
