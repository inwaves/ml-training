import pandas as pd # data manipulation
import matplotlib.pyplot as plt # visualisation
import eli5 # for permutation importance
from pdpbox import pdp, get_dataset, info_plots # for partial dependence plotting 
from IPython.display import display # for permutation importance display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, plot_confusion_matrix


def preprocess_data():
    """ Pre-processes the dataset with common data cleaning methods.
    """

    diabetes_df = pd.read_csv('../data/diabetes-classification.csv')

    diabetes_df.columns = [col.lower() for col in diabetes_df.columns]

    X = diabetes_df.drop(['outcome', 'age', 'skinthickness'], axis=1) # dropping correlated variables
    y = diabetes_df['outcome']

    return train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

weights = []
def show_model_permutations(X_test=None, y_test=None, lin_model=None):
    """ Appends permutation importance for the labels on one model to a list.
        Prints when called with no model.
    """
    if lin_model is None:         
        for display_weights in weights:
            display(display_weights)
        return
    
    perm_importance = calc_model_perm_importance(X_test, y_test, lin_model)
    weights.append(eli5.show_weights(perm_importance, feature_names = X_test.columns.tolist()))
    
def calc_model_perm_importance(X_test, y_test, lin_model):
    """ Calculates the permutation importance for the labels on one model. 
    """

    return eli5.sklearn.PermutationImportance(lin_model,
                                                        random_state=1).fit(X_test, y_test)

def print_results(X_test, y_test, lin_model, predictions_lin_model, solver,
                 confusion_matrix=False):
    """ Prints the confusion matrix and precision, accuracy, recall and ROC AUC.
    """
    print(solver)
    print("-----------------")
    if confusion_matrix:
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

def show_partial_dep_plots(lin_model, X_test):
    """ Prints partial dependence plots for each feature in the dataset.
    """
    for feat_name in X_test.columns:
        pdp_dist = pdp.pdp_isolate(model=lin_model, dataset=X_test, 
                                   model_features=X_test.columns, feature=feat_name)
        pdp.pdp_plot(pdp_dist, feat_name)
        plt.show() 
    
def parameterise_model(X_train, y_train, solver, C=1.0, max_iter=100):
    """ Generates and fits a logistic regression model
    """
    lin_model = LogisticRegression(solver=solver, random_state=0,
                              C=C, max_iter=max_iter)
    
    lin_model.fit(X_train, y_train)
    
    return lin_model

def logistic_regression(X_train, X_test, y_train, y_test):
    """ Spawns several logistic regression models, displays their permutation importances, then scores their predictions.
    """
    for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
        lin_model = parameterise_model(X_train, y_train,
                                        solver, C=0.8,
                                        max_iter=1000)
        show_model_permutations(X_test, y_test, lin_model)
        
        predictions_lin_model = lin_model.predict(X_test)
        print_results(X_test, y_test, lin_model, predictions_lin_model, solver)
    
    show_partial_dep_plots(parameterise_model(X_train, y_train, solver='newton-cg'), X_test)
    show_model_permutations() # comment out if you don't want to see permutation importance 

def main():
    X_train, X_test, y_train, y_test = preprocess_data()
    logistic_regression(X_train, X_test, y_train, y_test)
    
if __name__ == '__main__':
    main()