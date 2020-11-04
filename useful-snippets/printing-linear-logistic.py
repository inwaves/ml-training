import matplotlib.pyplot as plt # visualisation
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, plot_confusion_matrix

def print_logistic_results(X_test, y_test, logistic_model, predictions_logistic_model, solver,
                 confusion_matrix=False):
    """ Prints the confusion matrix and precision, accuracy, recall and ROC AUC.
        X_test : a pandas DataFrame containing the test set of features
        y_test : a pandas DataFrame containing the test set of target variable
        logistic_model : a sklearn.linear_model.LogisticRegression object
        predictions_logistic_model : predictions made by the model
        solver : only for printing purposes, shows which solver was used in the model
        confusion_matrix : controls whether matrix is printed or not (can be verbose)
    """
    print(solver)
    print("-----------------")
    if confusion_matrix:
        disp = plot_confusion_matrix(logistic_model, X_test, y_test,
                                    display_labels=['positive', 'negative'],
                                    cmap=plt.cm.Blues,
                                    values_format= '.0f')
        disp.ax_.set_title("Confusion matrix")
        plt.show()
    print("Accuracy: {}".format(accuracy_score(y_test, predictions_logistic_model)))
    print("Precision: {}".format(precision_score(y_test, predictions_logistic_model)))
    print("Recall: {}".format(recall_score(y_test, predictions_logistic_model)))
    print("ROC AUC: {} \n".format(roc_auc_score(y_test, predictions_logistic_model)))

def main():
    # You should preprocess your data, train your model here

    print_logistic_results(X_test, y_test, logistic_model, predictions_logistic_model, solver, confusion_matrix=True)