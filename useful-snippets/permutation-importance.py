import eli5
import pandas as pd
from IPython.display import display # to display multiple permutations

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