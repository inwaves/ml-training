from custom_pdp import show_partial_dep_plots
from permutation_importance import show_model_permutations
from printing_linear_logistic import print_logistic_results
from sklearn.linear_model import LogisticRegression


def fit_param_model(X_train, y_train, solver, C=1.0, max_iter=100):
    """Generates and fits a logistic regression model"""
    lin_model = LogisticRegression(
        solver=solver, random_state=0, C=C, max_iter=max_iter
    )

    lin_model.fit(X_train, y_train)

    return lin_model


def logistic_regression(X_train, X_test, y_train, y_test):
    """Spawns several logistic regression models, displays their permutation importances, then scores their predictions."""
    for solver in ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]:
        lin_model = fit_param_model(X_train, y_train, solver, C=0.8, max_iter=1000)
        show_model_permutations(X_test, y_test, lin_model)

        predictions_lin_model = lin_model.predict(X_test)
        print_logistic_results(X_test, y_test, lin_model, predictions_lin_model, solver)

    show_partial_dep_plots(
        fit_param_model(X_train, y_train, solver="newton-cg"), X_test
    )
    show_model_permutations()  # comment out if you don't want to see permutation importance


print()
