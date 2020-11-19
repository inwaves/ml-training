import matplotlib.pyplot as plt
from pdpbox import pdp  # for partial dependence plotting


def show_partial_dep_plots(lin_model, X_test):
    """Prints partial dependence plots for each feature in the dataset."""
    for feat_name in X_test.columns:
        pdp_dist = pdp.pdp_isolate(
            model=lin_model,
            dataset=X_test,
            model_features=X_test.columns,
            feature=feat_name,
        )
        pdp.pdp_plot(pdp_dist, feat_name)
        plt.show()
