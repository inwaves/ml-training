import shap


def shap_single_value(X_test, explainer, my_model, row_to_show):
    """ Calculates the Shap values for a single row of data, displays the force plot.
        Use when explaining a single prediction.
    """
    data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

    my_model.predict_proba(data_for_prediction_array) # raw prediction, KernelExplainer uses this

    # Calculate Shap values for a single row
    shap_values = explainer.shap_values(data_for_prediction)

    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction) #[0] contains Shap values for negative outcome

def shap_agg_plot(X_test, explainer):
    """ Calculates the Shap values across all rows, plots the summary for the values.
        Gives an overview of which features were impactful in making a prediction.
    """
    shap_values = explainer.shap_values(X_test) # entire set, not one row

    # Make plot. [0] contains Shap values for negative outcome, [1] for positive
    shap.summary_plot(shap_values[1], X_test)

def shap_dependence_plot(X, explainer, fst_feature, snd_feature=None):
    """ Generates a Shap value interaction plot, which shows how a pair of features interact and how
        they change the value of the prediction, on average.
        Good to study how features interact.
        If snd_feature is blank, it will automatically determine the most interesting interaction.
    """
    shap_values = explainer.shap_values(X) # train on entire set

    shap.dependence_plot(fst_feature, shap_values[1], X, interaction_index=snd_feature)

def main():
    # You should preprocess your data, train your model here

    # Create object that can calculate Shap values: can be TreeExplainer, DeepExplainer, KernelExplainer
    explainer = shap.TreeExplainer(my_model)
    shap_single_value(X_test, explainer, my_model, row_to_show)
    shap_agg_plot(X_test, explainer)
    shap_dependence_plot(X, explainer, fst_feature, snd_feature)