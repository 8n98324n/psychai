import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def normalize_and_add_by_variable(input_data, target, normalized_base, normalized_value, check_p_value=False):
    # Build and fit the linear regression model
    formula = f"{target} ~ {normalized_base}"
    model = smf.ols(formula, data=input_data).fit()
    p_value = model.f_pvalue

    if check_p_value and p_value < 0.05:
        # Calculate the prediction for the normalized value
        normalized_data = pd.DataFrame({normalized_base: [normalized_value]})
        prediction_at_normalized_value = model.predict(normalized_data).iloc[0]

        # Adjust the target variable based on the model's predictions
        adjusted_predictions = model.predict(input_data[normalized_base]) - model.params[1] * normalized_value
        normalized_result = input_data[target] - adjusted_predictions + prediction_at_normalized_value
    else:
        # If p-value is high, don't normalize
        normalized_result = input_data[target]

    # Add the normalized result to the input data
    normalized_column_name = f"{target}_{normalized_value}"
    input_data[normalized_column_name] = normalized_result

    return input_data, p_value

# Example usage
# data_valid, p_value = normalized_by(data_valid, 'MAP', 'HR', 75)

def normalize_and_add_by_variables(input_data, new_variable_list, normalized_base, normalized_value, check_p_value=False):
    output_data = input_data.copy()

    for new_variable_name in new_variable_list:
        # Check if the variable is a column in the DataFrame
        if new_variable_name in output_data.columns:
            # Normalize the variable
            output_data, _ = normalize_and_add_by_variable(output_data, new_variable_name, normalized_base, normalized_value, check_p_value)

    return output_data

