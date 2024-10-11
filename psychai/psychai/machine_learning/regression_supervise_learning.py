import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from statsmodels.formula.api import ols
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import shap
import psychai.statistics.bland_altman_analysis.bland_altman_analysis as baa

def regression_supervised_learning(data_input, prediction_methods, dependent_variables, independent_variables_input, parameters=None):
    """
    Runs machine learning models to predict target variables using different algorithms.
    It includes cross-validation, outlier handling, SHAP analysis, and evaluation metrics.

    Args:
        data_input (DataFrame): The dataset containing both independent and dependent variables.
        prediction_methods (list): List of machine learning methods to use (e.g., 'Linear Regression', 'SVM').
        dependent_variables (list): List of target columns (dependent variables) to predict.
        independent_variables_input (list): List of independent variables (features) used in prediction.
        parameters (optional): Additional parameters for specific models (e.g., hidden layers for MLP).

    Returns:
        None
    """

    # Set font size for all plots globally
    font_size = 16
    plt.rcParams.update({'font.size': font_size})

    # Cross-validation parameters
    cross_validation = 10  # Number of folds for cross-validation
    number_of_repeat = 1  # Number of repetitions for cross-validation

    # Loop over each target (dependent variable)
    for target_index in range(len(dependent_variables)):
        prediction_target_column = dependent_variables[target_index]

        # Filter the data to remove rows with missing target values
        data_wt_target = data_input.dropna(subset=[prediction_target_column])
        y = data_wt_target[prediction_target_column]

        # Preprocessing: standardize the independent variables
        scaler = StandardScaler()
        X = data_wt_target[independent_variables_input]
        X_scaled = scaler.fit_transform(X)

        # Initialize KFold cross-validation
        kf = KFold(n_splits=cross_validation, shuffle=True, random_state=42)

        # Loop over each prediction method specified
        for learning_index in range(len(prediction_methods)):
            all_r2_scores, all_rmse_scores, all_mae_scores = [], [], []
            prediction_results = []
            aggregated_shap_values, aggregated_X_test = [], []

            # Repeat cross-validation (in case you want to perform multiple repetitions)
            for repeat_index in range(number_of_repeat):
                # Cross-validation loop
                for train_index, test_index in kf.split(X_scaled):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Initialize the appropriate model based on the method
                    if prediction_methods[learning_index] == "Linear Regression":
                        model = LinearRegression().fit(X_train, y_train)
                    elif prediction_methods[learning_index] == "Decision Tree":
                        model = DecisionTreeRegressor().fit(X_train, y_train)
                    elif prediction_methods[learning_index] == "Bagged Tree":
                        model = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10).fit(X_train, y_train)
                    elif prediction_methods[learning_index] == "Random Forest":
                        model = RandomForestRegressor().fit(X_train, y_train)
                    elif prediction_methods[learning_index] == "SVM":
                        model = SVR().fit(X_train, y_train)
                    elif prediction_methods[learning_index] == "MLP":
                        # Pass parameters if specified (e.g., hidden layers)
                        model = MLPRegressor(hidden_layer_sizes=parameters, max_iter=500, random_state=42).fit(X_train, y_train)

                    # Make predictions on the test set
                    predictions = model.predict(X_test)

                    # Handle outliers by replacing them with the median
                    # Calculate IQR for y_train to detect outliers
                    Q1, Q3 = np.percentile(y_train, 25), np.percentile(y_train, 75)
                    IQR = Q3 - Q1
                    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    median_y_train = np.median(y_train)

                    # Replace predictions outside the IQR with the median
                    predictions = np.where((predictions < lower_bound) | (predictions > upper_bound), median_y_train, predictions)

                    # Store actual and predicted values for evaluation
                    prediction_results.append(np.column_stack((predictions, y_test)))  # Changed the order: Predicted, Actual

                    # Calculate performance metrics
                    r2 = r2_score(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    mae = mean_absolute_error(y_test, predictions)

                    all_r2_scores.append(r2)
                    all_rmse_scores.append(rmse)
                    all_mae_scores.append(mae)

                    # SHAP analysis (interpretability of model predictions)
                    explainer = shap.Explainer(model, X_train)
                    shap_values = explainer(X_test, check_additivity=False)
                    aggregated_shap_values.append(shap_values.values)
                    aggregated_X_test.append(X_test)

            # Calculate and print average scores across folds
            avg_r2 = np.mean(all_r2_scores)
            avg_rmse = np.mean(all_rmse_scores)
            avg_mae = np.mean(all_mae_scores)
            std_r2, std_mae = np.std(all_r2_scores), np.std(all_mae_scores)

            print(f"Predicted: {prediction_target_column}, Method: {prediction_methods[learning_index]}")
            print(f"Average R2: {avg_r2:.2f}, Average RMSE: {avg_rmse:.2f}, Average MAE: {avg_mae:.2f}")

            # Convert results to a DataFrame for easier manipulation and plotting
            prediction_results_df = pd.DataFrame(np.vstack(prediction_results), columns=['Predicted', 'Actual'])  # Changed column order

            # Scatter plot to compare actual vs. predicted values
            plt.figure(figsize=(6, 4))
            plt.scatter(prediction_results_df['Predicted'], prediction_results_df['Actual'])  # Changed x: Predicted, y: Actual
            plt.title(f'{prediction_target_column} [R2:{avg_r2:.2f}({std_r2:.2f}), MAE:{avg_mae:.2f}({std_mae:.2f})]', fontsize=font_size)
            plt.xlabel('Predicted')  # Changed label to Predicted for x-axis
            plt.ylabel('Actual')  # Changed label to Actual for y-axis
            plt.show()

            # Bland-Altman analysis (for checking agreement between actual and predicted)
            analysis_results = baa.BlandAltmanAnalysis(prediction_results_df['Actual'], prediction_results_df['Predicted'])
            print("Bland-Altman Analysis Results:")
            print(f"Mean Difference: {analysis_results['mean_diff']:.2f}")
            print(f"Standard Deviation of Difference: {analysis_results['sd_diff']:.2f}")
            print(f"Upper Limit of Agreement: {analysis_results['upper_limit']:.2f}")
            print(f"Lower Limit of Agreement: {analysis_results['lower_limit']:.2f}")
            print(f"Number of Data Points: {analysis_results['n']}")

            # Aggregate SHAP values and features for all folds
            aggregated_shap_values = np.concatenate(aggregated_shap_values, axis=0)
            aggregated_X_test = np.concatenate(aggregated_X_test, axis=0)

            # SHAP summary plot for interpretability
            shap.summary_plot(aggregated_shap_values, aggregated_X_test, feature_names=independent_variables_input)

            # Identify top features based on SHAP values
            number_features_shown = 6  # Number of top features to display
            top_features = np.argsort(np.mean(np.abs(aggregated_shap_values), axis=0))[-number_features_shown:]

            # Set font sizes for axis labels and tick labels
            plt.rcParams.update({
                'xtick.labelsize': font_size,  # X-axis tick label font size
                'ytick.labelsize': font_size,  # Y-axis tick label font size
                'axes.labelsize': font_size    # Axis label font size
            })
            

            # Create SHAP summary plot for top features
            shap_values_top = aggregated_shap_values[:, top_features]
            shap.summary_plot(shap_values_top, aggregated_X_test[:, top_features], 
                              feature_names=[independent_variables_input[i] for i in top_features])
