
import os
import pandas as pd
import pickle
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import psychai.data_visualization.chart
import psychai.data_preparation.preprocessing
import psychai.machine_learning.classification_supervise_learning

import os
import sys

import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Extend system path to current directory (for importing modules if needed)
current_dir = os.path.abspath('../')
sys.path.append(current_dir)

import local_data_loader
import local_utilities
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind, pearsonr, f_oneway  # For statistical tests
from itertools import combinations  # For generating combinations of values
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from scipy.stats import boxcox
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
import psychai.feature.feature_engineering.feature_engineering
import psychai.data_preparation.data_preparation
import psychai.data_visualization.chart
# import psychai.data_preparation.preprocessing
import psychai.machine_learning.classifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer


class SingleModalityAnalysis:

    def __init__(self):
        pass

    def prosss_folder(self, base_directory, filters, user_ids, merging_pdf, modality_value= None):

        file_filter = FileFilter(base_directory, attribute_filters=filters, modality_value=modality_value)

        df = file_filter.get_matching_files(user_ids)

        # Merge the two dataframes based on the 'user_id' column
        df_updated = pd.merge(df, merging_pdf[['user_id', 'Group']], on='user_id', how='left')

        # Find the row with the largest file_size for each user_id
        #df_largest = df_updated.loc[df_updated.groupby('user_id')['file_size'].idxmax()]
        df_largest = df_updated.loc[df_updated.groupby(['user_id', 'part_2'])['file_size'].idxmax()]
        

        # Reset the index if needed
        df_largest.reset_index(drop=True, inplace=True)

        # Display the result
        print(df_largest)

        return df_largest

class EmotionClassificationTask:
    def __init__(self):
        pass    

    def classify(
        self,
        df_input,
        group_labels:dict,               # Dictionary for relabeling classes (e.g., {"1": "positive", "2": "negative"})
        model_name = "Random Forest",
        task_name = "",                  # Name of the classification task for display
        feature_columns = [],
        target_column = "Group",
        group_column = "user_id",
        ticklabel = [],
        n_cross_validation = 10,
        remove_high_correlation = False,
        corr_threshold = 0.95,
        random_state = 1,
        do_anova = False,
        percentage_kept_by_anova = 0.5,
        do_rfe = False,
        rfe_model = None,
        n_kept = 10,
        do_pca = False,
        explained_variance_threshold = 0.85,
        do_model_grid_search = False,
        model_grid_search_parameters = {},
        verbose = False,
        visualization = True,
        show_percentage = False,
        n_top_features = 10,  # Adjust `n` to your desired number of top features
        chart_helper: psychai.data_visualization.chart.Chart = None,
        model_initial_parameters = {}
    ):
        """
        A function to run a machine learning pipeline, including data preparation, feature selection,
        hyperparameter tuning, and evaluation metrics.
        Returns accuracy and F1 scores for each class.
        """
        fe_helper = psychai.feature.feature_engineering.feature_engineering.FeatureEngineeringHelper()
        print(f"random_state:{random_state}")

        df = df_input.copy()
        df[target_column] = df[target_column].map(group_labels)


        # Prepare feature columns
        if len(feature_columns)==0:
            feature_columns = df.columns.tolist()

        columns_to_filter_out = [target_column,group_column,"Unnamed: 0","seq"]
        feature_columns = [col for col in feature_columns if col not in columns_to_filter_out]

        #feature_columns = df.columns[3:]  # Exclude target and group columns
        # target_column = "Group"
        # group_column = "user_id"

        X = df[feature_columns].to_numpy()
        y = df[target_column].to_numpy()

        # Initialize cross-validator
        if group_column:
            cv = GroupKFold(n_splits=n_cross_validation)
            groups = df[group_column].to_numpy()
        else:
            cv = KFold(n_splits=n_cross_validation, shuffle=True, random_state=random_state)
            groups = None

        # Initialize Random Forest model
        # model = RandomForestClassifier(random_state=random_state)

        # Store cross-validation scores
        cv_scores = []
        all_y_true = []
        all_y_pred = []
        all_y_test_binary = []
        all_y_prob = []
        all_groups = []

        # Initialize an empty list to store SHAP values and feature names across splits
        # all_shap_values = []
        # all_feature_names = []
        # Initialize an empty DataFrame to store all iterations
        df_all_feature_importance = pd.DataFrame()

        iteration_indx = 1
        for train_idx, val_idx in cv.split(X, y, groups):

            kept_columns = feature_columns
            # Split data into train and validation sets
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if groups is not None:
                groups_train = groups[train_idx]
                groups_val = groups[val_idx]
            else:
                groups_train = None
                groups_val = None
            # ===================== Scale features ========================================================
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # ===================== Remove High Correlated Columns ========================================

            if remove_high_correlation:
                X_train, X_val, kept_columns = fe_helper.remove_high_correlation(X_train, X_val , kept_columns, corr_threshold = corr_threshold, verbose= verbose)

            # ===================== Keep only high Anova F-values =========================================
            if do_anova:
                n_features_kept_by_anova = int(len(kept_columns)*percentage_kept_by_anova)
                if verbose:
                    print(f"{n_features_kept_by_anova} columns out of {len(kept_columns)} columns selected by anova")
                X_train, X_val, kept_columns = fe_helper.keep_high_anova(X_train, X_val, y_train, kept_columns, n_features_kept_by_anova, verbose= verbose)

            # ===================== Perform Recursive Feature Elimination (RFE) ===========================
            if do_rfe:
                X_train, X_val, kept_columns = fe_helper.RFE(X_train, X_val,  y_train, random_state, kept_columns, n_kept, verbose= verbose, rfe_model = rfe_model)

            #====================== PCA ===================================================================
            if do_pca:
                X_train, X_val = fe_helper.pca(X_train, X_val ,explained_variance_threshold, verbose= verbose)
                
            # Train the Random Forest model
            # Shuffle training data
            if groups is not None:
                X_train, y_train, groups_train = shuffle(X_train, y_train, groups_train, random_state=random_state)
            else:
                X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

            #X_train, y_train, groups_train = shuffle(X_train, y_train, groups_train, random_state=random_state)

            classifier = psychai.machine_learning.classifier.Classifier(
                X_train = X_train , y_train = y_train, X_test = X_val, y_test = y_val, model_name = model_name,
                do_model_grid_search = do_model_grid_search, grid_search_parameters = model_grid_search_parameters,
                group_info = groups_train, model_initial_parameters = model_initial_parameters
            )
            classifier.fit()

            # Evaluate the model on the validation set
            y_pred = classifier.predict()
            
            score = accuracy_score(y_val, y_pred)
            cv_scores.append(score)

            if model_name == "Random Forest" and not do_pca:
                trained_model = classifier.get_trained_model() 
                feature_importances = trained_model.feature_importances_  # Get feature importances
                df_feature_importance = pd.DataFrame([feature_importances], columns=kept_columns)
                df_all_feature_importance = pd.concat([df_all_feature_importance, df_feature_importance], axis=0, ignore_index=True)

            # if model_name == "SVM" and not do_pca:
            #     trained_model = classifier.get_trained_model() 
            #     feature_importances = trained_model.coef_.flatten()  # Get the coefficients (feature importance)
            #     # Create a DataFrame for feature importance
            #     df_feature_importance = pd.DataFrame([feature_importances], columns=kept_columns)
            #     df_all_feature_importance = pd.concat([df_all_feature_importance, df_feature_importance], axis=0, ignore_index=True)
            

            if verbose:
                print(f"The {iteration_indx}-th run fold score: {score}")
                iteration_indx = iteration_indx+1

            # Store results for evaluation after all folds
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            if groups is not None:
                all_groups.extend(groups_val)

            if len(np.unique(y_val)) == 2:
                lb = LabelBinarizer()
                y_test_binary = lb.fit_transform(y_val).ravel()  # Binarize the labels (0 and 1)
                # Calculate the predicted probabilities and AUC score
                y_prob = classifier.predict_proba()[:, 1]
                all_y_test_binary.extend(y_test_binary)
                all_y_prob.extend(y_prob)

        # import shap
        # trained_model = classifier.get_trained_model() 
        # explainer = shap.KernelExplainer(trained_model.predict_proba, X_train)
        # shap_values = explainer.shap_values(X_val)
        # aggregated_shap_values = shap_values.mean(axis=2) 
        # shap.summary_plot(aggregated_shap_values, X_val, feature_names=kept_columns)


        if verbose:
            print(f"Cross-Validation Scores: {cv_scores}")
            print(f"Mean Accuracy: {np.mean(cv_scores)}")


        # Overall metrics
        accuracy = accuracy_score(all_y_true, all_y_pred)
        recall = recall_score(all_y_true, all_y_pred, average='weighted')
        f1 = f1_score(all_y_true, all_y_pred, average='weighted')

        # if len(ticklabel)>0:
        #     cm = confusion_matrix(all_y_true, all_y_pred,  labels=ticklabel)
        # else:
        #     cm = confusion_matrix(all_y_true, all_y_pred)
        # # Reorder the confusion matrix columns and rows based on ticklabels
        # label_to_index = {label: i for i, label in enumerate(ticklabel)}
        # indices = [label_to_index[label] for label in ticklabel]
        # cm = cm[indices, :][:, indices]



        # Print metrics
        print(f"Total Accuracy: {accuracy}")
        print(f"Total Recall: {recall}")
        print(f"Total F1 Score: {f1}")

        # Call the function with an option for percentage
        if visualization:
            title = f"{task_name} (acc:{accuracy:.2f}, recall:{recall:.2f}, f1:{f1:.2f})"
            if chart_helper:
                if len(ticklabel)>0:
                    cm = confusion_matrix(all_y_true, all_y_pred,  labels=ticklabel)
                    chart_helper.plot_confusion_matrix(cm, ticklabels=ticklabel, show_percentage=show_percentage, task_name=task_name, title = title)
                else:
                    cm = confusion_matrix(all_y_true, all_y_pred)
                    chart_helper.plot_confusion_matrix(cm, ticklabels=np.unique(all_y_true), show_percentage=show_percentage, task_name=task_name, title = title)

            
        # Step 6: If binary classification, calculate ROC-AUC score
        if visualization:
            if len(np.unique(y_val)) == 2:
                # lb = LabelBinarizer()
                # y_test_binary = lb.fit_transform(y_val).ravel()  # Binarize the labels (0 and 1)
                # # Calculate the predicted probabilities and AUC score
                # y_prob = classifier.predict_proba()[:, 1]
                auc = roc_auc_score(all_y_test_binary, all_y_prob)
                print(f"ROC-AUC for {task_name}: {auc:.4f}")

                # Plot ROC Curve
                fpr, tpr, thresholds = roc_curve(all_y_test_binary, all_y_prob)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for {task_name}')
                plt.legend(loc="lower right")
                plt.show()


        # feature_importance = pd.DataFrame()
        if model_name == "Random Forest" and not do_pca:
            # Fill NaN values with 0 (if missing features are assumed unimportant)
            df_all_feature_importance = df_all_feature_importance.fillna(0)

            # Calculate the mean value of each feature across iterations
            mean_feature_importances = df_all_feature_importance.mean()

            # Select the top n features
            
            top_features = mean_feature_importances.nlargest(n_top_features)

            top_features = top_features.sort_values()

            # Print the column names as a list
            print(f"Top {n_top_features} Features by Mean Importance:{top_features.index.tolist()}")

            if visualization:
                # Plot the top n features as a horizontal bar chart
                plt.figure(figsize=(10, 6))
                top_features.plot(kind='barh', color='skyblue')  # Sort for better visualization
                plt.xlabel("Mean Feature Importance")
                plt.title(f"Top {n_top_features} Features by Mean Importance")
                plt.tight_layout()
                plt.show()


        #     # Combine feature importances across all splits
        #     mean_feature_importances = np.mean(all_feature_importances, axis=0)

        #     # Sort features by importance and select the top 10
        #     feature_names = np.array(kept_columns)  # Assuming `kept_columns` is the list of feature names
        #     sorted_idx = np.argsort(mean_feature_importances)[::-1]  # Indices of features sorted by importance (descending)
        #     top_10_idx = sorted_idx[:10]  # Get indices of the top 10 features

        #     # Extract top 10 features and their importance
        #     top_10_features = feature_names[top_10_idx]
        #     top_10_importances = mean_feature_importances[top_10_idx]

        #     # Plot the top 10 feature importance chart
        #     plt.figure(figsize=(10, 6))
        #     plt.barh(top_10_features, top_10_importances, align='center')
        #     plt.xlabel("Mean Feature Importance")
        #     plt.ylabel("Features")
        #     plt.title("Top 10 Feature Importance across all Splits")
        #     plt.gca().invert_yaxis()  # Invert y-axis for better readability (highest importance on top)
        #     plt.tight_layout()
        #     plt.show()

        #     feature_importance = pd.DataFrame([mean_feature_importances], columns=feature_names)

        feature_importance = []
        return accuracy, f1, feature_importance, all_y_pred, all_y_true, all_groups

    def run_emotion_classification_tasks_Test(self, df_merged, 
                                         correlation_threshold = 0.9, 
                                         remove_redundant_features = True,
                                         column_to_split = "",
                                         cv = 3,
                                         random_state = 1,
                                         model_type = "Random Forest", # "Random Forest"
                                         grid_search_parameters = {},
                                         model_parameters = {},
                                         n_top_features= 5,
                                         outlier_strategies=[]):

        # Step 2: Define feature columns and target label
        # We assume that 'Group' is the label column and all other columns from the third onward are feature columns.
        x_columns = df_merged.columns[2:]  # Columns from the 3rd onward are features
        y_column = "Group"  # Target label column

        # Step 3: Define parameter grid for Random Forest model
        # The parameter grid is used to test different configurations for optimal performance.
        if grid_search_parameters == {}:
            if model_type =="XGBoost":
                grid_search_parameters = {
                    # Number of trees (boosting rounds)
                    'n_estimators': [50, 100, 200],  # Common choices for boosting rounds

                    # Maximum depth of each tree, controlling complexity
                    'max_depth': [3, 6, 10],  # Typical values are 3, 6, 10

                    # Learning rate to control the contribution of each tree
                    'learning_rate': [0.01, 0.1, 0.2],  # Common values to try, slower rates can improve accuracy

                    # Fraction of columns (features) to be used by each tree
                    'colsample_bytree': [0.6, 0.8, 1.0],  # Reduces overfitting by using a subset of features

                    # Fraction of rows to be used by each tree
                    'subsample': [0.8, 1.0],  # Usually set to values between 0.5 and 1.0 to prevent overfitting

                    # Minimum loss reduction required to make a further partition
                    'gamma': [0, 0.1, 0.2],  # Controls tree splitting; use 0 or small values initially

                    # Minimum child weight, a regularization to control tree depth
                    'min_child_weight': [1, 5, 10],  # Higher values prevent deeper splits, reducing overfitting

                    # L2 regularization term on weights (ridge regression)
                    'lambda': [1, 10],  # Helps reduce overfitting

                    # Seed for reproducibility
                    'random_state': [1],  # Standard value for reproducibility
                }

            else:
                grid_search_parameters = {
                    'bootstrap': [True],                # Whether to use bootstrap samples
                    'ccp_alpha': [0.0],                 # Complexity parameter used for minimal cost-complexity pruning
                    'class_weight': ['balanced'],       # Adjust weights to handle class imbalance
                    'criterion': ['gini'],              # Splitting criterion
                    'max_depth': [None],        # Limits tree depth to avoid overfitting
                    'max_features': ['sqrt'],           # Number of features considered at each split
                    'max_leaf_nodes': [None],           # Maximum leaf nodes
                    'max_samples': [None],              # If bootstrap is true, it limits samples
                    'min_impurity_decrease': [0.0],     # Threshold for node impurity
                    'min_samples_leaf': [1],         # Minimum samples at each leaf node
                    'min_samples_split': [2],        # Minimum samples required to split a node
                    'min_weight_fraction_leaf': [0.0],  # Minimum fraction of samples at each leaf node
                    'n_estimators': [100],    # Number of trees in the forest
                    'n_jobs': [None],                   # Number of jobs to run in parallel
                    'oob_score': [False],               # Use out-of-bag samples for validation
                    'verbose': [0],                     # Controls verbosity of output
                    'warm_start': [False],              # Reuse previous solution for additional trees
                    'random_state': [random_state]      # Seed for reproducibility
                }
                # Task 3: 2x2 Analysis with Original Group Labels
        machine_learning = MachineLearning(df_merged.copy())
        # group_labels_task3 = {
        #     1: "ELVT",
        #     2: "CMPS",
        #     3: "ADMR",
        #     4: "CTRL"
        # }
        group_labels_task3 = {
            1: 0,
            2: 1,
            3: 2,
            4: 3
        }
        accuracy, f1_scores, feature_importance = machine_learning.run_machine_learning(
            group_labels_task3,
            "Multi-Class Classification",
            x_columns=x_columns,
            y_column=y_column,
            cv = cv,
            remove_redundant_features=remove_redundant_features,
            grid_search_parameters=grid_search_parameters,
            correlation_threshold = correlation_threshold,
            column_to_split = column_to_split,
            model_type = model_type,
            use_pca= False,
            n_top_features = n_top_features,
            outlier_strategies=outlier_strategies
        )
        return accuracy, f1_scores, feature_importance 

        # Step 4: Define group relabeling and machine learning tasks
        # In each task, the label column 'Group' is relabeled to create different analysis perspectives.

        # Task 1: Compassionate vs Non-Compassionate
        # machine_learning = MachineLearning(df_merged.copy())
        # group_labels_task1 = {1: "compassion", 2: "compassion", 3: "non-compassion", 4: "non-compassion"}
        # machine_learning.run_machine_learning(
        #     group_labels_task1,
        #     "compassion vs Non-compassion",
        #     x_columns=x_columns,
        #     y_column=y_column,
        #     remove_redundant_features=remove_redundant_features,
        #     parameters=parameters,
        #     correlation_threshold = correlation_threshold,
        #     column_to_split = column_to_split
        # )

    def extend_list(self, input_list, requested_length):
        # If the input list is shorter than the requested length, add empty strings
        if len(input_list) < requested_length:
            input_list.extend([""] * (requested_length - len(input_list)))
        return input_list
    
    def run_emotion_classification_datasets_test(self, df_list, n_top_features=5, correlation_threshold = 0.7):
        df_evaluations = pd.DataFrame()
        df_features = pd.DataFrame()
        
        for df in df_list:
            # Run emotion classification task and get results
            accuracy_1, f1_scores_dict_1, feature_importance_1 = self.run_emotion_classification_tasks_Test(
                df, n_top_features=n_top_features, correlation_threshold = correlation_threshold)
            
            # Prepare evaluation result (accuracy and f1_scores_dict)
            result_evaluation = [
                f"{accuracy_1:.2f}",
                *[f"{score:.2f}" for score in f1_scores_dict_1.values()],
            ]
            
            # Convert the evaluation result to a DataFrame row and append it to df_evaluations
            df_evaluation_row = pd.DataFrame([result_evaluation])  # Make it a 2D array for rows
            df_evaluations = pd.concat([df_evaluations, df_evaluation_row], ignore_index=True)
            
            # Reverse feature importance and extend to match n_top_features
            feature_importance_1.reverse()
            result_feature = self.extend_list(feature_importance_1, n_top_features)
            
            # Convert feature importance to a DataFrame row and append it to df_features
            df_feature_row = pd.DataFrame([result_feature])  # Make it a row (2D array)
            df_features = pd.concat([df_features, df_feature_row], ignore_index=True)
        
        return df_evaluations, df_features


    def run_emotion_classification_datasets(self, df_list, 
                                            correlation_threshold = 0.7, 
                                            n_top_features = 5,
                                            k =1.5,
                                            random_state = 1,
                                            cv = 10,
                                            outlier_strategies = ['Replace with Median'],
                                            column_to_split = "",
                                            use_pca = False):
        df_evaluations = pd.DataFrame()
        df_features = pd.DataFrame()
        for df in df_list:
            if (len(df.columns)<20):
                correlation_threshold = 1

            accuracy_1, f1_scores_dict_1, feature_importance_1, accuracy_2, f1_scores_dict_2, feature_importance_2, accuracy_3, f1_scores_dict_3, feature_importance_3 = self.run_emotion_classification_tasks(
                df, 
                n_top_features= n_top_features,
                correlation_threshold = correlation_threshold,
                outlier_strategies = outlier_strategies,
                random_state = random_state,
                cv = cv,
                k = k,
                column_to_split= column_to_split,
                use_pca = use_pca)
            
            result_evaluation = [
                f"{accuracy_1:.2f}",
                *[f"{score:.2f}" for score in f1_scores_dict_1.values()],
                f"{accuracy_2:.2f}",
                *[f"{score:.2f}" for score in f1_scores_dict_2.values()],
                f"{accuracy_3:.2f}",
                *[f"{score:.2f}" for score in f1_scores_dict_3.values()],
            ]
            df_evaluation_row = pd.DataFrame([result_evaluation])
            df_evaluations = pd.concat([df_evaluations, df_evaluation_row], ignore_index=True)
            feature_importance_1.reverse()
            feature_importance_2.reverse()
            feature_importance_3.reverse()
            result_feature =  self.extend_list(feature_importance_1,n_top_features) + self.extend_list(feature_importance_2,n_top_features) + self.extend_list(feature_importance_3,n_top_features)
            df_feature_row = pd.DataFrame([result_feature])  # Make it a row (2D array)
            df_features = pd.concat([df_features, df_feature_row], ignore_index=True)
        return df_evaluations, df_features


    def run_emotion_classification_tasks(self, 
                                         df_prepared, parameters = {}  
                                        ):
        
        model_name = "Random Forest"
        random_state = 42
        n_cross_validation = 5
        remove_high_correlation = False
        corr_threshold = 0.95
        do_anova = True
        percentage_kept_by_anova = 0.7
        do_rfe = True
        rfe_model = None
        n_kept = 6
        do_pca = False
        explained_variance_threshold = 0.7
        verbose = True
        visualization = True
        do_model_grid_search = False
        model_grid_search_parameters = {}
        run_1 = False
        run_2 = False
        run_3 = False
        model_initial_parameters = {}

        all_y_true_list = []
        all_y_pred_list = []
        all_group_list = []

        if parameters.get("model_name") is not None:
            model_name = parameters.get("model_name")
        if parameters.get("random_state") is not None:
            random_state = parameters.get("random_state")
        if parameters.get("group_column") is not None:
            group_column = parameters.get("group_column")
        if parameters.get("n_cross_validation") is not None:
            n_cross_validation = parameters.get("n_cross_validation")
        if parameters.get("remove_high_correlation") is not None:
            remove_high_correlation = parameters.get("remove_high_correlation")
        if parameters.get("corr_threshold") is not None:
            corr_threshold = parameters.get("corr_threshold")
        if parameters.get("do_anova") is not None:
            do_anova = parameters.get("do_anova")
        if parameters.get("percentage_kept_by_anova") is not None:
            percentage_kept_by_anova = parameters.get("percentage_kept_by_anova")
        if parameters.get("do_rfe") is not None:
            do_rfe = parameters.get("do_rfe")
        if parameters.get("rfe_model") is not None:
            rfe_model = parameters.get("rfe_model")
        if parameters.get("n_kept") is not None:
            n_kept = parameters.get("n_kept")
        if parameters.get("do_pca") is not None:
            do_pca = parameters.get("do_pca")
        if parameters.get("explained_variance_threshold") is not None:
            explained_variance_threshold = parameters.get("explained_variance_threshold")
        if parameters.get("verbose") is not None:
            verbose = parameters.get("verbose")
        if parameters.get("visualization") is not None:
            visualization = parameters.get("visualization")
        if parameters.get("do_model_grid_search") is not None:
            do_model_grid_search = parameters.get("do_model_grid_search")
        if parameters.get("model_grid_search_parameters") is not None:
            model_grid_search_parameters = parameters.get("model_grid_search_parameters")
        if parameters.get("run_1") is not None:
            run_1 = parameters.get("run_1")
        if parameters.get("run_2") is not None:
            run_2 = parameters.get("run_2")
        if parameters.get("run_3") is not None:
            run_3 = parameters.get("run_3")
        if parameters.get("model_initial_parameters") is not None:
            model_initial_parameters = parameters.get("model_initial_parameters")

        # outlier_multiplier = 3
        # random_state = 42
        # n_cross_validation = 5
        # remove_high_correlation = False
        # corr_threshold = 0.95
        # do_anova = True
        # percentage_kept_by_anova = 0.7
        # do_RFE = True
        # n_kept = 6
        # do_pca = False
        # explained_variance_threshold = 0.7
        # verbose = True
        # do_model_grid_search = False

        # model_grid_search_parameters = {
        #     'n_estimators': [100, 200, 500],
        #     'max_depth': [10, 20, None],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'max_features': ['sqrt', 'log2', None],
        #     'bootstrap': [True, False],
        # }
        # model_grid_search_parameters = {
        #     'n_estimators': [200],
        #     'max_depth': [None],
        #     'min_samples_split': [2],
        #     'min_samples_leaf': [1],
        # }
        chart_helper = psychai.data_visualization.chart.Chart()
        accuracy_1 = 0
        accuracy_2 = 0
        accuracy_3 = 0
        if run_1:
            group_labels_task1 = {1: "CMPS", 2: "CMPS", 3: "N-CMPS", 4: "N-CMPS"}
            accuracy_1, f1_scores_dict_1, feature_importance_1, all_y_pred, all_y_true, all_groups =  self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = group_column,
                group_labels = group_labels_task1,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_rfe,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                chart_helper= chart_helper,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model,
                visualization = visualization,
                task_name="Task1:Compassion"
            )
            all_y_true_list.extend(all_y_true)
            all_y_pred_list.extend(all_y_pred)
            all_group_list.extend(all_groups)
        if run_2:
            group_labels_task2 = {1: "INSP", 3: "INSP", 2: "U-INSP", 4: "U-INSP"}
            accuracy_2, f1_scores_dict_1, feature_importance_1, all_y_pred, all_y_true, all_groups = self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = group_column,
                group_labels = group_labels_task2,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_rfe,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                chart_helper= chart_helper,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model,
                visualization = visualization,
                task_name="Task2:Inspiration"
            )
            all_y_true_list.extend(all_y_true)
            all_y_pred_list.extend(all_y_pred)
            all_group_list.extend(all_groups)
        if run_3:
            
            group_labels_task3 = {1: "ELVT",2: "SMPS",3: "ADMR",4: "CTRL"}
            ticklabel= ["ELVT","SMPS","ADMR","CTRL"]

            if model_name in ["XGBoost"]:
                group_labels_task3 = {1:0,2:1,3:2,4:3}
                ticklabel = [0,1,2,3]
            accuracy_3, f1_scores_dict_1, feature_importance_1, all_y_pred, all_y_true, all_groups = self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = group_column,
                group_labels = group_labels_task3,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_rfe,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                chart_helper= chart_helper,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                ticklabel= ticklabel,
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model,
                visualization = visualization,
                task_name="Task3:Moral Elevation"
            )
            all_y_true_list.extend(all_y_true)
            all_y_pred_list.extend(all_y_pred)
            all_group_list.extend(all_groups)
            return accuracy_1, accuracy_2, accuracy_3, all_y_pred_list,  all_y_true_list, all_group_list
    def run_emotion_classification_tasks_2(self, 
                                            df_prepared,
                                            model_name = "Random Forest", 
                                            random_state = 42,
                                            n_cross_validation = 5,
                                            remove_high_correlation = False,
                                            corr_threshold = 0.95,
                                            do_anova = True,
                                            percentage_kept_by_anova = 0.7,
                                            do_RFE = True,
                                            rfe_model = None,
                                            n_kept = 6,
                                            do_pca = False,
                                            explained_variance_threshold = 0.7,
                                            verbose = True,
                                            do_model_grid_search = False, 
                                            model_grid_search_parameters = {},
                                            run_1 = True,
                                            run_2 = True,
                                            run_3 = True,
                                            model_initial_parameters = {}
                                        ):

        # outlier_multiplier = 3
        # random_state = 42
        # n_cross_validation = 5
        # remove_high_correlation = False
        # corr_threshold = 0.95
        # do_anova = True
        # percentage_kept_by_anova = 0.7
        # do_RFE = True
        # n_kept = 6
        # do_pca = False
        # explained_variance_threshold = 0.7
        # verbose = True
        # do_model_grid_search = False

        # model_grid_search_parameters = {
        #     'n_estimators': [100, 200, 500],
        #     'max_depth': [10, 20, None],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'max_features': ['sqrt', 'log2', None],
        #     'bootstrap': [True, False],
        # }
        # model_grid_search_parameters = {
        #     'n_estimators': [200],
        #     'max_depth': [None],
        #     'min_samples_split': [2],
        #     'min_samples_leaf': [1],
        # }
        chart_helper = psychai.data_visualization.chart.Chart()

        if run_1:
            group_labels_task1 = {1: "compassion", 2: "compassion", 3: "non-compassion", 4: "non-compassion"}
            accuracy_1, f1_scores_dict_1, feature_importance_1 =  self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = "user_id",
                group_labels = group_labels_task1,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_RFE,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                chart_helper= chart_helper,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model
            )
        if run_2:
            group_labels_task2 = {1: "inspired", 3: "inspired", 2: "uninspired", 4: "uninspired"}
            accuracy_1, f1_scores_dict_1, feature_importance_1 = self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = "user_id",
                group_labels = group_labels_task2,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_RFE,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                chart_helper= chart_helper,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model
            )
        if run_3:
            group_labels_task3 = {1: "ELVT",2: "CMPS",3: "ADMR",4: "CTRL"}
            accuracy_1, f1_scores_dict_1, feature_importance_1 = self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = "user_id",
                group_labels = group_labels_task3,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_RFE,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                chart_helper= chart_helper,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                ticklabel= ["ELVT","CMPS","ADMR","CTRL"],
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model
            )


    # def run_emotion_classification_tasks(self, df_merged, 
    #                                      correlation_threshold = 0.7, 
    #                                      remove_redundant_features = True,
    #                                      column_to_split = "",
    #                                      random_state = 1,
    #                                      model_type = "Random Forest",
    #                                      cv = 10,
    #                                      grid_search_parameters = {},
    #                                      model_parameters = {},
    #                                      n_top_features = 5,
    #                                      k = 1.5,
    #                                      use_pca = False,
    #                                      outlier_strategies = ['Replace with Median']):
        
    #     do_task_1 = True
    #     do_task_2 = True
    #     do_task_3 = True
    #     accuracy_1 = 0
    #     f1_scores_dict_1 = {}
    #     feature_importance_1 = []
    #     accuracy_2 = 0
    #     f1_scores_dict_2 = {}
    #     feature_importance_2 = []
    #     accuracy_3 = 0
    #     f1_scores_dict_3 = {}
    #     feature_importance_3 = []
        
    #     # Scramble columns
    #     selected_columns = df_merged.columns.copy()
    #     selected_columns = selected_columns[2:]
    #     # Set the random seed for reproducibility
    #     np.random.seed(random_state)
    #     selected_columns = np.random.permutation(selected_columns).tolist()
    #     selected_columns.insert(0, "user_id")
    #     selected_columns.insert(0, "Group")
    #     df_scrambled = df_merged[selected_columns]

    #     # Step 2: Define feature columns and target label
    #     # We assume that 'Group' is the label column and all other columns from the third onward are feature columns.
    #     x_columns = df_scrambled.columns[2:]  # Columns from the 3rd onward are features
    #     y_column = "Group"  # Target label column

    #     # Task 1: Compassionate vs Non-Compassionate
    #     if do_task_1:
    #         machine_learning = MachineLearning(df_scrambled.copy())
    #         group_labels_task1 = {1: "compassion", 2: "compassion", 3: "non-compassion", 4: "non-compassion"}
    #         accuracy_1, f1_scores_dict_1, feature_importance_1 = machine_learning.run_machine_learning(
    #             group_labels_task1,
    #             "compassion vs Non-compassion",
    #             x_columns=x_columns,
    #             y_column=y_column,
    #             remove_redundant_features=remove_redundant_features,
    #             grid_search_parameters=grid_search_parameters,
    #             cv = cv,
    #             correlation_threshold = correlation_threshold,
    #             column_to_split = column_to_split,
    #             model_type = model_type,
    #             n_top_features = n_top_features,
    #             outlier_strategies = outlier_strategies,
    #             random_state= random_state,
    #             use_pca= use_pca,
    #             k = k,
    #             model_parameters = model_parameters
    #         )
        

    #     # Task 2: Inspired vs Uninspired
    #     if do_task_2:
    #         machine_learning = MachineLearning(df_scrambled.copy())
    #         group_labels_task2 = {1: "inspired", 3: "inspired", 2: "uninspired", 4: "uninspired"}
    #         accuracy_2, f1_scores_dict_2, feature_importance_2 = machine_learning.run_machine_learning(
    #             group_labels_task2,
    #             "Inspired vs Uninspired",
    #             x_columns=x_columns,
    #             y_column=y_column,
    #             remove_redundant_features=remove_redundant_features,
    #             grid_search_parameters=grid_search_parameters, 
    #             correlation_threshold = correlation_threshold,
    #             column_to_split = column_to_split,
    #             model_type = model_type,
    #             cv = cv,
    #             n_top_features = n_top_features,
    #             outlier_strategies = outlier_strategies,
    #             random_state= random_state,
    #             k = k
    #         )


    #     # Task 3: 2x2 Analysis with Original Group Labels
    #     if do_task_3:
    #         machine_learning = MachineLearning(df_scrambled.copy())
    #         group_labels_task3 = {
    #             1: "ELVT",
    #             2: "CMPS",
    #             3: "ADMR",
    #             4: "CTRL"
    #         }
    #         accuracy_3, f1_scores_dict_3, feature_importance_3 = machine_learning.run_machine_learning(
    #             group_labels_task3,
    #             "Multi-Class Classification",
    #             x_columns=x_columns,
    #             y_column=y_column,
    #             remove_redundant_features=remove_redundant_features,
    #             grid_search_parameters=grid_search_parameters,
    #             correlation_threshold = correlation_threshold,
    #             column_to_split = column_to_split,
    #             model_type = model_type,
    #             cv = cv,
    #             n_top_features = n_top_features,
    #             outlier_strategies = outlier_strategies,
    #             random_state= random_state,
    #             k = k
    #         )

    #     # Prepare the formatted values as a list (exclude feature_importance_3 for print output)
    #     results = [
    #         f"{accuracy_1:.2f}",
    #         *[f"{score:.2f}" for score in f1_scores_dict_1.values()],
    #         f"{accuracy_2:.2f}",
    #         *[f"{score:.2f}" for score in f1_scores_dict_2.values()],
    #         f"{accuracy_3:.2f}",
    #         *[f"{score:.2f}" for score in f1_scores_dict_3.values()],
    #     ]
        
    #     # Print the formatted results (excluding feature_importance_3)
    #     print(*results)
        
    #     # Return the results list, including feature_importance_3
    #     return accuracy_1, f1_scores_dict_1, feature_importance_1, accuracy_2, f1_scores_dict_2, feature_importance_2, accuracy_3, f1_scores_dict_3, feature_importance_3,

class MachineLearning:
    def __init__(self, df):
        self.df = df.copy()
        pass

    # # Define the outlier detection and replacement function
    # def replace_outliers_with_group_median(self, df, k=1.5):
    #     """
    #     This function replaces outliers in each feature column of a DataFrame with the median of the corresponding group.
    #     Outliers are identified using the IQR method where:
    #         mins = df[column] < Q1 - k * IQR
    #         maxs = df[column] > Q3 + k * IQR
    #     Additionally, NaN values are treated as outliers and replaced by the group's median.
        
    #     Arguments:
    #     df : pandas DataFrame with 'Group', 'user_id', and feature columns
    #     k : multiplier for IQR to define outliers (default is 1.5)
        
    #     Returns:
    #     df : DataFrame with outliers and NaN values replaced by group medians
    #     """
    #     # List of feature columns (excluding Group and user_id)
    #     feature_columns = [col for col in df.columns if col not in ['Group', 'user_id']]
        
    #     # Group the DataFrame by 'Group'
    #     grouped_df = df.groupby('Group')
        
    #     # Iterate over each feature column
    #     for column in feature_columns:
    #         # Iterate over each group
    #         for group, group_df in grouped_df:
    #             # Calculate Q1, Q3, and IQR for the current group and column
    #             Q1 = group_df[column].quantile(0.25)
    #             Q3 = group_df[column].quantile(0.75)
    #             IQR = Q3 - Q1
                
    #             # Calculate the lower and upper bounds for outliers
    #             lower_bound = Q1 - k * IQR
    #             upper_bound = Q3 + k * IQR
                
    #             # Identify the outliers: below lower bound, above upper bound, or NaN
    #             is_outlier = (group_df[column] < lower_bound) | (group_df[column] > upper_bound) | pd.isna(group_df[column])
                
    #             # Get the median of the group for the current feature
    #             group_median = group_df[column].median()
                
    #             # Replace outliers with the median of the group
    #             df.loc[(df['Group'] == group) & is_outlier, column] = group_median

    #     return df

    # Sample usage
    # Assuming `df` is your DataFrame with columns 'Group', 'user_id', and feature columns
    # For example, df = pd.DataFrame(...)

    def prepare_data(self, group_labels, x_columns, y_column, column_to_split= "",
                            explained_variance_threshold=0.95,
                            remove_redundant_features=True, remove_low_variance=False, 
                            correlation_threshold=0.9, 
                            variance_threshold=0.01, use_pca=False,
                            outlier_strategies = ['Replace with Median'],
                            random_state = 1,
                            k=1.5):
        
        # if len(outlier_strategies)>0:
        #     for outlier_strategy in outlier_strategies:
        #         if outlier_strategy == 'Replace with Median':
        #             self.df = self.replace_outliers_with_group_median(self.df , k=1.5)


        # Create a copy of the DataFrame and relabel the 'Group' column according to task specifications
        if group_labels != {}:
            self.df["Group"] = self.df["Group"].map(group_labels)

        data_preprocessor = psychai.data_preparation.preprocessing.DataPreprocessor()
        # Prepare data
        X_train, y_train, X_test, y_test = data_preprocessor.prepare_training_data(self.df, x_columns, y_column,
                        column_to_split=column_to_split, explained_variance_threshold=explained_variance_threshold, 
                        correlation_threshold=correlation_threshold, 
                        remove_redundant_features=remove_redundant_features, remove_low_variance=remove_low_variance, 
                        variance_threshold=variance_threshold, use_pca=use_pca, outlier_strategies= outlier_strategies, k=k, 
                        random_state = random_state)
        return X_train, y_train, X_test, y_test
    
    def run_machine_learning_with_splitted_data(self, task_name, x_columns, y_column, 
                                                X_train, y_train, X_test, y_test, parameters= {}):
        if parameters == {}:
            parameters = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 3, 5, 7],
            }
        
        # Run grid search for best model parameters
        
        rf_classifier = psychai.machine_learning.classification_supervise_learning.Classification(X_train.copy(), 
                        y_train.copy(), X_test.copy(), y_test.copy(), model_name=f"Random Forest - {task_name}", column_name=x_columns)
        grid_search_results = rf_classifier.grid_search_by_name("Random Forest", parameters=parameters, cv=10)
        
        # Display best hyperparameters and CV score
        print(f"Task: {task_name}")
        print("Best Hyperparameters:", grid_search_results.best_params_)
        print("Best CV Score:", grid_search_results.best_score_)
        
        # Evaluate the best model on the test set
        rf_classifier.evaluate_model()
        print("\n")

        rf_classifier.feature_importance(10)


    from sklearn.metrics import accuracy_score, f1_score

    # def run_machine_learning(
    #     self, 
    #     group_labels,               # Dictionary for relabeling classes (e.g., {"1": "positive", "2": "negative"})
    #     task_name,                  # Name of the classification task for display
    #     x_columns,                  # List of feature columns
    #     y_column,                   # Target column to predict
    #     column_to_split="",         # Optional column to split data for training/testing (if needed)
    #     cv = 10,
    #     #explained_variance_threshold=0.95, 
    #     explained_variance_threshold=1, 
    #     correlation_threshold=0.9, 
    #     remove_redundant_features=True, 
    #     remove_low_variance=False, 
    #     variance_threshold=0.01, 
    #     use_pca=False,
    #     model_type = "Random Forest", # "Random Forest"
    #     grid_search_parameters={},               # Parameter grid for hyperparameter tuning
    #     n_top_features = 10,
    #     outlier_strategies = ['Replace with Median'],
    #     k = 1.5,
    #     random_state = 1,
    #     model_parameters = {}
    # ):
    #     """
    #     A function to run a machine learning pipeline, including data preparation, feature selection,
    #     hyperparameter tuning, and evaluation metrics.
    #     Returns accuracy and F1 scores for each class.
    #     """

    #    # Step 1: Prepare Training and Testing Data
    #     # Calls the prepare_data method to preprocess and split the data.
    #     X_train, y_train, X_test, y_test = self.prepare_data(
    #         group_labels,              # Relabel the groups based on `group_labels`
    #         x_columns,                 # Feature columns
    #         y_column,                  # Target column
    #         column_to_split=column_to_split,
    #         correlation_threshold=correlation_threshold,
    #         explained_variance_threshold=explained_variance_threshold, 
    #         remove_redundant_features=remove_redundant_features, 
    #         remove_low_variance=remove_low_variance, 
    #         variance_threshold=variance_threshold, 
    #         use_pca=use_pca,
    #         outlier_strategies = outlier_strategies,
    #         k = k,
    #         random_state = random_state
    #     )

    #     # Step 2: Set Default Parameters if None Provided
    #     # Defines the parameter grid for Random Forest. If `parameters` is empty, a default set is used.
    #     # if not parameters:
    #     #     parameters = {
    #     #         'n_estimators': [50, 100, 200, 300],  # Number of trees
    #     #     }

    #     # Step 3: Initialize Classifier and Perform Grid Search
    #     # Initializes Random Forest classifier and performs grid search on the specified parameters.
    #     classifier = psychai.machine_learning.classification_supervise_learning.Classification(
    #         X_train, 
    #         y_train, 
    #         X_test, 
    #         y_test, 
    #         model_label=f"{model_type}-{task_name}", 
    #         model_name=model_type, 
    #         column_name=x_columns,
    #         random_state = random_state,
    #         model_parameters = model_parameters
    #     )
    #     # Grid search to find best parameters using cross-validation
    #     if grid_search_parameters == {}:
    #         classifier.train_model()

    #     else:
    #         classifier.grid_search(
    #             grid_search_parameters=grid_search_parameters, 
    #             cv=cv  # 10-fold cross-validation
    #         )

    #     # grid_search_results = rf_classifier.grid_search_by_name(
    #     #     "XGBoost", 
    #     #     #parameters=parameters, 
    #     #     cv=2  # 10-fold cross-validation
    #     # )

    #     # Step 4: Display Best Hyperparameters and CV Score
    #     # Shows the best parameters and the highest cross-validation score found during grid search.
    #     print(f"Task: {task_name}")
    #     # print("Best Hyperparameters:", grid_search_results.best_params_)
    #     # print("Best CV Score:", grid_search_results.best_score_)

    #     # Step 5: Ensure Unique Label Order for Evaluation
    #     # Collects unique class labels, ordered based on `group_labels` values for consistent display.
    #     if group_labels!={}:
    #         unique_labels = list(dict.fromkeys(group_labels.values()))
    #     else:
    #         unique_labels =  list(set(y_train))

    #     # Step 6: Get Predictions from the Best Model
    #     # Uses the best model from grid search to make predictions on the test set.
    #     # best_model = grid_search_results.best_estimator_
    #     #y_pred = best_model.predict(X_test)

    #     # # Step 7: Calculate Accuracy
    #     # # Computes overall accuracy as the proportion of correct predictions.
    #     # accuracy = accuracy_score(y_test, y_pred)
    #     # print(f"Prediction Accuracy: {accuracy:.4f}")

    #     # # Step 8: Calculate F1 Scores for Each Class
    #     # # Computes F1 scores for each class (useful for imbalanced datasets).
    #     # # `average=None` returns an array of F1 scores per class.
    #     # f1_scores = f1_score(y_test, y_pred, average=None, labels=unique_labels)
    #     # f1_scores_dict = dict(zip(unique_labels, f1_scores))
        
    #     # # Display F1 score for each class
    #     # for label, f1 in f1_scores_dict.items():
    #     #     print(f"F1 Score for {label}: {f1:.4f}")

    #     # Step 9: Display Confusion Matrix
    #     # Visualizes the confusion matrix to assess the model's classification performance.
    #     accuracy, f1_scores = classifier.evaluate_model(labels_order=unique_labels, font_size=16)

    #     # Step 10: Display Feature Importances
    #     # Displays the top features contributing to the model's decisions.
    #     feature_importance = None
    #     if model_type == "Random Forest":
    #         feature_importance = classifier.feature_importance(n_top_features)

    #     if model_type == "Decision Tree":
    #         from sklearn.tree import plot_tree
    #         import matplotlib.pyplot as plt

    #         model = classifier.get_trained_model()
    #         #plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)'

    #         plt.figure(figsize=(20, 10))  # Set the width and height of the plot
    #         plot_tree(
    #             model, 
    #             feature_names=x_columns, 
    #             filled=True, 
    #             fontsize=12  # Increase font size
    #         )
    #         plt.show()

    #     # Return accuracy and F1 scores if further use is needed
    #     return accuracy, f1_scores, feature_importance

    def run_machine_learning(
        self,
        group_labels,               # Dictionary for relabeling classes (e.g., {"1": "positive", "2": "negative"})
        task_name,                  # Name of the classification task for display
        x_columns,                  # List of feature columns
        y_column,                   # Target column to predict
        column_to_split="",         # Optional column to split data for training/summarizetesting (if needed)
        cv=10,
        explained_variance_threshold=1,
        correlation_threshold=0.9,
        remove_redundant_features=True,
        remove_low_variance=False,
        variance_threshold=0.01,
        use_pca=False,
        model_type="Random Forest",  # "Random Forest" or "Decision Tree"
        grid_search_parameters={},
        n_top_features=10,
        outlier_strategies=['Replace with Median'],
        k=1.5,
        random_state=1,
        model_parameters={}
    ):
        """
        A function to run a machine learning pipeline, including data preparation, feature selection,
        hyperparameter tuning, and evaluation metrics.
        Returns accuracy and F1 scores for each class.
        """

        # Step 1: Prepare Training and Testing Data
        X_train, y_train, X_test, y_test = self.prepare_data(
            group_labels,
            x_columns,
            y_column,
            column_to_split=column_to_split,
            correlation_threshold=correlation_threshold,
            explained_variance_threshold=explained_variance_threshold,
            remove_redundant_features=remove_redundant_features,
            remove_low_variance=remove_low_variance,
            variance_threshold=variance_threshold,
            use_pca=use_pca,
            outlier_strategies=outlier_strategies,
            k=k,
            random_state=random_state
        )

        # Step 2: Initialize Classifier
        classifier = psychai.machine_learning.classification_supervise_learning.Classification(
            X_train,
            y_train,
            X_test,
            y_test,
            model_label=f"{model_type}-{task_name}",
            model_name=model_type,
            column_name=x_columns,
            random_state=random_state,
            model_parameters=model_parameters
        )

        # Step 3: Train or Perform Grid Search
        if not grid_search_parameters:
            classifier.train_model()
        else:
            classifier.grid_search(
                grid_search_parameters=grid_search_parameters,
                cv=cv
            )

        # Step 4: Ensure Consistent Label Order
        unique_labels = list(dict.fromkeys(group_labels.values())) if group_labels else list(set(y_train))

        # Step 5: Evaluate Model
        accuracy, f1_scores = classifier.evaluate_model(labels_order=unique_labels, font_size=16)

        # Step 6: Display Feature Importances or Decision Tree
        feature_importance = None
        if model_type == "Random Forest":
            feature_importance = classifier.feature_importance(n_top_features)

        if model_type == "Decision Tree":
            from sklearn.tree import plot_tree
            import matplotlib.pyplot as plt

            # Get the trained Decision Tree model
            model = classifier.get_trained_model()

            # Plot the decision tree with proper feature names and labels
            plt.figure(figsize=(20, 10))
            plot_tree(
                model,
                feature_names=x_columns,      # Ensure feature names are displayed correctly
                class_names=unique_labels,   # Ensure result labels are displayed
                filled=True,
                fontsize=12
            )
            plt.show()

        # Return results for further use if needed
        return accuracy, f1_scores, feature_importance


class DataHelper:

    def __init__(self):
        pass
   
    def merge_dataframe(self, dataframes):
        # Verify each DataFrame contains 'user_id' and 'Group' columns
        for i, df in enumerate(dataframes):
            if 'user_id' not in df.columns or 'Group' not in df.columns:
                raise KeyError(f"DataFrame at index {i} is missing 'user_id' or 'Group' column.")

        # Concatenate all DataFrames on 'user_id', performing an outer join and excluding 'Group' columns
        merged_data = pd.concat([df.set_index('user_id').drop(columns='Group') for df in dataframes], axis=1, join="outer").reset_index()

        # Combine 'Group' columns by taking the first available 'Group' for each 'user_id'
        group_data = pd.concat([df[['user_id', 'Group']].drop_duplicates(subset='user_id') for df in dataframes])
        group_data = group_data.drop_duplicates(subset='user_id', keep='first')

        # Merge the 'Group' column back into the final DataFrame as the second column
        df_merged = pd.merge(group_data, merged_data, on='user_id', how='outer')

        return df_merged

    # Define a function to clean up the text in each row
    def clean_llm_returned_text(self, text):
        # Check if both markers exist in the text
        start_marker = "<|notimestamps|>"
        end_marker = "<|endoftext|>"
        
        if start_marker in text and end_marker in text:
            # Find the start and end positions of the markers
            start_idx = text.find(start_marker) + len(start_marker)
            end_idx = text.find(end_marker)
            
            # Extract the text between the markers
            return text[start_idx:end_idx].strip()  # .strip() to remove any leading/trailing spaces
        else:
            # If markers are not found, return the original text
            return text

    # Group by user_id
    def calculate_mean_by_user_id(self, df):
        # Step 1: Check consistency in non-numerical columns
        df_consistent = df.groupby('user_id').apply(self.check_non_numerical_consistency).reset_index(drop=True)

        # Step 2: Calculate mean for numerical columns
        df_mean = df.groupby('user_id').mean(numeric_only=True).reset_index()

        # Step 3: Add back consistent non-numerical columns
        # Ensure 'user_id' is present in non_numerical_columns subset
        non_numerical_columns = df_consistent[['user_id'] + df_consistent.select_dtypes(include=['object']).columns.tolist()].drop_duplicates(subset=['user_id'])
        
        df_mean = pd.merge(df_mean, non_numerical_columns, on='user_id', how='left')

        return df_mean

    # Function to check consistency in non-numerical columns
    def check_non_numerical_consistency(self, group):
        for col in group.select_dtypes(include=['object']).columns:
            if group[col].nunique() > 1:
                raise ValueError(f"Inconsistent values found in column '{col}' for user_id '{group.name}'")
        return group.iloc[0]  # Return the first row for consistent non-numerical columns


class FileFilter:
    def __init__(self, base_directory, attribute_filters, modality_value):
        """
        Initializes the FileFilter object with base directory, list of attribute filters, and modality value.

        Args:
            base_directory (str): The base directory that contains the user folders (e.g., 001, 002, ..., 101).
            attribute_filters (list of tuples): List of (attribute_index, attribute_value) pairs to filter by.
            modality_value (str): The value you are looking for in the modality folder (e.g., 'PPG', 'Segment_Audio').
        """
        self.base_directory = base_directory  # Base directory path where the user folders are located
        self.attribute_filters = attribute_filters  # List of (attribute_index, attribute_value) tuples
        self.modality_value = modality_value  # Modality folder name
        self.matching_files = []  # List to store matching file information

    def filter_files_by_attributes(self, file_name):
        """
        Filters files based on multiple attributes in the file name.

        Args:
            file_name (str): The name of the file to check (e.g., PS-9_001_1_10_21_14_30_45.txt).

        Returns:
            bool: True if the file matches all the desired attributes, otherwise False.
        """
        # Split the file name into parts by the underscore character
        parts = file_name.split('_')
        
        # Check if the file name format has enough parts
        for attribute_index, attribute_value in self.attribute_filters:
            if attribute_index < len(parts) and parts[attribute_index] == attribute_value:
                return True
        return False

    def process_user_folder(self, user_id):
        """
        Processes the files in a specific user folder and checks for matching files.

        Args:
            user_id (int): The ID of the user folder to process (e.g., 1 for '001').
        """
        # Format the user folder ID as a three-digit string (e.g., 001, 002, etc.)
        user_folder = f"{user_id:03d}"

        # Construct the full path to the modality folder inside the user's directory
        modality_folder = os.path.join(self.base_directory, user_folder, self.modality_value)

        # Check if the modality folder exists for this user
        if os.path.exists(modality_folder):
            # Iterate through all files in the modality folder
            for file_name in os.listdir(modality_folder):
                # Check if the file matches the desired attributes
                if self.filter_files_by_attributes(file_name):
                    # Split the file name into parts and append a dictionary with all necessary information
                    parts = file_name.split('_')
                    file_path = os.path.join(modality_folder, file_name)
                    file_info = {
                        'user_id': user_id,
                        'modality': self.modality_value,
                        'attribute_index': 2, #[index for index, _ in self.attribute_filters],
                        'attribute_value': parts[2], #[value for _, value in self.attribute_filters],
                        'files': file_path,
                        'file_size': os.path.getsize(file_path)
                    }
                    # Add parts as separate columns
                    for i, part in enumerate(parts):
                        file_info[f'part_{i}'] = part
                    
                    self.matching_files.append(file_info)
        else:
            # Print a message if the modality folder is not found
            print(f"{self.modality_value} folder not found for user {user_folder}")

        pass

    def find_matching_files(self, user_range):
        """
        Loops through the user folders to find files that match the specified attributes.

        Args:
            user_range (range): The range of user IDs to check (e.g., range(1, 14) for users 001 to 013).
        """
        # Iterate through each user folder in the specified range
        for user_id in user_range:
            # Process the files for each user folder
            self.process_user_folder(user_id)

        # Print out the total number of matching files found
        print(f"Total matching files found: {len(self.matching_files)}")

    def get_matching_files(self, user_range):
        """
        Finds the matching files and returns them as a DataFrame, including parts of the file name.

        Args:
            user_range (range): The range of user IDs to check (e.g., range(1, 14) for users 001 to 013).

        Returns:
            pd.DataFrame: A DataFrame containing 'user_id', 'attribute_index', 'attribute_value', 'files',
                          and individual parts of the file name.
        """
        # Clear previous matching files
        self.matching_files = []

        # Find matching files in the specified user range
        self.find_matching_files(user_range)

        # Convert matching files to a pandas DataFrame
        df = pd.DataFrame(self.matching_files)

        return df




class IO:
    def __init__(self):
        pass

    def cache_object(self, obj_name, folder, create_function, override=False):
        """
        Cache the object using pickle.
        
        Parameters:
        - obj_name (str): Name of the object to be cached, used as the filename.
        - create_function (function): A function that generates the object.
        - override (bool): If True, always recreate the object and overwrite the cache.
        
        Returns:
        - The object, either from cache or newly created.
        """
        # Create the filename based on the object's name
        filename = f"{folder}\\{obj_name}.pkl"
        
        # Check if the file exists and whether override is False
        if os.path.exists(filename) and not override:
            # Load and return the object from cache
            with open(filename, 'rb') as f:
                print(f"Loading {obj_name} from cache.")
                return pickle.load(f)
        else:
            # Create the object using the provided function
            print(f"Creating {obj_name} using the provided function.")
            obj = create_function()
            
            # Save the object to the cache file
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)
                print(f"{obj_name} saved to cache.")
            
            return obj