from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# import xgboost as xgb
from sklearn.model_selection import GridSearchCV
# Importing libraries for evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
from matplotlib import font_manager
import psychai.data_visualization.chart

class Classification:

    random_forest = "Random Forest"
    

    def __init__(self, X_train, y_train, X_test, y_test, column_name=[], model_name="Model"):
        """
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.trained_model = None
        self.column_name = column_name
        self.chart_helper = psychai.data_visualization.chart.Chart()

    def grid_search_by_name(self, model_name, parameters=None, cv=10):

        if model_name == "Random Forest":
            # 1. Random Forest:
            # Random Forest is an ensemble model that builds multiple decision trees during training.
            # It improves accuracy by reducing overfitting through aggregation of many trees.
            # Random Forest is robust to noise and works well with both classification and regression tasks.
            #model = RandomForestClassifier(random_state=random_state)
            model = RandomForestClassifier()

            # Hyperparameters for Random Forest
            # 'n_estimators' defines the number of trees in the forest.
            # 'max_depth' controls the depth of each tree, preventing overfitting.
            # 'min_samples_split' defines the minimum number of samples required to split an internal node.
            if parameters==None:
                parameters = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                }
        
        elif model_name == "Logistic Regression":

            # 2. Logistic Regression:
            # Logistic Regression is a simple, linear model that works well for binary and multiclass classification.
            # It is easy to interpret and often used as a baseline model.
            model = LogisticRegression(random_state=42, max_iter=1000)

            # Hyperparameters for Logistic Regression
            # 'C' is the inverse of regularization strength. Smaller values specify stronger regularization.
            # 'penalty' defines the type of regularization applied (L1, L2, or none).
            if parameters==None:
                parameters = {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],  # L2 is the most common penalty used
                }

            # 3. XGBoost:
            # XGBoost is a gradient boosting algorithm known for its high accuracy and speed.
            # It works by iteratively adding models (weak learners) that minimize prediction errors of previous models.
            # XGBoost is highly effective in handling unbalanced data and complex relationships.
            # xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

            # Hyperparameters for XGBoost
            # 'n_estimators' defines the number of boosting rounds.
            # 'learning_rate' controls the contribution of each tree, a smaller learning rate may improve performance but requires more trees.
            # 'max_depth' controls the depth of each tree.
            # xgb_params = {
            #     'n_estimators': [100, 200],
            #     'learning_rate': [0.01, 0.1],
            #     'max_depth': [3, 6],
            # }
        if model != None:
            return self.grid_search_by_model(model, parameters, cv)
        return None

    def grid_search_by_model(self, model, params, cv=10):
        """
        Function to perform GridSearchCV on the given model with the provided hyperparameters.
        - model: the machine learning model to be tuned
        - params: dictionary of hyperparameters for the model
        - X_train: features of the training set
        - y_train: target labels of the training set
        Returns: Best model found by GridSearchCV
        """
        self.trained_model = model
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv, n_jobs=-1, verbose=1)

        # Fit the model on the training data
        grid_search.fit(self.X_train, self.y_train)

        self.trained_model = grid_search.best_estimator_

        # Return the best model with the best hyperparameters
        return grid_search
    
   
    
    def evaluate_model(self, labels_order=[], font_size = 14, cmap_color = "Blues"):
        """
        Function to evaluate the performance of the model on the test set.
        - model: The trained model to be evaluated
        - X_test: Test features
        - y_test: True labels of the test set
        - model_name: Name of the model (for display purposes)
        """

        if not self.trained_model:
            return
        
        # Use the model to predict on the test data
        y_pred = self.trained_model.predict(self.X_test)
        
        # Print the classification report, which includes precision, recall, F1-score, and accuracy
        print(f"\nClassification Report for {self.model_name}:\n")
        print(classification_report(self.y_test, y_pred))
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"{self.model_name} Accuracy: {accuracy:.4f}")
        
        # Confusion Matrix

        # Confusion Matrix with Custom Font Sizes and Colors
        print(f"\nConfusion Matrix for {self.model_name}:\n")

        # Generate confusion matrix and plot
        if len(labels_order) > 0:

            cm = confusion_matrix(self.y_test, y_pred, labels=labels_order)
            self.chart_helper.draw_heatmap_with_matrix(cm, title = f"Confusion Matrix for {self.model_name}",
                                   x_label= "Predicted Label", y_label = "True Label", ticklabels = labels_order,
                                   title_font_size = font_size, annotation_font_size = font_size, label_font_size = font_size, ticker_font_size = font_size, cmap_color = cmap_color)
        else:
            cm = confusion_matrix(self.y_test, y_pred)
            self.chart_helper.draw_heatmap_with_matrix(cm, title = f"Confusion Matrix for {self.model_name}",
                            x_label= "Predicted Label", y_label = "True Label",  ticklabels=np.unique(self.y_test),
                            title_font_size = font_size, annotation_font_size = font_size, label_font_size = font_size, ticker_font_size = font_size, cmap_color = cmap_color)

        # # Generate confusion matrix and plot
        # if len(labels_order) > 0:
        #     cm = confusion_matrix(self.y_test, y_pred, labels=labels_order)
        #     sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_color, 
        #                 xticklabels=labels_order, yticklabels=labels_order, 
        #                 annot_kws={"size": font_size})  # Set annotation font size
        # else:
        #     cm = confusion_matrix(self.y_test, y_pred)
        #     sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_color, 
        #                 xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test),
        #                 annot_kws={"size": font_size})  # Set annotation font size

        # # Customize font sizes and add labels
        # plt.title(f"Confusion Matrix for {self.model_name}", fontsize=title_font_size)
        # plt.xlabel("Predicted Label", fontsize=font_size)
        # plt.ylabel("True Label", fontsize=font_size)
        # plt.xticks(fontsize=font_size)
        # plt.yticks(fontsize=font_size)
        # plt.show()

        # print(f"\nConfusion Matrix for {self.model_name}:\n")
        # if len(labels_order)>0:
        #     cm = confusion_matrix(self.y_test, y_pred, labels=labels_order)
        #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order)
        # else:
        #     cm = confusion_matrix(self.y_test, y_pred)
        #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test))
        
        # plt.title(f"Confusion Matrix for {self.model_name}")
        # plt.xlabel("Predicted Label")
        # plt.ylabel("True Label")
        # plt.show()
        
        # Check if the problem is binary classification (for ROC-AUC score)
        if len(np.unique(self.y_test)) == 2:

            # Convert y_test to binary values
            lb = LabelBinarizer()
            y_test_binary = lb.fit_transform(self.y_test).ravel()  # This will give binary labels (0 and 1)

            # Calculate the probabilities and AUC
            y_prob = self.trained_model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(y_test_binary, y_prob)
            print(f"ROC-AUC for {self.model_name}: {auc:.4f}")

            # Plot the ROC curve
            fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {self.model_name}')
            plt.legend(loc="lower right")
            plt.show()




    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.pyplot as plt
    import numpy as np

    def feature_importance(self, top_number=10):
        """
        Display feature importance for the Random Forest model.
        - top_number: Number of top features to display (if > 0).
        """
        # Ensure the model has been trained before proceeding
        if not self.trained_model:
            return
        
        feature_importances = self.trained_model.feature_importances_
        sorted_idx = np.argsort(feature_importances)

        # Select the top features if top_number is specified
        if top_number > 0:
            top_number = min(top_number, len(sorted_idx))  # Limit top_number to the number of features
            sorted_idx = sorted_idx[-top_number:]  # Get indices of top features

        # Set font properties to support Chinese characters and adjust sizes
        plt.rcParams['font.sans-serif'] = ['SimHei']  # For Windows: SimHei (to support Chinese)
        plt.rcParams['axes.unicode_minus'] = False    # Display minus signs correctly
        label_font_size = 16  # Font size for the x and y labels
        title_font_size = 16  # Font size for the plot title
        tick_font_size = 16   # Font size for the tick labels

        # Set up the plot
        plt.figure(figsize=(8, 6))

        # Check if column names are provided; otherwise, generate generic names
        if len(self.column_name) > 0:
            sorted_idx = sorted_idx.astype(int) if not isinstance(sorted_idx, list) else sorted_idx
            feature_labels = [self.column_name[i] for i in sorted_idx]
        else:
            feature_labels = [f"column_name_{i+1}" for i in sorted_idx]

        # Plot feature importances with custom font sizes
        plt.barh(feature_labels, feature_importances[sorted_idx])

        # Set the labels and title with custom font sizes
        plt.xlabel("Feature Importance", fontsize=label_font_size)
        plt.ylabel("Features", fontsize=label_font_size)
        plt.title("Top Feature Importances for Random Forest" if top_number > 0 else "Feature Importance for Random Forest", 
                fontsize=title_font_size)

        # Customize tick label font size
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)

        # Display the plot
        plt.show()


    # def feature_importance(self, top_number=0):
    #     """
    #     Display feature importance for the Random Forest model.
    #     - model: The trained Random Forest model
    #     - top_number: Number of top features to display (if > 0).
    #     """
    #     if not self.trained_model:
    #         return
        
    #     feature_importances = self.trained_model.feature_importances_
    #     sorted_idx = np.argsort(feature_importances)

    #     # Select the top features if top_number is specified

    #     if top_number > 0:
    #         top_number = min(top_number, len(sorted_idx))  # Limit top_number to the number of features
    #         sorted_idx = sorted_idx[-top_number:]  # Get indices of top features


    #     # Set the font to support Chinese characters
    #     plt.rcParams['font.sans-serif'] = ['SimHei']  # For Windows: SimHei
    #     plt.rcParams['axes.unicode_minus'] = False    # Ensure minus signs are displayed correctly

    #     # The rest of your plotting code
    #     plt.figure(figsize=(8, 6))

    #     # Check if column names are provided; otherwise, generate generic names
    #     if len(self.column_name) > 0:
    #         # Convert sorted_idx to an integer array if needed
    #         sorted_idx = sorted_idx.astype(int) if not isinstance(sorted_idx, list) else sorted_idx

    #         # Plot feature importances
    #         plt.barh([self.column_name[i] for i in sorted_idx], feature_importances[sorted_idx])
    #     else:
    #         # Generate generic column names like "column_name_1", "column_name_2", etc.
    #         generated_column_names = [f"column_name_{i+1}" for i in sorted_idx]
    #         plt.barh(generated_column_names, feature_importances[sorted_idx])

    #     plt.xlabel("Feature Importance")
    #     plt.title("Top Feature Importances for Random Forest" if top_number > 0 else "Feature Importance for Random Forest")
    #     plt.show()

        # plt.figure(figsize=(8, 6))

        # # Check if column names are provided; otherwise, generate generic names
        # if len(self.column_name) > 0:
        #     # Convert sorted_idx to an integer array if needed
        #     sorted_idx = sorted_idx.astype(int) if not isinstance(sorted_idx, list) else sorted_idx

        #     # Plot feature importances
        #     plt.barh([self.column_name[i] for i in sorted_idx], feature_importances[sorted_idx])
        #     #plt.barh(self.column_name[sorted_idx], feature_importances[sorted_idx])
        # else:
        #     # Generate generic column names like "column_name_1", "column_name_2", etc.
        #     generated_column_names = [f"column_name_{i+1}" for i in sorted_idx]
        #     plt.barh(generated_column_names, feature_importances[sorted_idx])

        # plt.xlabel("Feature Importance")
        # plt.title("Top Feature Importances for Random Forest" if top_number > 0 else "Feature Importance for Random Forest")
        # plt.show()
