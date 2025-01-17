from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier



class Classification:

    random_forest = "Random Forest"
    

    def __init__(self, X_train, y_train, X_test, y_test, column_name=[], 
                 model_name="Model", model_label = "Model", random_state = 1,
                 model_parameters = {}):
        """
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.model_label = model_label
        self.trained_model = None
        self.column_name = column_name
        self.chart_helper = psychai.data_visualization.chart.Chart()
        self.random_state = random_state
        self.model_parameters = model_parameters


    def get_trained_model(self):
        if self.trained_model != None:
            return self.trained_model
        else:
            return None

    def get_model_by_name(self):

        if self.model_name == "Random Forest":
            # 1. Random Forest:
            # Random Forest is an ensemble model that builds multiple decision trees during training.
            # It improves accuracy by reducing overfitting through aggregation of many trees.
            # Random Forest is robust to noise and works well with both classification and regression tasks.
            #model = RandomForestClassifier(random_state=random_state)
            self.trained_model = RandomForestClassifier(random_state=self.random_state)

        if self.model_name == "Decision Tree":
            max_depth=5
            if 'max_depth' in self.model_parameters:
                max_depth = self.model_parameters.get('max_depth',5)
                print(f"max_depth:{max_depth}")
            self.trained_model  = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=self.random_state)

        elif self.model_name == "Logistic Regression":

            # 2. Logistic Regression:
            # Logistic Regression is a simple, linear model that works well for binary and multiclass classification.
            # It is easy to interpret and often used as a baseline model.
            self.trained_model = LogisticRegression(random_state=self.random_state, max_iter=1000)

        elif self.model_name == "XGBoost":
            # XGBoost is a gradient boosting algorithm known for its high accuracy and speed.
            # It works by iteratively adding models (weak learners) that minimize prediction errors of previous models.
            # XGBoost is highly effective in handling unbalanced data and complex relationships.
            self.trained_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=self.random_state)

        elif self.model_name == "SVM":
            self.trained_model = SVC(kernel='linear', C=1.0, probability=True,  random_state=self.random_state)

        elif self.model_name == "MLP":
            hidden_layer_sizes = (100,)
            max_iter = 500
            alpha=0.001
            if 'hidden_layer_sizes' in self.model_parameters:
                hidden_layer_sizes = self.model_parameters.get('hidden_layer_sizes', (100,))
                print(f"hidden_layer_sizes:{hidden_layer_sizes}")
            if 'max_iter' in self.model_parameters:
                max_iter = self.model_parameters.get('max_iter', 500)
                print(f"max_iter:{max_iter}")
            if 'alpha' in self.model_parameters:
                alpha = self.model_parameters.get('alpha', 0.001)
                print(f"alpha:{alpha}")

            self.trained_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                               activation='relu', solver='adam', random_state=self.random_state, 
                                               alpha=alpha,  # L2 regularization
                                               max_iter=max_iter)


    def train_model(self):
        self.get_model_by_name()

        if self.trained_model != None:
            self.trained_model.fit(X=self.X_train,y=self.y_train)

    def grid_search(self, grid_search_parameters=None, cv=10):

        self.get_model_by_name()

        if self.trained_model != None:
            return self.grid_search_by_model(self.trained_model, grid_search_parameters, cv)
        
        # return None

    # def grid_search_by_model(self, model, params, cv=2):
    #     """
    #     Function to perform GridSearchCV on the given model with the provided hyperparameters.
    #     - model: the machine learning model to be tuned
    #     - params: dictionary of hyperparameters for the model
    #     - X_train: features of the training set
    #     - y_train: target labels of the training set
    #     Returns: Best model found by GridSearchCV
    #     """
    #     Encode the string labels to numerical values
    #     label_encoder = LabelEncoder()
    #     y_encoded = label_encoder.fit_transform(self.y_train)  # Encode labels for training

    #     self.trained_model = model
    #     Initialize GridSearchCV
    #     grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv, n_jobs=-1, verbose=1)

    #     Fit the model on the training data
    #     grid_search.fit(self.X_train, y_encoded)

    #     self.trained_model = grid_search.best_estimator_

    #     Return the best model with the best hyperparameters
    #     return grid_search

    def grid_search_by_model(self, model, grid_search_parameters, cv=10):
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
        grid_search = GridSearchCV(estimator=model, param_grid=grid_search_parameters, cv=cv, n_jobs=-1, verbose=1)

        # Fit the model on the training data
        grid_search.fit(self.X_train, self.y_train)

        self.trained_model = grid_search.best_estimator_

        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best CV Score:", grid_search.best_score_)

        # # Return the best model with the best hyperparameters
        # return grid_search
    

    # from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.preprocessing import LabelBinarizer
    # from sklearn.metrics import roc_auc_score, roc_curve

    def evaluate_model(self, labels_order=[], font_size=14, cmap_color="Blues"):
        """
        Function to evaluate the performance of the model on the test set.
        - model: The trained model to be evaluated
        - X_test: Test features
        - y_test: True labels of the test set
        - model_name: Name of the model (for display purposes)
        """

        if not self.trained_model:
            return
        
        # Step 1: Use the model to predict on the test data
        y_pred = self.trained_model.predict(self.X_test)
        
        # Step 2: Print the classification report
        print(f"\nClassification Report for {self.model_label}:\n")
        report = classification_report(self.y_test, y_pred, output_dict=True)  # Get report as a dictionary
        print(classification_report(self.y_test, y_pred))
        
        # Step 3: Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"{self.model_label} Accuracy: {accuracy:.4f}")
        
        # Step 4: Calculate F1 score for each class
        f1_scores_dict = {}
        for label in report.keys():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:  # Skip non-class labels
                f1_scores_dict[label] = report[label]['f1-score']
                print(f"F1 Score for {label}: {report[label]['f1-score']:.4f}")
        
        # Step 5: Confusion Matrix
        print(f"\nConfusion Matrix for {self.model_label}:\n")
        if len(labels_order) > 0:
            cm = confusion_matrix(self.y_test, y_pred, labels=labels_order)
            # self, matrix, title = "", x_label= "", y_label = "", font_size = 12,
            #                      cmap_color = "Blues", ticklabels = []
            
            self.chart_helper.draw_heatmap_with_matrix(cm, title=f"Confusion Matrix for {self.model_label}",
                                                    x_label="Predicted Label", y_label="True Label", 
                                                    ticklabels=labels_order,
                                                    font_size=font_size, 
                                                    cmap_color=cmap_color)
        else:
            cm = confusion_matrix(self.y_test, y_pred)
            self.chart_helper.draw_heatmap_with_matrix(cm, title=f"Confusion Matrix for {self.model_label}",
                                                    x_label="Predicted Label", y_label="True Label",  
                                                    ticklabels=np.unique(self.y_test),
                                                    font_size=font_size, 
                                                    cmap_color=cmap_color)
        
        # Step 6: If binary classification, calculate ROC-AUC score
        if len(np.unique(self.y_test)) == 2:
            lb = LabelBinarizer()
            y_test_binary = lb.fit_transform(self.y_test).ravel()  # Binarize the labels (0 and 1)

            # Calculate the predicted probabilities and AUC score
            y_prob = self.trained_model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(y_test_binary, y_prob)
            print(f"ROC-AUC for {self.model_label}: {auc:.4f}")

            # Plot ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {self.model_label}')
            plt.legend(loc="lower right")
            plt.show()
        
        # Return the accuracy and the dictionary of F1 scores
        return accuracy, f1_scores_dict
  
    
    # def evaluate_model(self, labels_order=[], font_size = 14, cmap_color = "Blues"):
    #     """
    #     Function to evaluate the performance of the model on the test set.
    #     - model: The trained model to be evaluated
    #     - X_test: Test features
    #     - y_test: True labels of the test set
    #     - model_name: Name of the model (for display purposes)
    #     """

    #     if not self.trained_model:
    #         return
        
    #     # Use the model to predict on the test data
    #     y_pred = self.trained_model.predict(self.X_test)
        
    #     # Print the classification report, which includes precision, recall, F1-score, and accuracy
    #     print(f"\nClassification Report for {self.model_name}:\n")
    #     print(classification_report(self.y_test, y_pred))
        
    #     # Calculate accuracy
    #     accuracy = accuracy_score(self.y_test, y_pred)
    #     print(f"{self.model_name} Accuracy: {accuracy:.4f}")
        
    #     # Confusion Matrix

    #     # Confusion Matrix with Custom Font Sizes and Colors
    #     print(f"\nConfusion Matrix for {self.model_name}:\n")

    #     # Generate confusion matrix and plot
    #     if len(labels_order) > 0:

    #         cm = confusion_matrix(self.y_test, y_pred, labels=labels_order)
    #         self.chart_helper.draw_heatmap_with_matrix(cm, title = f"Confusion Matrix for {self.model_name}",
    #                                x_label= "Predicted Label", y_label = "True Label", labels_order = labels_order,
    #                                title_font_size = font_size, annotation_font_size = font_size, label_font_size = font_size, ticker_font_size = font_size, cmap_color = cmap_color)
    #     else:
    #         cm = confusion_matrix(self.y_test, y_pred)
    #         self.chart_helper.draw_heatmap_with_matrix(cm, title = f"Confusion Matrix for {self.model_name}",
    #                         x_label= "Predicted Label", y_label = "True Label",  labels_order=np.unique(self.y_test),
    #                         title_font_size = font_size, annotation_font_size = font_size, label_font_size = font_size, ticker_font_size = font_size, cmap_color = cmap_color)

        
    #     # Check if the problem is binary classification (for ROC-AUC score)
    #     if len(np.unique(self.y_test)) == 2:

    #         # Convert y_test to binary values
    #         lb = LabelBinarizer()
    #         y_test_binary = lb.fit_transform(self.y_test).ravel()  # This will give binary labels (0 and 1)

    #         # Calculate the probabilities and AUC
    #         y_prob = self.trained_model.predict_proba(self.X_test)[:, 1]
    #         auc = roc_auc_score(y_test_binary, y_prob)
    #         print(f"ROC-AUC for {self.model_name}: {auc:.4f}")

    #         # Plot the ROC curve
    #         fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob)
    #         plt.figure(figsize=(8, 6))
    #         plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    #         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #         plt.xlim([0.0, 1.0])
    #         plt.ylim([0.0, 1.05])
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.title(f'ROC Curve for {self.model_name}')
    #         plt.legend(loc="lower right")
    #         plt.show()




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

        return feature_labels


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
