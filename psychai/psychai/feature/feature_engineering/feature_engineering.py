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
from sklearn.decomposition import PCA

class FeatureEngineeringHelper:

    def __init__(self):
        pass
    
    def remove_high_correlation(self, X_train_input, X_val_input, feature_columns, corr_threshold=0.95, verbose = False):

        X_train = X_train_input.copy()
        X_val = X_val_input.copy()

        corr_matrix = pd.DataFrame(X_train).corr().abs()  # Calculate correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Upper triangle of the matrix
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]  # Columns to drop
        # Drop the highly correlated columns
        X_train_removed = (pd.DataFrame(X_train).drop(columns=to_drop)).to_numpy()
        X_val_emoved = (pd.DataFrame(X_val).drop(columns=to_drop)).to_numpy()
        # Remove elements based on indices
        kept_columns = [col for idx, col in enumerate(feature_columns) if idx not in to_drop]
        filtered_out_columns = [col for idx, col in enumerate(feature_columns) if idx in to_drop]
        if verbose:
            print(f"{len(filtered_out_columns)} out of {len(feature_columns)} Columns removed: {filtered_out_columns}")
        return X_train_removed, X_val_emoved, kept_columns

        # X_train = X_train_removed
        # X_val = X_val_emoved

    def keep_high_anova(self, X_train_input, X_val_input, y_train, kept_columns, n_features_kept_by_anova, verbose = False):

        X_train = X_train_input.copy()
        X_val = X_val_input.copy()

        kept_columns_list = list(kept_columns)
        # Calculate F-values and p-values for ANOVA
        f_values, p_values = f_classif(X_train, y_train)

        # Create a DataFrame to store feature names and their F-values
        f_values_df = pd.DataFrame({
            'Feature': kept_columns_list,
            'F_value': f_values
        })

        # Sort features by F-value in descending order
        f_values_df = f_values_df.sort_values(by='F_value', ascending=False)

        # Select the top 10 features
        top_features = f_values_df['Feature'].head(n_features_kept_by_anova).tolist()

        # Filter the train and validation datasets for the top 10 features
        top_feature_indices = [kept_columns_list.index(feature) for feature in top_features]
        X_train_selected = X_train[:, top_feature_indices]
        X_val_selected = X_val[:, top_feature_indices]
        kept_columns = top_features

        if verbose:
            print(f"ANOVA Selected Features: {top_features}")

        return X_train_selected, X_val_selected, kept_columns
    

    def RFE(self, X_train_input, X_val_input, y_train, random_state, kept_columns, n_kept, verbose = False, rfe_model= None):

        X_train_selected = X_train_input.copy()
        X_val_selected = X_val_input.copy()
        # Perform Recursive Feature Elimination (RFE)
        # rfe_model = SVC(kernel="linear", random_state=random_state)
        
        
        # rfe_model = RandomForestClassifier(random_state=random_state)
        if rfe_model is None:
            rfe_model = SVC(kernel="linear", random_state=random_state)

        rfe = RFE(estimator=rfe_model, n_features_to_select=n_kept)
        rfe.fit(X_train_selected, y_train)

        # Update train and validation sets with selected features
        X_train_selected = X_train_selected[:, rfe.support_]
        X_val_selected = X_val_selected[:, rfe.support_]

        # Extract the selected feature names
        selected_features = [kept_columns[i] for i in range(len(kept_columns)) if rfe.support_[i]]

        if verbose:
            print(f"RFE:{selected_features}")

        return X_train_selected, X_val_selected, selected_features
    

    def pca(self,  X_train_input, X_val_input, explained_variance_threshold=0.85, verbose = False):

        X_train_selected = X_train_input.copy()
        X_val_selected = X_val_input.copy()

        pca = PCA()
        pca.fit_transform(X_train_selected)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        optimal_num_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
        pca = PCA(n_components=optimal_num_components)
        selected_columns=[f"PCA_{i+1}" for i in range(optimal_num_components)]
        X_train_selected = pd.DataFrame(pca.fit_transform(X_train_selected), columns=selected_columns)
        X_val_selected = pd.DataFrame(pca.transform(X_val_selected), columns=selected_columns) 
        # X_train_selected = pd.DataFrame(pca.fit_transform(X_train_selected), columns=[f"PCA_{i+1}" for i in range(optimal_num_components)])
        # X_val_selected = pd.DataFrame(pca.transform(X_val_selected), columns=[f"PCA_{i+1}" for i in range(optimal_num_components)]) 
        if verbose:
            print(f"Optimal number of PCA components: {optimal_num_components}")
            print(f"Shape of Features after PCA (X): {X_train_selected.shape}")

        return X_train_selected, X_val_selected