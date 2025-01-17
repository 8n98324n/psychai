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

class DataPreprocessor:
    def __init__(self):
        """
        """
        #self.df = df

    def remove_outliers(self, input_data, column_name, taylor_multiplier):
        try:
            # Calculate Q1 and Q3
            Q1 = input_data[column_name].quantile(0.25)
            Q3 = input_data[column_name].quantile(0.75)

            # Calculate IQR and thresholds
            IQR = Q3 - Q1
            lower_threshold = Q1 - taylor_multiplier * IQR
            higher_threshold = Q3 + taylor_multiplier * IQR

            # Filter out outliers
            return_data = input_data[(input_data[column_name] > lower_threshold) & (input_data[column_name] < higher_threshold)]
            return return_data
        except Exception as e:
            print(f"ERROR: {column_name}: {e}")
            return input_data

    # # Define the outlier detection and replacement function
    # def replace_outliers_in_pd_with_group_median(self, df, k=1.5, group_column = 'Group', skip_columns = ['Group', 'user_id']):
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
    #     feature_columns = [col for col in df.columns if col not in skip_columns]

    #     # Group the DataFrame by 'Group'
    #     grouped_df = df.groupby(group_column)

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
    #             df.loc[(df[group_column] == group) & is_outlier, column] = group_median

    #     return df


    # def replace_outliers_in_training_set_with_group_median(self, X, y, k=1.5):
    #     """
    #     This function replaces outliers in each feature column of a DataFrame with the median of the corresponding group.
    #     Outliers are identified using the IQR method where:
    #         mins = df[column] < Q1 - k * IQR
    #         maxs = df[column] > Q3 + k * IQR
    #     Additionally, NaN values are treated as outliers and replaced by the group's median.

    #     Arguments:
    #     X : pandas DataFrame containing the feature columns (excluding 'Group' and 'user_id')
    #     y : pandas Series containing the target variable (the 'Group' column)
    #     k : multiplier for IQR to define outliers (default is 1.5)

    #     Returns:
    #     X : DataFrame with outliers replaced by group medians
    #     """

    #     y = pd.Series(y)

    #     # Ensure that X is a pandas DataFrame and y is a pandas Series
    #     if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
    #         raise TypeError("X should be a DataFrame and y should be a Series")

    #     # Add the Group column (y) temporarily to the X DataFrame to allow grouping
    #     X['Group'] = y

    #     # List of feature columns (excluding 'Group')
    #     feature_columns = [col for col in X.columns if col != 'Group']

    #     # Group the DataFrame by 'Group'
    #     grouped_df = X.groupby('Group')

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

    #             # Replace outliers with the median of the group in the original X DataFrame
    #             X.loc[(X['Group'] == group) & is_outlier, column] = group_median

    #     # Drop the 'Group' column after processing
    #     X = X.drop(columns=['Group'])

    #     # Return the DataFrame with outliers replaced
    #     return X

    def prepare_training_data_given_X_y(self, df,  X, y, X_columns, y_column, column_to_split="", correlation_threshold = 0.05,
                                        explained_variance_threshold=0.9,
                     remove_redundant_features=True, remove_low_variance=True,
                     variance_threshold=0.01, use_pca=False, outlier_strategies = ['Replace with Median'],
                     random_state= 1, model_name = "Random Forest"):
        """
        Prepares data for machine learning by extracting features and labels, handling missing values,
        scaling the features, and optionally removing redundant and/or low variance features and/or applying PCA.

        Parameters:
        - X_columns: List of columns to be used as features.
        - y_column: Column name of the target variable.
        - explained_variance_threshold: Proportion of variance to retain for PCA (default is 0.95).
        - remove_redundant_features: Whether to remove redundant features based on correlation (default is True).
        - correlation_threshold: remove_redundant_features threshold (default is 0.9)
        - remove_low_variance: Whether to remove features with low variance (default is True).
        - variance_threshold: The threshold for variance to determine low variance features (default is 0.01).
        - use_pca: Whether to apply PCA for dimensionality reduction (default is False).

        Returns:
        - X_train, y_train: Training features and labels.
        - X_test, y_test: Testing features and labels.
        """
   # Step 2: Handling Missing Values
        print("Missing values in features (X):", X.isnull().sum().sum())
        print("Missing values in target (y):", pd.Series(y).isnull().sum())
        X = X.fillna(X.mean())  # Fill missing values with column means

        # Step 3: Remove Low Variance Features
        if remove_low_variance:
            selector = VarianceThreshold(threshold=variance_threshold)
            X_selected = selector.fit_transform(X)
            # Get column names of retained features after variance threshold
            retained_features = [X_columns[i] for i in range(len(X_columns)) if selector.variances_[i] > variance_threshold]
            X = pd.DataFrame(X_selected, columns=retained_features)  # Convert back to DataFrame
            print(f"Retained features after low variance filtering: {retained_features}")
            print(f"Removed {len(X_columns) - len(retained_features)} low variance features out of {len(X_columns)} total features.")

        # Step 4: Remove Redundant Features Based on Correlation
        if remove_redundant_features:
            correlation_matrix = X.corr().abs()
            upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
            X = X.drop(columns=to_drop)
            print(f"Removed {len(to_drop)} redundant features out of {len(X_columns)} total features based on correlation: {to_drop}")
            #print(upper_tri)

        # Step 5: Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Standardize features
        X = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame to retain column names

        # RFE = true
        # if RFE:
        #     if model_name == "Random Forest":
        #         from sklearn.ensemble import RandomForestClassifier
        #         model = RandomForestClassifier(random_state=self.random_state)
        #         # Recursive Feature Elimination (RFE)
        #         # Select top 5 features
        #         rfe = RFE(estimator=model, n_features_to_select=10)
        #         rfe.fit(X_train, y_train)

        #         # Get selected features
        #         selected_features = X_train.columns[rfe.support_]

        # Step 6: Dimensionality Reduction with PCA (Optional)
        if use_pca:
            pca = PCA()
            X_pca_temp = pca.fit_transform(X)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            optimal_num_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
            pca = PCA(n_components=optimal_num_components)
            X = pd.DataFrame(pca.fit_transform(X), columns=[f"PCA_{i+1}" for i in range(optimal_num_components)])
            print(f"Optimal number of PCA components: {optimal_num_components}")
            print(f"Shape of Features after PCA (X): {X.shape}")



        # Step 7: Splitting the Data into Training and Testing Sets
        if column_to_split and column_to_split in df.columns:
            # Split by unique values in column_to_split from `self.df`
            unique_values = df[column_to_split].unique()
            train_values, test_values = train_test_split(unique_values, test_size=0.2, random_state=random_state)

            # Get train and test indices based on column_to_split in self.df
            train_indices = df[df[column_to_split].isin(train_values)].index
            test_indices = df[df[column_to_split].isin(test_values)].index

            # Use the train and test indices to subset X and y
            X_train = X.loc[X.index.intersection(train_indices)]
            X_test = X.loc[X.index.intersection(test_indices)]
            y_train = y[X_train.index]
            y_test = y[X_test.index]
        else:
            # Default row-wise split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

        # Step 8: Class Distribution Check
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print("Class distribution in training set:", dict(zip(unique_train, counts_train)))

        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print("Class distribution in testing set:", dict(zip(unique_test, counts_test)))


        from sklearn.utils import shuffle

        # Shuffle training data
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=random_state)

        # Shuffle testing data
        X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state=random_state)
                # Return the prepared data for use in model training and testing
        return X_train_shuffled, y_train_shuffled, X_test_shuffled, y_test_shuffled


    # from sklearn.feature_selection import RFE

    # def prepare_training_data_given_X_y(
    #     self,
    #     df,
    #     X,
    #     y,
    #     X_columns,
    #     y_column,
    #     column_to_split="",
    #     correlation_threshold=0.05,
    #     explained_variance_threshold=0.9,
    #     remove_redundant_features=True,
    #     remove_low_variance=True,
    #     variance_threshold=0.01,
    #     use_pca=False,
    #     use_rfe=True,
    #     rfe_num_features=10,
    #     random_state=1,
    #     model_name="Random Forest"
    # ):
    #     """
    #     Prepares data for machine learning by splitting the data first,
    #     and then applying data preparation steps like scaling, PCA, RFE, and feature selection.

    #     Parameters:
    #     - X_columns: List of columns to be used as features.
    #     - y_column: Column name of the target variable.
    #     - explained_variance_threshold: Proportion of variance to retain for PCA (default is 0.9).
    #     - remove_redundant_features: Whether to remove redundant features based on correlation (default is True).
    #     - correlation_threshold: Threshold for removing redundant features (default is 0.05).
    #     - remove_low_variance: Whether to remove features with low variance (default is True).
    #     - variance_threshold: The threshold for variance to determine low variance features (default is 0.01).
    #     - use_pca: Whether to apply PCA for dimensionality reduction (default is False).
    #     - use_rfe: Whether to apply Recursive Feature Elimination (RFE) for feature selection (default is False).
    #     - rfe_num_features: Number of features to select using RFE (default is 10).

    #     Returns:
    #     - X_train_shuffled, y_train_shuffled: Training features and labels.
    #     - X_test_shuffled, y_test_shuffled: Testing features and labels.
    #     """
    #     # Step 1: Splitting the Data into Training and Testing Sets
    #     if column_to_split and column_to_split in df.columns:
    #         unique_values = df[column_to_split].unique()
    #         train_values, test_values = train_test_split(unique_values, test_size=0.2, random_state=random_state)

    #         train_indices = df[df[column_to_split].isin(train_values)].index
    #         test_indices = df[df[column_to_split].isin(test_values)].index

    #         X_train = X.loc[X.index.intersection(train_indices)]
    #         X_test = X.loc[X.index.intersection(test_indices)]
    #         y_train = y[X_train.index]
    #         y_test = y[X_test.index]
    #     else:
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X, y, test_size=0.2, stratify=y, random_state=random_state
    #         )

    #     # Step 2: Handle Missing Values
    #     print("Missing values in features (X_train):", X_train.isnull().sum().sum())
    #     print("Missing values in features (X_test):", X_test.isnull().sum().sum())
    #     X_train = X_train.fillna(X_train.mean())
    #     X_test = X_test.fillna(X_train.mean())  # Use training mean to fill test set

    #     # Step 3: Remove Low Variance Features (on training data only)
    #     if remove_low_variance:
    #         selector = VarianceThreshold(threshold=variance_threshold)
    #         X_train_selected = selector.fit_transform(X_train)
    #         retained_features = [X_columns[i] for i in range(len(X_columns)) if selector.variances_[i] > variance_threshold]
    #         X_train = pd.DataFrame(X_train_selected, columns=retained_features)
    #         X_test = X_test[retained_features]  # Apply the same feature selection to the test set
    #         print(f"Retained features after low variance filtering: {retained_features}")

    #     # Step 4: Remove Redundant Features Based on Correlation (on training data only)
    #     if remove_redundant_features:
    #         correlation_matrix = X_train.corr().abs()
    #         upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    #         to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    #         X_train = X_train.drop(columns=to_drop)
    #         X_test = X_test.drop(columns=to_drop, errors='ignore')
    #         print(f"Removed {len(to_drop)} redundant features based on correlation: {to_drop}")

    #     # Step 5: Recursive Feature Elimination (RFE)
    #     if use_rfe:
    #         if model_name == "Random Forest":
    #             from sklearn.ensemble import RandomForestClassifier
    #             model = RandomForestClassifier(random_state=random_state)
    #         elif model_name == "Linear Regression":
    #             model = LinearRegression()
    #         else:
    #             raise ValueError(f"RFE not supported for model: {model_name}")

    #         rfe_selector = RFE(estimator=model, n_features_to_select=rfe_num_features)
    #         rfe_selector.fit(X_train, y_train)

    #         rfe_selected_features = X_train.columns[rfe_selector.support_]
    #         X_train = X_train[rfe_selected_features]
    #         X_test = X_test[rfe_selected_features]
    #         print(f"Selected features after RFE: {list(rfe_selected_features)}")

    #     # Step 6: Feature Scaling (using training data only)
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)
    #     X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    #     X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    #     # Step 7: Dimensionality Reduction with PCA (Optional)
    #     if use_pca:
    #         pca = PCA()
    #         X_pca_temp = pca.fit_transform(X_train)
    #         cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    #         optimal_num_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
    #         pca = PCA(n_components=optimal_num_components)
    #         X_train = pd.DataFrame(pca.fit_transform(X_train), columns=[f"PCA_{i+1}" for i in range(optimal_num_components)])
    #         X_test = pd.DataFrame(pca.transform(X_test), columns=[f"PCA_{i+1}" for i in range(optimal_num_components)])
    #         print(f"Optimal number of PCA components: {optimal_num_components}")

    #     # Step 8: Class Distribution Check
    #     unique_train, counts_train = np.unique(y_train, return_counts=True)
    #     print("Class distribution in training set:", dict(zip(unique_train, counts_train)))

    #     unique_test, counts_test = np.unique(y_test, return_counts=True)
    #     print("Class distribution in testing set:", dict(zip(unique_test, counts_test)))

    #     # Shuffle training and testing data
    #     from sklearn.utils import shuffle
    #     X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=random_state)
    #     X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state=random_state)

    #     # Return the prepared data
    #     return X_train_shuffled, y_train_shuffled, X_test_shuffled, y_test_shuffled


    def replace_outliers_with_median(self, df, k=1.5, skip_columns=['Group', 'user_id']):
        """
        This function replaces outliers in each feature column of a DataFrame with the overall median
        of that column. Outliers are identified using the IQR method where:
            mins = df[column] < Q1 - k * IQR
            maxs = df[column] > Q3 + k * IQR
        Additionally, NaN values are treated as outliers and replaced by the column's median.

        Arguments:
        df : pandas DataFrame with 'Group', 'user_id', and feature columns
        k : multiplier for IQR to define outliers (default is 1.5)

        Returns:
        df : DataFrame with outliers and NaN values replaced by the column's median
        """
        # List of feature columns (excluding Group and user_id)
        feature_columns = [col for col in df.columns if col not in skip_columns]

        # Iterate over each feature column
        for column in feature_columns:
            # Calculate Q1, Q3, and IQR for the entire DataFrame column
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate the lower and upper bounds for outliers
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR

            # Identify the outliers: below lower bound, above upper bound, or NaN
            is_outlier = (df[column] < lower_bound) | (df[column] > upper_bound) | pd.isna(df[column])

            # Get the overall median of the column
            column_median = df[column].median()

            # Replace outliers with the column's median
            df.loc[is_outlier, column] = column_median

        return df



    def calculate_cosine_similarity(self, df):
        """
        This function calculates the cosine similarity between each row of the DataFrame.
        It returns a DataFrame where each cell contains the cosine similarity score
        between the rows in the original DataFrame.

        Arguments:
        df : pandas DataFrame with numerical features

        Returns:
        similarity_df : pandas DataFrame with cosine similarity scores
        """
        # Ensure the DataFrame has no non-numeric columns (e.g., 'Group', 'user_id')
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate the cosine similarity matrix
        similarity_matrix = cosine_similarity(numeric_df)

        # Convert to DataFrame for better readability
        similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

        return similarity_df

    def replace_outliers_with_most_similar_median(self, df, similarity_df,  k=1.5, skip_columns=['Group', 'user_id']):
        """
        This function replaces outliers in each feature column of a DataFrame with the median
        of the 5 most similar rows (based on cosine similarity). Outliers are identified using the IQR method:
            mins = df[column] < Q1 - k * IQR
            maxs = df[column] > Q3 + k * IQR
        Additionally, NaN values are treated as outliers and replaced by the median of the 5 most similar rows.

        Arguments:
        df : pandas DataFrame with 'Group', 'user_id', and feature columns
        k : multiplier for IQR to define outliers (default is 1.5)

        Returns:
        df : DataFrame with outliers and NaN values replaced by the median of the 5 most similar rows
        """
        # List of feature columns (excluding Group and user_id)
        feature_columns = [col for col in df.columns if col not in skip_columns]


        # Iterate over each feature column
        for column in feature_columns:
            # Calculate Q1, Q3, and IQR for the entire DataFrame column
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate the lower and upper bounds for outliers
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR

            # Identify the outliers: below lower bound, above upper bound, or NaN
            is_outlier = (df[column] < lower_bound) | (df[column] > upper_bound) | pd.isna(df[column])

            # Replace outliers with the median of the 5 most similar rows
            for i, is_out in enumerate(is_outlier):
                if is_out:  # If the value is an outlier
                    try:
                        number_to_compare = int(0.1*len(feature_columns))
                        # Get the row index of the 5 most similar rows based on cosine similarity
                        similar_rows = similarity_df.iloc[i].nlargest(number_to_compare+1).index[1:]  # Exclude the row itself
                        # Get the median of these 5 rows for the current column
                        similar_medians = df.loc[similar_rows, column]
                        median_of_similar = similar_medians.median()

                        # Replace the outlier with the median of the 5 most similar rows
                        df.at[i, column] = median_of_similar
                    except Exception as e:
                        print(f"Errorat row {i}:{e} ")
        return df
    # # Example usage:

    # # Sample DataFrame with some outliers and NaN values
    # data = {
    #     'Group': ['A', 'B', 'A', 'B', 'A'],
    #     'user_id': [1, 2, 3, 4, 5],
    #     'Feature1': [10, 200, 30, 400, 50],
    #     'Feature2': [5, 50, 15, 500, 25]
    # }
    # df = pd.DataFrame(data)

    # # Replace outliers with the median of the 5 most similar rows
    # df_cleaned = replace_outliers_with_most_similar_median(df)

    # # Display the cleaned DataFrame
    # print(df_cleaned)

    def remove_rows_with_m_outliers(self, df, k = 1.5, m=10, skip_columns=['Group', 'user_id']):
        """
        This function removes rows from the DataFrame where k or more of the feature columns are outliers.

        Outliers are identified using the IQR method:
            mins = df[column] < Q1 - k * IQR
            maxs = df[column] > Q3 + k * IQR
        Arguments:
        df : pandas DataFrame with 'Group', 'user_id', and feature columns
        k : minimum number of outliers required in a row to remove that row
        skip_columns : columns to be excluded from feature analysis (e.g., 'Group', 'user_id')

        Returns:
        df : DataFrame with rows removed where k or more columns are outliers
        """
        # List of feature columns (excluding Group and user_id)
        feature_columns = [col for col in df.columns if col not in skip_columns]

        # Create a boolean DataFrame for identifying outliers
        outlier_mask = pd.DataFrame(False, index=df.index, columns=feature_columns)

        # Calculate the IQR for each feature column
        for column in feature_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate the lower and upper bounds for outliers
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR

            # Identify outliers (True if outlier, False otherwise)
            outlier_mask[column] = (df[column] < lower_bound) | (df[column] > upper_bound) | pd.isna(df[column])

        # Count the number of outliers in each row
        outlier_counts = outlier_mask.sum(axis=1)

        # Remove rows where outliers count >= m
        df_cleaned = df[outlier_counts < m]

        return df_cleaned

    # # Define a function to handle outliers
    # # Define a function to handle outliers with a parameter k
    # def replace_outliers_with_bounds(self, col, k=1.5):
    #     # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    #     Q1 = col.quantile(0.25)
    #     Q3 = col.quantile(0.75)
    #     # Calculate the IQR
    #     IQR = Q3 - Q1
    #     # Calculate lower and upper bounds using k
    #     lower_bound = Q1 - k * IQR
    #     upper_bound = Q3 + k * IQR
    #     # Replace outliers with bounds
    #     return col.apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

    # # Define a function to handle outliers with a parameter k and NaN values
    # def replace_outliers_with_bounds(self, col, k=1.5):
    #     # Skip columns with all NaN values
    #     if col.isna().all():
    #         return col  # Return column unchanged if all values are NaN

    #     # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    #     Q1 = col.quantile(0.25)
    #     Q3 = col.quantile(0.75)
    #     # Calculate the IQR
    #     IQR = Q3 - Q1
    #     # Calculate lower and upper bounds using k
    #     lower_bound = Q1 - k * IQR
    #     upper_bound = Q3 + k * IQR
    #     # Replace outliers with bounds, and preserve NaN values
    #     return col.apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

  # Define a function to handle outliers with a parameter k, ensuring columns are numerical
    def replace_outliers_with_bounds(self, df, k=1.5):
        # Iterate through each column
        for col in df.columns:
            # Check if the column is numerical
            if pd.api.types.is_numeric_dtype(df[col]):
                # Skip columns with all NaN values
                if df[col].isna().all():
                    continue

                # Calculate Q1 (25th percentile) and Q3 (75th percentile)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                # Calculate the IQR
                IQR = Q3 - Q1
                # Calculate lower and upper bounds using k
                lower_bound = Q1 - k * IQR
                upper_bound = Q3 + k * IQR

                # Replace outliers with bounds, preserving NaN values
                df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
        return df

    def prepare_training_data(self,df, X_columns, y_column, column_to_split="", explained_variance_threshold=0.95,
                              correlation_threshold=0.9, remove_redundant_features=True, remove_low_variance=False,
                     variance_threshold=0.01, use_pca=False, outlier_strategies = ['Replace with Median'], k=1.5, random_state= 1):
        """
        Prepares data for machine learning by extracting features and labels, handling missing values,
        scaling the features, and optionally removing redundant and/or low variance features and/or applying PCA.

        Parameters:
        - X_columns: List of columns to be used as features.
        - y_column: Column name of the target variable.
        - explained_variance_threshold: Proportion of variance to retain for PCA (default is 0.95).
        - remove_redundant_features: Whether to remove redundant features based on correlation (default is True).
        - remove_low_variance: Whether to remove features with low variance (default is True).
        - variance_threshold: The threshold for variance to determine low variance features (default is 0.01).
        - use_pca: Whether to apply PCA for dimensionality reduction (default is False).

        Returns:
        - X_train, y_train: Training features and labels.
        - X_test, y_test: Testing features and labels.
        """

        # if len(outlier_strategies)>0:
        #     for outlier_strategy in outlier_strategies:
        #         if outlier_strategy == 'Remove Multi-Column Outlier':
        #             num_rows_before = df.shape[0]

        #             m = int((0.2/k)*len(X_columns))
        #             if (m<3):
        #                 m = 3
        #             df = self.remove_rows_with_m_outliers(df.copy(), m=m, k=k)
        #             df= df.reset_index(drop=True)
        #             num_rows_after = df.shape[0]
        #             print(f"Remove outliers for {m} columns with k={k}: {num_rows_before-num_rows_after} rows out of {num_rows_before} rows removed.")
        #         if outlier_strategy == 'Replace with Median':
        #             df = self.replace_outliers_with_median(df.copy(), k=k)
        #             print("Replace with Median")
        #         if outlier_strategy == 'Replace with Similar Median':
        #             similarity_df = self.calculate_cosine_similarity(df)
        #             df = self.replace_outliers_with_most_similar_median(df.copy(), similarity_df, k=k)
        #             print(f"Replace with Similar Median. k={k}")
        #         if outlier_strategy == 'Replace with Boundary':
        #             df = self.replace_outliers_with_bounds(df, k=1.5)
        #             print(f"Replace with Boundary. k={k}")


        # Step 1: Extracting Features and Target Labels
        X = df[X_columns].copy()  # Extract features as DataFrame
        y = df[y_column].values   # Extract target variable as numpy array

        return self.prepare_training_data_given_X_y(df, X,y,X_columns, y_column, column_to_split = column_to_split,
                                                    correlation_threshold = correlation_threshold,
                                                    explained_variance_threshold = explained_variance_threshold,
                     remove_redundant_features = remove_redundant_features, remove_low_variance = remove_low_variance,
                     variance_threshold = variance_threshold, use_pca = use_pca, outlier_strategies = outlier_strategies,
                     random_state = random_state)

        # # Step 2: Handling Missing Values
        # print("Missing values in features (X):", X.isnull().sum().sum())
        # print("Missing values in target (y):", pd.Series(y).isnull().sum())
        # X = X.fillna(X.mean())  # Fill missing values with column means

        # # Step 3: Remove Low Variance Features
        # if remove_low_variance:
        #     selector = VarianceThreshold(threshold=variance_threshold)
        #     X_selected = selector.fit_transform(X)
        #     # Get column names of retained features after variance threshold
        #     retained_features = [X_columns[i] for i in range(len(X_columns)) if selector.variances_[i] > variance_threshold]
        #     X = pd.DataFrame(X_selected, columns=retained_features)  # Convert back to DataFrame
        #     print(f"Retained features after low variance filtering: {retained_features}")
        #     print(f"Removed {len(X_columns) - len(retained_features)} low variance features")

        # # Step 4: Remove Redundant Features Based on Correlation
        # if remove_redundant_features:
        #     correlation_matrix = X.corr().abs()
        #     upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        #     to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
        #     X = X.drop(columns=to_drop)
        #     print(f"Removed {len(to_drop)} redundant features based on correlation: {to_drop}")

        # # Step 5: Feature Scaling
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)  # Standardize features
        # X = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame to retain column names

        # # Step 6: Dimensionality Reduction with PCA (Optional)
        # if use_pca:
        #     pca = PCA()
        #     X_pca_temp = pca.fit_transform(X)
        #     cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        #     optimal_num_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
        #     pca = PCA(n_components=optimal_num_components)
        #     X = pd.DataFrame(pca.fit_transform(X), columns=[f"PCA_{i+1}" for i in range(optimal_num_components)])
        #     print(f"Optimal number of PCA components: {optimal_num_components}")
        #     print(f"Shape of Features after PCA (X): {X.shape}")


        # # Step 7: Splitting the Data into Training and Testing Sets
        # if column_to_split and column_to_split in self.df.columns:
        #     # Split by unique values in column_to_split from `self.df`
        #     unique_values = self.df[column_to_split].unique()
        #     train_values, test_values = train_test_split(unique_values, test_size=0.2, random_state=42)

        #     # Get train and test indices based on column_to_split in self.df
        #     train_indices = self.df[self.df[column_to_split].isin(train_values)].index
        #     test_indices = self.df[self.df[column_to_split].isin(test_values)].index

        #     # Use the train and test indices to subset X and y
        #     X_train = X.loc[X.index.intersection(train_indices)]
        #     X_test = X.loc[X.index.intersection(test_indices)]
        #     y_train = y[X_train.index]
        #     y_test = y[X_test.index]
        # else:
        #     # Default row-wise split
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # # Step 8: Class Distribution Check
        # unique_train, counts_train = np.unique(y_train, return_counts=True)
        # print("Class distribution in training set:", dict(zip(unique_train, counts_train)))

        # unique_test, counts_test = np.unique(y_test, return_counts=True)
        # print("Class distribution in testing set:", dict(zip(unique_test, counts_test)))

        # # Return the prepared data for use in model training and testing
        # return X_train, y_train, X_test, y_test
