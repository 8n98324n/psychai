import os
import sys
import warnings
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreparationHelper:
    """
    A helper class to prepare and preprocess data for machine learning tasks.
    Includes functionalities for missing value handling, outlier removal, scaling, feature selection, and more.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataPreparationHelper with a DataFrame.

        Parameters:
        - df (pd.DataFrame): The input data to be preprocessed.
        """
        self.df = df

    def remove_all_nan_rows(self):
         # Remove rows that contain only NaN values
        self.df = self.df.dropna(how='all').reset_index(drop=True)


    def prepare_data(self, parameters:dict = {}):
        if parameters.get("remove_all_nan_rows") is not None:
                    remove_all_nan_rows = parameters.get("remove_all_nan_rows")
                    if remove_all_nan_rows:
                        self.remove_all_nan_rows()

        if parameters.get("drop_missing_data") is not None:
                    drop_missing_data = parameters.get("drop_missing_data")
                    if drop_missing_data:
                        self.drop_missing_data(threshold=0.1)
        if parameters.get("replace_missing_data_with_mean") is not None:
                    replace_missing_data_with_mean = parameters.get("replace_missing_data_with_mean")
                    if replace_missing_data_with_mean:
                        self.handle_missing_values("mean", self.df.columns[2:].tolist())       
        if parameters.get("replace_missing_data_with_median") is not None:
                replace_missing_data_with_median = parameters.get("replace_missing_data_with_median")
                if replace_missing_data_with_median:
                    self.handle_missing_values("median", self.df.columns[2:].tolist())                     

                


        # data_prepare_helper.process_outliers(strategy = 'remove', method = 'iqr', columns = df_speech_acoustics.columns[2:], threshold=threshold)

        if parameters.get("log_transform") is not None:
            log_transform = parameters.get("log_transform")
            if log_transform:
                self.data_transform(self.df.columns[2:], transformation_method="log", only_skewed_columns=True, skewness_threshold=1, create_new_column=False)             

        #df_prepared = data_prepare_helper.get_data_frame()
        # data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="negative exp", only_skewed_columns=True, skewness_threshold=1, create_new_column=True)
        if parameters.get("cap_data") is not None and parameters.get("outlier_multiplier") is not None:
                    cap_data = parameters.get("cap_data")
                    if cap_data:
                        outlier_multiplier = parameters.get("outlier_multiplier")
                        #self.data_prepare_helper.handle_missing_values("mean", self.df.columns[2:].tolist())   
                        df_prepared = self.get_data_frame()                  
                        df_prepared = self.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'remove', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        df_prepared = self.get_data_frame()
        return df_prepared

    def handle_missing_values(self, strategy: str = 'median', columns: list = None) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Parameters:
        - strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', 'constant').
        - columns (list): List of columns to apply the imputation. If None, applies to all columns.

        Returns:
        - pd.DataFrame: DataFrame with imputed missing values.
        """
        imputer = SimpleImputer(strategy=strategy)
        if columns:
            self.df[columns] = imputer.fit_transform(self.df[columns])
        else:
            self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
        return self.df

    def drop_missing_data(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Drop rows or columns with a high percentage of missing values.

        Parameters:
        - threshold (float): Proportion of missing values above which the column/row will be dropped.

        Returns:
        - pd.DataFrame: Cleaned DataFrame.
        """
        self.df = self.df.dropna(axis=0, thresh=int((1 - threshold) * self.df.shape[1]))  # Drop rows with too many missing values
        return self.df

    def remove_outliers_iqr(self, columns: list, threshold: float, drop: str = "rows") -> pd.DataFrame:
        """
        Detect and handle outliers using the Interquartile Range (IQR) method.

        Parameters:
        - columns (list): List of columns to analyze for outliers.
        - threshold (float): Maximum allowable percentage of outliers (0 to 1).
        - drop (str): 'rows' to drop rows with outliers, 'columns' to drop columns with excessive outliers.

        Returns:
        - pd.DataFrame: DataFrame with outliers handled.
        """
        outlier_flags = pd.DataFrame(index=self.df.index)  # To track outlier flags for rows

        for column in columns:
            # Calculate IQR bounds
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outlier_flags[column] = ~self.df[column].between(lower_bound, upper_bound)

        if drop == "rows":
            # Drop rows exceeding the threshold of outliers
            outlier_count = outlier_flags.sum(axis=1)  # Count outliers per row
            row_threshold = len(columns) * threshold  # Maximum allowable outliers per row
            self.df = self.df[outlier_count <= row_threshold]
        elif drop == "columns":
            # Drop columns exceeding the threshold of outliers
            outlier_percentage = outlier_flags.mean(axis=0)  # Outlier percentage per column
            self.df = self.df.loc[:, outlier_percentage <= threshold]

        return self.df

    def remove_outliers_zscore(self, columns: list, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.

        Parameters:
        - columns (list): List of columns to analyze for outliers.
        - threshold (float): Z-score threshold to define outliers.

        Returns:
        - pd.DataFrame: DataFrame with outliers removed.
        """
        for column in columns:
            z_scores = np.abs(stats.zscore(self.df[column]))
            self.df = self.df[z_scores < threshold]
        return self.df

    def remove_outliers(self, threshold=0.5, method: str = 'iqr', columns: list = None) -> pd.DataFrame:
        """
        Remove outliers based on the specified method ('iqr' or 'zscore').

        Parameters:
        - threshold (float): Threshold for identifying outliers.
        - method (str): Method for outlier detection ('iqr' or 'zscore').
        - columns (list): Columns to analyze for outliers.

        Returns:
        - pd.DataFrame: DataFrame with outliers removed.
        """
        if method == 'iqr':
            self.df = self.remove_outliers_iqr(columns, threshold=threshold)
        elif method == 'zscore':
            self.df = self.remove_outliers_zscore(columns, threshold=threshold)
        return self.df

    def process_outliers(self, strategy: str = 'remove', method: str = 'iqr', columns: list = None, threshold: float = 0.5, outlier_multiplier=3) -> pd.DataFrame:
        """
        Process outliers based on the strategy ('remove' or 'cap').

        Parameters:
        - strategy (str): Strategy for handling outliers ('remove', 'cap').
        - method (str): Method for outlier detection ('iqr' or 'zscore').
        - columns (list): Columns to analyze for outliers.
        - threshold (float): Threshold for identifying outliers.

        Returns:
        - pd.DataFrame: DataFrame with outliers processed.
        """
        if strategy == 'remove':
            self.df = self.remove_outliers(method=method, threshold=threshold, columns=columns)
        elif strategy == 'cap':
            # Implement capping strategy
            for column in columns:
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_cap = Q1 - outlier_multiplier * IQR
                upper_cap = Q3 + outlier_multiplier * IQR
                self.df[column] = np.clip(self.df[column], lower_cap, upper_cap)
        return self.df
    
    
    def data_transform(self, columns: list, transformation_method = ["log"], only_skewed_columns=True, skewness_threshold=1, create_new_column = True, verbose=False) -> pd.DataFrame:
        """
        Apply log transformation to selected columns, ensuring all values are larger than a positive constant.

        Parameters:
        - columns: List of columns to apply the log transformation.
        - only_skewed_columns: Whether to apply transformation only to highly skewed columns.
        - skewness_threshold: Threshold to determine skewness if `only_skewed_columns` is True.
        - verbose: Whether to print details of processing.

        Returns:
        - List of processed columns.
        """
        columns_to_process = columns.copy()

        if only_skewed_columns:
            # Find highly skewed columns based on threshold
            columns_to_process = self.find_highly_skewed_columns(skewness_threshold=skewness_threshold)
            if verbose:
                print(f"Skewed columns identified: {columns_to_process}")

        for column in columns_to_process:  # Process numeric columns

            if transformation_method=="log":
                new_column_name = f"log_{column}"
                
                # Add a positive constant to make all values > 0
                min_value = self.df[column].min()
                constant = abs(min_value) + 1 if min_value <= 0 else 0
                
                if verbose:
                    print(f"Column: {column}, Min value: {min_value}, Constant added: {constant}")
                
                # Apply log transformation
                if create_new_column:
                    self.df[new_column_name] = np.log(self.df[column] + constant)
                else:
                    self.df[column] = np.log(self.df[column] + constant)
                    
            if transformation_method=="log1p":
                new_column_name = f"log_{column}"
                
                # Add a positive constant to make all values > 0
                min_value = self.df[column].min()
                constant = abs(min_value) + 1 if min_value <= 0 else 0
                
                if verbose:
                    print(f"Column: {column}, Min value: {min_value}, Constant added: {constant}")
                
                # Apply log transformation
                if create_new_column:
                    self.df[new_column_name] = np.log1p(self.df[column] + constant)
                else:
                    self.df[column] = np.log1p(self.df[column] + constant)

            elif transformation_method=="negative exp":
                
                new_column_name = f"ne_{column}"
                
                # Apply log transformation
                if create_new_column:
                    self.df[new_column_name] = 1-np.exp(-1*self.df[column])
                else:
                    self.df[column] = 1- np.exp(-1*self.df[column])               
        
        return columns_to_process

    # def add_log_transform(self, columns: list, only_skewed_columns=True, skewness_threshold=1, verbose=False, create_new_column = False) -> pd.DataFrame:
    #     """
    #     Apply log transformation to selected columns.

    #     Parameters:
    #     - columns (list): List of columns to apply the log transformation.
    #     - only_skewed_columns (bool): Apply transformation only to highly skewed columns.
    #     - skewness_threshold (float): Skewness threshold for transformation.
    #     - verbose (bool): Print details of processing if True.

    #     Returns:
    #     - list: List of processed columns.
    #     """
    #     columns_to_process = columns.copy()

    #     if only_skewed_columns:
    #         # Find highly skewed columns based on threshold
    #         columns_to_process = self.find_highly_skewed_columns(skewness_threshold=skewness_threshold)
    #         if verbose:
    #             print(f"Skewed columns identified: {columns_to_process}")

    #     # columns_to_process = ["love"]
    #     for column in columns_to_process:
    #         new_column_name = f"log_{column}"

    #         # Add a positive constant to make all values > 0
    #         min_value = self.df[column].min()
    #         constant = abs(min_value) + 0.1 if min_value <= 0 else 0.1

    #         if verbose:
    #             print(f"Column: {column}, Min value: {min_value}, Constant added: {constant}")

    #         # Apply log transformation
    #         if create_new_column:
    #             self.df[new_column_name] = np.log(self.df[column] + constant)
    #         else:
    #             self.df[column] = np.log(self.df[column] + constant)

    #     return columns_to_process

    def find_highly_skewed_columns(self, skewness_threshold=1):
        """
        Find columns in a DataFrame with skewness greater than a given threshold.

        Parameters:
        - skewness_threshold (float): The skewness threshold.

        Returns:
        - list: A list of column names with skewness greater than the threshold.
        """
        skewed_columns = []

        for column in self.df.select_dtypes(include=['float64', 'int64']).columns:
            skewness = self.df[column].skew()
            if abs(skewness) > skewness_threshold:
                skewed_columns.append(column)

        return skewed_columns

    def data_splitting(self, X_columns, y_column, test_size=0.2, column_to_split=None):
        """
        Split the data into training and testing sets.

        Parameters:
        - X_columns (list): List of feature columns.
        - y_column (str): Target column.
        - test_size (float): Proportion of the data to include in the test split.
        - column_to_split (str): Column to split data uniquely, if specified.

        Returns:
        - tuple: Split training and testing data (X_train, X_test, y_train, y_test).
        """
        X = self.df[X_columns]
        y = self.df[y_column]

        if column_to_split and column_to_split in self.df.columns:
            unique_values = self.df[column_to_split].unique()
            train_values, test_values = train_test_split(unique_values, test_size=test_size, random_state=random_state)

            train_indices = self.df[self.df[column_to_split].isin(train_values)].index
            test_indices = self.df[self.df[column_to_split].isin(test_values)].index

            self.X_train = X.loc[X.index.intersection(train_indices)]
            self.X_test = X.loc[X.index.intersection(test_indices)]
            self.y_train = y[self.X_train.index]
            self.y_test = y[self.X_test.index]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        self.X_train = pd.DataFrame(self.X_train)
        self.X_test = pd.DataFrame(self.X_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def remove_highly_correlated_features(self, threshold: float = 0.9, verbose=False) -> pd.DataFrame:
        """
        Remove features that are highly correlated with each other.

        Parameters:
        - threshold (float): Correlation threshold for removing features.
        - verbose (bool): Print details if True.

        Returns:
        - list: List of dropped columns.
        """
        corr_matrix = self.X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

        if verbose:
            print(f"{len(to_drop)} out of {len(self.X_train.columns)} columns dropped due to high correlation: {to_drop}.")

        self.X_train.drop(columns=to_drop, inplace=True)
        self.X_test.drop(columns=to_drop, inplace=True)

        return to_drop

    def standard_scaler(self, columns_to_scale):
        """
        Standardize selected columns in the dataset.

        Parameters:
        - columns_to_scale (list): List of columns to standardize.

        Returns:
        - tuple: Scaled training and testing data (X_train, X_test, y_train, y_test).
        """
        scaler = StandardScaler()
        self.X_train[columns_to_scale] = scaler.fit_transform(self.X_train[columns_to_scale])
        self.X_test[columns_to_scale] = scaler.transform(self.X_test[columns_to_scale])
        return self.X_train, self.X_test, self.y_train, self.y_test

    def recursive_feature_elimination(self, columns: list, n_features: int = 10) -> pd.DataFrame:
        """
        Perform Recursive Feature Elimination (RFE) to select the best features.

        Parameters:
        - columns (list): List of columns to perform RFE on.
        - n_features (int): Number of top features to select.

        Returns:
        - list: Selected feature names.
        """
        from sklearn.feature_selection import RFE
        from sklearn.svm import SVC

        model = SVC(kernel="linear")
        X_train_subset = self.X_train[columns]

        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(X_train_subset, self.y_train)

        selected_columns = X_train_subset.columns[rfe.support_]

        self.X_train = self.X_train[selected_columns]
        self.X_test = self.X_test[selected_columns]

        return selected_columns

    def run_pca(self, explained_variance_threshold=0.8) -> pd.DataFrame:
        """
        Perform PCA to reduce dimensionality while retaining a specified variance threshold.

        Parameters:
        - explained_variance_threshold (float): Variance threshold to retain.

        Returns:
        - tuple: Reduced training and testing data (X_train, X_test, y_train, y_test).
        """
        pca = PCA()
        pca.fit_transform(self.X_train)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        optimal_num_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
        pca = PCA(n_components=optimal_num_components)
        self.X_train = pd.DataFrame(pca.fit_transform(self.X_train), columns=[f"PCA_{i+1}" for i in range(optimal_num_components)])
        self.X_test = pd.DataFrame(pca.transform(self.X_test), columns=[f"PCA_{i+1}" for i in range(optimal_num_components)])
        print(f"Optimal number of PCA components: {optimal_num_components}")
        print(f"Shape of Features after PCA (X): {self.X_train.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_splitted_data(self):
        """
        Retrieve the training and testing data.

        Returns:
        - tuple: Split training and testing data (X_train, X_test, y_train, y_test).
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data_frame(self):
        """
        Retrieve the current DataFrame.

        Returns:
        - pd.DataFrame: The current DataFrame.
        """
        return self.df
