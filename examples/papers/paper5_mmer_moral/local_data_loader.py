import pandas as pd
import local_utilities
from sklearn.feature_extraction.text import TfidfVectorizer
import psychai.feature.feature_extraction.feature_processor
from dotenv import load_dotenv
import os

class DataLoader:

    def __init__(self):
        self.data_helper = local_utilities.DataHelper()
        self.df = pd.DataFrame()
        load_dotenv(override=True)
        self.huggingface_cache_location = os.getenv("huggingface_cache_location")
        self.datasets_cache_location = os.getenv("datasets_cache_location")
        self.resutls_location = os.getenv("results_location")
        pass

    # def add_summary_columns(self, df: pd.DataFrame, strategies = []):

    #     # Add the "seq" column
    #     df['seq'] = df.groupby('user_id').cumcount()

    #     # Summarize features using the specified strategies
    #     processor = psychai.feature.feature_extraction.feature_processor.FeatureProcessor()
    #     df_summarized = processor.summarize_features_by_column(df.iloc[:,1:], strategies, group_by_column="user_id")
    #     df_merged = pd.merge(df[:-1], df_summarized, on='user_id', how='left')
    #     df_merged
    #     return df_merged
    
    # def add_summary_columns(self, df: pd.DataFrame, strategies=[]):
    #     """
    #     Add summary columns to the DataFrame and return both merged and extended summarized DataFrames.

    #     Parameters:
    #     - df (pd.DataFrame): Input DataFrame containing a 'user_id' column.
    #     - strategies (list): List of strategies for summarizing features.

    #     Returns:
    #     - tuple: A tuple containing:
    #     - df_merged (pd.DataFrame): Merged DataFrame with summarized features.
    #     - df_summarized_extended (pd.DataFrame): Summarized DataFrame with an additional 'Group' column.
    #     """
    #     # Add the "seq" column
    #     df['seq'] = df.groupby('user_id').cumcount()

    #     # Extract 'user_id' and 'Group' information
    #     user_group_mapping = df[['user_id', 'Group']].drop_duplicates().set_index('user_id')['Group']

    #     # Summarize features using the specified strategies
    #     processor = psychai.feature.feature_extraction.feature_processor.FeatureProcessor()
    #     df_summarized = processor.summarize_features_by_column(df.iloc[:, 1:], strategies, group_by_column="user_id")

    #     # Add 'Group' information to the summarized DataFrame
    #     df_summarized_extended = df_summarized.copy()
    #     df_summarized_extended['Group'] = df_summarized_extended['user_id'].map(user_group_mapping)
        

    #     # Merge the original DataFrame with the summarized DataFrame
    #     df_merged = pd.merge(df[:-1], df_summarized, on='user_id', how='left')

    #     return df_merged, df_summarized_extended

    def add_summary_columns(self, df: pd.DataFrame, strategies=[]):
        """
        Add summary columns to the DataFrame and return both merged and extended summarized DataFrames.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing a 'user_id' column.
        - strategies (list): List of strategies for summarizing features.

        Returns:
        - tuple: A tuple containing:
        - df_merged (pd.DataFrame): Merged DataFrame with summarized features.
        - df_summarized_extended (pd.DataFrame): Summarized DataFrame with an additional 'Group' column.
        """
        # Add the "seq" column
        df['seq'] = df.groupby('user_id').cumcount()

        # Extract 'user_id' and 'Group' information
        user_group_mapping = df[['user_id', 'Group']].drop_duplicates().set_index('user_id')['Group']

        # Summarize features using the specified strategies
        processor = psychai.feature.feature_extraction.feature_processor.FeatureProcessor()
        df_summarized = processor.summarize_features_by_column(df.iloc[:, 1:], strategies, group_by_column="user_id")

        # Add 'Group' information to the summarized DataFrame
        df_summarized_extended = df_summarized.copy()
        df_summarized_extended['Group'] = df_summarized_extended['user_id'].map(user_group_mapping)

        # Reorder columns: Move 'Group' and 'user_id' to the front and drop the original first column
        # df_summarized_extended = df_summarized_extended.drop(columns=df_summarized_extended.columns[0])
        last_two_columns = ['Group', 'user_id']
        other_columns = [col for col in df_summarized_extended.columns if col not in last_two_columns]
        df_summarized_extended = df_summarized_extended[last_two_columns + other_columns]

        # Merge the original DataFrame with the summarized DataFrame
        df_merged = pd.merge(df[:-1], df_summarized, on='user_id', how='left')

        return df_merged, df_summarized_extended


    def remove_outliers_and_calculate_median(self, df_baseline, columns_to_process = [], remove_outlier = False,  outlier_multiplier=1.5):
        """
        Remove outliers for each column in df_baseline according to IQR criteria for each user_id,
        then calculate the mean of each user_id for each column and return the resulting DataFrame.
        
        Args:
            df_baseline (pd.DataFrame): DataFrame containing a 'user_id' column and other numeric columns.

        Returns:
            pd.DataFrame: A new DataFrame with user_id as a column and the median for each column.
        """
        # Separate user_ids

        if remove_outlier:
            user_groups = df_baseline.groupby('user_id')

            # Function to remove outliers from a column based on IQR
            def remove_outliers(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_multiplier * IQR
                upper_bound = Q3 + outlier_multiplier * IQR
                return series[(series >= lower_bound) & (series <= upper_bound)]

            # Process each group and calculate medians after outlier removal
            filtered_data = []
            for user_id, group in user_groups:
                filtered_group = group.copy()
                for column in group.columns:
                    if column != 'user_id':
                        filtered_group[column] = remove_outliers(group[column])
                filtered_data.append(filtered_group)

            filtered_df = pd.concat(filtered_data)
        else:
            filtered_df = df_baseline

        # Calculate median for each user_id
        df_baseline_median = filtered_df.groupby('user_id').median().reset_index()
        if columns_to_process != []:
            df_results = df_baseline_median["user_id" + columns_to_process]
        else:
            df_results = df_baseline_median

        return df_results

    def remove_baseline_from_target_df(self, df_target, df_baseline_median, column_to_process = [], outlier_multiplier = 3):
        """
        Subtract all values of each user_id in all columns in df_target that exist in df_baseline_median.
        
        Args:
            df_target (pd.DataFrame): DataFrame containing a 'user_id' column and other numeric columns.
            df_baseline_median (pd.DataFrame): DataFrame with user_id as a column and numeric columns.

        Returns:
            pd.DataFrame: A new DataFrame with the same structure as df_target with values adjusted.
        """



        if len(column_to_process) == 0:
            column_to_process = [col for col in df_baseline_median.columns if col in df_target.columns and col != 'user_id']


        # Merge df_target with only common columns from df_baseline_median
        df_merged = pd.merge(
            df_target,
            df_baseline_median[column_to_process.append(pd.Index(['user_id']))],
            on='user_id',
            suffixes=('', '_baseline'),
            how='left',  # Ensures all rows in df_target are kept
        )

        # Subtract baseline values from target values for matching columns
        for col in column_to_process:
            df_merged[col] = df_merged[col] - df_merged[f'{col}_baseline']

        # Retain only columns from df_target
        df_merged = df_merged[df_target.columns]

        return df_merged
    

    def remove_outliers_by_iqr(self, df_target, column_to_process=[], outlier_multiplier=3):
        """
        Remove outliers in the specified column(s) in df_target based on the interquartile range (IQR) for each user_id.

        Args:
            df_target (pd.DataFrame): DataFrame containing a 'user_id' column and other numeric columns.
            column_to_process (list): List of column names to process. If empty, all numeric columns except 'user_id' will be processed.
            outlier_multiplier (float): The multiplier for the IQR to determine outlier thresholds.

        Returns:
            pd.DataFrame: A DataFrame with outliers removed for the specified columns.
        """
        # import pandas as pd

        if len(column_to_process) == 0:
            column_to_process = [col for col in df_target.columns if col != 'user_id' and pd.api.types.is_numeric_dtype(df_target[col])]

        # Process each column for each user_id
        def remove_column_outliers(group):
            for col in column_to_process:
                q1 = group[col].quantile(0.25)
                q3 = group[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - outlier_multiplier * iqr
                upper_bound = q3 + outlier_multiplier * iqr
                group = group[(group[col] >= lower_bound) & (group[col] <= upper_bound)]
            return group

        # Apply the outlier removal function grouped by user_id
        df_filtered = df_target.groupby('user_id', group_keys=False).apply(remove_column_outliers)

        # Reset index to retain original structure
        df_filtered = df_filtered.reset_index(drop=True)

        return df_filtered


    def cap_outliers_by_iqr(self, df_target, column_to_process=[], outlier_multiplier=3):
        """
        Cap outliers in the specified column(s) in df_target based on the interquartile range (IQR) for each user_id.

        Args:
            df_target (pd.DataFrame): DataFrame containing a 'user_id' column and other numeric columns.
            column_to_process (list): List of column names to process. If empty, all numeric columns except 'user_id' will be processed.
            outlier_multiplier (float): The multiplier for the IQR to determine outlier thresholds.

        Returns:
            pd.DataFrame: A DataFrame with outliers capped for the specified columns.
        """
        # import pandas as pd

        if len(column_to_process) == 0:
            column_to_process = [col for col in df_target.columns if col != 'user_id' and pd.api.types.is_numeric_dtype(df_target[col])]

        # Process each column for each user_id
        def cap_column_outliers(group):
            for col in column_to_process:
                q1 = group[col].quantile(0.25)
                q3 = group[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - outlier_multiplier * iqr
                upper_bound = q3 + outlier_multiplier * iqr
                group[col] = group[col].clip(lower=lower_bound, upper=upper_bound)
            return group

        # Apply the outlier capping function grouped by user_id
        df_capped = df_target.groupby('user_id').apply(cap_column_outliers)

        # Reset index to retain original structure
        df_capped = df_capped.reset_index(drop=True)

        return df_capped


    def trim_data_frame(self, df:pd.DataFrame, feature_columns = [], prefix = "" ):
        if len(feature_columns) == 0:
            trimmed_columns = df.columns
        else:
            if type(feature_columns) == type([]):
                trimmed_columns = feature_columns
                trimmed_columns.insert(0, "user_id")
                trimmed_columns.insert(0, "Group")
            elif type(feature_columns) == type(df.columns):
                trimmed_columns = feature_columns
                trimmed_columns = trimmed_columns.insert(0, "user_id")
                trimmed_columns = trimmed_columns.insert(0, "Group")                
        df_trimmed = df.copy()[trimmed_columns]
        if prefix != "":
            df_trimmed.columns = [col if i < 2 else f"{prefix}-{col}" for i, col in enumerate(trimmed_columns)]
        else:
            df_trimmed.columns = [col if i < 2 else f"{col}" for i, col in enumerate(trimmed_columns)]
        return df_trimmed
    
    def get_concatenated_speech_sentiment_data_frame(self):
        file_path = r"./results/speech_text/csv/result_speech_sentiment_concatenated_20241112.csv"
        df_read = pd.read_csv(file_path)
        x_columns = df_read.columns[3:]
        prefix =  "concatenated_sentiment"
        return self.trim_data_frame(df_read, x_columns, prefix)

    def get_facial_df(self, file_target_path, file_baseline_path , outlier_multiplier = 3, to_trim = True, only_summary = True, prefix= "facial"):
        df_target = self.get_facial_df_from_csv(file_target_path, prefix)
        df_base_line = self.get_facial_df_from_csv(file_baseline_path, prefix)
        def trimm_df(df, to_trim):
            if to_trim:
                # List of columns containing the keyword 'AU'
                # columns_with_au = [col for col in df.columns if 'AU' in col]

                # Additional features to plot
                # columns_to_keep = [col for col in df_prepared.columns if all(substr not in col for substr in ["FaceRect","x_", "y_","distance_","frame","Frame","seq"])]
                columns_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
                columns_pose = ["Pitch","Roll","Yaw"]
                # Filter columns containing any of the substrings
                # columns_with_emotions = [col for col in df.columns if any(keyword in col for keyword in columns_emotions)]

                columns_to_run = ['Group','user_id']
                columns_to_run.extend([col for col in df if any(key in col for key in columns_emotions)])
                columns_to_run.extend([col for col in df if any(key in col for key in columns_pose)])
                columns_to_run.extend([col for col in df if any(key in col for key in ['AU'])])
                columns_to_run.extend([col for col in df if any(key in col for key in ['Face'])])
                # # columns_to_run.extend([col for col in df_prepared if any(key in col for key in ['distance'])])
                # df_train = df_prepared[columns_to_run]

                # basic_info = ["Group", "user_id"]
                # # Extend the columns_with_au list with features_to_plot
                # basic_info.extend(columns_with_au)
                # basic_info.extend(columns_emotions)
            
            else:
                columns_to_exclude = ["frame",'Group','user_id']
                columns_to_run = ['Group','user_id']
                columns_to_run.extend([col for col in df if not any(key in col for key in columns_to_exclude)])               

            df_trimmed = df[columns_to_run]

            # Add the "seq" column
            df_trimmed['seq'] = df_trimmed.groupby('user_id').cumcount()

            df_trimmed = df_trimmed[df["AU01"]>0]


            # Assuming your DataFrame is named 'df'
            # Remove duplicate rows
            df_trimmed = df_trimmed.drop_duplicates()

            # Reset index
            df_trimmed = df_trimmed.reset_index(drop=True)

            return df_trimmed
        
        df_target_trimmed = trimm_df(df_target, to_trim)
        df_baseline_trimmed = trimm_df(df_base_line, to_trim)

           
        df_baseline_median = self.remove_outliers_and_calculate_median(df_baseline_trimmed, remove_outlier= False,  outlier_multiplier= outlier_multiplier)
        df_target_centered = self.remove_baseline_from_target_df(df_target_trimmed, df_baseline_median, column_to_process=df_target_trimmed.columns[2:])
        # Extract features using multiple strategies and plot them
        strategies = ['median', 'variance', 'max', 'end_to_begin']
        # strategies = ['median','variance']
        
        df_with_summary, df_summary = self.add_summary_columns(df_target_centered, strategies)

        threshold = 0.3
        outlier_multiplier = 3
        if only_summary:
            data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary.copy())
        else:
            data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_with_summary.copy())
        # data_prepare_helper = DataPreparationHelper(df_summary.copy())
        #data_prepare_helper.drop_missing_data(threshold=0.1)
        # data_prepare_helper.handle_missing_values("mean", df_summary.columns[2:].tolist())
        # data_prepare_helper.process_outliers(strategy = 'remove', method = 'iqr', columns = df_speech_acoustics.columns[2:], threshold=threshold)
        #data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="log", only_skewed_columns=True, skewness_threshold=1, create_new_column=False)
        #df_prepared = data_prepare_helper.get_data_frame()
        # data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="negative exp", only_skewed_columns=True, skewness_threshold=1, create_new_column=True)
        #df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'remove', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared

    def get_facial_df_from_csv(self, file_path, prefix= "facial"):

        df_read = pd.read_csv(file_path)

        # Convert the existing selection to a list and add the new column at the beginning
        selected_columns = df_read.columns[16:].tolist()
        # selected_columns = selected_columns.insert(0, "user_id")
        # selected_columns = selected_columns.insert(0, "Group")

        # Reorder df_filtered_speech_acoustics with the new column order
        # Ensure the new column exists in the DataFrame; otherwise, this will raise an error
        # df_read = df_read[selected_columns]

        # df_filtered_facial_expression = df_read.loc[:, ~df_read.columns.str.contains('frame')]
        df_read = df_read[~df_read.isnull().any(axis=1)]
        df_read= df_read.reset_index(drop=True)
        # x_columns = df_read.columns
        prefix =  ""
        return self.trim_data_frame(df_read, feature_columns = selected_columns,prefix=  prefix)
    
    def get_facial_expression_data_frame(self, file_path):
        
        # file_path = r"./results/facial/facial_features_cache_20241101.csv"
        df_read = pd.read_csv(file_path)
        # Define the list of strings to search for
        keywords = ["AU", "happiness", "sadness", "disgust", "anger", "fear", "surprise", "neutral"]

        # Create a regex pattern that matches any of the keywords
        pattern = "|".join(keywords)  # This will create a pattern like "AU|Face"

        # Filter columns based on the pattern
        x_columns = df_read.columns[df_read.columns.str.contains(pattern, case=False)]

        selected_columns = x_columns
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")

        # Reorder df_filtered_speech_acoustics with the new column order
        # Ensure the new column exists in the DataFrame; otherwise, this will raise an error
        df_read_explainable = df_read[selected_columns]
        return df_read_explainable
    
    # def get_mean_eye_movement_data_frame(self, record_path):
    #     df= self.get_eye_movement_from_csv(record_path)
    #     df_mean = self.data_helper.calculate_mean_by_user_id(df)
    #     return df_mean

    
    def get_eye_movement_df(self, record_path, prefix = "eye",strategies = ['median','max']):
        df = self.get_eye_movement_df_from_csv(record_path, prefix)
        # Extract features using multiple strategies and plot them
        # strategies = ['median','max']
        df_with_summary, df_summary = self.add_summary_columns(df, strategies)
        threshold = 0.3
        outlier_multiplier = 3
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary)
        #data_prepare_helper.drop_missing_data(threshold=0.1)
        # data_prepare_helper.handle_missing_values("mean", df_summary.columns[2:].tolist())
        # data_prepare_helper.process_outliers(strategy = 'remove', method = 'iqr', columns = df_speech_acoustics.columns[2:], threshold=threshold)
        #data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="log", only_skewed_columns=True, skewness_threshold=1, create_new_column=False)
        #df_prepared = data_prepare_helper.get_data_frame()
        # data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="negative exp", only_skewed_columns=True, skewness_threshold=1, create_new_column=True)
        #df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'remove', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
    
    
    def get_eye_movement_df_from_csv(self, record_path, prefix = "eye"):
        # record_path = r"./results/eye_movement/csv/eye_movement_features_cache_20241106.csv"
        df_filtered = pd.read_csv(record_path)
        df_filtered = df_filtered[~df_filtered.isnull().any(axis=1)]
        df_filtered= df_filtered.reset_index(drop=True)
        selected_columns = df_filtered.columns[16:]
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_trimmed = df_filtered[selected_columns]
        df_trimmed = df_trimmed.loc[:, ~df_trimmed.columns.str.contains('frame')]
        x_columns = df_trimmed.columns[2:]
        return self.trim_data_frame(df_trimmed, x_columns, prefix)
    
    def get_pose_features_without_landmarks_data_frame(self, record_path, prefix = "pose"):
        # record_path = r"./results/pose/pose_features_cache_20241101_2.csv"
        # df_read = pd.read_csv(record_path)
        # df_read = df_read.loc[:, ~df_read.columns.str.contains('landmark')]
        # return df_read
        #record_path = r"./results/pose/pose_features_cache_20241101_2.csv"
        df_read = pd.read_csv(record_path)
        selected_columns = df_read.columns[16:]

        keywords = ["landmark"]
        # Create a regex pattern that matches any of the keywords
        pattern = "|".join(keywords)  # This will create a pattern like "AU|Face"   
        selected_columns = selected_columns[~selected_columns.str.contains(pattern, case=False)]        
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_read = df_read[selected_columns]

        df_filtered = df_read[~df_read.isnull().any(axis=1)]
        df_filtered= df_filtered.reset_index(drop=True)

        x_columns = df_filtered.columns[2:]
        trimmed_df= self.trim_data_frame(df_filtered, x_columns, prefix)
        return trimmed_df
    
    def get_pose_features_landmarks_data_frame(self):
        record_path = r"./results/pose/pose_features_cache_20241101_2.csv"
        df_read = pd.read_csv(record_path)
        selected_columns = df_read.columns[16:]

        keywords = ["landmark"]
        # Create a regex pattern that matches any of the keywords
        pattern = "|".join(keywords)  # This will create a pattern like "AU|Face"   
        selected_columns = selected_columns[selected_columns.str.contains(pattern, case=False)]        
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_read = df_read[selected_columns]

        df_filtered = df_read[~df_read.isnull().any(axis=1)]
        df_filtered= df_filtered.reset_index(drop=True)

        x_columns = df_filtered.columns[2:]
        prefix =  "pose"
        return self.trim_data_frame(df_filtered, x_columns, prefix)
    
    def get_r_ppg_df(self, record_path, df_info, prefix = "r_ppg"):
        df = self.get_r_ppg_df_from_csv(record_path, df_info, prefix)
        # Extract features using multiple strategies and plot them
        strategies = ['median']
        df_with_summary, df_summary = self.add_summary_columns(df, strategies)
        threshold = 0.3
        outlier_multiplier = 3
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary)
        #data_prepare_helper.drop_missing_data(threshold=0.1)
        # data_prepare_helper.handle_missing_values("mean", df_summary.columns[2:].tolist())
        # data_prepare_helper.process_outliers(strategy = 'remove', method = 'iqr', columns = df_speech_acoustics.columns[2:], threshold=threshold)
        #data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="log", only_skewed_columns=True, skewness_threshold=1, create_new_column=False)
        #df_prepared = data_prepare_helper.get_data_frame()
        # data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="negative exp", only_skewed_columns=True, skewness_threshold=1, create_new_column=True)
        #df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'remove', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
    
    def get_r_ppg_df_from_csv(self, record_path, df_info, prefix = "r_ppg"):
        #record_path = r"./results/remote_ppg/csv_output/results_20241108.csv"
        df_read = pd.read_csv(record_path)
        df_read['user_id'] = df_read['File_Name'].str.extract(r'PS-9_(\d{3})')[0].astype(int)
        # df_group = pd.read_csv('./resources/data/rct/rct.csv')
        df_merged = pd.merge(df_read, df_info, on='user_id', how='left')
        df_merged= df_merged.reset_index(drop=True)
        selected_columns = df_merged.columns.copy()
        selected_columns = selected_columns[1:-3]
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_trimmed = df_merged[selected_columns]
        x_columns = df_trimmed.columns[2:]
        return self.trim_data_frame(df_trimmed, x_columns, prefix)
    
    def get_mean_remote_ppg_data_frame(self):
        df= self.get_r_ppg_df_from_csv()
        df_mean = self.data_helper.calculate_mean_by_user_id(df)
        return df_mean
    
    def get_thermal_df(self, record_path, prefix =  "thermal"):

        df = self.get_thermal_df_from_csv(record_path, prefix)
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df)
        # data_prepare_helper = DataPreparationHelper(df_summary.copy())
        #data_prepare_helper.drop_missing_data(threshold=0.1)
        # data_prepare_helper.handle_missing_values("mean", df_summary.columns[2:].tolist())
        # data_prepare_helper.process_outliers(strategy = 'remove', method = 'iqr', columns = df_speech_acoustics.columns[2:], threshold=threshold)
        #data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="log", only_skewed_columns=True, skewness_threshold=1, create_new_column=False)
        #df_prepared = data_prepare_helper.get_data_frame()
        # data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="negative exp", only_skewed_columns=True, skewness_threshold=1, create_new_column=True)
        #df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'remove', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier = outlier_multiplier)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
        
    def get_thermal_df_from_csv(self, record_path, prefix =  "thermal"):

        df_read = pd.read_csv(record_path)
        selected_columns = df_read.columns[16:]
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_read = df_read[selected_columns]
        x_columns = df_read.columns[2:]
        return self.trim_data_frame(df_read, x_columns, prefix)
    
    def get_mm_llm_df(self, file_path, cap_outlier = True, outlier_multiplier= 3, prefix =  "mm_llm"):
        df_prepared = self.get_mm_llm_df_from_csv(file_path, prefix = prefix)
        threshold = 0.1
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_prepared)
        data_prepare_helper.drop_missing_data(threshold=0.1)
        data_prepare_helper.handle_missing_values("mean", df_prepared.columns[0:].tolist())
        #data_prepare_helper.process_outliers(strategy = 'remove', method = 'iqr', columns = df_with_summary.columns[2:], threshold=threshold)
        #df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier= outlier_multiplier)
        # df_prepared  = data_prepare_helper.get_data_frame()
        # data_prepare_helper.add_log_transform(df_prepared.columns[2:], only_skewed_columns=True, skewness_threshold=1)
        if cap_outlier:
            df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier= outlier_multiplier)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
    def get_mm_llm_df_from_csv(self, file_path, prefix =  "mm_llm"):
        #record_path = r"./results/mm_llm/csv_output/results_20241109_processed_and_merged.csv"
        df_read = pd.read_csv(file_path)
        selected_columns = df_read.columns
        selected_columns = selected_columns[3:-1].tolist()
        keywords = ['frame', 'Group', 'sequence']
        selected_columns = [col for col in selected_columns if not any(keyword in col for keyword in keywords)]
        selected_columns.insert(0, "user_id")
        selected_columns.insert(0, "Group")
        df_read = df_read[selected_columns]
        x_columns = df_read.columns[2:]
        return self.trim_data_frame(df_read, x_columns, prefix)
        
    def get_speech_sentiment_data_frame(self, file_path):
        #file_path = r"./results/speech_text/csv/result_speech_sentiment_20241112.csv"
        # df_read = pd.read_csv(file_path)
        # x_columns = df_read.columns[3:]
        # prefix =  "sentiment"
        # return self.process_data_frame(df_read, x_columns, prefix)
    
        df_read = pd.read_csv(file_path)
        selected_columns = df_read.columns
        selected_columns = selected_columns[17:].tolist()
        keywords = ['frame', 'Group', 'sequence']
        selected_columns = [col for col in selected_columns if not any(keyword in col for keyword in keywords)]
        prefix =  "sentiment-"
        return self.trim_data_frame(df_read, selected_columns, prefix)
    
    def get_mean_speech_sentiment_data_frame(self):
        df= self.get_speech_sentiment_data_frame()
        df_mean = self.data_helper.calculate_mean_by_user_id(df)
        return df_mean
    
    def get_speech_text_df(self, file_path, parameters: dict =  {}, prefix = "speech_text"): # do_log_transform = False, remove_outlier = False):
        df_prepared = self.get_speech_text_df_from_csv(file_path, prefix = prefix)
        df_prepared = df_prepared[df_prepared.columns[:2].append(df_prepared.columns[-28:])]
        strategies = ['median','variance','max']
        df_with_summary, df_summary = self.add_summary_columns(df_prepared, strategies)
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary)
        df_prepared = data_prepare_helper.prepare_data()
        # data_prepare_helper.drop_missing_data(threshold=0.1)
        # data_prepare_helper.handle_missing_values("mean", df_prepared.columns[0:].tolist())
        # df_prepared = data_prepare_helper.get_data_frame()
        # # data_prepare_helper.process_outliers(strategy = 'remove', method = 'iqr', columns = df_with_summary.columns[2:], threshold=threshold)      
        # # df_prepared  = data_prepare_helper.get_data_frame()
        # if parameters.get("do_log_transform") is not None:
        #     do_log_transform = parameters.get("do_log_transform")
        #     if do_log_transform:
        #         data_prepare_helper.data_transform(df_prepared.columns[2:], transformation_method="log", only_skewed_columns=True, skewness_threshold=1, create_new_column=False)
        #     #data_prepare_helper.add_log_transform(df_prepared.columns[2:], only_skewed_columns=True, skewness_threshold=1, create_new_column=False)
        # # if remove_outlier:
        # #     df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier= outlier_multiplier)
        # df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
    
    
    def get_speech_text_df_from_csv(self, file_path, prefix = "speech_text"):
        df_read = pd.read_csv(file_path)
        selected_columns = df_read.columns
        selected_columns = selected_columns[3:].tolist()
        keywords = ['frame', 'Group', 'sequence','attribute_value']
        selected_columns = [col for col in selected_columns if not any(keyword in col for keyword in keywords)]
        return self.trim_data_frame(df_read, selected_columns, prefix)
    
    def get_mean_speech_text_data_frame(self):
        df= self.get_speech_text_df_from_csv()
        df_mean = self.data_helper.calculate_mean_by_user_id(df)
        return df_mean
    
    def get_acoustics_df(self, path_target, path_baseline, prefix="acoustics"):
        df_target = self.get_acoustics_df_from_csv(begin_column=14, record_path= path_target, prefix = prefix)
        df_baseline = self.get_acoustics_df_from_csv(begin_column=3, record_path= path_baseline)
        df_baseline_median = self.remove_outliers_and_calculate_median(df_baseline, outlier_multiplier= 3)
        df_target_centered = self.remove_baseline_from_target_df(df_target, df_baseline_median, column_to_process=df_target.columns[2:])
        strategies =  ['median','variance','max']
        df_with_summary, df_summary = self.add_summary_columns(df_target_centered, strategies)
        #df_with_summary.to_csv(path_with_summary)
        #df_summary.to_csv(path_summary)
        #df_summary = pd.read_csv(path_summary)
        #df_with_summary = pd.read_csv(path_with_summary)
        #df_prepared = df_summary.copy()
        # threshold = 0.1
        # outlier_multiplier = 3
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary)
        parameter = {
            "cap_data": True,
            "outlier_multiplier": 3,
            "replace_missing_data_with_mean":True
        }
        df_prepared = data_prepare_helper.prepare_data(parameter)
        # data_prepare_helper.drop_missing_data(threshold=0.1)
        # data_prepare_helper.handle_missing_values("mean", df_prepared.columns[0:].tolist())
        # #data_prepare_helper.process_outliers(strategy = 'remove', method = 'iqr', columns = df_with_summary.columns[2:], threshold=threshold)
        # df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier= outlier_multiplier)
        # # df_prepared  = data_prepare_helper.get_data_frame()
        # # data_prepare_helper.add_log_transform(df_prepared.columns[2:], only_skewed_columns=True, skewness_threshold=1)
        # df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared

    def get_acoustics_df_from_csv(self, begin_column = None, end_column = None, record_path = "", prefix="acoustics"):
        if record_path == "":
            record_path =  os.path.join(self.resutls_location, "examples", "paper5_mmer_moral", "audio","audio_features_merged_20241219.csv")
        df_read = pd.read_csv(record_path)
        if begin_column != None and end_column != None:
            x_columns = df_read.columns[begin_column:end_column]
        elif begin_column:
            x_columns = df_read.columns[begin_column:]
        elif begin_column:
            x_columns = df_read.columns[:end_column]
        else:
            x_columns = df_read.columns

        # Define the substrings to match for removal
        removal_substrings = ["seq","part","attribute","file","Group","user_id",]
        #removal_substrings = []
        # Use a list comprehension to filter out columns containing any of the substrings
        x_columns_filtered = [col for col in x_columns if not any(substring in col for substring in removal_substrings)]
        matching_columns = [col for col in x_columns if any(sub in col for sub in removal_substrings)]
        print(f"Filtered out columns: {matching_columns}")

        df_read =  self.trim_data_frame(df_read, x_columns_filtered, prefix)

        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('seq')]
        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('user_id')]
        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('attribute')]
        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('file_size')]
        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('part_1')]
        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('part_2')]
        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('part_3')]
        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('part_4')]
        # df_filtered = df_read.loc[:, ~df_read.columns.str.contains('Group')]

        #df_filtered = df_read[~df_read.isnull().any(axis=1)]
        df_cleaned = df_read.dropna(how='all')
        df_cleaned= df_cleaned.reset_index(drop=True)
        #df_filtered= df_filtered.reset_index(drop=True)

        return df_cleaned

    def get_mean_acoustics_data_frame(self):
        df_read = self.get_acoustics_df_from_csv()
        df_mean_speech_acoustics = self.data_helper.calculate_mean_by_user_id(df_read)
        return df_mean_speech_acoustics

