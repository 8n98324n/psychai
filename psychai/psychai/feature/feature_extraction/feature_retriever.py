import os
import numpy as np
import pickle
from tqdm.notebook import tqdm
import warnings
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


class FeatureRetriever:
    def __init__(self, df_user_info, feature_extractors= None, summarized_feature_extractors= None, strategies = None, cache_file_path=""):
        """
        Initializes the FeatureProcessor object with a DataFrame, feature extraction functions, 
        and an optional cache file for saving/loading processed features.

        Args:
            df (pd.DataFrame): The DataFrame containing a 'files' column with paths or identifiers 
                               for the data items (e.g., audio files, images, text files).
            feature_extractors (list of callable): A list of functions that will be used to extract 
                                                   features from the data items.
            cache_file_path (str): Path to a cache file to store or load the DataFrame with extracted features.
                                   If no cache file path is provided, no caching will be done.
        """
        self.df_user_info = df_user_info  # The DataFrame that holds file paths or data identifiers
        self.merged = None
        self.feature_extractors = feature_extractors  # List of feature extraction functions
        if len(cache_file_path) > 0:
            self.allow_saving_cache = True
            self.cache_file_path = cache_file_path  # Cache file path for storing processed data
        else:
            self.allow_saving_cache = False
        # self.cache_df = self.load_cache()  # Load cached data if available
        self.summarized_feature_extractors = summarized_feature_extractors
        self.strategies = strategies
        

    def load_cache(self, cache_file_path):
        """
        Loads the cached DataFrame from the cache file if it exists.

        Returns:
            pd.DataFrame: The cached DataFrame with previously extracted features, 
                          or an empty DataFrame if no cache exists.
        """
        if self.allow_saving_cache:
            if os.path.isfile(cache_file_path):
                with open(cache_file_path, 'rb') as handle:
                    return pickle.load(handle)

        return pd.DataFrame()  # Return an empty DataFrame if no cache exists

    # def save_cache(self):
    #     """
    #     Saves the current DataFrame (with extracted features) to the cache file.
    #     """
    #     with open(self.cache_file_path, 'wb') as handle:
    #         pickle.dump(self.df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_cache(self, df, cache_file_path):
        """
        Saves the current DataFrame (with extracted features) to the cache file.
        
        This function ensures that the directory for the cache file exists.
        If the directory doesn't exist, it is created.
        """
        if self.allow_saving_cache:
            # Ensure the directory exists
            directory = os.path.dirname(cache_file_path)
            
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesn't exist
                print(f"Directory '{directory}' created.")

            # Save the DataFrame to the cache file using pickle
            with open(cache_file_path, 'wb') as handle:
                pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Cache file saved at '{cache_file_path}'.")


    def process_item(self, item):
            
        if self.feature_extractors is not None:
            for feature_extractor in self.feature_extractors:
                # Extract features and names (assuming the function returns a DataFrame)
                features_df = feature_extractor(item)

    def process_item(self, item):


        if self.feature_extractors is not None:
            # Initialize a list to collect individual feature DataFrames
            feature_dfs = []

            # Apply each feature extraction function to the item
            for feature_extractor in self.feature_extractors:
                # Extract features as a DataFrame (assumed to return a DataFrame)
                features_df = feature_extractor(item)

                if isinstance(features_df, pd.DataFrame):
                    # If the feature extractor returns a DataFrame, append it to the list
                    feature_dfs.append(features_df)
                else:
                    print(f"Warning: Feature extractor did not return a DataFrame for item {item}")

            # Concatenate all DataFrames along columns (axis=1) to form a single row
            if feature_dfs:
                # Concatenate along columns (axis=1) to create a one-row DataFrame
                combined_features_df = pd.concat(feature_dfs, axis=1)

                # add the 'item' (file path) as the first column if needed
                combined_features_df.insert(0, 'files', item)

                return combined_features_df
            else:
                print(f"No features extracted for item {item}")
                return None
        else:
            print("No feature extractors available.")
            return None


    def extract_features(self, override=False):
        """
        Extracts features from all items in the DataFrame by applying the provided 
        feature extraction functions in parallel.

        Returns:
            pd.DataFrame: A DataFrame with the original data and the newly extracted features.
        """
        df_cache = pd.DataFrame()
        # Check for cache and update it with new features
        if override:
             # Create a set of files already in the cache to skip
            cached_files = set()
            items_to_process = [item for item in self.df['files']]

            print(f"Processing {len(items_to_process)} items.")

            df_results = pd.DataFrame()
                
            with ThreadPoolExecutor() as executor:
                # Submit tasks to the thread pool for only uncached items
                futures = {executor.submit(self.process_item, item): item for item in items_to_process}

                # Track the progress using tqdm, and collect results as they are completed
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                    try:
                        # Process results and collect them into a DataFrame
                        result_df = future.result()
                        df_results = pd.concat([df_results, result_df], ignore_index=True)
                    except Exception as e:
                        print(f"Error processing item {futures[future]}: {e}")

            # Merge the new features into the original DataFrame on the 'files' column
            if df_results.size > 0:
                self.merged = pd.merge(self.df_user_info, df_results, on='files', how='left')
                # df_cache = self.df.copy()  # Update the cache with the entire DataFrame
                # Save the updated cache to the file
                self.save_cache(df_results, self.cache_file_path)           
        else:
            df_cache = self.load_cache(self.cache_file_path)
            print("Loading from cache...")
            # Create a set of files already in the cache to skip
            cached_files = set(df_cache['files']) if not df_cache.empty else set()
            items_to_process = [item for item in self.df_user_info['files'] if item not in cached_files]

            if not items_to_process:
                print("All items are already cached. No new processing required.")
                self.merged = pd.merge(self.df_user_info, df_cache, on='files', how='left')
                return self.merged

            print(f"Processing {len(items_to_process)} new items out of {len(self.df_user_info['files'])} total items.")

            # Use ThreadPoolExecutor to process multiple files in parallel, speeding up feature extraction
            if df_cache.empty:
                df_results = pd.DataFrame()
            else:
                df_results = df_cache
                
            with ThreadPoolExecutor() as executor:
                # Submit tasks to the thread pool for only uncached items
                futures = {executor.submit(self.process_item, item): item for item in items_to_process}

                # Track the progress using tqdm, and collect results as they are completed
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                    try:
                        # Process results and collect them into a DataFrame
                        result_df = future.result()
                        df_results = pd.concat([df_results, result_df], ignore_index=True)
                    except Exception as e:
                        print(f"Error processing item {futures[future]}: {e}")

            # Merge the new features into the original DataFrame on the 'files' column
            if df_results.size > 0:
                self.merged = pd.merge(self.df_user_info, df_results, on='files', how='left')
                # self.df = pd.merge(self.df, df_results, on='files', how='left')
                # df_cache = self.df.copy()  # Update the cache with the entire DataFrame

                # Save only the feature part of the data frame
                self.save_cache(df_results, self.cache_file_path)     
                
        return self.merged
