import numpy as np
import pandas as pd

class FeatureProcessor:

    def __init__(self):
        pass


    def summarize_features_by_column(self, data: pd.DataFrame, strategies: list, group_by_column = "user_id", order_by_column = "seq"):
        """
        Summarize the extracted features for each user_id based on the provided strategies (e.g., mean, variance),
        ensuring the data is ordered by the 'frame' column within each group.

        Args:
        data (pd.DataFrame): Data containing features, a 'user_id' column, and a 'frame' column.
        strategies (list): List of summarization strategies to apply.

        Returns:
        pd.DataFrame: A summarized DataFrame with one row per user_id and summarized features.
        """
        # Ensure 'user_id' and 'frame' columns exist in the data
        if group_by_column not in data.columns or order_by_column not in data.columns:
            raise ValueError(f"Both '{group_by_column}' and '{order_by_column}' columns must be present in the data.")

        summarized_data = []

        # Group the data by user_id
        grouped_data = data.groupby(group_by_column)

        for user_id, group in grouped_data:
            all_features = []
            combined_feature_names = []

            # Order the group by 'frame' column
            group = group.sort_values(by=order_by_column)

            # Convert columns to numeric, setting non-numeric values to NaN
            numeric_df = group.drop(columns=[group_by_column, order_by_column]).apply(pd.to_numeric, errors='coerce')

            # Identify and keep only numeric columns
            numeric_cols = numeric_df.columns[numeric_df.notna().all()].tolist()
            pose_features = numeric_df[numeric_cols].to_numpy()
            all_feature_names = [name for name in numeric_cols]

            # Apply each summarization strategy
            for strategy in strategies:
                try:
                    if strategy == 'mean':
                        summarized_features = self.summarize_mean(pose_features)
                    if strategy == 'median':
                        summarized_features = self.summarize_median(pose_features)
                    elif strategy == 'variance':
                        summarized_features = self.summarize_variance(pose_features)
                    elif strategy == 'end_to_begin':
                        summarized_features = self.summarize_end_to_begin(pose_features, len(pose_features))
                    elif strategy == 'max':
                        summarized_features = self.summarize_max(pose_features)
                    else:
                        raise ValueError(f"Invalid strategy '{strategy}'")
                except Exception as e:
                    print(f"Error for user_id {user_id} with strategy {strategy}: {e}")
                    return None

                feature_names = self.add_strategy_prefix(strategy, all_feature_names)
                all_features.append(summarized_features)
                combined_feature_names.extend(feature_names)

            # Combine all features for this user_id into a single row
            user_summary = pd.DataFrame([np.hstack(all_features)], columns=combined_feature_names)
            user_summary[group_by_column] = user_id
            summarized_data.append(user_summary)

        # Concatenate all summarized rows into a single DataFrame
        result_df = pd.concat(summarized_data, ignore_index=True)
        return result_df


    # def summarize_features_with_data(self, data: pd.DataFrame , strategies):
    #     """
    #     Summarize the extracted features based on the provided strategies (e.g., mean, variance).

    #     Args:
    #     pose_features (np.array): Extracted pose features for each frame.
    #     strategies (list): List of summarization strategies to apply.
    #     all_feature_names (list): Original feature names.

    #     Returns:
    #     np.array: Combined summarized features.
    #     list: Updated list of feature names with strategy prefixes.
    #     """
    #     all_features = []
    #     combined_feature_names = []

    #     # Step 2: Convert columns to numeric, setting non-numeric values to NaN
    #     numeric_df = data.apply(pd.to_numeric, errors='coerce')

    #     # Step 3: Identify and keep only numeric columns
    #     numeric_cols = numeric_df.columns[numeric_df.notna().all()].tolist()
    #     pose_features = numeric_df[numeric_cols].to_numpy()
    #     all_feature_names = [name for name in numeric_cols]

    #     # Apply each summarization strategy (e.g., mean, variance) to the features
    #     for strategy in strategies:
    #         try:
    #             if strategy == 'mean':
    #                 summarized_features = self.summarize_mean(pose_features)
    #             elif strategy == 'variance':
    #                 summarized_features = self.summarize_variance(pose_features)
    #             elif strategy == 'end_to_begin':
    #                 summarized_features = self.summarize_end_to_begin(pose_features, len(pose_features))
    #             elif strategy == 'max':
    #                 summarized_features = self.summarize_max(pose_features)
    #             else:
    #                 raise ValueError(f"Invalid strategy '{strategy}'")
    #         except Exception as e:
    #             print(f"Error: {e}")
    #             return None, None   


    #         feature_names = self.add_strategy_prefix(strategy, all_feature_names)
    #         all_features.append(summarized_features)
    #         combined_feature_names.extend(feature_names)

    #     df_result = pd.DataFrame([np.hstack(all_features)], columns=combined_feature_names)
    #     return df_result
    

    def summarize_features(self, pose_features, strategies, all_feature_names):
        """
        Summarize the extracted features based on the provided strategies (e.g., mean, variance).

        Args:
        pose_features (np.array): Extracted pose features for each frame.
        strategies (list): List of summarization strategies to apply.
        all_feature_names (list): Original feature names.

        Returns:
        np.array: Combined summarized features.
        list: Updated list of feature names with strategy prefixes.
        """
        all_features = []
        combined_feature_names = []

        # Step 1: Convert pose_features to a DataFrame to check for numeric columns
        pose_df = pd.DataFrame(pose_features, columns=all_feature_names)

        # Step 2: Convert columns to numeric, setting non-numeric values to NaN
        numeric_df = pose_df.apply(pd.to_numeric, errors='coerce')

        # Step 3: Identify and keep only numeric columns
        numeric_cols = numeric_df.columns[numeric_df.notna().all()].tolist()
        pose_features = numeric_df[numeric_cols].to_numpy()
        all_feature_names = [name for name in numeric_cols]

        # Apply each summarization strategy (e.g., mean, variance) to the features
        for strategy in strategies:
            try:
                if strategy == 'mean':
                    summarized_features = self.summarize_mean(pose_features)
                elif strategy == 'variance':
                    summarized_features = self.summarize_variance(pose_features)
                elif strategy == 'end_to_begin':
                    summarized_features = self.summarize_end_to_begin(pose_features, len(pose_features))
                elif strategy == 'max':
                    summarized_features = self.summarize_max(pose_features)
                else:
                    raise ValueError(f"Invalid strategy '{strategy}'")
            except Exception as e:
                print(f"Error: {e}")
                return None, None   


            feature_names = self.add_strategy_prefix(strategy, all_feature_names)
            all_features.append(summarized_features)
            combined_feature_names.extend(feature_names)

        return np.hstack(all_features), combined_feature_names
    
    
    # Helper functions for summarization and feature names (unchanged from original code)
    def summarize_median(self, pose_features):
        return np.median(pose_features, axis=0) if pose_features.size else np.array([])
    
    def summarize_mean(self, pose_features):
        return np.mean(pose_features, axis=0) if pose_features.size else np.array([])

    def summarize_variance(self, pose_features):
        return np.var(pose_features, axis=0) if pose_features.size else np.array([])

    def summarize_end_to_begin(self, pose_features, total_frames):
        # Determine tenth_frames based on total_frames
        if total_frames > 30:
            tenth_frames = total_frames // 10
        elif total_frames >= 10:
            tenth_frames = 3
        elif total_frames >= 5:
            tenth_frames = 2
        else:
            tenth_frames = 1
        
        # Adjust first_tenth and last_tenth if total frames are larger than 5
        if total_frames > 5:
            first_tenth = pose_features[1:tenth_frames]  # From the second frame to the tenth frame
            last_tenth = pose_features[-tenth_frames:-1] # From the -tenth frame to the second-to-last frame
        else:
            first_tenth = pose_features[:tenth_frames]
            last_tenth = pose_features[-tenth_frames:]
        
        # Calculate means for the selected frames
        first_mean = np.mean(first_tenth, axis=0)
        last_mean = np.mean(last_tenth, axis=0)
        
        # Return the ratio of last_mean to first_mean, avoiding division by zero
        return last_mean / (first_mean + 1e-6)


    def summarize_max(self, pose_features):
        return np.max(np.abs(pose_features), axis=0) if pose_features.size else np.array([])

    def add_strategy_prefix(self, strategy, feature_names):
        return [f"{strategy}_{name}" for name in feature_names]
