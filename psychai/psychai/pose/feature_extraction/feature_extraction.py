import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseFeatureExtractor:
    def __init__(self, video_path, frame_rate=30, downsample_factor=1, custom_features=None):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.downsample_factor = downsample_factor
        self.custom_features = custom_features

    def process_frame(self, frame, pose):
        # Convert frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # If pose landmarks detected, extract and process them
        if results.pose_landmarks:
            frame_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(frame_landmarks)
        return None

    def process_video(self, strategies=['mean'], features_to_plot=None, use_parallel=True):
        cap = cv2.VideoCapture(self.video_path)

        pose_features = []
        total_frames = 0
        processed_frames = 0
        all_feature_names = self.get_feature_names()

        frame_index = 0
        frame_tasks = []
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            if use_parallel:
                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor() as executor:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Downsample the frame processing
                        if frame_index % self.downsample_factor == 0:
                            frame_tasks.append(executor.submit(self.process_frame, frame, pose))
                        
                        frame_index += 1
                    cap.release()

                    # Process frames as they are completed
                    for future in as_completed(frame_tasks):
                        frame_landmarks = future.result()
                        if frame_landmarks is not None:
                            # Add custom features if specified
                            if self.custom_features:
                                custom_values, custom_names = self.calculate_custom_features_row(frame_landmarks, self.custom_features)
                                frame_landmarks = np.hstack([frame_landmarks, custom_values])
                                if not hasattr(self, 'custom_names_extended'):
                                    all_feature_names.extend(custom_names)
                                    self.custom_names_extended = True

                            pose_features.append(frame_landmarks)
                            processed_frames += 1
            else:
                # Sequential processing
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Downsample the frame processing
                    if frame_index % self.downsample_factor == 0:
                        frame_landmarks = self.process_frame(frame, pose)
                        if frame_landmarks is not None:
                            # Add custom features if specified
                            if self.custom_features:
                                custom_values, custom_names = self.calculate_custom_features_row(frame_landmarks, self.custom_features)
                                frame_landmarks = np.hstack([frame_landmarks, custom_values])
                                if not hasattr(self, 'custom_names_extended'):
                                    all_feature_names.extend(custom_names)
                                    self.custom_names_extended = True

                            pose_features.append(frame_landmarks)
                            processed_frames += 1
                    
                    frame_index += 1
                cap.release()

        pose_features = np.array(pose_features)

        # Initialize variables for combined results
        all_features = []
        combined_feature_names = []
        all_feature_names_raw = all_feature_names

        # Plotting logic
        if features_to_plot is None or len(features_to_plot) == 0:
            features_to_plot = []
        elif features_to_plot == ['All']:
            features_to_plot = all_feature_names_raw

        # Apply summarization strategies
        for strategy in strategies:
            if strategy == 'mean':
                summarized_features = self.summarize_mean(pose_features)
                feature_names = self.add_strategy_prefix('mean', all_feature_names)
            elif strategy == 'variance':
                summarized_features = self.summarize_variance(pose_features)
                feature_names = self.add_strategy_prefix('variance', all_feature_names)
            elif strategy == 'end_to_begin':
                summarized_features = self.summarize_end_to_begin(pose_features, processed_frames)
                feature_names = self.add_strategy_prefix('end_to_begin', all_feature_names)
            elif strategy == 'max':
                summarized_features = self.summarize_max(pose_features)
                feature_names = self.add_strategy_prefix('max', all_feature_names)
            else:
                raise ValueError(f"Invalid strategy '{strategy}'")

            all_features.append(summarized_features)
            combined_feature_names.extend(feature_names)

        # Combine all features
        all_features = np.hstack(all_features)

        feature_indices = [all_feature_names_raw.index(f) for f in features_to_plot]

        if feature_indices:
            time_stamps = np.arange(0, processed_frames) * (self.downsample_factor / self.frame_rate)
            plt.figure(figsize=(12, 6))

            for idx in feature_indices:
                plt.plot(time_stamps, pose_features[:, idx], label=all_feature_names_raw[idx])

            plt.xlabel('Time (s)')
            plt.ylabel('Feature Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title('Pose Features Over Time')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()

        return all_features, combined_feature_names
    
    # Summarization strategies (unchanged)
    def summarize_mean(self, pose_features):
        if pose_features.size == 0:
            return np.array([])
        return np.mean(pose_features, axis=0)

    def summarize_variance(self, pose_features):
        if pose_features.size == 0:
            return np.array([])
        return np.var(pose_features, axis=0)

    def summarize_end_to_begin(self, pose_features, total_frames):
        if pose_features.size == 0:
            return np.array([])
        tenth_frames = max(10, total_frames // 10)
        first_tenth = pose_features[:tenth_frames] if pose_features.shape[0] > tenth_frames else pose_features[:]
        last_tenth = pose_features[-tenth_frames:] if pose_features.shape[0] > tenth_frames else pose_features[:]
        first_mean = np.mean(first_tenth, axis=0)
        last_mean = np.mean(last_tenth, axis=0)
        return last_mean / (first_mean + 1e-6)

    def summarize_max(self, pose_features):
        if pose_features.size == 0:
            return np.array([])
        max_abs_indices = np.argmax(np.abs(pose_features), axis=0)
        max_values = pose_features[max_abs_indices, np.arange(pose_features.shape[1])]
        return max_values

    # Feature name functions (unchanged)
    def get_feature_names(self):
        feature_names = []
        for i in range(33):
            feature_names.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        return feature_names

    def add_strategy_prefix(self, strategy, feature_names):
        return [f"{strategy}_{name}" for name in feature_names]

    def calculate_custom_features_row(self, pose_row, custom_features):
        custom_feature_values = []
        custom_feature_names = []

        for custom in custom_features:
            operation = custom[0]
            if operation == 'sum' or operation == 'difference':
                feature_a, feature_b, new_name = custom[1], custom[2], custom[3]
                a_idx = feature_a * 3
                b_idx = feature_b * 3
                a_values = pose_row[a_idx:a_idx + 3]
                b_values = pose_row[b_idx:b_idx + 3]
                if operation == 'sum':
                    result_values = a_values + b_values
                elif operation == 'difference':
                    result_values = a_values - b_values
                custom_feature_values.extend(result_values)
                custom_feature_names.extend([f"{new_name}_x", f"{new_name}_y", f"{new_name}_z"])
            elif operation == 'angle':
                point_0, point_1, point_2, new_name = custom[1], custom[2], custom[3], custom[4]
                p0 = pose_row[point_0 * 3: point_0 * 3 + 3]
                p1 = pose_row[point_1 * 3: point_1 * 3 + 3]
                p2 = pose_row[point_2 * 3: point_2 * 3 + 3]
                v1 = p0 - p1
                v2 = p2 - p1
                dot_product = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    angle = 0.0
                else:
                    cos_theta = dot_product / (norm_v1 * norm_v2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle = np.arccos(cos_theta)
                custom_feature_values.append(angle)
                custom_feature_names.append(new_name)

        return np.array(custom_feature_values), custom_feature_names


# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt

# # Initialize MediaPipe Pose and Drawing modules
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils


# # Class to handle multiple strategies and custom feature calculation
# class PoseFeatureExtractor:
#     def __init__(self, video_path, frame_rate=30, downsample_factor=1, custom_features=None):
#         self.video_path = video_path
#         self.frame_rate = frame_rate  # Added frame rate
#         self.downsample_factor = downsample_factor  # Downsampling factor
#         self.custom_features = custom_features

#     def process_video(self, strategies=['mean'], features_to_plot=None):
#         cap = cv2.VideoCapture(self.video_path)
        
#         pose_features = []
#         total_frames = 0
#         processed_frames = 0
#         all_feature_names = self.get_feature_names()

#         frame_index = 0  # To track the actual frame number
#         with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Process only every 'd' frames (down-sampling factor)
#                 if frame_index % self.downsample_factor == 0:
#                     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     results = pose.process(image_rgb)

#                     if results.pose_landmarks:
#                         frame_landmarks = []
#                         for landmark in results.pose_landmarks.landmark:
#                             frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

#                         # Convert the frame landmarks to NumPy array
#                         frame_landmarks = np.array(frame_landmarks)

#                         # Add custom features if specified
#                         if self.custom_features:
#                             custom_values, custom_names = self.calculate_custom_features_row(frame_landmarks, self.custom_features)
#                             frame_landmarks = np.hstack([frame_landmarks, custom_values])
#                             # Extend the base feature names with custom names only once
#                             if not hasattr(self, 'custom_names_extended'):
#                                 all_feature_names.extend(custom_names)
#                                 self.custom_names_extended = True  # Flag to ensure it runs only once

#                         pose_features.append(frame_landmarks)
#                         processed_frames += 1  # Count the number of processed frames

#                 frame_index += 1  # Increment the frame index

#             cap.release()
#             cv2.destroyAllWindows()

#         pose_features = np.array(pose_features)

#         # Initialize variables for combined results
#         all_features = []
#         combined_feature_names = []
#         all_feature_names_raw = all_feature_names

#         # Plotting logic
#         if features_to_plot is None or len(features_to_plot) == 0:
#             features_to_plot = []
#         elif features_to_plot == ['All']:
#             features_to_plot = all_feature_names_raw  # Plot all features if list is empty

#         # Apply multiple strategies to the entire pose data (including custom features)
#         for strategy in strategies:
#             if strategy == 'mean':
#                 summarized_features = self.summarize_mean(pose_features)
#                 feature_names = self.add_strategy_prefix('mean', all_feature_names)
#             elif strategy == 'variance':
#                 summarized_features = self.summarize_variance(pose_features)
#                 feature_names = self.add_strategy_prefix('variance', all_feature_names)
#             elif strategy == 'end_to_begin':
#                 summarized_features = self.summarize_end_to_begin(pose_features, processed_frames)
#                 feature_names = self.add_strategy_prefix('end_to_begin', all_feature_names)
#             elif strategy == 'max':
#                 summarized_features = self.summarize_max(pose_features)
#                 feature_names = self.add_strategy_prefix('max', all_feature_names)
#             else:
#                 raise ValueError(f"Invalid strategy '{strategy}'. Choose from 'mean', 'variance', 'end-to-begin', or 'max'.")

#             # Append the summarized features and feature names
#             all_features.append(summarized_features)
#             combined_feature_names.extend(feature_names)

#         # Combine all features into a single array
#         all_features = np.hstack(all_features)


#         feature_indices = [all_feature_names_raw.index(f) for f in features_to_plot]

#         if feature_indices:
#             # Adjust the time stamps considering the downsampling factor
#             time_stamps = np.arange(0, processed_frames) * (self.downsample_factor / self.frame_rate)

#             # Set the figure size (e.g., 12 inches wide, 6 inches tall)
#             plt.figure(figsize=(12, 6))  # Adjust the width (first value) and height (second value)

#             for idx in feature_indices:
#                 plt.plot(time_stamps, pose_features[:, idx], label=all_feature_names_raw[idx])

#             plt.xlabel('Time (s)')
#             plt.ylabel('Feature Value')
            
#             # Moving the legend outside the plot
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Moves the legend to the right of the plot
#             plt.title('Pose Features Over Time')
            
#             # Adjust the layout to make space for the legend
#             plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin
#             plt.show()

#         return all_features, combined_feature_names
    
#     # Summarization strategies (unchanged)
#     def summarize_mean(self, pose_features):
#         if pose_features.size == 0:
#             return np.array([])
#         return np.mean(pose_features, axis=0)

#     def summarize_variance(self, pose_features):
#         if pose_features.size == 0:
#             return np.array([])
#         return np.var(pose_features, axis=0)

#     def summarize_end_to_begin(self, pose_features, total_frames):
#         if pose_features.size == 0:
#             return np.array([])

#         tenth_frames = max(10, total_frames // 10)
        
#         first_tenth = pose_features[:tenth_frames] if pose_features.shape[0] > tenth_frames else pose_features[:]
#         last_tenth = pose_features[-tenth_frames:] if pose_features.shape[0] > tenth_frames else pose_features[:]
        
#         first_mean = np.mean(first_tenth, axis=0)
#         last_mean = np.mean(last_tenth, axis=0)

#         ratio = last_mean / (first_mean + 1e-6)  # Add epsilon to avoid division by zero
#         return ratio

#     def summarize_max(self, pose_features):
#         if pose_features.size == 0:
#             return np.array([])
        
#         max_abs_indices = np.argmax(np.abs(pose_features), axis=0)
#         max_values = pose_features[max_abs_indices, np.arange(pose_features.shape[1])]
        
#         return max_values

#     # Function to generate feature names (unchanged)
#     def get_feature_names(self):
#         feature_names = []
#         for i in range(33):  # 33 pose landmarks
#             feature_names.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
#         return feature_names

#     # Function to add strategy name as a prefix to feature names (unchanged)
#     def add_strategy_prefix(self, strategy, feature_names):
#         return [f"{strategy}_{name}" for name in feature_names]

#     # Function to calculate custom features based on specified operations
#     def calculate_custom_features_row(self, pose_row, custom_features):
#         custom_feature_values = []
#         custom_feature_names = []

#         for custom in custom_features:
#             operation = custom[0]

#             if operation == 'sum' or operation == 'difference':
#                 feature_a, feature_b, new_name = custom[1], custom[2], custom[3]

#                 # Extract the x, y, z values of feature_a and feature_b
#                 a_idx = feature_a * 3
#                 b_idx = feature_b * 3

#                 a_values = pose_row[a_idx:a_idx + 3]
#                 b_values = pose_row[b_idx:b_idx + 3]

#                 # Perform sum or difference operation on x, y, z values
#                 if operation == 'sum':
#                     result_values = a_values + b_values
#                 elif operation == 'difference':
#                     result_values = a_values - b_values
#                 else:
#                     raise ValueError("Invalid operation. Use 'sum' or 'difference'.")

#                 # Append the result and the new feature name
#                 custom_feature_values.extend(result_values)
#                 custom_feature_names.extend([f"{new_name}_x", f"{new_name}_y", f"{new_name}_z"])

#             elif operation == 'angle':
#                 point_0, point_1, point_2, new_name = custom[1], custom[2], custom[3], custom[4]

#                 # Extract the x, y, z values of points 0, 1, and 2
#                 p0 = pose_row[point_0 * 3: point_0 * 3 + 3]
#                 p1 = pose_row[point_1 * 3: point_1 * 3 + 3]
#                 p2 = pose_row[point_2 * 3: point_2 * 3 + 3]

#                 # Vectors between points
#                 v1 = p0 - p1  # Vector from point 1 to point 0
#                 v2 = p2 - p1  # Vector from point 1 to point 2

#                 # Calculate the angle between v1 and v2 using the dot product
#                 dot_product = np.dot(v1, v2)
#                 norm_v1 = np.linalg.norm(v1)
#                 norm_v2 = np.linalg.norm(v2)

#                 # Avoid division by zero in case of very small vectors
#                 if norm_v1 == 0 or norm_v2 == 0:
#                     angle = 0.0
#                 else:
#                     cos_theta = dot_product / (norm_v1 * norm_v2)
#                     # Clamp cos_theta to the valid range [-1, 1] to avoid numerical issues
#                     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#                     angle = np.arccos(cos_theta)  # Angle in radians

#                 # Append the angle value (in radians) and feature name
#                 custom_feature_values.append(angle)
#                 custom_feature_names.append(new_name)

#         return np.array(custom_feature_values), custom_feature_names


# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Initialize MediaPipe Pose and Drawing modules
# # mp_pose = mp.solutions.pose
# # mp_drawing = mp.solutions.drawing_utils

# # # Summarization strategies (unchanged)
# # def summarize_mean(pose_features):
# #     if pose_features.size == 0:
# #         return np.array([])
# #     return np.mean(pose_features, axis=0)

# # def summarize_variance(pose_features):
# #     if pose_features.size == 0:
# #         return np.array([])
# #     return np.var(pose_features, axis=0)

# # def summarize_end_to_begin(pose_features, total_frames):
# #     if pose_features.size == 0:
# #         return np.array([])

# #     tenth_frames = max(10, total_frames // 10)
    
# #     first_tenth = pose_features[:tenth_frames] if pose_features.shape[0] > tenth_frames else pose_features[:]
# #     last_tenth = pose_features[-tenth_frames:] if pose_features.shape[0] > tenth_frames else pose_features[:]
    
# #     first_mean = np.mean(first_tenth, axis=0)
# #     last_mean = np.mean(last_tenth, axis=0)

# #     ratio = last_mean / (first_mean + 1e-6)  # Add epsilon to avoid division by zero
# #     return ratio

# # def summarize_max(pose_features):
# #     if pose_features.size == 0:
# #         return np.array([])
    
# #     max_abs_indices = np.argmax(np.abs(pose_features), axis=0)
# #     max_values = pose_features[max_abs_indices, np.arange(pose_features.shape[1])]
    
# #     return max_values

# # # Function to generate feature names (unchanged)
# # def get_feature_names():
# #     feature_names = []
# #     for i in range(33):  # 33 pose landmarks
# #         feature_names.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
# #     return feature_names

# # # Function to add strategy name as a prefix to feature names (unchanged)
# # def add_strategy_prefix(strategy, feature_names):
# #     return [f"{strategy}_{name}" for name in feature_names]

# # # Function to calculate custom features based on specified operations
# # def calculate_custom_features_row(pose_row, custom_features):
# #     custom_feature_values = []
# #     custom_feature_names = []

# #     for custom in custom_features:
# #         operation = custom[0]

# #         if operation == 'sum' or operation == 'difference':
# #             feature_a, feature_b, new_name = custom[1], custom[2], custom[3]

# #             # Extract the x, y, z values of feature_a and feature_b
# #             a_idx = feature_a * 3
# #             b_idx = feature_b * 3

# #             a_values = pose_row[a_idx:a_idx + 3]
# #             b_values = pose_row[b_idx:b_idx + 3]

# #             # Perform sum or difference operation on x, y, z values
# #             if operation == 'sum':
# #                 result_values = a_values + b_values
# #             elif operation == 'difference':
# #                 result_values = a_values - b_values
# #             else:
# #                 raise ValueError("Invalid operation. Use 'sum' or 'difference'.")

# #             # Append the result and the new feature name
# #             custom_feature_values.extend(result_values)
# #             custom_feature_names.extend([f"{new_name}_x", f"{new_name}_y", f"{new_name}_z"])

# #         elif operation == 'angle':
# #             point_0, point_1, point_2, new_name = custom[1], custom[2], custom[3], custom[4]

# #             # Extract the x, y, z values of points 0, 1, and 2
# #             p0 = pose_row[point_0 * 3: point_0 * 3 + 3]
# #             p1 = pose_row[point_1 * 3: point_1 * 3 + 3]
# #             p2 = pose_row[point_2 * 3: point_2 * 3 + 3]

# #             # Vectors between points
# #             v1 = p0 - p1  # Vector from point 1 to point 0
# #             v2 = p2 - p1  # Vector from point 1 to point 2

# #             # Calculate the angle between v1 and v2 using the dot product
# #             dot_product = np.dot(v1, v2)
# #             norm_v1 = np.linalg.norm(v1)
# #             norm_v2 = np.linalg.norm(v2)

# #             # Avoid division by zero in case of very small vectors
# #             if norm_v1 == 0 or norm_v2 == 0:
# #                 angle = 0.0
# #             else:
# #                 cos_theta = dot_product / (norm_v1 * norm_v2)
# #                 # Clamp cos_theta to the valid range [-1, 1] to avoid numerical issues
# #                 cos_theta = np.clip(cos_theta, -1.0, 1.0)
# #                 angle = np.arccos(cos_theta)  # Angle in radians

# #             # Append the angle value (in radians) and feature name
# #             custom_feature_values.append(angle)
# #             custom_feature_names.append(new_name)

# #     return np.array(custom_feature_values), custom_feature_names

# # # Class to handle multiple strategies and custom feature calculation
# # class PoseFeatureExtractor:
# #     def __init__(self, video_path, frame_rate=30, downsample_factor=1, custom_features=None):
# #         self.video_path = video_path
# #         self.frame_rate = frame_rate  # Added frame rate
# #         self.downsample_factor = downsample_factor  # Downsampling factor
# #         self.custom_features = custom_features

# #     def process_video(self, strategies=['mean'], features_to_plot=None):
# #         cap = cv2.VideoCapture(self.video_path)
        
# #         pose_features = []
# #         total_frames = 0
# #         processed_frames = 0
# #         all_feature_names = get_feature_names()

# #         frame_index = 0  # To track the actual frame number
# #         with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# #             while cap.isOpened():
# #                 ret, frame = cap.read()
# #                 if not ret:
# #                     break

# #                 # Process only every 'd' frames (down-sampling factor)
# #                 if frame_index % self.downsample_factor == 0:
# #                     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #                     results = pose.process(image_rgb)

# #                     if results.pose_landmarks:
# #                         frame_landmarks = []
# #                         for landmark in results.pose_landmarks.landmark:
# #                             frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

# #                         # Convert the frame landmarks to NumPy array
# #                         frame_landmarks = np.array(frame_landmarks)

# #                         # Add custom features if specified
# #                         if self.custom_features:
# #                             custom_values, custom_names = calculate_custom_features_row(frame_landmarks, self.custom_features)
# #                             frame_landmarks = np.hstack([frame_landmarks, custom_values])
# #                             # Extend the base feature names with custom names only once
# #                             if not hasattr(self, 'custom_names_extended'):
# #                                 all_feature_names.extend(custom_names)
# #                                 self.custom_names_extended = True  # Flag to ensure it runs only once

# #                         pose_features.append(frame_landmarks)
# #                         processed_frames += 1  # Count the number of processed frames

# #                 frame_index += 1  # Increment the frame index

# #             cap.release()
# #             cv2.destroyAllWindows()

# #         pose_features = np.array(pose_features)

# #         # Initialize variables for combined results
# #         all_features = []
# #         combined_feature_names = []
# #         all_feature_names_raw = all_feature_names

# #         # Plotting logic
# #         if features_to_plot is None or len(features_to_plot) == 0:
# #             features_to_plot = []
# #         elif features_to_plot == ['All']:
# #             features_to_plot = all_feature_names_raw  # Plot all features if list is empty

# #         # Apply multiple strategies to the entire pose data (including custom features)
# #         for strategy in strategies:
# #             if strategy == 'mean':
# #                 summarized_features = summarize_mean(pose_features)
# #                 feature_names = add_strategy_prefix('mean', all_feature_names)
# #             elif strategy == 'variance':
# #                 summarized_features = summarize_variance(pose_features)
# #                 feature_names = add_strategy_prefix('variance', all_feature_names)
# #             elif strategy == 'end_to_begin':
# #                 summarized_features = summarize_end_to_begin(pose_features, processed_frames)
# #                 feature_names = add_strategy_prefix('end_to_begin', all_feature_names)
# #             elif strategy == 'max':
# #                 summarized_features = summarize_max(pose_features)
# #                 feature_names = add_strategy_prefix('max', all_feature_names)
# #             else:
# #                 raise ValueError(f"Invalid strategy '{strategy}'. Choose from 'mean', 'variance', 'end-to-begin', or 'max'.")

# #             # Append the summarized features and feature names
# #             all_features.append(summarized_features)
# #             combined_feature_names.extend(feature_names)

# #         # Combine all features into a single array
# #         all_features = np.hstack(all_features)


# #         feature_indices = [all_feature_names_raw.index(f) for f in features_to_plot]

# #         if feature_indices:
# #             # Adjust the time stamps considering the downsampling factor
# #             time_stamps = np.arange(0, processed_frames) * (self.downsample_factor / self.frame_rate)

# #             # Set the figure size (e.g., 12 inches wide, 6 inches tall)
# #             plt.figure(figsize=(12, 6))  # Adjust the width (first value) and height (second value)

# #             for idx in feature_indices:
# #                 plt.plot(time_stamps, pose_features[:, idx], label=all_feature_names_raw[idx])

# #             plt.xlabel('Time (s)')
# #             plt.ylabel('Feature Value')
            
# #             # Moving the legend outside the plot
# #             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Moves the legend to the right of the plot
# #             plt.title('Pose Features Over Time')
            
# #             # Adjust the layout to make space for the legend
# #             plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin
# #             plt.show()

# #         return all_features, combined_feature_names
