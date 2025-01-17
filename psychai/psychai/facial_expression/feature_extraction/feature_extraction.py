# import numpy as np
# import re
# import pandas as pd
# import psychai.feature.feature_extraction.feature_processor
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# class FacialExpresionFeatureExtractor:
    
#     def __init__(self,
#                     landmark_model="mobilefacenet",
#                     au_model='xgb',
#                     emotion_model="resmasknet",
#                     identity_model="facenet",
#                     device="cpu"
#                     ):

#         self.fex_detector  = Detector(
#             landmark_model=landmark_model,
#             au_model=au_model,
#             emotion_model=emotion_model,
#             identity_model=identity_model,
#             device=device,
#         )

#     def check_path_or_create(self, path):
#         # Check if the path exists
#         if not os.path.exists(path):
#             # If it does not exist, create the directory
#             os.makedirs(path)
#         return path       

#     def process_video_folder(self, folder_path, sampling_time=None):

#         files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#         # loop over the images
#         index = 1
#         for file in files:
#             print("[INFO] Processing video: {}/{}".format(index, len(files)))
#             self.process_video(file, sampling_time)
#             index = index+1
#         print('finished')

#     def process_image(self, file_path, show_frame_images = False, face_detection_threshold = 0.95, add_distance = False):

#         result = self.fex_detector.detect(file_path, face_detection_threshold=face_detection_threshold )
#         df = pd.DataFrame(result)
#         df = df.drop(columns=['input'])
#         if show_frame_images:
#             figs = result.plot_detections(poses=True)

#         if add_distance:
#             df = self.add_distance_measures(df)

#         # remove identity
#         df = df.loc[:, ~df.columns.str.contains('Identity')]
            
#         return np.array(df), df.columns.tolist()

#     def add_distance_measures(self, df_facial_features):
        
#         # Function to calculate Euclidean distance
#         def euclidean_distance(x1, y1, x2, y2):
#             return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#         # Initialize a new DataFrame to store distances
#         # distance_df = pd.DataFrame(index=df_facial_features.index)

#         # Extract columns for x and y coordinates
#         x_cols = [col for col in df_facial_features.columns if re.match(r'x_\d+', col)]
#         y_cols = [col for col in df_facial_features.columns if re.match(r'y_\d+', col)]

#         # Sort columns to ensure corresponding x and y pairs are aligned
#         x_cols.sort()
#         y_cols.sort()

#         # Get the number of points
#         num_points = len(x_cols)

#         # Loop through each pair of points and calculate the distance
#         for i in range(num_points):
#             for j in range(i + 1, num_points):
#                 x1 = df_facial_features[x_cols[i]]
#                 y1 = df_facial_features[y_cols[i]]
#                 x2 = df_facial_features[x_cols[j]]
#                 y2 = df_facial_features[y_cols[j]]

#                 distance = euclidean_distance(x1, y1, x2, y2)
#                 df_facial_features[f'distance_{i}_{j}'] = distance

#         return df_facial_features

#     def get_frame_rate(self, file_path):
#         # Open the video file
#         video = cv2.VideoCapture(file_path)
        
#         # Retrieve the frame rate
#         frame_rate = video.get(cv2.CAP_PROP_FPS)
        
#         # Release the video file
#         video.release()
        
#         # Return the frame rate
#         return frame_rate

#     def get_total_frames(self, video_path):
#             """
#             Get the total number of frames in a video using OpenCV.

#             Parameters:
#                 video_path (str): Path to the video file.

#             Returns:
#                 int: Total number of frames in the video.
#             """
#             # Open the video file
#             cap = cv2.VideoCapture(video_path)

#             if not cap.isOpened():
#                 raise ValueError(f"Unable to open video file: {video_path}")

#             # Get the total frame count
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#             # Release the video capture object
#             cap.release()

#             return total_frames

#     def process_video(self, file_path, sampling_frames = 1800, sampling_time=None, show_frame_images= False):
#         """
#         Processes a video file by extracting features from every nth frame and stores the features in a DataFrame.
        
#         Args:
#             video_path (str): The path to the video file.
#             skip_n_frames (int): The number of frames to skip before processing the next frame.
#             features_extractor (object): The object containing the process_image method to extract features from an image.
        
#         Returns:
#             pd.DataFrame: A DataFrame containing features extracted from each processed frame.
#         """

#         if sampling_time is not None:
#             # Convert sampling_time to skip_frames
#             frame_rate = self.get_frame_rate(file_path)
#             sampling_frames = int(sampling_time * frame_rate)            
        
#         # # Default to a sampling time equivalent to skip_frames=900 if not provided
#         # if sampling_time is None and sampling_frame is None:
#         #     sampling_time = 30

#         total_frames = self.get_total_frames(file_path)

#         print(f"skip frames {sampling_frames}/{total_frames}")


#         # Open the video file
#         cap = cv2.VideoCapture(file_path)

#         if not cap.isOpened():
#             print(f"Error: Unable to open video file {file_path}")
#             return None

#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
#         frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get frame rate

#         print(f"Processing video with {frame_count} frames at {frame_rate} FPS. Skip {sampling_frames} frames for sampling.")

#         # Initialize an empty list to store the extracted features for each frame
#         features_list = []
#         feature_names = None

#         # Iterate through the frames of the video
#         for frame_idx in range(frame_count):
#             ret, frame = cap.read()

#             if not ret:
#                 print(f"Error: Failed to read frame {frame_idx}")
#                 break

#             # Skip the first n frames (skip frames by checking if frame_idx is divisible by skip_n_frames)
#             if frame_idx % sampling_frames != 0:
#                 continue

#             # Get the file name without extension
#             file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]


#             # Create a temporary file to save the frame as an image
#             temp_filename = f"temp_{file_name_without_extension}_frame_{frame_idx}.jpg"
#             cv2.imwrite(temp_filename, frame)  # Save the frame as an image

#             # Call the feature extractor function on the image file
#             features, feature_names = self.process_image(temp_filename, show_frame_images= show_frame_images, add_distance=True)

#             #extended_features = np.insert(features[0], 0, frame_idx)
#             # Append the features to the list
#             frame_index = feature_names.index('frame')
#             features[0][frame_index]=  frame_idx
#             features_list.append(features[0])

#             # Optionally, delete the temporary image after processing
#             os.remove(temp_filename)

#         # Release the video capture object
#         cap.release()

#         # Convert the list of features to a DataFrame, with the feature names as columns
#         if feature_names is not None:
#             #feature_names = ["frame"] + feature_names
#             df_features = pd.DataFrame(features_list, columns=feature_names)
#             print(f"Extracted features for {len(features_list)} frames.")
#             return df_features
#         else:
#             print("No features extracted.")
#             return None
    
#     def summarized_process_folder(self,input_file_fullname):

#         # Extract features using multiple strategies and plot them
#         strategies = ['mean', 'variance', 'max', 'end_to_begin']
        
#         try:
#             features, feature_names = self.process_video(input_file_fullname, sampling_time=30, show_frame_images = False)
#         except Exception as e:
#             print(f"Error: {e}")
#             return None, None   

#         # Summarize features using the specified strategies
#         processor = psychai.feature.feature_extraction.feature_processor.FeatureProcessor()
#         #processor = FeatureProcessor()
#         summarized_features, summarized_feature_names = processor.summarize_features(features, strategies, feature_names)

#         return summarized_features, summarized_feature_names       


import numpy as np
import re
import pandas as pd
import psychai.feature.feature_extraction.feature_processor
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
# Import the `Detector` class from the `feat.detector` module, which is specifically designed for face detection.
# This class helps in detecting faces within images, which can be used in further facial feature extraction.
from feat import Detector

class FacialExpresionFeatureExtractor:
    
    def __init__(self,
                    landmark_model="mobilefacenet",
                    au_model='xgb',
                    emotion_model="resmasknet",
                    identity_model="facenet",
                    device="cpu"
                    ):

        self.fex_detector  = Detector(
            landmark_model=landmark_model,
            au_model=au_model,
            emotion_model=emotion_model,
            identity_model=identity_model,
            device=device,
        )

    def check_path_or_create(self, path):
        # Check if the path exists
        if not os.path.exists(path):
            # If it does not exist, create the directory
            os.makedirs(path)
        return path       

    def process_video_folder(self, folder_path, sampling_time=None):

        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # loop over the images
        index = 1
        for file in files:
            print("[INFO] Processing video: {}/{}".format(index, len(files)))
            self.process_video(file, sampling_time)
            index = index+1
        print('finished')

    def process_image(self, file_path, show_frame_images = False, face_detection_threshold = 0.95, add_distance = False):

        result = self.fex_detector.detect(file_path, face_detection_threshold=face_detection_threshold )
        df = pd.DataFrame(result)
        df = df.drop(columns=['input'])
        if show_frame_images:
            figs = result.plot_detections(poses=True)

        if add_distance:
            df = self.add_distance_measures(df)

        # remove identity
        df = df.loc[:, ~df.columns.str.contains('Identity')]
            
        return np.array(df), df.columns.tolist()

    def add_distance_measures(self, df_facial_features):
        
        # Function to calculate Euclidean distance
        def euclidean_distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Initialize a new DataFrame to store distances
        # distance_df = pd.DataFrame(index=df_facial_features.index)

        # Extract columns for x and y coordinates
        x_cols = [col for col in df_facial_features.columns if re.match(r'x_\d+', col)]
        y_cols = [col for col in df_facial_features.columns if re.match(r'y_\d+', col)]

        # Sort columns to ensure corresponding x and y pairs are aligned
        x_cols.sort()
        y_cols.sort()

        # Get the number of points
        num_points = len(x_cols)

        # Loop through each pair of points and calculate the distance
        for i in range(num_points):
            for j in range(i + 1, num_points):
                x1 = df_facial_features[x_cols[i]]
                y1 = df_facial_features[y_cols[i]]
                x2 = df_facial_features[x_cols[j]]
                y2 = df_facial_features[y_cols[j]]

                distance = euclidean_distance(x1, y1, x2, y2)
                df_facial_features[f'distance_{i}_{j}'] = distance

        return df_facial_features

    def get_frame_rate(self, file_path):
        # Open the video file
        video = cv2.VideoCapture(file_path)
        
        # Retrieve the frame rate
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        
        # Release the video file
        video.release()
        
        # Return the frame rate
        return frame_rate

    def get_total_frames(self, video_path):
            """
            Get the total number of frames in a video using OpenCV.

            Parameters:
                video_path (str): Path to the video file.

            Returns:
                int: Total number of frames in the video.
            """
            # Open the video file
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Unable to open video file: {video_path}")

            # Get the total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Release the video capture object
            cap.release()

            return total_frames

    def process_video(self, file_path, sampling_frames = 1800, sampling_time=None, show_frame_images= False):
        """
        Processes a video file by extracting features from every nth frame and stores the features in a DataFrame.
        
        Args:
            video_path (str): The path to the video file.
            skip_n_frames (int): The number of frames to skip before processing the next frame.
            features_extractor (object): The object containing the process_image method to extract features from an image.
        
        Returns:
            pd.DataFrame: A DataFrame containing features extracted from each processed frame.
        """

        if sampling_time is not None:
            # Convert sampling_time to skip_frames
            frame_rate = self.get_frame_rate(file_path)
            sampling_frames = int(sampling_time * frame_rate)            
        
        # # Default to a sampling time equivalent to skip_frames=900 if not provided
        # if sampling_time is None and sampling_frame is None:
        #     sampling_time = 30

        total_frames = self.get_total_frames(file_path)

        print(f"skip frames {sampling_frames}/{total_frames}")


        # Open the video file
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video file {file_path}")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get frame rate

        print(f"Processing video with {frame_count} frames at {frame_rate} FPS. Skip {sampling_frames} frames for sampling.")

        # Initialize an empty list to store the extracted features for each frame
        features_list = []
        feature_names = None

        # Iterate through the frames of the video
        for frame_idx in range(frame_count):
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Failed to read frame {frame_idx}")
                break

            # Skip the first n frames (skip frames by checking if frame_idx is divisible by skip_n_frames)
            if frame_idx % sampling_frames != 0:
                continue

            # Get the file name without extension
            file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]


            # Create a temporary file to save the frame as an image
            temp_filename = f"temp_{file_name_without_extension}_frame_{frame_idx}.jpg"
            cv2.imwrite(temp_filename, frame)  # Save the frame as an image

            # Call the feature extractor function on the image file
            features, feature_names = self.process_image(temp_filename, show_frame_images= show_frame_images, add_distance=True)

            #extended_features = np.insert(features[0], 0, frame_idx)
            # Append the features to the list
            frame_index = feature_names.index('frame')
            features[0][frame_index]=  frame_idx
            features_list.append(features[0])

            # Optionally, delete the temporary image after processing
            os.remove(temp_filename)

        # Release the video capture object
        cap.release()

        # Convert the list of features to a DataFrame, with the feature names as columns
        if feature_names is not None:
            #feature_names = ["frame"] + feature_names
            df_features = pd.DataFrame(features_list, columns=feature_names)
            print(f"Extracted features for {len(features_list)} frames.")
            return df_features
        else:
            print("No features extracted.")
            return None
        
    def summarized_data(self,df):

        # Extract features using multiple strategies and plot them
        strategies = ['mean', 'variance', 'max', 'end_to_begin']
        
        # Summarize features using the specified strategies
        processor = psychai.feature.feature_extraction.feature_processor.FeatureProcessor()
        df_results = processor.summarize_features_by_user(df, strategies, user_column = "user_id", order_by_column = "frame")

        return df_results
 