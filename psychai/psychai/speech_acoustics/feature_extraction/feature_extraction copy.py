import librosa
import opensmile
import numpy as np
import os
import pickle
import soundfile as sf
import psychai.feature.feature_extraction.feature_retriever
import psychai.feature.feature_extraction.feature_processor

class SpeechFeatureExtractor:
    def __init__(self, segment_length=5, pick_chche_folder=r"./"):
        self.segment_length = segment_length  # Length of each segment in seconds
        self.picke_cache_folder = pick_chche_folder

    # Segment audio and process each segment
    def segment_and_extract_features(self, file, extraction_methods=["librosa"]):
        audio_time_series, sample_rate = librosa.load(file)
        segment_samples = int(self.segment_length * sample_rate)  # Convert segment length to samples
        num_segments = len(audio_time_series) // segment_samples  # Calculate number of full segments
        base_name = os.path.basename(file)
        # Process each segment
        all_features = []
        feature_names = []

        for i in range(num_segments):
            segment = audio_time_series[i * segment_samples : (i + 1) * segment_samples]
            segment_features = []

            try:
                for method in extraction_methods:
                    if method == "librosa":
                        features, names = self.extract_librosa_features(segment, sample_rate, base_name)
                    elif method == "opensmile":
                        features, names = self.extract_opensmile_features(segment, sample_rate, base_name)
                    else:
                        raise ValueError("Invalid extraction method. Choose 'librosa' or 'opensmile'.")
                    
                    # Extend features and names
                    segment_features.extend(features)
                    if i == 0:
                        feature_names.extend(names)  # Only add names once
            except Exception as e:
                print(f"Error for {file} segment:{i}. method:{method}. Error:{e}")

            all_features.append(segment_features)

        # Convert features list to numpy array for uniform output
        return np.array(all_features), feature_names

    # Feature extraction function for audio using librosa
    def extract_librosa_features(self, segment, sample_rate, file_name):
        stft = np.abs(librosa.stft(segment))
        
        # Extract various features
        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=segment, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(segment), sr=sample_rate).T, axis=0)

        # Combine all extracted features
        result = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        
        # Create corresponding feature names
        feature_names = []
        feature_names.extend([f'librosa_mfcc_{i}' for i in range(len(mfccs))])
        feature_names.extend([f'librosa_chroma_{i}' for i in range(len(chroma))])
        feature_names.extend([f'librosa_mel_{i}' for i in range(len(mel))])
        feature_names.extend([f'librosa_contrast_{i}' for i in range(len(contrast))])
        feature_names.extend([f'librosa_tonnetz_{i}' for i in range(len(tonnetz))])
        print(f'[File:{file_name}. Processed by librosa.]')
        return result, feature_names

    # Feature extraction function using OpenSmile
    def extract_opensmile_features(self, segment, sample_rate, file_name):
        # Save segment to a temporary WAV file (needed for OpenSmile)
        temp_file = os.path.join(self.picke_cache_folder, file_name)
        sf.write(temp_file, segment, sample_rate)  # Using soundfile to write the WAV file
        
        try:
            # Initialize OpenSmile and process the file
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            features = smile.process_file(temp_file).values.flatten()
            print(f'[File:{file_name}. Processed by opensmile.]')
            
            # Generate feature names for OpenSmile features
            feature_names = [f'opensmile_feature_{i}' for i in range(len(features))]
        finally:
            # Delete the temporary file after processing
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return features, feature_names


    def extract_features_with_cache(self, file_path, pickle_folder=""):
        # Check if pickle_folder is specified
        if pickle_folder:
            # Ensure the pickle folder exists
            os.makedirs(pickle_folder, exist_ok=True)
            
            # Generate the pickle file path based on the input file name
            base_name = os.path.basename(file_path)
            pickle_file_path = os.path.join(pickle_folder, f"{base_name}.pkl")
            
            # Check if the pickle file already exists
            if os.path.exists(pickle_file_path):
                # Load features from the pickle file
                try:
                    with open(pickle_file_path, "rb") as f:
                        features, feature_names = pickle.load(f)
                    print(f"Loaded features from {pickle_file_path}")
                    return features, feature_names
                except Exception as e:
                    print(f"Error loading pickle file: {e}")
        
        # If pickle_folder is empty or pickle file does not exist, run the feature extraction
        try:
            features, feature_names = self.segment_and_extract_features(
                file_path, extraction_methods=["librosa", "opensmile"]
            )
            
            # Save the result to a pickle file if pickle_folder is specified
            if pickle_folder:
                with open(pickle_file_path, "wb") as f:
                    pickle.dump((features, feature_names), f)
                print(f"Features saved to {pickle_file_path}")
                
            return features, feature_names
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    def summarized_process_folder(self,file_path):

        # Extract features using multiple strategies and plot them
        strategies = ['mean', 'variance', 'max', 'end_to_begin']
        feature_names = []
        features, feature_names = self.extract_features_with_cache(file_path,self.picke_cache_folder)

        if len(feature_names)>0:
            # Summarize features using the specified strategies
            processor = psychai.feature.feature_extraction.feature_processor.FeatureProcessor()
            summarized_features, summarized_feature_names = processor.summarize_features(features, strategies, feature_names)

            return summarized_features, summarized_feature_names     
        else:
            return np.array([]),[]

