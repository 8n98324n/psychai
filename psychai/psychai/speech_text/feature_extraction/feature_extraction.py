# Import necessary libraries
import torch  # PyTorch, used for deep learning models and GPU utilization
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # Models and processors from Hugging Face
import librosa  # Library for audio signal processing
import numpy as np  # For numerical operations and array handling

class SpeechTextFeatureExtractor:
    """
    A class for extracting text from audio files using the Whisper model.
    
    This class is designed to load a pre-trained Whisper model from Hugging Face,
    handle audio input, and perform speech-to-text conversion by processing audio files.
    
    Attributes:
    ----------
    huggingface_cache_location : str
        The cache directory for storing pre-trained Hugging Face models.
    processor : WhisperProcessor
        The processor used to preprocess input audio data and tokenize the output text.
    model : WhisperForConditionalGeneration
        The Whisper model used for generating text from audio.
    device : torch.device
        The device on which the model and tensors will run (GPU if available, otherwise CPU).
    """
    
    def __init__(self, model_cache_location = "", processor_input = None, model_input = None):
        """
        Initialize the SpeechTextFeatureExtractor object.
        
        This method loads environment variables, sets up the Hugging Face cache directory,
        and loads the Whisper model and processor. It also checks for GPU availability 
        and moves the model to the appropriate device (CPU or GPU).
        """

        # Set the location for the Hugging Face model cache.
        # The cache directory is used to store pre-trained models to avoid downloading them every time.
        # The path is obtained from environment variables and converted to an absolute path.
        self.huggingface_cache_location = model_cache_location

        # Check if a GPU is available, otherwise default to the CPU
        # This ensures the model runs on the fastest available hardware.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Print device information for confirmation

    
    def model_initialize(self):

        # Load the Whisper processor and model from the Hugging Face hub.
        # The processor converts audio data into input features that the model can understand.
        self.processor = processor_input
        if self.processor == None:
            self.processor = WhisperProcessor.from_pretrained(
                "openai/whisper-large-v2", 
                cache_dir=self.huggingface_cache_location
            )

        # Load the Whisper model, which will generate text from input audio features.
        self.model = model_input
        if self.model == None:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v2", 
                cache_dir=self.huggingface_cache_location
            )

            # Disable forced decoder IDs to allow more flexible text generation
            # This setting prevents the model from forcing specific tokens (e.g., specific language tokens).
            self.model.config.forced_decoder_ids = None
            
            # Move the model to the appropriate device (GPU if available, otherwise CPU)
            self.model = self.model.to(self.device)

    def extract_text(self, file: str) -> (np.ndarray, list):
        """
        Extract text (transcription) from an audio file using the Whisper model.
        
        This method processes an audio file, converts it into the format required by the Whisper model,
        and generates a transcription of the spoken content.
        
        Parameters:
        ----------
        file : str
            The path to the audio file to be transcribed. The file should be in a format 
            that can be processed by `librosa.load()` (e.g., WAV).

        Returns:
        -------
        transcription : np.ndarray
            The transcription of the audio file in the form of a NumPy array.
        ["Text"] : list
            A list containing a single string "Text", used for compatibility purposes.
        """
        
        # Load the audio file
        # The audio file is loaded with librosa, a library for audio processing in Python.
        # Setting `sr=None` keeps the original sampling rate of the file.
        signal, sr = librosa.load(file, sr=None)  # Load the file and retain the original sampling rate

        # Resample the audio signal to 16 kHz
        # Whisper models expect input audio to have a sampling rate of 16 kHz.
        resampled_signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)

        # Process the resampled signal using the Whisper processor
        # The processor takes the resampled audio and converts it into input features (tensors).
        # `return_tensors="pt"` means it returns a PyTorch tensor.
        input_features = self.processor(resampled_signal, sampling_rate=16000, return_tensors="pt").input_features 

        # Move the input features to the same device as the model (CPU or GPU)
        input_features = input_features.to(self.device)

        # Generate the predicted token ids from the input features
        # The model uses the processed input audio features to predict token ids that represent the transcription.
        predicted_ids = self.model.generate(input_features)

        # Decode the token ids into the transcription (text)
        # The processor's `batch_decode()` method converts the token ids into human-readable text.
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=False)

        # Return the transcription as a NumPy array and a list ["Text"]
        # The second return value ["Text"] is likely for compatibility with other parts of the system.
        return np.hstack([transcription]), ["Text"]
