import os
import sys
import torch
from pathlib import Path
from infer import TranskunInfer
import librosa
import numpy as np

# Initialize the model
infer = TranskunInfer(device="cuda" if torch.cuda.is_available() else "cpu")


def read_audio(path, normalize=True):
    y, sr = librosa.load(path)
    y = y.reshape(-1, 1)
    if normalize:
        y = np.float32(y) / 2**15
    return sr, y


def create_output_path(input_path, output_dir, extension=".mid"):
    """
    Create the output file path preserving the directory structure in the output directory.
    
    Args:
        input_path (str): The original input file path.
        output_dir (str): The base directory where the output files will be saved.
        extension (str): The extension for the output file (default: ".mid").
    
    Returns:
        str: The full path for the output file.
    """
    relative_path = os.path.relpath(input_path, start=input_dir)  # Relative path from input_dir
    output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + extension)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directories if they don't exist
    return output_path


def process_directory(input_dir, output_dir, extension):
    """
    Process all files with the given extension in the specified directory.
    
    Args:
        input_dir (str): The directory to search for files.
        output_dir (str): The directory to save the output files.
        extension (str): The file extension to search for (e.g. ".wav").
    """
    # Make sure the extension starts with a dot
    if not extension.startswith("."):
        extension = f".{extension}"

    # Find all files with the specified extension in the directory
    print(f"Searching for files with extension '{extension}' in '{input_dir}'")
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(extension):
                # Create output path preserving directory structure
                output_path = create_output_path(file_path, output_dir, extension=".mid")
                print(f"Transcribing: {file_path} -> {output_path}")
                
                # Read the audio file and transcribe to MIDI
                sr, audio = read_audio(file_path)
                midi_data = infer.get_midi(audio=audio, fs=sr)
                
                # Write the MIDI data to the output file
                with open(output_path, "wb") as midi_file:
                    midi_file.write(midi_data)

            else:
                print(f"Skipping invalid file: {file_path}")


if __name__ == "__main__":
    # Check if the directory and extension arguments are provided
    if len(sys.argv) < 4:
        print("Usage: python transcribe_directory.py <input_directory> <output_directory> <extension>")
        sys.exit(1)

    # Get the input and output directories and file extension from the command line arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    extension = sys.argv[3]

    # Process the directory
    process_directory(input_dir, os.path.join(output_dir, os.path.basename(input_dir)), extension)
