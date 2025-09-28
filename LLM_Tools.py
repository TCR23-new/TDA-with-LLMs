"""
LLM Tools for Topological Data Analysis Agent

This module provides tools for loading, processing, and analyzing neural spiking data
and neural network representations using topological data analysis methods.
"""

import json
import io
import base64
import numpy as np
import pickle
from ripser import Rips
from PIL import Image
from langchain_community.utilities.awslambda import LambdaWrapper


# Data Loading and Formatting Functions
def load_data(filename: str) -> np.ndarray:
    """
    Load spiking data from pickle file.
    
    Args:
        filename: Path to pickle file containing spiking data
        
    Returns:
        Numpy array of spiking data with shape (trials, cells)
    """
    # Handle multiline filename input
    if "\n" in filename and '.pickle' in filename:
        filename_list = filename.split('\n')
        filename = [f for f in filename_list if '.pickle' in f][0]
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data[0]


def format_spk_data(data: np.ndarray) -> str:
    """
    Format spiking data array into string representation.
    
    Args:
        data: Numpy array with shape (trials, cells)
        
    Returns:
        Formatted string with trial and cell information
    """
    num_cells = data.shape[1]
    num_trials = data.shape[0]
    data_str = ""
    
    for t in range(num_trials):
        data_str += f'Trial {t}:'
        for i in range(num_cells):
            data_str += f'Cell {i+1}: {data[t,i]}'
        data_str += "\n"
    
    return data_str


def unformat(data_str: str) -> np.ndarray:
    """
    Convert formatted string back to numpy array.
    
    Args:
        data_str: Formatted string from format_spk_data
        
    Returns:
        Numpy array with shape (trials, cells)
    """
    trial_sections = [s for s in data_str.split('Trial') if 'Cell' in s]
    num_trials = len(trial_sections)
    num_cells = len(trial_sections[0].split('Cell')) - 1
    
    data = np.zeros((num_trials, num_cells))
    
    for i in range(num_trials):
        cell_data = trial_sections[i].split('Cell')[1:]  # Skip empty first element
        for j, cell_str in enumerate(cell_data):
            data[i, j] = float(cell_str.split(':')[1])
    
    return data


# Topological Data Analysis Functions
def compute_topo_summary(data: np.ndarray) -> np.ndarray:
    """
    Compute persistent homology using Ripser.
    
    Args:
        data: Input data array for topological analysis
        
    Returns:
        Persistence diagram (birth-death pairs) for 0-dimensional features
    """
    rips = Rips(verbose=False)
    diagrams = rips.fit_transform(data)
    return diagrams[0]  # Return 0-dimensional persistence diagram


def topo_format(data: np.ndarray) -> str:
    """
    Format topological features into readable string.
    
    Args:
        data: Persistence diagram array
        
    Returns:
        Formatted string showing birth-death times for each feature
    """
    num_features = data.shape[0]
    birth_death_times = data.shape[1]
    data_str = ""
    
    for i in range(num_features):
        data_str += f"Feat {i+1}: "
        for j in range(birth_death_times):
            data_str += f"{data[i,j]} "
        data_str += "\n"
    
    return data_str


# Image Processing Functions
def load_image(img_name: str) -> str:
    """
    Load image and convert to base64 JSON format for AWS Lambda.
    
    Args:
        img_name: Path to image file
        
    Returns:
        JSON string with base64-encoded image data
    """
    # Handle multiline input
    if "\n" in img_name:
        img_name = img_name.split('.png')[0] + '.png'
    
    # Load, resize, and encode image
    image = Image.open(img_name).convert('RGB').resize((32, 32))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    input_data = {"image_data": base64_string, "trim_model": 'true'}
    return json.dumps(input_data)


# Tool Functions (LangChain Tool Interface)
def Tool1(filename: str) -> str:
    """
    Tool 1: Load and format spiking data from pickle file.
    
    Args:
        filename: Path to pickle file
        
    Returns:
        Formatted string representation of spiking data
    """
    return format_spk_data(load_data(filename))


def Tool2(data: str) -> str:
    """
    Tool 2: Compute topology of formatted spiking data.
    
    Args:
        data: Formatted string from Tool1
        
    Returns:
        Formatted topological summary
    """
    return topo_format(compute_topo_summary(unformat(data)))


def Tool3(data: str) -> str:
    """
    Tool 3: Generate neural network representation of image using AWS Lambda.
    
    Args:
        data: Image filename
        
    Returns:
        Neural network representation string
    """
    rep_tool = LambdaWrapper(function_name="representation-tool")
    input_str = load_image(data)
    response = rep_tool.run(input_str)
    rep_dict = json.loads(response[8:])  # Skip first 8 characters
    return rep_dict['representation']


def Tool4(data: str) -> str:
    """
    Tool 4: Write neural network representation to file.
    
    Args:
        data: Representation data to write
        
    Returns:
        Empty string (file operation)
    """
    file_name = "/Users/tevinrouse/Desktop/TestImages/nn_representations.txt"
    
    try:
        with open(file_name, "x") as f:
            f.write(f"Trial: {data}\n")
    except FileExistsError:
        with open(file_name, "a") as f:
            f.write(f"Trial: {data}\n")
    
    return ""


def Tool5(filename: str) -> str:
    """
    Tool 5: Compute topology of neural network representations from text file.
    
    Args:
        filename: Path to text file containing representations
        
    Returns:
        Formatted topological summary
    """
    threshold = 10
    
    # Handle multiline input
    if "\n" in filename:
        filename = filename.split('.txt')[0] + '.txt'
    
    representations = []
    
    with open(filename, 'rb') as f:
        lines = f.readlines()
        lines = [line for line in lines if len(line) > threshold]
        
        for line in lines:
            line = line.decode("utf-8")
            
            # Skip header lines
            if "Trial" not in line or "Obs" not in line:
                values = line.split(',')
                representation = []
                
                for ix, value in enumerate(values):
                    if ix == threshold:
                        break
                    
                    # Clean up trial prefix
                    if 'Trial:' in value:
                        value = value.replace('Trial:', '').rstrip()
                    
                    try:
                        representation.append(float(value))
                    except ValueError:
                        continue  # Skip invalid values
                
                representations.append(representation)
    
    # Convert to numpy array and compute topology
    data = np.array([rep for rep in representations if len(rep) == threshold])
    return topo_format(compute_topo_summary(data))
