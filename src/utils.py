"""
utils.py

Utility functions for the LLM fine-tuning project.
"""

import os
import tempfile
import torch
from pathlib import Path

def setup_environment(verbose=True):
    """Setup environment variables and directories"""
    
    # Set HuggingFace cache directories
    cache_dir = "E:/huggingface_cache"
    temp_dir = "E:/temp_processing"
    
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = f"{cache_dir}/transformers"
    os.environ['HF_DATASETS_CACHE'] = f"{cache_dir}/datasets"
    
    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set temp directory
    tempfile.tempdir = temp_dir
    
    if verbose:
        print(f"📁 Cache directory: {cache_dir}")
        print(f"📁 Temp directory: {temp_dir}")

def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())

def ensure_complete_sentence(text: str) -> str:
    """Ensure text ends with a complete sentence"""
    sentence_endings = ['.', '!', '?']
    last_complete = -1
    
    for i in range(len(text) - 1, -1, -1):
        if text[i] in sentence_endings:
            # Check if this is likely end of sentence (not abbreviation)
            if i < len(text) - 1 and text[i + 1] in [' ', '\\n', '\\t']:
                last_complete = i
                break
            elif i == len(text) - 1:  # End of text
                last_complete = i
                break
    
    if last_complete == -1:
        return text  # No sentence ending found, return as is
    
    return text[:last_complete + 1].strip()

def clear_gpu_cache():
    """Clear GPU cache if CUDA is available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_device_info():
    """Get device information"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name()
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {device} ({memory:.1f}GB)"
    else:
        return "CPU only"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"