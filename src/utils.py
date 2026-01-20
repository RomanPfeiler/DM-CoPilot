# src/utils.py

import re

def clean_transcript_text(file_path: str) -> str:
    """
    Reads a raw text file and removes timestamps.
    
    Args:
        file_path (str): Path to the transcript file.
        
    Returns:
        str: Cleaned text without timestamps (e.g., '0:00') or extra spaces.
    """
    # 1. Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 2. Define Regex pattern for timestamps (e.g., 0:00, 10:15, 1:05:00)
    # \d+ matches numbers, : matches colon
    timestamp_pattern = r'\d+:\d+(?::\d+)?'
    
    # 3. Replace timestamps with empty string
    clean_text = re.sub(timestamp_pattern, '', raw_text)
    
    # 4. Collapse multiple spaces into one
    clean_text = " ".join(clean_text.split())
    
    return clean_text

def get_sliding_windows(text: str, chunk_size: int, overlap: int):
    """
    Generator that creates overlapping chunks of text.
    
    Args:
        text (str): The full input text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words to overlap.
        
    Yields:
        str: A chunk of text.
    """
    words = text.split()
    
    # Loop through words with a step size of (chunk_size - overlap)
    for i in range(0, len(words), chunk_size - overlap):
        # Slice the list of words
        chunk_words = words[i : i + chunk_size]
        
        # Join back into a string
        yield " ".join(chunk_words)
        
        # Stop if we reach the end
        if i + chunk_size >= len(words):
            break