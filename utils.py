"""
Utility functions for SentinelRAG.
"""

import os
import re
import json
from typing import List, Dict, Any
import numpy as np

def load_env_keys() -> None:
    """
    Loads API keys from environment variables.
    Raises errors if required keys are missing.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is missing or empty.")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing or empty.")

def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncates text preserving word boundaries.
    
    Args:
        text (str): Text to truncate.
        max_chars (int): Maximum allowed characters.
        
    Returns:
        str: Truncated text.
    """
    if len(text) <= max_chars:
        return text
        
    truncated = text[:max_chars]
    # Try to find the last space to avoid cutting words in half
    last_space = truncated.rfind(' ')
    if last_space != -1:
        truncated = truncated[:last_space]
        
    return truncated + "..."

def generate_citation_id(source_type: str, index: int) -> str:
    """
    Generates citation IDs like WEB1, WIKI2, ARXIV3, PDF4.
    
    Args:
        source_type (str): The type of the source (web, wiki, arxiv, pdf).
        index (int): The index of the source.
        
    Returns:
        str: Formatted citation ID.
    """
    prefix_map = {
        'web': 'WEB',
        'wikipedia': 'WIKI',
        'arxiv': 'ARXIV',
        'pdf': 'PDF'
    }
    
    prefix = prefix_map.get(source_type.lower(), 'SRC')
    return f"{prefix}{index}"

def extract_citations(text: str) -> List[str]:
    """
    Extracts all citation references from text using regex.
    For example: [WEB1], [WIKI2] -> ['WEB1', 'WIKI2']
    
    Args:
        text (str): The text to extract citations from.
        
    Returns:
        List[str]: List of citation IDs found in the text.
    """
    # Regex to find citations in brackets, e.g., [WEB1], [WIKI2]
    pattern = r'\[(WEB\d+|WIKI\d+|ARXIV\d+|PDF\d+)\]'
    matches = re.findall(pattern, text)
    return matches

def format_source_context(sources: List[Dict[str, Any]]) -> str:
    """
    Formats sources into a context string for LLM prompts.
    
    Args:
        sources (List[dict]): List of source dictionaries.
        
    Returns:
        str: Formatted context string.
    """
    formatted_sources = []
    for source in sources:
        citation_id = source.get('citation_id', 'UNKNOWN')
        title = source.get('title', 'No Title')
        text = source.get('text', '')
        
        formatted_source = f"Source [{citation_id}]: {title}\nContent: {text}"
        formatted_sources.append(formatted_source)
        
    return "\n\n".join(formatted_sources)

def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Parses JSON from LLM responses, handling markdown code blocks.
    
    Args:
        text (str): The LLM response text.
        
    Returns:
        dict: Parsed JSON data.
    """
    # Remove markdown code block markers if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
        
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}. Returning empty dict.")
        return {}

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numpy softmax implementation for NLI score normalization.
    
    Args:
        x (np.ndarray): Input array.
        
    Returns:
        np.ndarray: Softmax normalized array.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
