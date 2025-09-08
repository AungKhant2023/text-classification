import unicodedata
import re

def normalize_english_to_calibri(text: str) -> str:
    """
    Convert fancy English letters (bold, italic, math symbols, etc.) 
    to normal ASCII letters (Calibri-style) while keeping all other characters intact.
    """
    result = ""
    for char in text:
        # Check if character is an English letter or number in fancy Unicode
        decomposed = unicodedata.normalize("NFKD", char)
        base = ''.join(c for c in decomposed if not unicodedata.combining(c))
        # Only replace if the base is ASCII letter or number
        if 'A' <= base <= 'Z' or 'a' <= base <= 'z' or '0' <= base <= '9':
            result += base
        else:
            # Keep original character (like Myanmar letters, punctuation, emoji, etc.)
            result += char
    return result