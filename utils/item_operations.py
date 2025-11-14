import re  
from datetime import datetime
from difflib import SequenceMatcher  

# ----------------------- Item Operations -------------------------------

# --- Normalization / Cleaning ---
province_abbreviations = {  
    "BC": "British Columbia",  
    "ON": "Ontario",  
    "QC": "Quebec",  
    "AB": "Alberta",  
    "MB": "Manitoba",  
    "SK": "Saskatchewan",  
    "NS": "Nova Scotia",  
    "NB": "New Brunswick",  
    "NL": "Newfoundland and Labrador",  
    "PE": "Prince Edward Island",  
    "NT": "Northwest Territories",  
    "YT": "Yukon",  
    "NU": "Nunavut",  
}  

"""
Normalize input text by converting to lowercase, stripping whitespace,
replacing province abbreviations with full names, and removing non-alphanumeric characters.
Optionally remove numbers based on the flag.
""" 
def normalize_text(text, remove_numbers=False):  
    text = str(text).lower().strip()  
    for abbr, full in province_abbreviations.items():  
        text = re.sub(r"\b" + abbr.lower() + r"\b", full.lower(), text)  
    if remove_numbers:  
        text = re.sub(r"\d+", "", text)  
    text = "".join(char for char in text if char.isalnum() or char.isspace())  
    return " ".join(text.split())  

"""
Remove numbers with 1 or 2 digits from the input text.
"""
def remove_short_numbers(text):  
    return re.sub(r"\b\d{1,4}\b", "", text)  

# --- Extraction / Checking ---
"""
Extract all numbers from the input text and return them as a list of strings.
"""
def extract_numbers(text):  
    return re.findall(r"\d+", text)  

"""
Check whether given string date-time entry matches a given format.
"""
def inconsistent_datetime(date_str, fmt):
    # Catches inconsistent format and return true 
    try:
        datetime.strptime(date_str, fmt)
        return False 
    except ValueError:
        return True 

# --- Similarity / Comparison ---
"""
Calculate the similarity between two strings using the SequenceMatcher from difflib.
Return the similarity ratio.
"""
def string_similarity(str1, str2):  
    return SequenceMatcher(None, str1, str2).ratio()  


