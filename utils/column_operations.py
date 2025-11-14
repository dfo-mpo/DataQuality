import re  
import numpy as np  
import os
import statistics
import pandas as pd
from functools import partial
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from Levenshtein import ratio
from difflib import SequenceMatcher  
from ast import literal_eval
import io

# ----------------------- Column Operations -------------------------------

# --- Normalization / Cleaning ---
"""
For Accuracy A1 
Create a new column that flags previously existing nulls and empty strings. this prevents a false positive (if it is already a null then it shouldn't be counted as an instance of a symbol in numerics)
"""
def new_column(df, column_name):
    df[f"{column_name}_new"] = np.where(
        df[column_name].isnull() | (df[column_name].astype(str).str.strip() == ""), 1, 0)
    
    return df

"""
A1 
Use the new null flag column to find symbols in numerics. change existing nulls to "True" as preparation for an output dataset that only flags symbols in numerics and not anything else, 
change symbols in numerics to real NaN.
"""
def find_non_digits(df, column_name):
    new_column(df, column_name)
    new_col_name = f"{column_name}_new"
    to_numeric_with_coerce = partial(pd.to_numeric, errors='coerce')
    
    df[column_name] = np.where(df[new_col_name] == 0, df[column_name].apply(to_numeric_with_coerce), "True")
    
    df[column_name] = df[column_name].replace("nan", np.nan) # replace nan that was string, to real NaN 
    
    return df
    
# --- Extraction / Checking ---
"""
Extract unique non null values from specific column.
"""
def get_names_used_for_column(df, column_name):  
    unique_observations = pd.unique(df[column_name].dropna().values.ravel())  
    return unique_observations  

"""
Check if any number in the list has 1 or 2 digits.
"""
def contains_short_number(num_list):
    return any(len(num) <= 4 for num in num_list)

"""
Check if any number in the first list is present in the second list.
"""
def numbers_match(num_list1, num_list2):
    return any(num in num_list2 for num in num_list1)


# --- Similarity / Comparison ---
"""
Calculate the similarity between two lists of numbers by comparing each digit.
Return the proportion of matching digits.
"""
def numeric_similarity(num1_list, num2_list):  
    num1, num2 = " ".join(num1_list), " ".join(num2_list)  
    matches = sum(1 for a, b in zip(num1, num2) if a == b)  
    max_length = max(len(num1), len(num2))  
    return matches / max_length if max_length > 0 else 0  

"""
Calculate the cosine similarity between lists of texts using TF-IDF vectorization.
"""
def calculate_cosine_similarity(text_list, ref_list, stop_words):  
    vectorizer = TfidfVectorizer(stop_words=stop_words, analyzer="word", ngram_range=(1, 2))  
    ref_vec = vectorizer.fit_transform(ref_list)  
    text_vec = vectorizer.transform(text_list)  
    return cosine_similarity(text_vec, ref_vec)  

"""
Calculate the levenshtein similarity ratio between lists of texts.
"""
def calculate_levenshtein_similarity(text_list, ref_list):
    sim_matrix = np.zeros((len(text_list), len(ref_list)))

    for i, text in enumerate(text_list):
        for j, ref in enumerate(ref_list):
            sim_matrix[i, j] = ratio(text, ref)
    return sim_matrix

"""
Compare whether the column being tested resembles the reference data column
and create a DataFrame with new column(s) added
"""
def compare_datasets(df,selected_column, unique_observations):     
    # Iterate over each row in the selected column    
    column_results = []  
    for value in df[selected_column]:    
        column_results.append(np.where(pd.isnull(value), True, value in unique_observations))    
    
    # Add the results as a new column in the DataFrame  
    df[selected_column + '_comparison'] = column_results  
    
    return df  