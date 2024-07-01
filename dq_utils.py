import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os 
import re
from difflib import SequenceMatcher

# Function 0: Reading the dataset file
def read_data(dataset_path):  
    _, file_extension = os.path.splitext(dataset_path)
    if file_extension == '.csv':  
        df = pd.read_csv(dataset_path)    
    elif file_extension == '.xlsx':
        df = pd.read_excel(dataset_path)  
    else:
        print('Unsupported file type')  
        df = None  
    return df

# Consistency Type 1 (C1) function

# Dictionary mapping Canadian province abbreviations to their full names
province_abbreviations = {
    'BC': 'British Columbia',
    'ON': 'Ontario',
    'QC': 'Quebec',
    'AB': 'Alberta',
    'MB': 'Manitoba',
    'SK': 'Saskatchewan',
    'NS': 'Nova Scotia',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland and Labrador',
    'PE': 'Prince Edward Island',
    'NT': 'Northwest Territories',
    'YT': 'Yukon',
    'NU': 'Nunavut'
}

def normalize_text(text, remove_numbers=False):
    """
    Normalize input text by converting to lowercase, stripping whitespace,
    replacing province abbreviations with full names, and removing non-alphanumeric characters.
    Optionally remove numbers based on the flag.
    """
    text = str(text).lower().strip()
    for abbr, full in province_abbreviations.items():
        text = re.sub(r'\b' + abbr.lower() + r'\b', full.lower(), text)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return ' '.join(text.split())

def extract_numbers(text):
    """
    Extract all numbers from the input text and return them as a list of strings.
    """
    return re.findall(r'\d+', text)

def remove_short_numbers(text):
    """
    Remove numbers with 1 or 2 digits from the input text.
    """
    return re.sub(r'\b\d{1,4}\b', '', text)

def numeric_similarity(num1_list, num2_list):
    """
    Calculate the similarity between two lists of numbers by comparing each digit.
    Return the proportion of matching digits.
    """
    num1, num2 = ' '.join(num1_list), ' '.join(num2_list)
    matches = sum(1 for a, b in zip(num1, num2) if a == b)
    max_length = max(len(num1), len(num2))
    return matches / max_length if max_length > 0 else 0

def string_similarity(str1, str2):
    """
    Calculate the similarity between two strings using the SequenceMatcher from difflib.
    Return the similarity ratio.
    """
    return SequenceMatcher(None, str1, str2).ratio()

def calculate_cosine_similarity(text_list, ref_list, Stop_Words):
    """
    Calculate the cosine similarity between lists of texts using TF-IDF vectorization.
    """
    vectorizer = TfidfVectorizer(stop_words=Stop_Words, analyzer='word', ngram_range=(1, 2))
    ref_vec = vectorizer.fit_transform(ref_list)
    text_vec = vectorizer.transform(text_list)
    return cosine_similarity(text_vec, ref_vec)

def contains_short_number(num_list):
    """
    Check if any number in the list has 1 or 2 digits.
    """
    return any(len(num) <= 4 for num in num_list)

def numbers_match(num_list1, num_list2):
    """
    Check if any number in the first list is present in the second list.
    """
    return any(num in num_list2 for num in num_list1)

def calculate_combined_similarity(df, unique_observations, text_similarity_matrix):
    """
    Combine text and numeric similarities into a single similarity matrix.
    """
    # Make a copy of the text similarity matrix to modify it
    combined_sim_matrix = np.copy(text_similarity_matrix)
    
    # Extract numeric parts from each unique observation
    numeric_parts = [extract_numbers(obs) for obs in unique_observations]
    
    # Iterate over each pair of unique observations to calculate numeric similarity
    for i, num_i in enumerate(numeric_parts):
        for j, num_j in enumerate(numeric_parts):
            if i != j:
                # Calculate the numeric similarity for the current pair
                num_sim = numeric_similarity(num_i, num_j)
                
                # Update the combined similarity matrix with the maximum value between text and numeric similarity
                combined_sim_matrix[i, j] = max(combined_sim_matrix[i, j], num_sim)
    
    # Iterate over each pair of unique observations to calculate string similarity
    for i, obs_i in enumerate(unique_observations):
        for j, obs_j in enumerate(unique_observations):
            if i != j:
                # Calculate the string similarity for the current pair
                seq_sim = string_similarity(obs_i, obs_j)
                
                # Update the combined similarity matrix with the maximum value between existing and sequence matcher 
                combined_sim_matrix[i, j] = max(combined_sim_matrix[i, j], seq_sim)
    
    return combined_sim_matrix

def average_consistency_score(cosine_sim_matrix, threshold):
    """
    Calculate the average consistency score based on the cosine similarity matrix and a given threshold.
    """
    num_rows, num_columns = cosine_sim_matrix.shape
    inconsistency = 0

    for i in range(num_rows):
        if np.any((cosine_sim_matrix[i] > threshold) & (cosine_sim_matrix[i] <= 1.0000000)):
            inconsistency += 1
    
    return (num_rows - inconsistency) / num_rows

def process_and_calculate_similarity(dataset_path, column_names, threshold, Stop_Words=['the', 'and']):
    """
    Process the dataset, normalize the text, and calculate the similarity scores for multiple columns.
    """
    # Read the dataset from the provided Excel file path
    df = read_data(dataset_path)
    overall_consistency_scores = []

    # Iterate over each specified column
    for column_name in column_names:
        # Normalize the text in the specified column and store the results in a new column
        df[f'Normalized {column_name}'] = df[column_name].apply(normalize_text)
        
        # Get unique normalized observations by removing duplicates and NaN values
        unique_observations = pd.unique(df[f'Normalized {column_name}'].dropna().values.ravel())
        
        # Calculate the cosine similarity matrix for the unique normalized observations
        text_sim_matrix = calculate_cosine_similarity(unique_observations.tolist(), unique_observations.tolist(), Stop_Words)
        
        # Set the diagonal of the similarity matrix to 0 to ignore self-similarity
        np.fill_diagonal(text_sim_matrix, 0)
        
        # Combine text similarity with numeric similarity to get a final similarity matrix
        combined_sim_matrix = calculate_combined_similarity(df, unique_observations, text_sim_matrix)
        
        # Initialize columns in the dataframe to store the recommended organization matches and all matches
        df[f'Recommended {column_name}'] = None
        df[f'All Matches {column_name}'] = None

        # Iterate over each normalized organization in the dataframe
        for i, norm_org in enumerate(df[f'Normalized {column_name}']):
            # Find the index of the current normalized organization in the unique observations
            try:
                current_index = np.where(unique_observations == norm_org)[0][0]
            except IndexError:
                df.at[i, f'Recommended {column_name}'] = "No significant match"
                df.at[i, f'All Matches {column_name}'] = []
                continue
            
            # Get the similarities for the current organization from the combined similarity matrix
            similarities = combined_sim_matrix[current_index]
            
            # Find the indices and values of all matching organizations
            matched_indices = np.where(similarities >= threshold)[0]
            all_matches = [unique_observations[idx] for idx in matched_indices]
            all_match_scores = [similarities[idx] for idx in matched_indices]

            best_score = 0
            best_match = "No significant match"

            # Extract numbers from the current organization
            num_list_current = extract_numbers(norm_org)

            for idx in matched_indices:
                candidate_match = unique_observations[idx]
                num_list_candidate = extract_numbers(candidate_match)

                if contains_short_number(num_list_current) or contains_short_number(num_list_candidate):
                    # If short numbers are present, ensure they match; otherwise, skip this match
                    if not numbers_match(num_list_current, num_list_candidate):
                        continue
                    # Recalculate similarity excluding short numbers
                    norm_org_no_nums = remove_short_numbers(norm_org)
                    candidate_no_nums = remove_short_numbers(candidate_match)
                    recalculated_similarity = string_similarity(norm_org_no_nums, candidate_no_nums)
                    if recalculated_similarity > best_score:
                        best_score = recalculated_similarity
                        best_match = candidate_match
                else:
                    if similarities[idx] > best_score:
                        best_score = similarities[idx]
                        best_match = candidate_match

            # Assign the best match to the dataframe
            if best_score > threshold:
                df.at[i, f'Recommended {column_name}'] = f"{best_match} ({best_score:.2f})"
            else:
                df.at[i, f'Recommended {column_name}'] = "No significant match"

            # Store all matches
            df.at[i, f'All Matches {column_name}'] = ', '.join([f"{match} ({score:.2f})" for match, score in zip(all_matches, all_match_scores) if score > threshold])

        # Calculate the overall consistency score for the current column
        consistency_score = average_consistency_score(text_sim_matrix, threshold)
        overall_consistency_scores.append(consistency_score)

    # Calculate the overall consistency score as the average of individual consistency scores
    overall_consistency_score = np.mean(overall_consistency_scores)
    df['Overall Consistency Score'] = overall_consistency_score

    return df 