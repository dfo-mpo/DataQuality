import numpy as np  
import pandas as pd
from utils.column_operations import find_non_digits

# ----------------------- Table Operations -------------------------------

# --- Normalization / Cleaning ---
"""
A1
create new column(s) for output report based on the original input dataset with the additional newly generated column(s)
"""
def add_only_numbers_columns(df, selected_columns, original_df):    
    selected_columns = [col for col in df.columns if col in selected_columns]   
    
    for column_name in selected_columns: 
        non_digits = find_non_digits(df, column_name)
        
        original_df[column_name + '_Only_Numbers'] = non_digits[column_name].apply(
            lambda x: np.where(pd.isnull(x), False, True)
        )    

    return original_df

# --- Extraction / Checking ---
"""
Extract maximum similarity values, and corresponding project names
and create a DataFrame with this information.
"""
def get_max_similarity_values(combined_sim_matrix, unique_observations, column_names):
    # Store max values and names
    max_values = []
    max_names = []
    unique_project_names = np.array(unique_observations)
    
    # Iterate over each row to find max similarity values and corresponding project names
    for i, row in enumerate(combined_sim_matrix):
        # Ignore self-similarity by setting the diagonal to -1
        row[i] = -1
        
        # Get the index of the maximum similarity
        top_indices = np.argsort(row)[::-1]  # Sort in descending order
        
        # Get the maximum value, name, and ratio
        max_values.append(row[top_indices[0]])
        max_names.append(unique_project_names[top_indices[0]])
    
    # Create a DataFrame with the max values and corresponding project names
    max_values_df = pd.DataFrame({
        "Column Source": column_names,
        "Names Tested": unique_project_names,
        "Highest Similarity Names": max_names,
        "Similarity Score": max_values
    })
    
    return max_values_df

"""
Filter for column pairs that meet the threshold, with same column pairings and duplicates removed.
"""
def filter_corrs(corrs, threshold, subset=None):
    
    corrs = corrs.copy()
    # Remove same column pairings and subset sensitive features
    np.fill_diagonal(corrs.values, np.nan)
    corrs = corrs[subset].drop(subset) if subset is not None else corrs

    # Keep columns pairings with absolute correlation above the threshold
    corrs_thr = corrs[(abs(corrs) > threshold)].melt(ignore_index=False).reset_index().dropna()

    # Rename columns
    # used / because some features use _ in column name already so may confuse user reading output table
    corrs_thr.columns = ['var1', 'var2', 'corr_coeff']
    corrs_thr['features'] = ['/'.join(sorted((i.var1, i.var2))) for i in corrs_thr.itertuples()]
    
    # Remove duplicate column pairings and sort by descending correlation coefficients
    corrs_thr.drop_duplicates('features', inplace=True)
    corrs_thr.sort_values(by='corr_coeff', ascending=False, inplace=True)

    return corrs_thr
    
# --- Similarity / Comparison ---
"""
Calculate the average consistency score based on the cosine similarity matrix and a given threshold.
"""
def average_c1_consistency_score(cosine_sim_matrix, threshold=0.91):  
    num_rows, num_columns = cosine_sim_matrix.shape  
    inconsistency = 0  
    for i in range(num_rows):  
        if np.any((cosine_sim_matrix[i] > threshold) & (cosine_sim_matrix[i] <= 1.0000000)):  
            inconsistency += 1  
    return (num_rows - inconsistency) / num_rows  

"""
Calculate the average consistency score based on the cosine similarity matrix and a given threshold.
"""
def average_c2_consistency_score(cosine_sim_df, threshold=0.91):
    num_rows, num_columns = cosine_sim_df.shape
    total_count = 0  # This will count all values above or equal to the threshold

    for i in range(num_rows):
        if np.max(cosine_sim_df[i]) >= threshold:  # Include all comparisons
            total_count += 1
    total_observations = num_rows  # Total number of observations
    average_consistency_score = total_count / total_observations
    return average_consistency_score

"""
Calculate the average consistency score based on the levenshtein similarity ratio matrix and a given threshold.
"""
def average_c3_consistency_score(leven_dist_df, threshold=0.91):
    num_rows, num_columns = leven_dist_df.shape
    total_count = 0  # This will count all values above or equal to the threshold

    for i in range(num_rows):
        if np.max(leven_dist_df[i]) >= threshold:  # Include all comparisons
            total_count += 1
    total_observations = num_rows  # Total number of observations 
    average_consistency_score = total_count / total_observations
    return average_consistency_score
    
"""
Combine text and numeric similarities into a single similarity matrix.
"""
def calculate_combined_similarity(unique_observations, text_similarity_matrix):
    # Make a copy of the text similarity matrix to modify it
    combined_sim_matrix = np.copy(text_similarity_matrix)
    
    # Extract numeric parts from each unique observation, but remove short numbers
    numeric_parts = [extract_numbers(obs) for obs in unique_observations]
    
    # Iterate over each pair of unique observations to calculate numeric similarity
    for i, num_i in enumerate(numeric_parts):
        for j, num_j in enumerate(numeric_parts):
            if i != j:
                if not contains_short_number(numeric_parts):
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