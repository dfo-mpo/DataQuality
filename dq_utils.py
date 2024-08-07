# Set Up
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

# Consistency Test Type 2
# Function 1: Get names used for a single column  
def get_names_used_for_column(df, column_name):  
    unique_observations = pd.unique(df[column_name].dropna().values.ravel())  
    return unique_observations  

# Function 2: Calculate Cosine Similarity  
def calculate_cosine_similarity(text_list, ref_list, Stop_Words):  
    count_vectorizer = CountVectorizer(stop_words= Stop_Words)  
    ref_vec = count_vectorizer.fit_transform(ref_list).todense()  
    ref_vec_array = np.array(ref_vec) 
    text_vec = count_vectorizer.transform(text_list).todense()  
    text_vec_array = np.array(text_vec) 
    cosine_sim = np.round((cosine_similarity(text_vec_array, ref_vec_array)), 2)  
    return cosine_sim  

# Function 3: Average Consistency Score  
def average_consistency_score(cosine_sim_df, threshold=0.91):
    num_rows, num_columns = cosine_sim_df.shape
    total_count = 0  # This will count all values above or equal to the threshold  
    
    for i in range(num_rows):
        if np.max(cosine_sim_df[i]) >= threshold: #Include all comparisons 
            total_count += 1
    total_observations = num_rows  # Total number of observations  
    average_consistency_score = total_count / total_observations  
    return average_consistency_score 
   
def process_and_calculate_similarity_ref(dataset_path, column_mapping, ref_dataset_path = None, threshold = 0.91, Stop_Words = 'activity'):    
    #Read the data file  
    df = read_data(dataset_path)  
  
    # Initialize ref_df if a ref dataset is provided  
    if ref_dataset_path:  
        df_ref = read_data(ref_dataset_path)  
        ref_data = True #Flag to indicate we are using a ref dataset  
    else:  
        ref_data = False #No ref dataset, compare within the same dataset  
  
    all_consistency_scores = []    
      
    for selected_column, m_selected_column in column_mapping.items():    
        if ref_data:  
             # Compare to ref dataset    
            unique_observations = get_names_used_for_column(df_ref, m_selected_column)    
        else:    
            # Use own column for comparison    
            unique_observations = get_names_used_for_column(df, selected_column)  
              
        cosine_sim_matrix = calculate_cosine_similarity(df[selected_column].dropna(), unique_observations, Stop_Words=Stop_Words)    
        column_consistency_score = average_consistency_score(cosine_sim_matrix, threshold)    
        all_consistency_scores.append(column_consistency_score)    
  
    # Calculate the average of all consistency scores    
    overall_avg_consistency = sum(all_consistency_scores) / len(all_consistency_scores) if all_consistency_scores else None    
  
    return overall_avg_consistency

# Accuracy Type 1

# Function 1: Using isdigit to find non-numerical entries
def find_non_digits(s):  
    # Ensure the value is treated as a string  
    s = str(s)  
    return [char for char in s if not (char.isdigit() or char == '.')]  

# Function 2 : Calculate the score
def accuracy_score(dataset_path, selected_columns):
    adf = read_data(dataset_path)
    selected_columns = [col for col in adf.columns if col in selected_columns] 

    all_accuracy_scores = []
    
    for column_name in selected_columns:  
        # Drop NA, null, or blank values from column  
        column_data = adf[column_name].dropna()  
          
        total_rows = len(column_data)  
          
        if total_rows > 0:  # to avoid division by zero  
            non_digit_chars_per_row = column_data.apply(find_non_digits)  
            non_numerical_count = non_digit_chars_per_row.apply(lambda x: len(x) > 0).sum()   
            accuracy_score = (total_rows - non_numerical_count) / total_rows  
            all_accuracy_scores.append(accuracy_score)    
  
    overall_accuracy_score = sum(all_accuracy_scores) / len(all_accuracy_scores) if all_accuracy_scores else None   

    return overall_accuracy_score

# Accuracy Type 2

def find_outliers_iqr(dataset_path, selected_columns, groupby_column = None, threshold = 1.5, minimum_score= 0.85):  
    df = read_data(dataset_path)
    outliers_dict = {}

   # If a groupby column is specified, perform the IQR calculation within each group  
    if groupby_column:  
        grouped = df.groupby(groupby_column)  
        for column in selected_columns:  
            # Apply the outlier detection for each group  
            outliers = grouped[column].apply(lambda x: ((x < x.quantile(0.25) - threshold * (x.quantile(0.75) - x.quantile(0.25))) |  
                                                        (x > x.quantile(0.75) + threshold * (x.quantile(0.75) - x.quantile(0.25))))) 
            # Combine the outlier Series into a single Series that corresponds to the original DataFrame index  
            outliers_dict[column] = (1 - outliers.groupby(groupby_column).mean())
    else:  
        # Perform the IQR calculation on the whole column if no groupby column is specified  
        for column in selected_columns:  
            Q1 = df[column].quantile(0.25)  
            Q3 = df[column].quantile(0.75)  
            IQR = Q3 - Q1  
  
            lower_bound = Q1 - threshold * IQR  
            upper_bound = Q3 + threshold * IQR  
  
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)  
            outliers_dict[column] = (1 - outliers.mean())
    
    # compute final score  
    #total_groups = len(outliers_dict)  
    #groups_above = sum(1 for score in outliers_dict.values() if score > minimum_score)  
    #final_score = groups_above / total_groups if total_groups > 0 else 0  
    
    final_score = {}
    
    for key in outliers_dict.keys():
        arr = outliers_dict[key].values
        value_out = np.sum(arr > minimum_score)/len(arr)
        final_score[key] = value_out
  
    return outliers_dict, final_score   

# Accuracy Type 3

# function 1: finding duplicates
def find_duplicates_and_percentage(dataset_path):

    df = read_data(dataset_path)

    # Find duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)]
    
    # Calculate percentage of duplicate rows
    total_rows = len(df)
    total_duplicate_rows = len(duplicate_rows)
    percentage_duplicate = 1-(total_duplicate_rows / total_rows)
    
    # Print duplicate rows
    print("Duplicate Rows:")
    print(duplicate_rows)
    
    # Print percentage of duplicate rows
    print(f"\nDuplication Score: {percentage_duplicate*100}%")
    
# Completeness

def completeness_test(dataset_path, exclude_columns = [], threshold=0.75):  
    dataset = read_data(dataset_path)

    # Exclude the 'Comment' column if it exists in the dataset  
    if 'Comment' in dataset.columns:  
        dataset = dataset.drop(columns=['Comment'])  
  
    # Exclude columns in exclude_columns if they exist in the dataset    
    dataset = dataset.drop(columns=[col for col in exclude_columns if col in dataset.columns])
    
    # Calculate the percentage of non-null (non-missing) values in each column  
    is_null_percentage = dataset.isna().mean()  
      
    # Identify columns with non-null percentage less than or equal to the threshold  
    columns_to_keep = is_null_percentage[is_null_percentage <= threshold].index  
      
    # Keep columns that exceed the threshold of non-null values  
    dataset2 = dataset[columns_to_keep]  
      
    # Calculate the actual percentage of non-missing values in the dataset  
    total_non_missing = dataset2.notna().sum().sum()  
    total_obs = dataset2.shape[0] * dataset2.shape[1]  
    completeness_score = total_non_missing / total_obs  
      
    return completeness_score

# Timeliness

from datetime import datetime

def calc_timeliness(refresh_date, cycle_day):
    refresh_date = pd.to_datetime(refresh_date)
    unupdate_cycle = np.max([((datetime.now() - refresh_date).days/cycle_day)-1, 0])

    #unupdate_cycle = np.floor((datetime.now() - refresh_date).days/cycle_day)
    #print((datetime.now() - refresh_date).days/cycle_day)
    return np.max([0, 100 - (unupdate_cycle * (100/3))])

# Output reports
# Consistency Type 2

def compare_datasets(dataset_path, column_mapping, ref_dataset_path = None):      
    # Read the data file      
    df = read_data(dataset_path)      
      
    # Initialize ref_df if a ref dataset is provided      
    if ref_dataset_path:      
        df_ref = read_data(ref_dataset_path)      
        ref_data = True #Flag to indicate we are using a ref dataset      
    else:      
        ref_data = False #No ref dataset, compare within the same dataset      
      
    for selected_column, m_selected_column in column_mapping.items():        
        if ref_data:      
             # Compare to ref dataset        
            unique_observations = get_names_used_for_column(df_ref, m_selected_column)    
        else:        
            # Use own column for comparison        
            unique_observations = get_names_used_for_column(df, selected_column)    
              
        # Iterate over each row in the selected column    
        column_results = []  
        for value in df[selected_column]:    
            # Check if the value exists in unique_observations and append the result to column_results  
            if pd.isnull(value):  
                column_results.append(False) # or True, depending on how you want to handle NaN values  
            else:  
                column_results.append(value in unique_observations)  
          
        # Add the results as a new column in the DataFrame  
        df[selected_column + '_comparison'] = column_results  
        
    return df  

# Accuracy Type 1

# Function 1: Using isdigit to find non-numerical entries  
def find_non_digits(s):    
    # Ensure the value is treated as a string    
    s = str(s)    
    return [char for char in s if not (char.isdigit() or char == '.')]  
  
# Function 2 : Check if each row has only numbers in each selected column and add results as new columns  
def add_only_numbers_columns(dataset_path, selected_columns):  
    adf = read_data(dataset_path)  
    selected_columns = [col for col in adf.columns if col in selected_columns]   
  
    for column_name in selected_columns:    
        adf[column_name+'_Only_Numbers'] = adf[column_name].apply(lambda x: len(find_non_digits(x)) == 0)  
  
    return adf  