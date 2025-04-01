import re  
import numpy as np  
import os
import glob 
import pandas as pd  
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity  
from difflib import SequenceMatcher  

# ----------------------- Consistency Dimension Utils -------------------------------
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
Extract all numbers from the input text and return them as a list of strings.
"""
def extract_numbers(text):  
    return re.findall(r"\d+", text)  

"""
Remove numbers with 1 or 2 digits from the input text.
"""
def remove_short_numbers(text):  
    return re.sub(r"\b\d{1,4}\b", "", text)  

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
Calculate the similarity between two strings using the SequenceMatcher from difflib.
Return the similarity ratio.
"""
def string_similarity(str1, str2):  
    return SequenceMatcher(None, str1, str2).ratio()  
    
def get_names_used_for_column(df, column_name):  
    unique_observations = pd.unique(df[column_name].dropna().values.ravel())  
    return unique_observations  

"""
Calculate the cosine similarity between lists of texts using TF-IDF vectorization.
"""
def calculate_cosine_similarity(text_list, ref_list, stop_words):  
    vectorizer = TfidfVectorizer(stop_words=stop_words, analyzer="word", ngram_range=(1, 2))  
    ref_vec = vectorizer.fit_transform(ref_list)  
    text_vec = vectorizer.transform(text_list)  
    return cosine_similarity(text_vec, ref_vec)  

"""
Calculate the average consistency score based on the cosine similarity matrix and a given threshold.
"""
def average_consistency_score(cosine_sim_matrix, threshold=0.91):  
    num_rows, num_columns = cosine_sim_matrix.shape  
    inconsistency = 0  
    for i in range(num_rows):  
        if np.any((cosine_sim_matrix[i] > threshold) & (cosine_sim_matrix[i] <= 1.0000000)):  
            inconsistency += 1  
    return (num_rows - inconsistency) / num_rows  

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

# ----------------------- Accuracy Dimension Utils -------------------------------
"""
For Accuracy A1, find non-numerical characters in a string.
"""
# Function 1: Using isdigit to find non-numerical entries
def find_non_digits(s):
    # Ensure the value is treated as a string
    s = str(s)

    # Track whether we've seen the first "-" or the first "." for negative values or decimals
    first_dash_skipped = False
    first_period_skipped = False

    result = []
    for i, char in enumerate(s):
        if char in ["-", "."] :
            if (char == "-" and first_dash_skipped) or (char == "." and first_period_skipped):
                result.append(char)  # Keep dashes and periods that are not the first ones 
            elif char == "-":
                first_dash_skipped = True
            elif char == ".":
                first_period_skipped = True
        
        elif not char.isalnum():  # Keep only the symbols
            result.append(char)

    return "".join(result)  # Convert list back to a string

"""
For Accuracy A1, append columns that indicate true or false for whether there are symbols in numerics
and create a new DataFrame with new column(s)
"""
def add_only_numbers_columns(df, selected_columns):    
    selected_columns = [col for col in df.columns if col in selected_columns]   

    for column_name in selected_columns:    
        df[column_name + '_Only_Numbers'] = df[column_name].apply(
            lambda x: np.where(pd.isnull(x), True, len(find_non_digits(x)) == 0)
        )    

    return df

"""
For Accuracy A3, .
"""

# ----------------------- All Dimension Utils -------------------------------
# ANSI escape code for red text for console output 
RED = "\033[31m"  
RESET = "\033[0m" 

""" Reading the dataset file 
- Function to read either csv or xlsx data
"""
def read_data(dataset_path):
    _, file_extension = os.path.splitext(dataset_path)
    if file_extension == ".csv":
        try:  
            df = pd.read_csv(dataset_path, encoding="utf-8-sig")  
        except UnicodeDecodeError:  
            df = pd.read_csv(dataset_path, encoding="cp1252") 
    elif file_extension == ".xlsx":
        df = pd.read_excel(dataset_path)
    else:
        print("Unsupported file type")
        df = None
    return df

# Function to log a new row into the DQS_Output_Log_xx.xlsx file
def output_log_score(test_name, dataset_name, score, selected_columns, excluded_columns, isStandardTest, test_fail_comment, errors, dimension, metric_log_csv, threshold=None, minimum_score=None):
    # Convert score to a percentage
    percentage_score = score
    
    # Load the Excel file into a DataFrame
    log_file = "DQS_Output_Log_Test.xlsx"
    
    # Set threshold to "No threshold" if it is not provided
    if threshold is None:
        threshold_value = "no threshold"
    else:
        threshold_value = threshold

    # If selected_columns is None, assume "All" was tested
    if excluded_columns:
        columns_tested = "All columns excluding " + ", ".join(excluded_columns)
    elif selected_columns is None:
        columns_tested = "All columns"
    else:
        # Convert selected_columns list to a string if specific columns are provided
        columns_tested = ", ".join(selected_columns)
    
    # If isStandard then the test is a standard metric test, otherwise it is a custom test (not created by the SDPA team)
    standard_or_custom_value = "Standard" if isStandardTest else "Custom"
    
    # Including dimension (which may have to be defined by what class this data is in, but for testing purposes, it's hardcoded for now)
    # TODO 
    
    # Try loading the existing Excel file
    try:
        df = read_data(log_file)
    except FileNotFoundError:
        # Create an empty DataFrame if file doesn't exist (shouldn't be the case if you already created it)
        df = pd.DataFrame(columns=["Dataset", "Dimension", "Test", "Selected_Columns", "Threshold", "Score", "Run_Time_and_Date", "New_or_Existing_Test", "One_Line_Summary", "Errors", "Why_Did_the_Test_Fail"])

    # Prepare the new row as a DataFrame
    new_row = pd.DataFrame({
        "Dataset": [dataset_name],
        "Dimension": [dimension],
        "Test": [test_name],
        "Selected_Columns": [columns_tested],  # Add the list of columns tested
        "Threshold":[threshold_value],
        "Score": [percentage_score],
        "Run_Time_and_Date": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        "New_or_Existing_Test": [standard_or_custom_value],
        "One_Line_Summary": [get_onesentence_summary(test_name, metric_log_csv, threshold=minimum_score if minimum_score != None else threshold_value)] if metric_log_csv else None,
        "Errors": [errors], # TODO: expand
        "Why_Did_the_Test_Fail": [test_fail_comment] # TODO: expand
    })

    # Append the new row to the DataFrame
    df = pd.concat([df, new_row], ignore_index=True)

    # Save the updated DataFrame back to the Excel file
    df.to_excel(log_file, index=False)

"""
- Function to extract dataset name from a path
"""
def get_dataset_name(dataset_path):  
    # Extract the file name from the path (e.g., 'Dataset_A.csv')
    file_name = os.path.basename(dataset_path)
    # Split the file name to remove the extension (e.g., 'Dataset_A')
    dataset_name = os.path.splitext(file_name)[0]
    return dataset_name

"""
- Function to read metric log if it exists
"""
def get_onesentence_summary(metric: str, logging_path: str, threshold: int | None) -> str:
    try:
        df = pd.read_csv(logging_path)

        # Create 1 sentence summary
        if (metric == 'C1'):
            max_scores = df.groupby('Column Source')['Similarity Score'].max()
            filtered_sources = max_scores[max_scores > threshold]
            filtered_sources_str = ', '.join(filtered_sources.index.tolist()) 

            return "The following columns contain a score above the threshold " + filtered_sources_str + "."
        elif (metric == 'C2'):
            columns = df.columns

            # Find columns with _comparison  
            simular_columns = []  
            for column in columns:
                if f"{column}_comparison" in columns:  
                    simular_columns.append(column)
            simular_columns_str = ', '.join(simular_columns)

            return "The following columns resembles a reference data column: " + simular_columns_str + "."
        elif (metric == 'A1'):
            columns = df.columns

            # Find columns with _ONLY_NUMBERS equivalents  
            columns_with_equivalents = []  
            for column in columns:
                if f"{column}_Only_Numbers" in columns:  
                    columns_with_equivalents.append(column)
            columns_with_equivalents_str = ', '.join(columns_with_equivalents)

            return "Columns that contain only numbers " + columns_with_equivalents_str + "."
        elif (metric == 'A2'):
            # Find columns (headers) where any value in the column is above the threshold  
            columns_above_threshold = df.loc[:, (df > threshold).any()]  
            
            # Get the column names (headers) that meet the condition  
            headers_with_outliers = columns_above_threshold.columns.tolist()  
            
            # Output the results  
            return "Coloumns with outliers:"+ headers_with_outliers + "."
        elif (metric == 'P1'):
            columns = ', '.join(df.columns)

            return "Columns that exceed the threshold of non-null values: " + columns + "."
        elif (metric == 'U1'):

            return "Duplicate rows found in the dataset." if df.columns > 0 else "No duplicate rows found in the dataset."
        else: 
            return None
    except Exception as e:
        print(f"When trying to create one line summary for {metric}, the following error occured: {e}")
        return None