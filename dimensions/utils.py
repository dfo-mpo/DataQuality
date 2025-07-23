import re  
import numpy as np  
import os
import statistics
import pandas as pd
from functools import partial
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from difflib import SequenceMatcher  
import io

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
def average_c1_consistency_score(cosine_sim_matrix, threshold=0.91):  
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

# ----------------------- Accuracy Dimension Utils -------------------------------
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
    percentage_score = f"{float(score['value']) * 100:.2f}%" if score['value'] else '0%'
    
    # Load the Excel file into a DataFrame
    log_file = "DQS_Output_Log_Test.xlsx"
    
    # Set threshold to "No threshold" if it is not provided
    if threshold is None:
        threshold_value = "no threshold"
    else:
        threshold_value = threshold

    # If selected_columns is None, assume "All" was tested
    if excluded_columns and not excluded_columns == ['']:
        columns_tested = "All columns excluding " + ", ".join(excluded_columns)
    elif selected_columns is None or selected_columns == ['']:
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
        "One_Line_Summary": [get_onesentence_summary(test_name, metric_log_csv, selected_columns, threshold=minimum_score if minimum_score != None else threshold_value)] if metric_log_csv else None,
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
- Function to convert dataframe in CSV, 
    if a logging path is defined: Return is the name of the csv file created
    if no loggin path defined: Return the csv in memory
"""
def df_to_csv(logging_path: None|str, metric: str, final_df: pd.DataFrame):
    # Define csv file name
    base_filename=f"{logging_path}{metric}_output"
    version = 1
    while os.path.exists(f"{base_filename}_v{version}.csv"):
        version += 1
    
    # Create CSV file
    output_file = io.StringIO() if logging_path == None else f"{base_filename}_v{version}.csv"
    final_df.to_csv(output_file, index=False)

    return output_file

"""
- Function to read metric log if it exists
"""
def get_onesentence_summary(metric: str, logging_path: str|io.BytesIO, selected_columns: list[str], threshold: int | None) -> str:
    try:
        # Incase logging_path is an in memory file, reset its internal pointer first
        if not isinstance(logging_path, str):
            logging_path.seek(0)
        df = pd.read_csv(logging_path)

        # Create 1 sentence summary
        if (metric == 'C1'):
            max_scores = df.groupby('Column Source')['Similarity Score'].max()
            filtered_sources = max_scores[max_scores > threshold]
            filtered_sources_str = ', '.join(filtered_sources.index.tolist()) 

            return "The following columns contain a score above the threshold " + filtered_sources_str + "."
        elif (metric == 'C2'):
            columns = df.columns

            # Find columns with _comparison and a first entry value of 'False'
            simular_columns = []  
            for column in columns:
                if f"{column}_comparison" in columns and df[f"{column}_comparison"].iloc[0] == 'False':  
                    simular_columns.append(column)
            simular_columns_str = ', '.join(simular_columns)

            return "The following columns may have names that do not resemble a reference data column: " + simular_columns_str + "."
        elif (metric == 'A1'):
            columns = df.columns

            # Find columns with _ONLY_NUMBERS equivalents that contains a 'False' value 
            columns_with_equivalents = []  
            for column in columns:
                if f"{column}_Only_Numbers" in columns and df[f"{column}_Only_Numbers"].iloc[0] == 'False':  
                    columns_with_equivalents.append(column)
            columns_with_equivalents_str = ', '.join(columns_with_equivalents)

            return "Columns that may contain symbols: " + columns_with_equivalents_str + "."
        elif (metric == 'A2'):
            columns_below_threshold = []

            # Get groupby columns
            end_index = len(df.columns) - len(selected_columns) 
            groupby_cols = df.columns[:end_index]
            groupby_cols_str = ' (grouped by ' + ', '.join(groupby_cols) + ')'
            
            # Check if each selected column has a value below the threshold
            for column in selected_columns:
                min_value = df[column].min()
                if min_value < threshold:
                    # find average non outlier scores for column below threshold
                    outliers_avg = round(df[column].mean() * 100, 2)
                    columns_below_threshold.append(column) if (len(df.columns) == len(selected_columns)) else columns_below_threshold.append(column + " (Avg score: " + str(outliers_avg) + ")")
            columns_below_threshold_str = ', '.join(columns_below_threshold)
            
            # Output the results  
            return "There are at least 15% outliers existing in the following columns: "+ columns_below_threshold_str + "."
        elif (metric == 'P1'):
            columns = ', '.join(df.columns)

            return "Columns that exceed the threshold of non-null values: " + columns + "."
        elif (metric == 'U1'):

            return "Duplicate rows found in the dataset." if df.columns > 0 else "No duplicate rows found in the dataset."
        else: 
            return None
    except Exception as e:
        print(f"When trying to create one line summary for {metric}, the following error occurred: {e}")
        return None
    
""" Takes a list of scores from all metrics in a given dimension and calculates the dimension total score
    dimension_type: the name of the dimension being evaluated.
    scores: a list of dictionaries containing each metric and the score from it.
    weights: a multiply
    Return: Dictionary with dimension (the dimension's name) and score (the overall score for the dimension)
"""
def calculate_dimension_score(dimension_type: str, scores: list[dict], weights: dict) -> dict:
    # Custom weights are provided ensure it is correctly entered
    if (weights != {}):
        # Ensure number of weights is the name as the number of metric run (else use default weights)
        if len(weights) != len(scores):
            weights = {}
            print(f'{RED}Number of weights does not match number of metrics run, using default weights instead!{RESET}')
        
        # Ensure weights add to 1
        else:
            total_weight = 0
            for metric, weight in weights.items():
                total_weight += weight
            if total_weight < 1.0:
                weights = {}
                print(f'{RED}Weights do not add up to 1.0, using default weights instead!{RESET}')

    score_value = 0
    for score in scores:
        numeric_score = 0 if score['value'] is None else score['value'] # If test failed make the score 0
        weight = weights[score['metric']] if score['metric'] in weights else 1 / len(scores)
        score_value += numeric_score * weight
    
    return {"dimension": dimension_type, "score": score_value}

# Takes a list of scores (containing dimension name and total score) for each dimension.
# Determines a grade for the DQ based on the inputted score.
def calculate_DQ_grade(scores: list[dict]) -> str:
    total_score = 0
    for score in scores:
        # TODO: Check if Uniqueness and completeness are in score list
        total_score += score["score"]
    
    average_score = total_score / len(scores)

    # Based on conditions (raw score, ) return letter grade
    if average_score > 0.9: # TODO: add limit if required dimensions are not there
        return "A"
    elif average_score > 0.8:
        return "B"
    elif average_score > 0.7:
        return "C"
    else:
        return "D"
