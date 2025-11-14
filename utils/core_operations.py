import numpy as np  
import os
import pandas as pd
from datetime import datetime
from ast import literal_eval
import io

# ----------------------- Core Operations -------------------------------

# --- Utils Helpers ---
# ANSI escape code for red text for console output 
RED = "\033[31m"  
RESET = "\033[0m" 

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

# --- Input / Output ---
""" Reading the dataset file 
- Function to read either csv or xlsx data. If input is already a data frame, it will return the input.
"""
def read_data(dataset_path, dataset_name=None):
    # Case where input is already a dataframe
    if isinstance(dataset_path, pd.DataFrame):
        return dataset_path

    # Case name if provided as a separate input, required when handeling streamlit input
    if dataset_name:
        _, file_extension = os.path.splitext(dataset_name)
    else:
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
def output_log_score(test_name, dataset_name, score, selected_columns, excluded_columns, isStandardTest, test_fail_comment, 
                     errors, dimension, metric_log_csv, threshold=None, minimum_score=None, return_log=False):
    # Convert score to a percentage
    try:
        percentage_score = f"{float(score['value']) * 100:.2f}%" if score['value'] else '0%'
    except:
        percentage_score = '0%'
    
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

    # If return_log is set to true return the row, this allows the UI to visualize logs from metrics run
    if return_log:
        return new_row
    else:
        return None

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
        elif (metric == "C3"):
            inconsistent_columns = []

            # Find columns with Normalized {column}_comparison and 'False' entries
            comparison_columns = [col for col in df.columns if col.startswith('Normalized ') and col.endswith('_comparison')]
            for column in comparison_columns:
                if (df[column] == False).sum() > 0:
                    # Add original test column name
                    inconsistent_columns.append(column[len("Normalized "):-len("_comparison")])         
            
            return "The following columns may have names that do not resemble a province/territory: " + ', '.join(inconsistent_columns) + "."
        elif (metric == "C4"):
            inconsistent_columns = []

            # Find columns with _inconsistent and 'True' entries
            comparison_columns = [col for col in df.columns if col.endswith('_inconsistent')]
            for column in comparison_columns:
                if (df[column] == True).sum() > 0:
                    # Add original test column name
                    inconsistent_columns.append(column[:-len("_inconsistent")])   
            
            return "The following columns may have dates inconsistent with a date-time formatting: " + ', '.join(inconsistent_columns) + "."
        elif (metric == "C5"):
            invalid_columns = []

            # Find columns with _invalid and 'True' entries
            comparison_columns = [col for col in df.columns if col.endswith('_invalid')]
            for column in comparison_columns:
                if (df[column] == True).sum() > 0:
                    # Add original test column name
                    invalid_columns.append(column[:-len("_invalid")])   
            
            return "The following columns may have invalid latitude/longitude coordinates: " + ', '.join(invalid_columns) + "."
        elif (metric == "S1"):
            columns = df.columns
            if columns[0][0] > 0:
                return "Metadata exists for given dataset"
            else:
                return "Metadata does not exist for given dataset"
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
        elif (metric == "A3"):
            component_columns = ', '.join(selected_columns[:-1])
            agg_column = selected_columns[-1]
            
            return "The aggregated column " + agg_column + " may contain values not equal to the sums of its component columns: " + component_columns + "." if len(df) > 2 else "The aggregated column " + agg_column + " equals the sum of its component columns: " + component_columns + "."
        elif (metric == "A4"):
            invalid_pairs = []

            # Find column pairs not in chronological order
            n_pairs = int(len(selected_columns) / 2)
            column_pairs_check = df.iloc[:,-n_pairs:].columns
            for column_pair in column_pairs_check:
                if (df[column_pair] == True).sum() > 0:
                    invalid_pairs.append(column_pair)    
            # Add original test column pair names
            invalid_pairs_list = [tuple(s.split("_after_")) for s in invalid_pairs] 

            return "Column pairs that may contain dates not in chronological order: " + ", ".join(f"({a}, {b})" for a, b in invalid_pairs_list) + "."
        elif (metric == 'P1'):
            columns = ', '.join(df.columns)

            return "Columns that exceed the threshold of non-null values: " + columns + "."
        elif (metric == 'P2'):
            strength = ""
            
            if threshold < 0.5:
                strength = "little to no"
            elif threshold == 0.5:
                strength = "a possible"
            elif threshold > 0.5 and threshold < 0.75:
                strength = "a possibly moderate"
            elif threshold >= 0.75:
                strength = "a possibly strong"
                
            return f"There are {len(df['features'])} feature pair(s) with " + strength + f" association in missingness, given a correlation threshold of {threshold}."
        elif (metric == 'I1'):
            columns_above_threshold = ", ".join(df['var1'].unique())
        
            return f"Proxy variables whose correlation with sensitive features is higher than {threshold}: " + columns_above_threshold + "."
        elif (metric == 'U1'):

            return "Duplicate rows found in the dataset." if len(df.columns) > 0 else "No duplicate rows found in the dataset."
        else: 
            return None
    except Exception as e:
        print(f"When trying to create one line summary for {metric}, the following error occurred: {e}")
        return None

# --- Scoring / Validation ---
"""
- Determines if the given set of weights meet the follwing criteria
    - The number of weights match the number of metrics/dimensions
    - The weights add up to 1
    weights: the weights to evaluate.
    scores: a list of dictionaries containing each metric and the score from it.
    type: if set to 'metric' will using metrics in error message, otherwise uses dimensions.
    Return: Tuple of weights and boolean indicating if the weights needed to be set to default.
"""
def are_weights_valid(weights: dict, scores: list[dict], type='metric') -> tuple:
    weight_type = 'metrics' if type == 'metric' else 'dimensions'

    # Handle string inputes
    if weights == '' or weights == '{}':
        return {}, True
    if isinstance(weights, str):
        try:
            weights = weights.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"') # sanitize quates to prevent syntax errors
            weights = literal_eval(weights) if weights.strip() else {}
            if not isinstance(weights, dict):
                return {}, False
        except:
            return {}, False

    try:
        # Ensure number of weights is the name as the number of metric run (else use default weights)
        if len(weights) != len(scores):
            weights = {}
            print(f'{RED}Number of weights does not match number of {weight_type} run, using default weights instead!{RESET}')
            return weights, False
        
        # Ensure weights add to 1
        else:
            total_weight = 0
            for metric, weight in weights.items():
                total_weight += weight
            if total_weight < 1.0:
                weights = {}
                print(f'{RED}Weights do not add up to 1.0, using default weights instead!{RESET}')
                return weights, False
    except:
        print(f'{RED}Provided weights are not structured properly, ensure correct names and format is used. Using default weights instead!{RESET}')
        return {}, False
    
    return weights, True

""" Takes a list of scores from all metrics in a given dimension and calculates the dimension total score
    dimension_type: the name of the dimension being evaluated.
    scores: a list of dictionaries containing each metric and the score from it.
    weights: a multiply
    Return: Dictionary with dimension (the dimension's name) and score (the overall score for the dimension)
"""
def calculate_dimension_score(dimension_type: str, scores: list[dict], weights: dict) -> dict:
    # Custom weights are provided ensure it is correctly entered
    if (weights != {}):
        weights, valid = are_weights_valid(weights, scores)

    score_value = 0
    for score in scores:
        try:
            numeric_score = 0 if not score['value'] else score['value'] # If test failed make the score 0
            weight = weights[score['metric']] if score['metric'] in weights else 1.0 / len(scores)
            score_value += numeric_score * weight
        except: # Case where value from metric is 'No valid XX results generated
            score_value += 0
    
    return {"dimension": dimension_type, "score": score_value}

"""
Takes a list of scores (containing dimension name and total score) for each dimension.
Determines a grade for the DQ based on the inputted score.
"""
def calculate_DQ_grade(scores: list[dict], weights={}) -> str:
    # Custom weights are provided ensure it is correctly entered
    if (weights != {}):
        weights, valid = are_weights_valid(weights, scores, type='dimension')

    total_score = 0
    for score in scores:
        # TODO: Check if Uniqueness and completeness are in score list
        numeric_score = 0 if score['score'] is None else score['score']
        weight = weights[score['dimension']] if score['dimension'] in weights else 1 / len(scores)
        total_score += numeric_score * weight

    # Based on conditions (raw score, ) return letter grade
    if total_score > 0.9: # TODO: add limit if required dimensions are not there
        return "Exceptional"
    elif total_score > 0.8:
        return "High"
    elif total_score > 0.7:
        return "Good"
    elif total_score > 0.5:
        return "Minimum"
    else:
        return "Needs Improvement"

