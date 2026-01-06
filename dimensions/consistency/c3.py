import numpy as np  
import pandas as pd
from utils import core_operations, table_operations, column_operations, item_operations
from ui_tool.metadata import TestMetadata, ParameterType

TEST = "C3" 

""" Class to represent an individual test for the Consistency dimension.
    
    Goal: Ensure that data is consistent across different datasets and systems. 
    Consistent data follows the same formats, standards, and definitions, and there are no contradictions within the dataset.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only test scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
c3_column_names: columns used from the dataset for the C3 test.
c3_threshold: threshold for simulatrity score that is acceptable for C3 test.
"""
class Test:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, c3_column_names=[], c3_threshold=0.91, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

        self.c3_column_names = c3_column_names
        self.c3_threshold = c3_threshold

        self.threshold = self.c3_threshold
        self.selected_columns = self.c3_column_names 
    
    """ Consistency Type 3 (C3): Compares province/territory names (reference data) and string values in specified columns using Levenshtein Similarity Ratio.
    Levenshtein Similarity Ratio = 1 - (normalized Levenshtein Distance), where a score of 1 means the strings are identical.
    """
    def run_test(self):    
        df = core_operations.read_data(self.dataset_path) 
        all_consistency_scores = []
        compare_df = pd.DataFrame()
    
        # Initialize reference data (lowercased province/territory names)
        arr_ref_normalized = np.array([name.lower() for name in item_operations.province_abbreviations.values()])
    
        for column in self.c3_column_names:
    
            # Normalize entries 
            df[f"Normalized {column}"] = df[column].apply(item_operations.normalize_text)
    
            # Calculate Levenshtein Similarity Ratio matrix and average consistency score based on matrix and threshold
            levenshtein_sim_matrix = column_operations.calculate_levenshtein_similarity(df[f"Normalized {column}"].dropna(), arr_ref_normalized)    
            column_consistency_score = table_operations.average_c3_consistency_score(levenshtein_sim_matrix, self.c3_threshold)
            all_consistency_scores.append(column_consistency_score)

            # Compare to reference data and add comparison column to dataset
            compare_df = column_operations.compare_datasets(df, f"Normalized {column}", arr_ref_normalized)

        # Drop normalized columns for output report
        columns_to_drop = [f"Normalized {col}" for col in compare_df.columns]
        compare_df = compare_df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Take subset of data inconsistent with reference data 
        comparison_columns = [f"Normalized {col}_comparison" for col in self.c3_column_names]
        inconsistent = ~compare_df[comparison_columns].all(axis=1)
        cdf = compare_df[inconsistent].copy() 
            
        # Compute average score
        consistency_score = (
            sum(all_consistency_scores) / len(all_consistency_scores)
            if all_consistency_scores
            else None
        )
        
        # Conditional return logic
        if self.return_type == "score":
            return consistency_score, None
        elif self.return_type == "dataset":
            if not consistency_score: 
                return f"No valid {TEST} results generated", None

            output_file = core_operations.df_to_csv(self.logging_path, test=TEST.lower(), final_df=cdf)
            return consistency_score, output_file  # Return the file name
                    
        else:
            return df, None  # Default return value (DataFrame)
       
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for test
    c3_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    c3_metadata.add_parameter('c3_column_names', 'C3 Column Names', ParameterType.MULTI_SELECT, default=[])
    c3_metadata.add_parameter('c3_threshold', 'C3 Threshold', ParameterType.DECIMAL, value='0.91', step = 0.01)

    return c3_metadata 