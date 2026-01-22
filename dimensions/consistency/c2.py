import numpy as np  
import pandas as pd
from utils import core_operations, table_operations, column_operations
from ui_tool.metadata import TestMetadata, ParameterType

TEST = "C2" 

""" Class to represent an individual test for the Consistency dimension.
    
    Goal: Ensure that data is consistent across different datasets and systems. 
    Consistent data follows the same formats, standards, and definitions, and there are no contradictions within the dataset.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only test scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
c2_column_mapping: mapping of columns to evaluate and reference columns for each one in the C2 test. The pattern for comparison is 'dataset column' : 'reference column'.
c2_threshold: threshold for consistency score that is acceptable for C2 test.
ref_dataset_path: Reference dataset that selected dataset columns are compared to in C2 test.
"""
class Test:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, c2_column_mapping=[], c2_threshold=1.00, c2_stop_words=["activity"], ref_dataset_path=None, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

        self.c2_column_mapping = c2_column_mapping
        self.c2_threshold = c2_threshold
        self.c2_stop_words = c2_stop_words
        self.ref_dataset_path = ref_dataset_path

        self.threshold = self.c2_threshold
        self.selected_columns = self.c2_column_mapping 
    
    """ Consistency Type 2 (C2): Compares reference data and string values in specified columns.
    The compared columns in question must be identical to the ref list, otherwise they will be penalized more harshly.
    """
    def run_test(self):    
        # Read the data file
        df = core_operations.read_data(self.dataset_path)

        # Initialize ref_df if a ref dataset is provided
        if self.ref_dataset_path is not None:
            df_ref = core_operations.read_data(self.ref_dataset_path)
            ref_data = True  # Flag to indicate we are using a ref dataset
        else:
            ref_data = False  # No ref dataset, compare within the same dataset
        
        all_consistency_scores = []

        for selected_column, m_selected_column in self.c2_column_mapping.items():
            if ref_data:
                # Compare to ref dataset
                unique_observations = column_operations.get_names_used_for_column(df_ref, m_selected_column)
            else:
                # Use own column for comparison
                unique_observations = column_operations.get_names_used_for_column(df, selected_column)

            cosine_sim_matrix = column_operations.calculate_cosine_similarity(
                df[selected_column].dropna(), unique_observations, stop_words=self.c2_stop_words
            )
            column_consistency_score = table_operations.average_c2_consistency_score(
                cosine_sim_matrix, self.c2_threshold
            )
            all_consistency_scores.append(column_consistency_score)

        # Calculate the average of all consistency scores
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
            
            cdf = column_operations.compare_datasets(df, selected_column, unique_observations)  
            output_file = core_operations.df_to_csv(self.logging_path, test=TEST.lower(), final_df=cdf)
            return consistency_score, output_file  # Return the file name
        else:
            return df, None  # Default return value (DataFrame)
       
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for test
    c2_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    c2_metadata.add_parameter('c2_threshold', 'C2 Threshold', ParameterType.DECIMAL, value='1.00', step = 0.01)
    c2_metadata.add_parameter('c2_stop_words', 'C2 Stop Words', ParameterType.STRING_LIST, value=["activity"], suggestions=["activity"])
    c2_metadata.add_parameter('ref_dataset_path', 'C2 Reference Dataset File', ParameterType.FILE_UPLOAD)
    c2_metadata.add_parameter('c2_column_mapping', 'C2 Column Mapping', ParameterType.TEXT_INPUT, placeholder="e.g., {'Column1': 'Reference1', 'Column2': 'Reference2'}")
    
    return c2_metadata 