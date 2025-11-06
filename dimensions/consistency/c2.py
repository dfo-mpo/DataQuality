import numpy as np  
import pandas as pd
from utils import utils
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "C2" 

""" Class to represent an individual metric for the Consistency dimension.
    
    Goal: Ensure that data is consistent across different datasets and systems. 
    Consistent data follows the same formats, standards, and definitions, and there are no contradictions within the dataset.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
c2_column_mapping: mapping of columns to evaluate and reference columns for each one in the C2 metric. The pattern for comparison is 'dataset column' : 'reference column'.
c2_threshold: threshold for consistency score that is acceptable for C2 metric.
ref_dataset_path: Reference dataset that selected dataset columns are compared to in C2 metric.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, c2_column_mapping=[], c2_threshold=0.91, c2_stop_words=["activity"], ref_dataset_path=None, threshold=None, selected_columns=None):
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
    def run_metric(self):    
        # Read the data file
        df = utils.read_data(self.dataset_path)

        # Initialize ref_df if a ref dataset is provided
        if self.ref_dataset_path:
            df_ref = utils.read_data(self.ref_dataset_path)
            ref_data = True  # Flag to indicate we are using a ref dataset
        else:
            ref_data = False  # No ref dataset, compare within the same dataset

        all_consistency_scores = []

        for selected_column, m_selected_column in self.c2_column_mapping.items():
            if ref_data:
                # Compare to ref dataset
                unique_observations = utils.get_names_used_for_column(df_ref, m_selected_column)
            else:
                # Use own column for comparison
                unique_observations = utils.get_names_used_for_column(df, selected_column)

            cosine_sim_matrix = utils.calculate_cosine_similarity(
                df[selected_column].dropna(), unique_observations, stop_words=self.c2_stop_words
            )
            column_consistency_score = utils.average_c2_consistency_score(
                cosine_sim_matrix, self.c2_threshold
            )
            all_consistency_scores.append(column_consistency_score)

        # Calculate the average of all consistency scores
        overall_avg_consistency = (
            sum(all_consistency_scores) / len(all_consistency_scores)
            if all_consistency_scores
            else None
        )
        
        # add conditional return logic
        if self.return_type == "score":
            return overall_avg_consistency, None
        elif self.return_type == "dataset":
            if not overall_avg_consistency :
                return f"No valid {METRIC} results generated", None
            
            final_df = utils.compare_datasets(df, selected_column, unique_observations)  
            output_file = utils.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=final_df)
            return overall_avg_consistency, output_file  # Return the file name
        else:
            return df, None  # Default return value (DataFrame)
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for metric
    c2_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    c2_metadata.add_parameter('c2_threshold', 'C2 Threshold', ParameterType.DECIMAL, value='0.91', step = 0.01)
    c2_metadata.add_parameter('c2_stop_words', 'C2 Stop Words', ParameterType.TEXT_INPUT, value='["activity"]', hint="Words filtered for C2 metric simularity calculations")
    c2_metadata.add_parameter('ref_dataset_path', 'Reference Dataset File', ParameterType.FILE_UPLOAD)
    c2_metadata.add_parameter('c2_column_mapping', 'C2 Column Mapping', ParameterType.TEXT_INPUT, placeholder="e.g., {'Column1': 'Reference1', 'Column2': 'Reference2',}")
    
    return c2_metadata 