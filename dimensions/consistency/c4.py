import numpy as np  
import pandas as pd
from utils import core_operations, item_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "C4" 

""" Class to represent an individual metric for the Consistency dimension.
    
    Goal: Ensure that data is consistent across different datasets and systems. 
    Consistent data follows the same formats, standards, and definitions, and there are no contradictions within the dataset.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
c4_column_names: columns used from the dataset for the C4 metric.
c4_format: date-time format that selected dataset columns are compared to in C4 metric.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, c4_column_names=[], c4_format='%Y-%m-%d %H:%M:%S', threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

        self.c4_column_names = c4_column_names
        self.c4_format = c4_format

        self.threshold = None
        self.selected_columns = self.c4_column_names 
    
    """ Consistency Type 4 (C4): Checks whether the dataset follows standard date-time ISO 8601 formatting (or any format defined by the user).
    """
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)
        results = df.copy()
        all_consistency_scores = {}

        # Check date-time formating on the whole column
        for column in self.c4_column_names:
            # Remove NA values
            df_clean = df.dropna(subset=[column])
    
            # Calculate proportion of incorrectly formated values in each column
            results[f"{column}_inconsistent"] = df_clean[column].apply(lambda x: item_operations.inconsistent_datetime(str(x), self.c4_format))
            all_consistency_scores[column] = 1 - results[f"{column}_inconsistent"].mean()
    
        # Take subset of data with inconsistent date-time formatting
        comparison_columns = [f"{col}_inconsistent" for col in self.c4_column_names]
        inconsistent = results[comparison_columns].any(axis=1)
        inconsistent_df = results[inconsistent].copy() 
        
        # Compute average score  
        overall_consistency_score = sum(all_consistency_scores.values()) / len(all_consistency_scores)
    
        # add conditional return logic
        if self.return_type == "score":
            return overall_consistency_score, None
        elif self.return_type == "dataset":
            if not overall_consistency_score: 
                return f"No valid {METRIC} results generated", None
                
            final_df = inconsistent_df
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=final_df)
            return overall_consistency_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame) 
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for metric
    c4_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    c4_metadata.add_parameter('c4_column_names', 'C4 Column Names', ParameterType.MULTI_SELECT)
    c4_metadata.add_parameter('c4_format', 'C4 Format', ParameterType.STRING, value='%Y-%m-%d %H:%M:%S', hint="Date-time format that selected dataset columns are compared to. Use %Y (year), %M (months), and %D (days) separated by '-'. Use %H (hours), %M (minutes), and %S (seconds) separated by ':'." )
    
    return c4_metadata 