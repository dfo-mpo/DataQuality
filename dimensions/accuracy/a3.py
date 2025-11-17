import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "A3"

""" Class to represent an individual metric for the Accuracy dimension.

    Goal: Ensure that the data correctly represents the real-world values it is intended to model. 
    Accurate data is free from errors and is a true reflection of the actual values.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
a3_column_names: columns used from the dataset for the A3 metric.
a3_agg_column: aggregate column used to evaluate a3_column_names for A3 metric.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, a3_column_names=[], a3_agg_column=[], threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        self.a3_column_names = a3_column_names
        self.a3_agg_column = [a3_agg_column] if isinstance(a3_agg_column, str) else a3_agg_column

        self.threshold = None
        self.selected_columns = self.a3_column_names + self.a3_agg_column
    
    """ Accuracy Type 3 (A3): Checks whether aggregated column (eg. Total) values are equal to the expected sum of their component columns.
    """
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)

        # Fill NA with 0
        df_expected = df[self.a3_column_names].fillna(0)
        aggregated = df[self.a3_agg_column].fillna(0)
    
        # Compute expected total (row-wise sum)
        expected = df_expected.sum(axis=1)
    
        # Compare aggregrated to expected (flag inequal entries)
        matched = ~aggregated.eq(expected, axis=0)
    
        # Take subset of data where aggregated != expected
        inequal = matched.any(axis=1)
        adf = df[inequal].copy()
        
        # Compute score
        accuracy_score = 1 - (matched.sum() / len(matched)).iloc[0]

        # Conditional return logic
        if self.return_type == "score":
            return accuracy_score, None
        elif self.return_type == "dataset":
            if not accuracy_score: 
                return f"No valid {METRIC} results generated", None
                
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=adf)
            return accuracy_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame)
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Accuracy"
    
    # Define instance for metric
    a3_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    a3_metadata.add_parameter('a3_column_names', 'A3 Column Names', ParameterType.MULTI_SELECT, default=[])
    a3_metadata.add_parameter('a3_agg_column', 'A3 Aggregate Column', ParameterType.SINGLE_SELECT)
    
    return a3_metadata