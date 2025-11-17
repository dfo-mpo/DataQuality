import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

# TODO: Define metric name
METRIC = "R#" # Replace "R#" with the metric name, e.g., "R1"

""" Class to represent an individual metric for the Relevance dimension.
    Serves as a template for creating a new Relevance metric.
    
    Goal: Ensure that the data is relevant and useful for the intended purposes. 
    Relevant data meets the needs of the users and supports the business processes.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
"""
class Metric:
    # TODO: Define metric specific parameters
    # Every additional parameter required by your metric must be added to the __init__ header
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, # --- Add metric specific parameters here ---
                 threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

        # TODO: Assign metric specific attributes to a self variable 
        # Example:
        # self.r#_column_names = r#_column_names

        # TODO: Set threshold and selected columns for this metric (used in summary output) 
        self.threshold = None
        self.selected_columns = None 
    
    """ Relevance Type # (R#): 
    TODO: Provide a description of what this script does.
    """
    # TODO: Replace with the logic for this metric, where the final score should be called relevance_score
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)

        rdf = None # Placeholder for output report (returned when return_type="dataset")

        relevance_score = None # Placeholder for calculated metric score
        
        # Conditional return logic
        if self.return_type == "score":
            return relevance_score, None
        elif self.return_type == "dataset":
            if not relevance_score: 
                return f"No valid {METRIC} results generated", None
                
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=rdf)
            return relevance_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame)
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Relevance"

    # TODO: Define instance for metric
    # Example:
    # r#_metadata = MetricMetadata(dimension, METRIC)
    
    # TODO: Define each parameter needed for metric, use ParameterType when defining type
    # Example:
    # r#_metadata.add_parameter('r#_column_names', 'R# Column Names', ParameterType.MULTI_SELECT)
    
    return r#_metadata 