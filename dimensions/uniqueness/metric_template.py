import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

# TODO: Define metric name
METRIC = "U#" # Replace "U#" with the metric name, e.g., "U1"

""" Class to represent an individual metric for the Uniqueness dimension.
    Serves as a template for creating a new Uniqueness metric.
    
    Goal: Ensure that each record in the dataset is unique and there are no duplicate entries. 
    Unique data means there are no redundant records.

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
        # self.u#_column_names = u#_column_names

        # TODO: Set threshold and selected columns for this metric (used in summary output) 
        self.threshold = None
        self.selected_columns = None 
    
    """ Uniqueness Type # (U#): 
    TODO: Provide a description of what this script does.
    """
    # TODO: Replace with the logic for this metric, where the final score should be called uniqueness_score
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)

        uniqueness_score = None # Placeholder for calculated metric score

        # Conditional return logic
        if not uniqueness_score: 
            return f"No valid {METRIC} results generated", None
        else:
            return uniqueness_score, None
       
""" Creates a MetricMetadata instance for a single metric, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Uniqueness"

    # TODO: Define instance for metric
    # Example:
    # u#_metadata = MetricMetadata(dimension, METRIC)
    
    # TODO: Define each parameter needed for metric, use ParameterType when defining type
    # Example:
    # u#_metadata.add_parameter('u#_column_names', 'U# Column Names', ParameterType.MULTI_SELECT)

    # TODO: Replace "u#" with the metric name, e.g., "u1"
    return u#_metadata 