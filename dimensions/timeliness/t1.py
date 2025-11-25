import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "T1" 

""" Class to represent an individual metric for the Timeliness dimension.
    
    Goal: Ensure that the data is up-to-date and available when needed. 
    Timely data is delivered at the right time to support decision-making processes.

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
        # self.t1_column_names = t1_column_names

        # TODO: Set threshold and selected columns for this metric (used in summary output) 
        self.threshold = None
        self.selected_columns = None 
    
    """ Timeliness Type 1 (T1): 
    TODO: Provide a description of what this script does.
    """
    # TODO: Replace with the logic for this metric, where the final score should be called timeliness_score
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)

        timeliness_score = None # Placeholder for calculated metric score

        tdf = None # Placeholder for output report (returned when return_type="dataset")

        # Conditional return logic
        if self.return_type == "score":
            return timeliness_score, None
        elif self.return_type == "dataset":
            if not timeliness_score: 
                return f"No valid {METRIC} results generated", None
                
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=tdf)
            return timeliness_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame)
       
""" Creates a MetricMetadata instance for a single metric, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Timeliness"

    # Define instance for metric, replace with metric that requires parameters
    t1_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    # t1_metadata.add_parameter()
    
    return t1_metadata