import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "U1"

""" Class to represent an individual metric for the Uniqueness dimension.
    
    Goal: Ensure that each record in the dataset is unique and there are no duplicate entries. 
    Unique data means there are no redundant records.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

        self.threshold = None
        self.selected_columns = None 
    
    """ Uniqueness Type 1 (U1):
    Find duplicated rows (what used to be known as Accuracy Type 3)
    """
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)

        # Find duplicate rows
        duplicate_rows = df[df.duplicated(keep=False)]
        
        # Calculate percentage of duplicate rows
        total_rows = len(df)
        total_duplicate_rows = len(duplicate_rows)
        percentage_duplicate = 1-(total_duplicate_rows / total_rows)
        
        # Print duplicate rows
        print("Duplicate Rows:")
        print(duplicate_rows)
        
        # Print percentage of duplicate rows
        print(f"\nDuplication Score: {percentage_duplicate*100}%")
        
        # add conditional return logic
        if self.return_type == "score":
            return percentage_duplicate, None
        elif self.return_type == "dataset":
            if not total_rows :
                return f"No valid {METRIC} results generated", None
            
            final_df = duplicate_rows  
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=final_df)
            return percentage_duplicate, output_file  # Return the file name
            
        else:
            return df, None  # Default return value (DataFrame)
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Uniqueness"

    # Define instance for metric, replace with metric that requires parameters
    u1_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    # u1_metadata.add_parameter()

    return u1_metadata