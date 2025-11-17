import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "S1"

""" Class to represent an individual metric for the Accessibility dimension.

    Goal: Ensure that data is easily accessible to authorized users when needed. 
    Accessible data is stored in a way that makes it easy to retrieve and use, while also being secure from unauthorized access.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
s1_has_metadata: indicates whether a metadata file exists (true if exists, else false)
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, s1_has_metadata=None, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        self.s1_has_metadata = s1_has_metadata

        self.threshold = None
        self.selected_columns = None 
    
    """ Accessibility Type 1 (S1): Gives score based on if a metadata file exists for the given dataset.
    """  
    def run_metric(self):    
        # df = core_operations.read_data(self.dataset_path)
        # Add new metric asking if user has a meta data file
        accessibility_score = 1 if self.s1_has_metadata == True else 0

        sdf = False

        # Conditional return logic
        if self.return_type == "score":
            return accessibility_score, None
        elif self.return_type == "dataset":
            if not accessibility_score: 
                return f"No valid {METRIC} results generated", None
            df = pd.DataFrame({"Score": [accessibility_score]})    
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=df)
            return accessibility_score, output_file  # Return the file name
                
        else:
            return sdf, None  # Default return value (DataFrame)
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Accessibility"

    # Define instance for metric, replace with metric that requires parameters
    s1_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    s1_metadata.add_parameter("s1_has_metadata", "S1 Has Metadata", ParameterType.CHECKBOX, value=False)
    
    return s1_metadata 