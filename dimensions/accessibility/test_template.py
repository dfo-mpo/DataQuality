import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import TestMetadata, ParameterType

# TODO: Define test name
TEST = "S#" # Replace "S#" with the test name, e.g., "S1"

""" Class to represent an individual test for the Accessibility dimension.
    Serves as a template for creating a new Accessibility test.
    
    Goal: Ensure that data is easily accessible to authorized users when needed. 
    Accessible data is stored in a way that makes it easy to retrieve and use, while also being secure from unauthorized access.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only test scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
"""
class Test:
    # TODO: Define test specific parameters
    # Every additional parameter required by your test must be added to the __init__ header
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, # --- Add test specific parameters here ---
                 threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        # TODO: Assign test specific attributes to a self variable 
        # Example:
        # self.s#_column_names = s#_column_names

        # TODO: Set threshold and selected columns for this test (used in summary output) 
        self.threshold = None
        self.selected_columns = None 
    
    """ Accessibility Type # (S#): 
    TODO: Provide a description of what this script does.
    """    
    # TODO: Replace with the logic for this test, where the final score should be called accessibility_score
    def run_test(self):    
        df = core_operations.read_data(self.dataset_path)
        
        accessibility_score = None # Placeholder for calculated test score

        # Conditional return logic
        if not accessibility_score: 
            return f"No valid {TEST} results generated", None
        else:
            return accessibility_score, None
       
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Accessibility"

    # TODO: Define instance for test
    # Example:
    # s#_metadata = TestMetadata(dimension, TEST)
    
    # TODO: Define each parameter needed for test, use ParameterType when defining type
    # Example:
    # s#_metadata.add_parameter('s#_column_names', 'S# Column Names', ParameterType.MULTI_SELECT)

    # TODO: Replace "s#" with the test name, e.g., "s1"
    return s#_metadata 