import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import TestMetadata, ParameterType

# TODO: Define test name
TEST = "A#" # Replace "A#" with the test name, e.g., "A1"

""" Class to represent an individual test for the Accuracy dimension.
    Serves as a template for creating a new Accuracy test.
    
    Goal: Ensure that the data correctly represents the real-world values it is intended to model. 
    Accurate data is free from errors and is a true reflection of the actual values.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only test scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
# TODO: Add description of test specific parameters here
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
        # self.a#_column_names = a#_column_names

        # TODO: Set threshold and selected columns for this test (used in summary output) 
        self.threshold = None
        self.selected_columns = None 
    
    """ Accuracy Type # (A#): 
    TODO: Provide a description of what this script does.
    """
    # TODO: Replace with the logic for this test, where the final score should be called accuracy_score
    def run_test(self):    
        df = core_operations.read_data(self.dataset_path)
     
        accuracy_score = None # Placeholder for calculated test score

        # Conditional return logic
        if not accuracy_score: 
            return f"No valid {TEST} results generated", None
        else:      
             return accuracy_score, None
       
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Accuracy"

    # TODO: Define instance for test
    # Example:
    # a#_metadata = TestMetadata(dimension, TEST)
    
    # TODO: Define each parameter needed for test, use ParameterType when defining type
    # Example:
    # a#_metadata.add_parameter('a#_column_names', 'A# Column Names', ParameterType.MULTI_SELECT)

    # TODO: Replace "a#" with the test name, e.g., "a1"
    return a#_metadata