import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "A4"

""" Class to represent an individual metric for the Accuracy dimension.

    Goal: Ensure that the data correctly represents the real-world values it is intended to model. 
    Accurate data is free from errors and is a true reflection of the actual values.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
a4_column_pairs: related timestamp columns used from the dataset for the A4 metric.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, a4_column_pairs=None, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        self.a4_column_pairs = a4_column_pairs

        self.threshold = None
        self.selected_columns = [col for pair in self.a4_column_pairs for col in pair] if self.a4_column_pairs else None
    
    """ Accuracy Type 4 (A4): Checks whether related timestamp columns are in chronological order.
    Test will consider missing start and end dates as valid.
    """
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)
        results = df.copy()
        all_accuracy_scores = {}
        
        # Check whether column pairs are in chronological order (flags those not in chronological order)
        for start_col, end_col in self.a4_column_pairs:
            # Assumes entries are datetime
            col_name = f"{start_col}_after_{end_col}"
            results[col_name] = ~(
                (df[end_col] >= df[start_col]) | 
                df[end_col].isna() | 
                df[start_col].isna()
            )
    
            # Compute ratio not in chronological order for current column pair
            all_accuracy_scores[col_name] = 1 - results[col_name].mean()
        
        # Take subset of data not in chronological order
        check_columns = list(all_accuracy_scores.keys())
        invalid = results[check_columns].any(axis=1)
        adf = results[invalid].copy()
        
        # Compute average score
        accuracy_score = sum(all_accuracy_scores.values()) / len(all_accuracy_scores)

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
       
""" Creates a MetricMetadata instance for a single metric, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Accuracy"

    # Define instance for metric
    a4_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    a4_metadata.add_parameter('a4_column_pairs', 'A4 Column Pairs', ParameterType.PAIRS, value=[], hint="Pairs of related timestamp columns used from the dataset. Use format in placeholder replacing text with names of columns from your uploaded dataset.")
    
    return a4_metadata 