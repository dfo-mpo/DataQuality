import numpy as np  
import pandas as pd
from utils import core_operations, table_operations, column_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "A1"

""" Class to represent an individual metric for the Accuracy dimension.

    Goal: Ensure that the data correctly represents the real-world values it is intended to model. 
    Accurate data is free from errors and is a true reflection of the actual values.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
a1_column_names: columns used from the dataset for the A1 metric, should be all numeric columns.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, a1_column_names=[], threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        self.a1_column_names = a1_column_names

        self.threshold = None
        self.selected_columns = self.a1_column_names
    
    """ Accuracy Type 1 (A1): Determines whether there are symbols in numerics.
    Make the column a string, find symbols, and calculate the accuracy scores for multiple columns.
    """    
    def run_metric(self):    
        # Dataframes for output report reports
        original_df = core_operations.read_data(self.dataset_path) # This first original dataframe is used to compute a column of NaNs that are used for the accuracy calculations in the output report
        original_df_2 = core_operations.read_data(self.dataset_path) # This second original dataframe is used to write the output that needs to show what the original dataframe looked like
        
        # Dataframe for computing accuracy score
        adf = core_operations.read_data(self.dataset_path) # Dataframe that will be used to compute the accuracy score
        selected_columns = [col for col in adf.columns if col in self.a1_column_names] 

        all_accuracy_scores = []
        
        for column_name in selected_columns:  
            
            non_digit_chars_per_row = column_operations.find_non_digits(adf, column_name)
            # Drop NA, null, or blank values from column  
            column_data = adf.loc[adf[f"{column_name}_new"]==0]   
            total_rows = len(column_data)  
            
            if total_rows > 0:  # To avoid division by zero  
                non_numerical_count = non_digit_chars_per_row[column_name].apply(lambda x: np.where(pd.isna(x), 1, 0)).sum()   
                accuracy_score = (total_rows - non_numerical_count) / total_rows  
                all_accuracy_scores.append(accuracy_score)   

        # Compute final score
        accuracy_score = sum(all_accuracy_scores) / len(all_accuracy_scores) if all_accuracy_scores else None
        
        # Conditional return logic
        if self.return_type == "score":
            return accuracy_score, None
        elif self.return_type == "dataset":
            if not accuracy_score:
                return f"No valid {METRIC} results generated", None
            
            adf = table_operations.add_only_numbers_columns(original_df, selected_columns, original_df_2)  
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=adf)
            return accuracy_score, output_file  # Return the file name
            
        else:
            return adf, None  # Default return value (DataFrame)  
       
""" Creates a MetricMetadata instance for a single metric, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Accuracy"

    # Define instance for metric
    a1_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    a1_metadata.add_parameter('a1_column_names', 'A1 Column Names', ParameterType.MULTI_SELECT, default=[])
    
    return a1_metadata 