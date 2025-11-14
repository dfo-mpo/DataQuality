from dython.nominal import associations
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "P1"

""" Class to represent an individual metric for the Completeness dimension.
    
    Goal: Ensure that all required data is available and that there are no missing values. 
    Complete data includes all necessary records and fields needed for the intended use.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
p1_exclude_columns: columns to ingore for the P1 test.
p1_threshold: threshold for acceptible percentance of null values in a given column for P1 test
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, p1_exclude_columns=[], p1_threshold=0.75, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        self.p1_exclude_columns = p1_exclude_columns
        self.p1_threshold = p1_threshold

        self.threshold = self.p1_threshold
        self.selected_columns = None
    
    """ Completeness Type 1 (P1): Checks for whether there are blanks in the entire dataset.
    """
    def run_metric(self):
        dataset = core_operations.read_data(self.dataset_path)

        # Exclude the 'Comment' column if it exists in the dataset  
        if 'Comment' in dataset.columns:  
            dataset = dataset.drop(columns=['Comment'])  

        # Exclude columns in p1_exclude_columns if they exist in the dataset    
        dataset = dataset.drop(columns=[col for col in self.p1_exclude_columns if col in dataset.columns])
        
        # Calculate the percentage of non-null (non-missing) values in each column  
        is_null_percentage = dataset.isna().mean()  

        # Identify columns with non-null percentage less than or equal to the threshold  
        columns_to_keep = is_null_percentage[is_null_percentage <= self.p1_threshold].index  

        # Keep columns that exceed the threshold of non-null values  
        dataset2 = dataset[columns_to_keep]  

        # Calculate the actual percentage of non-missing values in the dataset  
        total_non_missing = dataset2.notna().sum().sum()  
        total_obs = dataset2.shape[0] * dataset2.shape[1]  
        completeness_score = total_non_missing / total_obs
        
        # add conditional return logic
        if self.return_type == "score":
            return completeness_score, None
        elif self.return_type == "dataset":
            if not total_non_missing : # if there are not rows with data
                return "No valid p1 results generated", None
            
            final_df = dataset2  
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=final_df)
            return completeness_score, output_file  # Return the file name
            
        else:
            return dataset, None  # Default return value (DataFrame) 
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Completeness"

    # Define instance for metric
    p1_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    p1_metadata.add_parameter('p1_exclude_columns', 'P1 Exclude Columns', ParameterType.MULTI_SELECT)
    p1_metadata.add_parameter('p1_threshold', 'P1 Threshold', ParameterType.DECIMAL, value='0.75', step = 0.05)
    
    return p1_metadata 