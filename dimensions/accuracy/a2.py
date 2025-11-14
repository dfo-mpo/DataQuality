import numpy as np  
import pandas as pd
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "A2"

""" Class to represent an individual metric for the Accuracy dimension.

    Goal: Ensure that the data correctly represents the real-world values it is intended to model. 
    Accurate data is free from errors and is a true reflection of the actual values.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
a2_column_names: columns used from the dataset for the A2 metric, should be all numeric columns.
a2_groupby_column: used by metric A2, groupby data from selected_columns by each unique a2_groupby_column entry. Score is calculated for each groupby then averaged for a2_groupby_column. If multiple groupby columns are provided, calculations are done on using each individual column then averaged together.
a2_threshold: threshold used in A2 interquartile range calculations to determine outliers.
a2_minimum_score: minimum acceptable score from interquartile range calculations.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, a2_column_names=[], a2_groupby_column=None, a2_threshold=1.5, a2_minimum_score = 0.85, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        self.a2_column_names = a2_column_names
        self.a2_groupby_column = a2_groupby_column
        self.a2_threshold = a2_threshold
        self.a2_minimum_score = a2_minimum_score

        self.threshold = self.a2_threshold
        self.selected_columns = self.a2_column_names
    
    """ Accuracy Type 2 (A2): Find outliers that are 1.5 (or any threshold) times away from the inter-quartile range
    The threshold for how many inter-quartile range is considered to be an outlier and percentage of the column selected that passes can be customized.
    """
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)
        outliers_dict = {}
        scores = {} # keep incase we want to view the final scores by column
        avg_score = 0

        # Detect outliers of given data
        def detect_outliers(x):
            Q1 = x.quantile(0.25)  
            Q3 = x.quantile(0.75)  
            IQR = Q3 - Q1  
            lower_bound = Q1 - self.a2_threshold * IQR  
            upper_bound = Q3 + self.a2_threshold * IQR  
            return (x < lower_bound) | (x > upper_bound)
            
        # If a groupby column is specified, perform the IQR calculation within each group  
        if self.a2_groupby_column:  
            grouped = df.groupby(self.a2_groupby_column)  
            total_groups = len(grouped)
            
            for column in self.a2_column_names:  
                # Apply the outlier detection for each group  
                outliers = grouped[column].apply(detect_outliers) 
  
                # Combine the outlier Series into a single Series that corresponds to the original DataFrame index  
                outliers_dict[column] = (1 - outliers.groupby(self.a2_groupby_column).mean())

                # Compute final score
                scores[column] = np.sum(outliers_dict[column] > self.a2_minimum_score) / total_groups if total_groups > 0 else 0
                avg_score += scores[column]
        else:
            # Perform the IQR calculation on the whole column if no groupby column is specified  
            for column in self.a2_column_names: 
                # Convert to numeric, remove none numeric values TODO: Do we keep this or adopt a different solution? 
                column_data = pd.to_numeric(df[column], errors='coerce').dropna()  
                
                # Apply the outlier detection
                outliers = detect_outliers(column_data)
                outliers_dict[column] = (1 - outliers.mean())  

                # Compute final score
                scores[column] = np.sum(outliers_dict[column] > self.a2_minimum_score)  
                avg_score += scores[column] 
        
        # Compute average of final scores across selected columns  
        avg_score = avg_score / len(self.a2_column_names)

        # add conditional return logic
        if self.return_type == "score":
            return avg_score, None
        elif self.return_type == "dataset":
            if not outliers_dict :
                return f"No valid {METRIC} results generated", None

            final_df = pd.DataFrame()
            if self.a2_groupby_column is not None: 
                final_df = final_df.from_dict(outliers_dict)
                final_df.reset_index(inplace=True)
                final_df.rename(columns={'index': 'GroupName'}, inplace=True)
            else:
                final_df = pd.DataFrame([outliers_dict])
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=final_df)
            return avg_score, output_file  # Return the file name
            
        else:
            return df, None  # Default return value (DataFrame)  
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Accuracy"

    # Define instance for metric
    a2_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    a2_metadata.add_parameter('a2_column_names', 'A2 Column Names', ParameterType.MULTI_SELECT)
    a2_metadata.add_parameter('a2_groupby_column', 'Groupby Column(s)', ParameterType.MULTI_SELECT, hint="Used by metric A2, groupby data from selected_columns by each unique a2_groupby_column entry. Score is calculated for each groupby then averaged for a2_groupby_column. If multiple groupby columns are provided, calculations are done on using each individual column then averaged together.")
    a2_metadata.add_parameter('a2_threshold', 'A2 Threshold', ParameterType.DECIMAL, value='1.5', step = 0.1)
    a2_metadata.add_parameter('a2_minimum_score', 'A2 Minimum Score', ParameterType.DECIMAL, value='0.85', step = 0.05)
    
    return a2_metadata 