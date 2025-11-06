import numpy as np  
import pandas as pd
from dython.nominal import associations
from utils import utils
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "I1" 

""" Class to represent an individual metric for the Interdependency dimension.

    Goal: Ensure that data across different systems and datasets are harmonized and can be integrated. 
    Interdependent data can be effectively combined and used together without discrepancies.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
i1_sensitive_columns: sensitive columns used from the dataset for the I1 metric.
i1_threshold: threshold for correlation coefficient that is acceptable for I1 test.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, i1_sensitive_columns=[], i1_threshold=0.75, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

        self.i1_sensitive_columns = i1_sensitive_columns
        self.i1_threshold = i1_threshold

        self.threshold = self.i1_threshold
        self.selected_columns = self.i1_sensitive_columns 
    
    """ Interdependency Type 1 (I1): Identifies proxy variables whose correlation with sensitive features is higher than 0.75 (or any threshold).
    Proxy variables indirectly capture information about sensitive features, often used as substitutes for other variables. 
    Given that correlation ranges from -1 to 1 (1 suggests perfect association, 0 suggests no relation), 0.75 will be used as threshold to suggest a high level of association.
    """    
    def run_metric(self):    
        df = utils.read_data(self.dataset_path)
        all_interdependency_scores = {}

        # Exclude the 'Comment' or 'Comments' column if it exists in the dataset  
        if 'Comment' in df.columns:  
            df = df.drop(columns=['Comment']) 
        elif 'Comments' in df.columns:
            df = df.drop(columns=['Comments'])

        # Convert timestamp columns to int for computational purposes (formatted as yyyymmdd)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y%m%d').astype('Int64')
                
        # Number of non-sensitive features
        n_non_sensitive = len(df.columns) - len(self.i1_sensitive_columns)
    
        # Computes correlation coeff of all variables in dataset 
        corrs = associations(df, nom_nom_assoc='cramer', num_num_assoc='pearson', compute_only=True, cramers_v_bias_correction=False)['corr']
        corrs_thr = utils.filter_corrs(corrs, self.i1_threshold, subset = self.i1_sensitive_columns)
    
        # Compute proportion that exceeds threshold for each sensitive column
        corrs_subset = corrs[self.i1_sensitive_columns].drop(self.i1_sensitive_columns)
        for column in self.i1_sensitive_columns:
            all_interdependency_scores[column] = 1 - (sum(1 for corr in corrs_subset[column] if corr > self.i1_threshold) / n_non_sensitive)
        
        # Compute average score 
        overall_interdependency_score = sum(all_interdependency_scores.values()) / len(all_interdependency_scores)
  
        # add conditional return logic
        if self.return_type == "score":
            return overall_interdependency_score, None
        elif self.return_type == "dataset":
            if not overall_interdependency_score: 
                return f"No valid {METRIC} results generated", None
                
            final_df = corrs_thr
            output_file = utils.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=final_df)
            return overall_interdependency_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame)
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Interdependency"

    # Define instance for metric
    i1_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    i1_metadata.add_parameter('i1_sensitive_columns', 'I1 Sensitive Columns', ParameterType.MULTI_SELECT)
    i1_metadata.add_parameter('i1_threshold', 'I1 Threshold', ParameterType.DECIMAL, value='0.75', step = 0.05)
    
    return i1_metadata 