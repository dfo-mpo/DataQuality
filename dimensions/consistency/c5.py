import numpy as np  
import pandas as pd
import re
from utils import core_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "C5"

""" Class to represent an individual metric for the Consistency dimension.
    
    Goal: Ensure that data is consistent across different datasets and systems. 
    Consistent data follows the same formats, standards, and definitions, and there are no contradictions within the dataset.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
c5_column_names: columns used from the dataset for the C5 metric.
c5_region: restricts latitude/longitude validation to Pacific region.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, c5_column_names=[], c5_region=None, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

        self.c5_column_names = c5_column_names
        self.c5_region = c5_region
        
        self.threshold = None
        self.selected_columns = self.c5_column_names 
    
    
    """ Consistency Type 5 (C5): Checks whether the dataset follows Decimal Degrees (DD) formatting and has valid latitude & longitude coordinates.
    Users can optionally check whether coordinates fall within DFO's administrative Pacific Region, otherwise defaults to global bounds.
    """
    def run_metric(self):    
        df = core_operations.read_data(self.dataset_path)
        results = df.copy()
        all_consistency_scores = {}
        lat_min, lat_max = -90, 90
        long_min, long_max = -180, 180

        # Compile regex patterns to detect latitude and longitude column names
        lat_pattern = re.compile(r'(lat|latitude)', flags=re.IGNORECASE)
        long_pattern = re.compile(r'(long|longitude)', flags=re.IGNORECASE)

        # Define coordinate bounds based on region
        if self.c5_region == 'Pacific':
            lat_min, lat_max = 48.309405570541784, 68.70812368168862
            long_min, long_max = -141.01414329229658, -114.05462020890663
        
        for column in self.c5_column_names:
            # Remove NA values
            df_clean = df.dropna(subset=[column])
            # Normalize column names by converting to lowercase and stripping whitespaces
            lower = column.lower().strip()
    
            # Check validity of coordinates depending on if latitude or longitude (flags those out of bounds)
            if lat_pattern.search(column):
                results[f"{column}_invalid"] = df_clean[column].apply(lambda x: False if lat_min <= x <= lat_max else True)
                all_consistency_scores[column] = 1 - results[f"{column}_invalid"].mean()
    
            elif long_pattern.search(column):
                results[f"{column}_invalid"] = df_clean[column].apply(lambda x: False if long_min <= x <= long_max else True)
                all_consistency_scores[column] = 1 - results[f"{column}_invalid"].mean()

        # Take subset of data with invalid coordinates 
        comparison_columns = [f"{col}_invalid" for col in self.c5_column_names]
        invalid = results[comparison_columns].any(axis=1)
        cdf = results[invalid].copy()
    
        # Compute average score
        consistency_score = sum(all_consistency_scores.values()) / len(all_consistency_scores)
    
        # Conditional return logic
        if self.return_type == "score":
            return consistency_score, None
        elif self.return_type == "dataset":
            if not consistency_score: 
                return f"No valid {METRIC} results generated", None
                
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=cdf)
            return consistency_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame)
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for metric
    c5_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    c5_metadata.add_parameter('c5_column_names', 'C5 Column Names', ParameterType.MULTI_SELECT, default=[])
    c5_metadata.add_parameter('c5_region', 'C5 Region', ParameterType.SINGLE_SELECT, value=["All", "Pacific"], hint="Restricts geographic coordinates to specified DFO region.")
    
    return c5_metadata 