import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['U1']

""" Class to represent all metric tests for the Uniqueness dimension
    Goal: Ensure that each record in the dataset is unique and there are no duplicate entries. Unique data means there are no redundant records.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
"""
class Uniqueness:
    def __init__(self, dataset_path, return_type="score", logging_path=None):
        self.dataset_path = dataset_path
        self.return_type = return_type
        self.logging_path = logging_path

    """ Uniqueness Type 1 (U1):
    Find duplicated rows (what used to be known as Accuracy Type 3)
    """    
    def _u1_metric(self, metric):

        df = utils.read_data(self.dataset_path)

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
                return "No valid U1 results generated", None
            
            final_df = duplicate_rows  
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return percentage_duplicate, output_file  # Return the file name
            
        else:
            return df, None  # Default return value (DataFrame)
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputted metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"U1": None} 
            columns = {"U1": None}

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                metric_log_csv = None # Ensure it exists even if errors occur
                overall_uniqueness_score = {"metric": None, "value": None}  # Ensure it exists even if errors occur

                try:
                    if metric == 'U1':
                        overall_uniqueness_score["metric"] = metric
                        uniqueness_score, metric_log_csv = self._u1_metric(metric.lower())
                        overall_uniqueness_score["value"] = uniqueness_score
                
                except FileNotFoundError as e:
                    print(f'{utils.RED}Did not find dataset, make sure you have provided the correct name.{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)
                except Exception as e:
                    print(f'{utils.RED} {type(e).__name__} error has occured!{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)

                outputs.append(overall_uniqueness_score)

                # output report of results
                utils.output_log_score(
                    test_name = metric, 
                    dataset_name = utils.get_dataset_name(self.dataset_path), 
                    score = overall_uniqueness_score, 
                    selected_columns = columns[metric], 
                    excluded_columns = [''],
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Uniqueness", 
                    threshold= thresholds[metric],
                    metric_log_csv = metric_log_csv)
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1