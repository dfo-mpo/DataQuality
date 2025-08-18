import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['R1']

""" Class to represent all metric tests for the relevance dimension
    Goal: Ensure that the data is relevant and useful for the intended purposes. 
    Relevant data meets the needs of the users and supports the business processes.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
"""
class Relevance:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None):
        self.dataset_path = dataset_path
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        # TODO: Set all the other variables

    """ Relevance Type 1 (R1):
    TODO: provide a description of what this script does. There can be multiple types (Type 1, Type 2, etc.), add more as needed
    Example: Determines the similarity between string values in specified columns.
    """    
    # TODO: Replace with the logic for this metric, where the final score should be called relevance_score
    def _r1_metric(self, metric):  
        dataset = utils.read_data(self.dataset_path)

        relevance_score = None

        rdf = None

        # add conditional return logic
        if self.return_type == "score":
            return relevance_score, None
        elif self.return_type == "dataset":
            if not relevance_score :
                return "No valid R1 results generated", None
            
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=None)
            return relevance_score, output_file  # Return the file name
            
        else:
            return rdf, None  # Default return value (Data Frame)
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default. return_logs returns the logging data so the UI can visualize test output details.
    """
    def run_metrics(self, metrics=ALL_METRICS, return_logs=False):
        # Verify that inputted metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"R1": None} # TODO: Update with thresholds use for each test
            columns = {"R1": None} # TODO: Update with columns use for each test

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                output_logs = []
                metric_log_csv = None # Ensure it exists even if errors occur
                overall_relevance_score = {"metric": None, "value": None}  # Ensure it exists even if errors occur

                try:
                    if metric == 'R1':
                        overall_relevance_score["metric"] = metric
                        relevance_score, metric_log_csv = self._r1_metric(metric)
                        overall_relevance_score["value"] = relevance_score

                except FileNotFoundError as e:
                    print(f'{utils.RED}Did not find dataset, make sure you have provided the correct name.{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)
                except Exception as e:
                    print(f'{utils.RED} {type(e).__name__} error has occured!{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)

                outputs.append(overall_relevance_score)

                # output report of results
                metric_output_log = utils.output_log_score(
                    test_name = metric, 
                    dataset_name = self.uploaded_file_name if self.uploaded_file_name else utils.get_dataset_name(self.dataset_path), 
                    score = overall_relevance_score, 
                    selected_columns = columns[metric], 
                    excluded_columns = [''],
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Relevance", 
                    threshold= thresholds[metric],
                    metric_log_csv = metric_log_csv,
                    return_log = return_logs)
                output_logs.append(metric_output_log)

            # Only return outputs logs if output_log_score has returned logs in memory 
            if return_logs:
                return outputs, output_logs
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1