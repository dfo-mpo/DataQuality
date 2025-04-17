import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['I1']

""" Class to represent all metric tests for the Interdependency dimension """
class Interdependency:
    def __init__(self, dataset_path, logging_path=None):
        self.dataset_path = dataset_path
        self.logging_path = logging_path
        # TODO: Set all the other variables

    """ Interdependency Type 1 (I1):
    TODO: provide a description of what this script does. There can be multiple types (Type 1, Type 2, etc.), add more as needed
    Example: Determines the similarity between string values in specified columns.
    """    
    # TODO: Replace with the logic for this metric, where the final score should be called interdependency_score
    def _i1_metric(self):  
        dataset = utils.read_data(self.dataset_path)

        interdependency_score = None

        return interdependency_score, None
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputted metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"I1": None} # TODO: Update with thresholds use for each test
            columns = {"I1": None} # TODO: Update with columns use for each test

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                metric_log_csv = None # Ensure it exists even if errors occur
                overall_interdependency_score = {"metric": None, "value": None}  # Ensure it exists even if errors occur

                try:
                    if metric == 'I1':
                        overall_interdependency_score["metric"] = metric
                        interdependency_score, metric_log_csv = self._i1_metric()
                        overall_interdependency_score["value"] = interdependency_score
                
                except FileNotFoundError as e:
                    print(f'{utils.RED}Did not find dataset, make sure you have provided the correct name.{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)
                except Exception as e:
                    print(f'{utils.RED} {type(e).__name__} error has occured!{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)

                outputs.append(overall_interdependency_score)

                # output report of results
                utils.output_log_score(
                    test_name = metric, 
                    dataset_name = utils.get_dataset_name(self.dataset_path), 
                    score = overall_interdependency_score, 
                    selected_columns = columns[metric], 
                    excluded_columns = [''],
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Interdependency", 
                    threshold= thresholds[metric],
                    metric_log_csv = metric_log_csv)
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        