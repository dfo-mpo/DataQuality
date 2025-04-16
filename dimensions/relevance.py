import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['R1']

""" Class to represent all metric tests for the relevance dimension """
class Relevance:
    def __init__(self, dataset_path, logging_path=None):
        self.dataset_path = dataset_path
        self.logging_path = logging_path
        # TODO: Set all the other variables

    """ Relevance Type 1 (R1):
    TODO: provide a description of what this script does. There can be multiple types (Type 1, Type 2, etc.), add more as needed
    Example: Determines the similarity between string values in specified columns.
    """    
    # TODO: Replace with the logic for this metric, where the final score should be called relevance_score
    def _r1_metric(self):  
        dataset = utils.read_data(self.dataset_path)

        relevance_score = None

        return relevance_score 
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
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
                overall_relevance_score = None  # Ensure it exists even if errors occur

                try:
                    if metric == 'R1':
                        overall_relevance_score = self._r1_metric()

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
                utils.output_log_score(
                    test_name = metric, 
                    dataset_name = utils.get_dataset_name(self.dataset_path), 
                    score = overall_relevance_score, 
                    selected_columns = columns[metric], 
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Relevance", 
                    threshold= thresholds[metric])
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1