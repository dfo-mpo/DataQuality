import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['i1']

""" Class to represent all metric tests for the Interdependency dimension """
class Interdependency:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # TODO: Set all the other variables

    """ Interdependency Type 1 (I1):
    TODO: provide a description of what this script does. There can be multiple types (Type 1, Type 2, etc.), add more as needed
    Example: Determines the similarity between string values in specified columns.
    """    
    # TODO: Replace with the logic for this metric, where the final score should be called interdependency_score
    def i1_metric(self):  
        dataset = utils.read_data(self.dataset_path)


        # log the results
        utils.log_score(test_name = "Interdependency (I1)", dataset_name = utils.get_dataset_name(self.dataset_path), score = interdependency_score) 

        return interdependency_score 
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # TODO: verify that inputted metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            for metric in metrics:
                try:
                    if metric == 'i1':
                        outputs.append(self.i1_metric())
                except Exception as e:
                    print(f'{utils.RED}Test failed!{utils.RESET}')
                    print(f'Error: {e}')
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        