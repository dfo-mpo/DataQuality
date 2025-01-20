import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['t1']

""" Class to represent all metric tests for the Timeliness dimension """
class Timeliness:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # TODO: Set all the other variables

    """ Timeliness Type 1 (T1):
    TODO: provide a description of what this script does. 
    Example: Determines the similarity between string values in specified columns.
    """    
    # TODO: Replace with the logic for this metric, where the final score should be called timeliness_score 
    def t1_metric(self):  
        dataset = utils.read_data(self.dataset_path)


        # log the results
        utils.log_score(test_name = "Timeliness (T1)", dataset_name = utils.get_dataset_name(self.dataset_path), score = timeliness_score) 

        return timeliness_score 
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # TODO: verify that inputted metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            for metric in metrics:
                try:
                    if metric == 't1':
                        outputs.append(self.t1_metric())
                except Exception as e:
                    print(f'{utils.RED}Test failed!{utils.RESET}')
                    print(f'Error: {e}')
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        