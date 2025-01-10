import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['t1']

""" Class to represent all metric tests for the Timeliness dimension """
class Timeliness:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self. # will need to complete this 

    """ Timeliness Type 1 (T1): Checks for whether there are blanks in the entire dataset.
    """    
    def t1_metric(self):  
        dataset = utils.read_data(self.dataset_path) # will need to continue this function 


        # log the results
        utils.log_score(test_name = "Timeliness (T1)", dataset_name = utils.get_dataset_name(self.dataset_path), score = timeliness_score) 

        return timeliness_score 
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            for metric in metrics:
                try:
                    if metric == 't1':
                        outputs.append(self.p1_metric())
                except Exception as e:
                    print(f'{utils.RED}Test failed!{utils.RESET}')
                    print(f'Error: {e}')
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        