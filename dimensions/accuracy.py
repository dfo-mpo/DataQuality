import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['c1', 'c2']

""" Class to represent all metric tests for the Accuracy dimension """
class Accuracy:
    def __init__(self, dataset_path, selected_columns):
        self.dataset_path = dataset_path  
        self.selected_columns = selected_columns 

    """ Accuracy Type 1 (A1): Determines whether there are symbols in numerics.
    Make the column a string, find symbols, and calculate the accuracy scores for multiple columns.
    """    
    def a1_metric(self):    
        # Read the dataset from the provided Excel file path
            adf = utils.read_data(self.dataset_path)

            # Check if all specified columns were extracted, if not raise Key error
            for column in self.selected_columns:
                if column not in adf.columns:
                    raise KeyError(column)
                
            self.selected_columns = [col for col in adf.columns if col in self.selected_columns]

            all_accuracy_scores = []

            for column_name in self.selected_columns:
                # Drop NA, null, or blank values from column
                column_data = adf[column_name].dropna()

                total_rows = len(column_data)

                if total_rows > 0:  # to avoid division by zero
                    non_digit_chars_per_row = column_data.apply(utils.find_non_digits)
                    non_numerical_count = non_digit_chars_per_row.apply(
                        lambda x: len(x) > 0
                    ).sum()
                    accuracy_score = (total_rows - non_numerical_count) / total_rows
                    all_accuracy_scores.append(accuracy_score)

            overall_accuracy_score = (
                sum(all_accuracy_scores) / len(all_accuracy_scores)
                if all_accuracy_scores
                else None
            )

            # log the results
            utils.log_score(
                test_name="Accuracy (A1)",
                dataset_name=utils.get_dataset_name(self.dataset_path),
                selected_columns=self.selected_columns,
                threshold=None,
                score=overall_accuracy_score,
            )

            return overall_accuracy_score   

    """ Accuracy Type 2 (C2): Compares reference data and string values in specified columns.
    The compared columns in question must be identical to the ref list, otherwise they will be penalized more harshly.
    """

    
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            for metric in metrics:
                try:
                    if metric == 'c1':
                        outputs.append(self.c1_metric())
                    elif metric == 'c2':
                        outputs.append(self.c2_metric())
                except MemoryError as e:
                    print(f'{utils.RED}Dataset is too large for this test, out of memory!{utils.RESET}')
                    print(f'Error: {e}')
                    outputs.append('Dataset is too large for this test, out of memory!')
                except KeyError as e:
                    print(f'{utils.RED}Issue with column names, are you sure you entered them correctly?{utils.RESET}')
                    print(f'Column name that fails: {e}')
                    print(f'List of all detected column names: {list(utils.read_data(self.dataset_path).columns)}')
                    outputs.append('Issue with column names, are you sure you entered them correctly?')
                except FileNotFoundError as e:
                    print(f'{utils.RED}Did not find dataset, make sure you have provided the correct name.{utils.RESET}')
                    print(f'Error: {e}')
                    outputs.append('Did not find dataset.')
                except Exception as e:
                    print(f'{utils.RED}Test failed to run!{utils.RESET}')
                    print(f'Error: {e}')
                    outputs.append('Test failed to run!')
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        