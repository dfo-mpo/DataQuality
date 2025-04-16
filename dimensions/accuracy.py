import numpy as np  
import pandas as pd
from . import utils
import os
import io

ALL_METRICS = ['A1', 'A2']

""" Class to represent all metric tests for the Accuracy dimension """
class Accuracy:
    def __init__(self, dataset_path, selected_columns, groupby_column=None, a2_threshold=1.5, a2_minimum_score = 0.85, return_type="score", logging_path=None):
        self.dataset_path = dataset_path  
        self.selected_columns = selected_columns
        self.groupby_column = groupby_column
        self.a2_threshold = a2_threshold
        self.a2_minimum_score = a2_minimum_score
        self.return_type = return_type
        self.logging_path = logging_path

    """ Accuracy Type 1 (A1): Determines whether there are symbols in numerics.
    Make the column a string, find symbols, and calculate the accuracy scores for multiple columns.
    """    
    def _a1_metric(self, metric):    
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
        
        # add conditional return logic
        if self.return_type == "score":
            return overall_accuracy_score, None
        elif self.return_type == "dataset":
            if not overall_accuracy_score :
                return "No valid a1 results generated"
            
            final_df = utils.add_only_numbers_columns(adf, self.selected_columns)  
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return overall_accuracy_score, output_file  # Return the file name
            
        else:
            return adf, None  # Default return value (DataFrame)    

    """ Accuracy Type 2 (A2): Find outliers that are 1.5 (or any threshold) times away from the inter-quartile range
    The threshold for how many inter-quartile range is considered to be an outlier and percentage of the column selected that passes can be customized.
    """
    def _a2_metric(self, metric): 
        df = utils.read_data(self.dataset_path)
        outliers_dict = {}

        # If a groupby column is specified, perform the IQR calculation within each group  
        if self.groupby_column:  
            grouped = df.groupby(self.groupby_column)  
            for column in self.selected_columns:  
                # Apply the outlier detection for each group  
                outliers = grouped[column].apply(lambda x: ((x < x.quantile(0.25) - self.a2_threshold * (x.quantile(0.75) - x.quantile(0.25))) |  
                                                            (x > x.quantile(0.75) + self.a2_threshold * (x.quantile(0.75) - x.quantile(0.25))))) 
                # Combine the outlier Series into a single Series that corresponds to the original DataFrame index  
                outliers_dict[column] = (1 - outliers.groupby(self.groupby_column).mean())
        else:
            # Perform the IQR calculation on the whole column if no groupby column is specified  
            for column in self.selected_columns: 
                # Convert to numeric, remove none numeric values TODO: Do we keep this or adopt a different solution? 
                column_data = pd.to_numeric(df[column], errors='coerce').dropna()  
                 
                # Calculate Q1 (25th percentile) and Q3 (75th percentile)  
                Q1 = column_data.quantile(0.25)  
                Q3 = column_data.quantile(0.75)  
                IQR = Q3 - Q1  

                lower_bound = Q1 - self.a2_threshold * IQR  
                upper_bound = Q3 + self.a2_threshold * IQR  
                
                outliers = (column_data < lower_bound) | (column_data > upper_bound)  
                outliers_dict[column] = (1 - outliers.mean()) 

        # compute final score  
        total_groups = len(outliers_dict)  
        groups_above = sum(1 for score in outliers_dict.values() if score > self.a2_minimum_score)  
        final_score = groups_above / total_groups if total_groups > 0 else 0  
        
        # final_score = {}
        
        # for key, value in outliers_dict.items():
        #     value_out = value > self.a2_minimum_score
        #     final_score[key] = value_out

        # add conditional return logic
        if self.return_type == "score":
            return final_score, None
        elif self.return_type == "dataset":
            if not outliers_dict :
                return "No valid a2 results generated"

            final_df = pd.DataFrame([outliers_dict])
            # final_df = pd.DataFrame([outliers_dict])
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return final_score, output_file  # Return the file name
            
        else:
            return df, None  # Default return value (DataFrame)  

    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"A1": None, "A2": self.a2_threshold, "A3": None}
            columns = {"A1": self.selected_columns, "A2": self.selected_columns, "A3": None}

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                overall_accuracy_score = None  # Ensure it exists even if errors occur
                try:
                    if metric == 'A1':
                        overall_accuracy_score, metric_log_csv = self._a1_metric(metric.lower())
                    elif metric == 'A2':
                        overall_accuracy_score, metric_log_csv = self._a2_metric(metric.lower())
                except KeyError as e:
                    print(f'{utils.RED}Issue with column names, are you sure you entered them correctly?{utils.RESET}')
                    print(f'Column name that fails: {e}')
                    print(f'List of all detected column names: {list(utils.read_data(self.dataset_path).columns)}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e) + ' column not found in dataset.'
                except Exception as e:
                    print(f'{utils.RED} {type(e).__name__} error has occured!{utils.RESET}')
                    print(e)
                    errors = type(e).__name__  
                    test_fail_comment = str(e)

                outputs.append(overall_accuracy_score)

                # output report of results
                utils.output_log_score(
                    test_name = metric, 
                    dataset_name = utils.get_dataset_name(self.dataset_path), 
                    score = overall_accuracy_score, 
                    selected_columns = columns[metric],
                    excluded_columns = [''],
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Accuracy", 
                    threshold= thresholds[metric],
                    metric_log_csv = metric_log_csv,
                    minimum_score = self.a2_minimum_score)
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        