import numpy as np  
import pandas as pd 
from . import utils
from dython.nominal import associations

ALL_METRICS = ['I1']

""" Class to represent all metric tests for the Interdependency dimension
    Goal: Ensure that data across different systems and datasets are harmonized and can be integrated. 
    Interdependent data can be effectively combined and used together without discrepancies.

dataset_path: path of the csv/xlsx to evaluate.
i1_sensitive_columns: sensitive columns used from the dataset for the I1 metric.
i1_threshold: threshold for correlation coefficient that is acceptable for I1 test.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
"""
class Interdependency:
    def __init__(self, dataset_path, i1_sensitive_columns, i1_threshold=0.75, return_type="score", logging_path=None, uploaded_file_name=None):
        self.dataset_path = dataset_path
        self.i1_sensitive_columns = i1_sensitive_columns
        self.i1_threshold = i1_threshold
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

    """ Interdependency Type 1 (I1): Identifies proxy variables whose correlation with sensitive features is higher than 0.75 (or any threshold).
    Proxy variables indirectly capture information about sensitive features, often used as substitutes for other variables. 
    Given that correlation ranges from -1 to 1 (1 suggests perfect association, 0 suggests no relation), 0.75 will be used as threshold to suggest a high level of association.
    """    
    def _i1_metric(self, metric):  
        df = utils.read_data(self.dataset_path)
        all_interdependency_scores = {}

        # Exclude the 'Comment' or 'Comments' column if it exists in the dataset  
        if 'Comment' in df.columns:  
            df = df.drop(columns=['Comment']) 
        elif 'Comments' in df.columns:
            df = df.drop(columns=['Comments'])
        
        # Number of non-sensitive features
        n_non_sensitive = len(df.columns) - len(self.i1_sensitive_columns)
    
        # Computes correlation coeff of all variables in dataset 
        corrs = associations(df, nom_nom_assoc='cramer', num_num_assoc='pearson', compute_only=True)['corr']
        corrs_thr = utils.filter_corrs(corrs, self.i1_threshold, subset = self.i1_sensitive_columns)
    
        # Compute proportion that exceeds threshold for each sensitive column
        corrs_subset = corrs[self.i1_sensitive_columns].drop(self.i1_sensitive_columns)
        for column in self.i1_sensitive_columns:
            all_interdependency_scores[column] = sum(1 for corr in corrs_subset[column] if corr > self.i1_threshold) / n_non_sensitive
        
        # Compute average score 
        overall_interdependency_score = sum(all_interdependency_scores.values()) / len(all_interdependency_scores)
        
        # may want to use for one line summary?
        summary = f"Found {len(corrs_thr)} feature pair(s) to sensitive attribute {self.i1_sensitive_columns} with correlation coefficient greater than defined threshold ({self.i1_threshold})"

        # add conditional return logic
        if self.return_type == "score":
            return overall_interdependency_score, None
        elif self.return_type == "dataset":
            if not overall_interdependency_score: 
                return "No valid I1 results generated", None
                
            final_df = corrs_thr
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return overall_interdependency_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame)
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default. return_logs returns the logging data so the UI can visualize test output details.
    """
    def run_metrics(self, metrics=ALL_METRICS, return_logs=False):
        # Verify that inputted metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"I1": self.i1_threshold} 
            columns = {"I1": self.i1_sensitive_columns} 

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                output_logs = []
                metric_log_csv = None # Ensure it exists even if errors occur
                overall_interdependency_score = {"metric": None, "value": None}  # Ensure it exists even if errors occur

                try:
                    if metric == 'I1':
                        overall_interdependency_score["metric"] = metric
                        interdependency_score, metric_log_csv = self._i1_metric(metric)
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
                metric_output_log = utils.output_log_score(
                    test_name = metric, 
                    dataset_name = self.uploaded_file_name if self.uploaded_file_name else utils.get_dataset_name(self.dataset_path), 
                    score = overall_interdependency_score, 
                    selected_columns = columns[metric], 
                    excluded_columns = [''],
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Interdependency", 
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
        