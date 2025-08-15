import numpy as np  
import pandas as pd
from . import utils

ALL_METRICS = ['A1', 'A2']

""" Class to represent all metric tests for the Accuracy dimension
    Goal: Ensure that the data correctly represents the real-world values it is intended to model. 
    Accurate data is free from errors and is a true reflection of the actual values.

dataset_path: path of the csv/xlsx to evaluate.
selected_columns: columns from the provided dataset to evaluate, used for metics A1 and A2, should be all numeric columns.
groupby_column: used by metric A2, __to do
a2_threshold: theshold used in A2 interquartile range calculations to determine outliers
a2_minimum_scure: minimum acceptible score from interquartile range calculations
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
"""
class Accuracy:
    def __init__(self, dataset_path, selected_columns, groupby_column=None, a2_threshold=1.5, a2_minimum_score = 0.85, return_type="score", logging_path=None, uploaded_file_name=None):
        self.dataset_path = dataset_path  
        self.selected_columns = selected_columns
        self.groupby_column = groupby_column
        self.a2_threshold = a2_threshold
        self.a2_minimum_score = a2_minimum_score
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

    """ Accuracy Type 1 (A1): Determines whether there are symbols in numerics.
    Make the column a string, find symbols, and calculate the accuracy scores for multiple columns.
    """    
    def _a1_metric(self, metric):    
        # dataframes for output report reports
        original_df = utils.read_data(self.dataset_path) # this first original dataframe is used to compute a column of NaNs that are used for the accuracy calculations in the output report
        original_df_2 = utils.read_data(self.dataset_path) # this second original dataframe is used to write the output that needs to show what the original dataframe looked like
        
        # dataframe for computing accuracy score
        adf = utils.read_data(self.dataset_path) # dataframe that will be used to compute the accuracy score
        selected_columns = [col for col in adf.columns if col in self.selected_columns] 

        all_accuracy_scores = []
        
        for column_name in selected_columns:  
            
            non_digit_chars_per_row = utils.find_non_digits(adf, column_name)
            # Drop NA, null, or blank values from column  
            column_data = adf.loc[adf[f"{column_name}_new"]==0]   
            total_rows = len(column_data)  
            
            if total_rows > 0:  # to avoid division by zero  
                non_numerical_count = non_digit_chars_per_row[column_name].apply(lambda x: np.where(pd.isna(x), 1, 0)).sum()   
                accuracy_score = (total_rows - non_numerical_count) / total_rows  
                all_accuracy_scores.append(accuracy_score)   

        # compute final score
        overall_accuracy_score = sum(all_accuracy_scores) / len(all_accuracy_scores) if all_accuracy_scores else None
        
        # add conditional return logic
        if self.return_type == "score":
            return overall_accuracy_score, None
        elif self.return_type == "dataset":
            if not overall_accuracy_score :
                return "No valid A1 results generated", None
            
            final_df = utils.add_only_numbers_columns(original_df, selected_columns, original_df_2)  
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
        scores = {} # keep incase we want to view the final scores by column
        avg_score = 0

        # Detect outliers of given data
        def detect_outliers(x):
            Q1 = x.quantile(0.25)  
            Q3 = x.quantile(0.75)  
            IQR = Q3 - Q1  
            lower_bound = Q1 - self.a2_threshold * IQR  
            upper_bound = Q3 + self.a2_threshold * IQR  
            return (x < lower_bound) | (x > upper_bound)
            
        # If a groupby column is specified, perform the IQR calculation within each group  
        if self.groupby_column:  
            grouped = df.groupby(self.groupby_column)  
            total_groups = len(grouped)
            for column in self.selected_columns:  
                # Apply the outlier detection for each group  
                outliers = grouped[column].apply(detect_outliers) 
  
                # Combine the outlier Series into a single Series that corresponds to the original DataFrame index  
                outliers_dict[column] = (1 - outliers.groupby(self.groupby_column).mean())

                # Compute final score
                scores[column] = np.sum(outliers_dict[column] > self.a2_minimum_score) / total_groups if total_groups > 0 else 0
                avg_score += scores[column]
        else:
            # Perform the IQR calculation on the whole column if no groupby column is specified  
            for column in self.selected_columns: 
                # Convert to numeric, remove none numeric values TODO: Do we keep this or adopt a different solution? 
                column_data = pd.to_numeric(df[column], errors='coerce').dropna()  
                
                # Apply the outlier detection
                outliers = detect_outliers(column_data)
                outliers_dict[column] = (1 - outliers.mean())  

                # Compute final score
                scores[column] = np.sum(outliers_dict[column] > self.a2_minimum_score)  
                avg_score += scores[column] 

        # Compute average of final scores across selected columns  
        avg_score = avg_score / len(self.selected_columns)   

        # add conditional return logic
        if self.return_type == "score":
            return avg_score, None
        elif self.return_type == "dataset":
            if not outliers_dict :
                return "No valid A2 results generated", None

            final_df = pd.DataFrame()
            if self.groupby_column is not None: 
                final_df = final_df.from_dict(outliers_dict)
                final_df.reset_index(inplace=True)
                final_df.rename(columns={'index': 'GroupName'}, inplace=True)
            else:
                final_df = pd.DataFrame([outliers_dict])
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return avg_score, output_file  # Return the file name
            
        else:
            return df, None  # Default return value (DataFrame)  

    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"A1": None, "A2": self.a2_threshold}
            columns = {"A1": self.selected_columns, "A2": self.selected_columns}

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                metric_log_csv = None # Ensure it exists even if errors occur
                overall_accuracy_score = {"metric": None, "value": None}  # Ensure it exists even if errors occur

                try:
                    if metric == 'A1':
                        overall_accuracy_score["metric"] = metric
                        accuracy_score, metric_log_csv = self._a1_metric(metric.lower())
                        overall_accuracy_score["value"] = accuracy_score
                    elif metric == 'A2':
                        overall_accuracy_score["metric"] = metric
                        accuracy_score, metric_log_csv = self._a2_metric(metric.lower())
                        overall_accuracy_score["value"] = accuracy_score
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
                    dataset_name = self.uploaded_file_name if self.uploaded_file_name else utils.get_dataset_name(self.dataset_path), 
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
        