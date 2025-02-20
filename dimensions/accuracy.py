import numpy as np  
from . import utils

ALL_METRICS = ['A1', 'A2', 'A3']

""" Class to represent all metric tests for the Accuracy dimension """
class Accuracy:
    def __init__(self, dataset_path, selected_columns, groupby_column=None, a2_threshold=1.5, a2_minimum_score = 0.85):
        self.dataset_path = dataset_path  
        self.selected_columns = selected_columns
        self.groupby_column = groupby_column
        self.a2_threshold = a2_threshold
        self.a2_minimum_score = a2_minimum_score

    """ Accuracy Type 1 (A1): Determines whether there are symbols in numerics.
    Make the column a string, find symbols, and calculate the accuracy scores for multiple columns.
    """    
    def _a1_metric(self):    
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

        return overall_accuracy_score   

    """ Accuracy Type 2 (A2): Find outliers that are 1.5 (or any threshold) times away from the inter-quartile range
    The threshold for how many inter-quartile range is considered to be an outlier and percentage of the column selected that passes can be customized.
    """
    def _a2_metric(self):  
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
                Q1 = df[column].quantile(0.25)  
                Q3 = df[column].quantile(0.75)  
                IQR = Q3 - Q1  

                lower_bound = Q1 - self.a2_threshold * IQR  
                upper_bound = Q3 + self.a2_threshold * IQR  
                
                outliers = (df[column] < lower_bound) | (df[column] > upper_bound)  
                outliers_dict[column] = (1 - outliers.mean()) 
        
        #compute final score
        total_groups = len(outliers_dict)
        groups_above = sum(1 for score in outliers_dict.values() if score > self.a2_minimum_score)
        final_score = groups_above / total_groups if total_groups > 0 else 0
    
        return outliers_dict, final_score
    
    """Accuracy Type 3 (A3): Find duplicated rows
    """
    def _a3_metric(self):

        df = utils.read_data(self.dataset_path)

        # Find duplicate rows
        duplicate_rows = df[df.duplicated(keep=False)]
        
        # Calculate percentage of duplicate rows
        total_rows = len(df)
        total_duplicate_rows = len(duplicate_rows)
        percentage_duplicate = 1-(total_duplicate_rows / total_rows)
        
        # Print duplicate rows
        print("Duplicate Rows:")
        print(duplicate_rows)
        
        # Print percentage of duplicate rows
        print(f"\nDuplication Score: {percentage_duplicate*100}%")
    
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
                        overall_accuracy_score = self._a1_metric()
                    elif metric == 'A2':
                        outliers, overall_accuracy_score = self._a2_metric()
                        outputs.append(outliers)
                    elif metric =='A3':
                        overall_accuracy_score = self._a3_metric()
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
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Accuracy", 
                    threshold= thresholds[metric])
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        