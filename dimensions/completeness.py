from . import utils
import os

ALL_METRICS = ['P1']

""" Class to represent all metric tests for the Completeness dimension """
class Completeness:
    def __init__(self, dataset_path, exclude_columns=[], p1_threshold=0.75, return_type="score", logging_path=""):
        self.dataset_path = dataset_path  
        self.exclude_columns = exclude_columns
        self.p1_threshold = p1_threshold
        self.return_type = return_type
        self.logging_path = logging_path

    """ Completeness Type 1 (P1): Checks for whether there are blanks in the entire dataset.
    """    
    def _p1_metric(self, metric):  
        dataset = utils.read_data(self.dataset_path)

        # Exclude the 'Comment' column if it exists in the dataset  
        if 'Comment' in dataset.columns:  
            dataset = dataset.drop(columns=['Comment'])  

        # Exclude columns in exclude_columns if they exist in the dataset    
        dataset = dataset.drop(columns=[col for col in self.exclude_columns if col in dataset.columns])
        
        # Calculate the percentage of non-null (non-missing) values in each column  
        is_null_percentage = dataset.isna().mean()  

        # Identify columns with non-null percentage less than or equal to the threshold  
        columns_to_keep = is_null_percentage[is_null_percentage <= self.p1_threshold].index  

        # Keep columns that exceed the threshold of non-null values  
        dataset2 = dataset[columns_to_keep]  

        # Calculate the actual percentage of non-missing values in the dataset  
        total_non_missing = dataset2.notna().sum().sum()  
        total_obs = dataset2.shape[0] * dataset2.shape[1]  
        completeness_score = total_non_missing / total_obs
        
        #return outliers_dict, final_score
        base_filename=f"{self.logging_path}{metric}_output"
        version = 1
        while os.path.exists(f"{base_filename}_v{version}.csv"):
            version += 1
        
        # add conditional return logic
        if self.return_type == "score":
            return completeness_score, None
        elif self.return_type == "dataset":
            if not total_non_missing : # if there are not rows with data
                return "No valid p1 results generated"
            
            final_df = dataset2  
            output_file = f"{base_filename}_v{version}.csv"
            final_df.to_csv(output_file, index=False)
            return completeness_score, output_file  # Return the file name
            
        else:
            return dataset, None  # Default return value (DataFrame) 
    
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"P1": self.p1_threshold}
            columns = {"P1":None}

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                overall_completeness_score = None  # Ensure it exists even if errors occur

                try:
                    if metric == 'P1':
                        overall_completeness_score, metric_log_csv = self._p1_metric(metric.lower())

                except FileNotFoundError as e:
                    print(f'{utils.RED}Did not find dataset, make sure you have provided the correct name.{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)
                except Exception as e:
                    print(f'{utils.RED} {type(e).__name__} error has occured!{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)
                
                outputs.append(overall_completeness_score)

                # output report of results
                utils.output_log_score(
                    test_name = metric, 
                    dataset_name = utils.get_dataset_name(self.dataset_path), 
                    score = overall_completeness_score, 
                    selected_columns = columns[metric], 
                    excluded_columns = self.exclude_columns,
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Completeness", 
                    threshold= thresholds[metric],
                    metric_log_csv = metric_log_csv)
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        