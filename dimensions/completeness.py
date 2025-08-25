from . import utils
from dython.nominal import associations

ALL_METRICS = ['P1', 'P2']

""" Class to represent all metric tests for the Completeness dimension 
    Goal: Ensure that all required data is available and that there are no missing values. 
    Complete data includes all necessary records and fields needed for the intended use.

dataset_path: path of the csv/xlsx to evaluate.
exclude_columns: columns to ingore for the P1 test.
p1_threshold: threshold for acceptible percentance of null values in a given column for P1 test
p2_threshold: threshold for correlation coefficient that is acceptable for P2 test.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
"""
class Completeness:
    def __init__(self, dataset_path, exclude_columns=[], p1_threshold=0.75, p2_threshold=0.5, return_type="score", logging_path=None, uploaded_file_name=None):
        self.dataset_path = dataset_path  
        self.exclude_columns = exclude_columns
        self.p1_threshold = p1_threshold
        self.p2_threshold = p2_threshold
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name

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
        
        # add conditional return logic
        if self.return_type == "score":
            return completeness_score, None
        elif self.return_type == "dataset":
            if not total_non_missing : # if there are not rows with data
                return "No valid p1 results generated", None
            
            final_df = dataset2  
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return completeness_score, output_file  # Return the file name
            
        else:
            return dataset, None  # Default return value (DataFrame) 

    """ Completeness Type 2 (P2): Finds column pairs with missing values whose correlation coefficient is higher than 0.5 (or any threshold).
    Given that correlation ranges from -1 to 1 (1 suggests perfect association, 0 suggests no relation), 0.5 will be used as a midpoint threshold to investigate whether an association exists. 
    """    
    def _p2_metric(self, metric):  
        df = utils.read_data(self.dataset_path)

        # Exclude the 'Comment' or 'Comments' column if it exists in the dataset  
        if 'Comment' in df.columns:  
            df = df.drop(columns=['Comment']) 
        elif 'Comments' in df.columns:
            df = df.drop(columns=['Comments'])
        
        # Identify columns with nulls (missing values)
        df_nulls = df.loc[:, df.isnull().sum() > 0]

        # Compute correlation coefficients on missing column values (true/false entries) 
        corrs = associations(df_nulls.isnull().astype(int), nom_nom_assoc='cramer', num_num_assoc='pearson', compute_only=True)['corr']

        # Number of unique column pairings
        n_pairs = len(corrs) * (len(corrs) - 1)/2

        # Keep columns pairings with absolute correlation above the threshold
        corrs_thr = utils.filter_corrs(corrs, self.p2_threshold)

        # Compute score 
        completeness_score = (len(corrs_thr) / n_pairs) if corrs_thr is not None else None
        # may want to use for one line summary?
        summary = f"Found {len(corrs_thr)} feature pair(s) with correlation coefficient greater than defined threshold ({self.p2_threshold})"
        
        # add conditional return logic
        if self.return_type == "score":
            return completeness_score, None
        elif self.return_type == "dataset":
            if not completeness_score: 
                return "No valid p2 results generated", None
            
            final_df = corrs_thr
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return completeness_score, output_file  # Return the file name
            
        else:
            return df, None  # Default return value (DataFrame)
            
    """ Run metrics: Will run specified metrics or all accuracy metrics by default. return_logs returns the logging data so the UI can visualize test output details.
    """
    def run_metrics(self, metrics=ALL_METRICS, return_logs=False):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"P1": self.p1_threshold, "P2": self.p2_threshold}
            columns = {"P1":None, "P2": None}

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                output_logs = []
                metric_log_csv = None # Ensure it exists even if errors occur
                overall_completeness_score = {"metric": None, "value": None}  # Ensure it exists even if errors occur

                try:
                    if metric == 'P1':
                        overall_completeness_score["metric"] = metric
                        completeness_score, metric_log_csv = self._p1_metric(metric.lower())
                        overall_completeness_score["value"] = completeness_score
                    elif metric == 'P2':
                        overall_completeness_score["metric"] = metric
                        completeness_score, metric_log_csv = self._p2_metric(metric.lower())
                        overall_completeness_score["value"] = completeness_score

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
                metric_output_log = utils.output_log_score(
                    test_name = metric, 
                    dataset_name = self.uploaded_file_name if self.uploaded_file_name else utils.get_dataset_name(self.dataset_path), 
                    score = overall_completeness_score, 
                    selected_columns = columns[metric], 
                    excluded_columns = self.exclude_columns,
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Completeness", 
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
        