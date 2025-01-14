from . import utils

ALL_METRICS = ['p1']

""" Class to represent all metric tests for the Completeness dimension """
class Completeness:
    def __init__(self, dataset_path, exclude_columns=None, p1_threshold=1.5):
        self.dataset_path = dataset_path  
        self.exclude_columns = exclude_columns
        self.p1_threshold = p1_threshold

    """ Completeness Type 1 (P1): Checks for whether there are blanks in the entire dataset.
    """    
    def p1_metric(self):  
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

        # log the results
        utils.log_score(test_name = "Completeness (P1)", dataset_name = utils.get_dataset_name(self.dataset_path), selected_columns = None, threshold = self.p1_threshold, score = completeness_score) 

        return completeness_score 
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            for metric in metrics:
                try:
                    if metric == 'p1':
                        outputs.append(self.p1_metric())
                except Exception as e:
                    print(f'{utils.RED}Test failed!{utils.RESET}')
                    print(f'Error: {e}')
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        