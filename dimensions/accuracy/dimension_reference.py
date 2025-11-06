from utils import utils
import os
import importlib

""" Class to manage and run all metrics for the Accuracy dimension. 

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
metric_params: dictionary of additional metric specific parameters 
"""
class Accuracy:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, metric_params=None):
        self.dataset_path = dataset_path 
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        self.metric_params = metric_params or {}

        # Looks for all metric classes in the accuracy folder
        self.ALL_METRICS = self.search_metrics() 

    """ Automatically finds and loads all metric classes defined in the accuracy folder.
    """
    def search_metrics(self):
        metrics = {}
        folder = os.path.dirname(__file__)
        # Go through every python file in this dimension's folder, find all metric files and import it
        for file in os.listdir(folder):
            if file.endswith(".py") and file not in ("__init__.py", "dimension_reference.py", "metric_template.py"):
                module_name = file[:-3]
                module = importlib.import_module(f".{module_name}", package=__package__)
                metrics[module_name.upper()] = module.Metric
        return metrics
        
    """ Run metrics: Will run specified metrics or all accuracy metrics by default. return_logs returns the logging data so the UI can visualize test output details.
    """
    def run_metrics(self, metrics=None, return_logs=False):
        if metrics is None:
            # Run all metrics by default
            metrics = self.ALL_METRICS
            
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(self.ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            output_logs = []
            threshold = None
            selected_columns = None

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                metric_log_csv = None # Ensure it exists even if errors occur
                overall_accuracy_score = {"metric": None, "value": None}  # Ensure it exists even if errors occur
                try:
                    # Get any user-defined parameters for this metric
                    params = self.metric_params.get(metric, {}) 

                    # Get the metric class corresponding to this metric
                    MetricClass = self.ALL_METRICS[metric]

                    # Create an instance of the metric class
                    metric_instance = MetricClass(
                        dataset_path = self.dataset_path,
                        return_type = self.return_type,
                        logging_path = self.logging_path,
                        **params
                    )

                    threshold = metric_instance.threshold
                    selected_columns = metric_instance.selected_columns

                    overall_accuracy_score["metric"] = metric
                    accuracy_score, metric_log_csv = metric_instance.run_metric()
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
                metric_output_log = utils.output_log_score(
                    test_name = metric, 
                    dataset_name = self.uploaded_file_name if self.uploaded_file_name else utils.get_dataset_name(self.dataset_path), 
                    score = overall_accuracy_score, 
                    selected_columns = selected_columns,
                    excluded_columns = [''],
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Accuracy", 
                    threshold= threshold,
                    metric_log_csv = metric_log_csv,
                    minimum_score = params.get(f"{metric.lower()}_minimum_score", None),
                    return_log = return_logs)
                output_logs.append(metric_output_log)
         
            # Only return outputs logs if output_log_score has returned logs in memory 
            if return_logs:
                return outputs, output_logs
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {self.ALL_METRICS}, inputted metrics: {metrics}')
            return -1

    """ Collects metadata class from all metrics in the accuracy folder.
    """
    def collect_metadata(self):
        metadata = []

        # Go through each metric file in this dimension's folder and import it
        for metric, metric_class in self.ALL_METRICS.items():
            module_name = metric_class.__module__
            module = importlib.import_module(module_name)

            # If this metric has a create_metadata() function, call it and append instance into metadata list
            if hasattr(module, "create_metadata"):
                metric_metadata = module.create_metadata()
                metadata.append(metric_metadata)
    
        return metadata