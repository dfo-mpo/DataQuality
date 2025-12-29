from utils import core_operations
import os
import importlib
from natsort import natsorted

""" Class to manage and run all tests for the Relevance dimension. 

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only test scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
test_params: dictionary of additional test specific parameters 
"""
class Relevance:
    # Finds and loads all test names defined in the relevance folder
    DIM_FOLDER = os.path.dirname(__file__) 
    ALL_TESTS = core_operations.list_test_names(DIM_FOLDER)

    """ Collects metadata class from all tests in the relevance folder.
        Returns a list of TestMetadata objects, one per test, or [] if no tests define additional input parameters.
    """
    @classmethod
    def collect_metadata(cls):
        metadata = []

        # Go through each test file in this dimension's folder and import it
        for test_name in natsorted(cls.ALL_TESTS):
            module = importlib.import_module(f".{test_name.lower()}", package=__package__)

            # If this test has a create_metadata() function, call it and append instance into metadata list
            if hasattr(module, "create_metadata"):
                metadata.append(module.create_metadata())

        return metadata
        
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, test_params=None):
        self.dataset_path = dataset_path 
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        self.test_params = test_params or {}

        # Looks for all test classes in the relevance folder
        self.ALL_TESTS = self.search_tests() 

    """ Automatically finds and loads all test classes defined in the relevance folder.
    """
    def search_tests(self):
        tests = {}
        folder = os.path.dirname(__file__)
        # Go through every python file in this dimension's folder, find all test files and import it
        for file in natsorted(os.listdir(folder)):
            if file.endswith(".py") and file not in ("__init__.py", "dimension_reference.py", "test_template.py"):
                module_name = file[:-3]
                module = importlib.import_module(f".{module_name}", package=__package__)
                tests[module_name.upper()] = module.Test
        return tests
        
    """ Runs specified tests or all relevance tests by default. 
        return_logs returns the logging data so the UI can visualize test output details.
    """
    def run_tests(self, tests=None, return_logs=False):
        if tests is None:
            # Run all tests by default
            tests = natsorted(self.ALL_TESTS)
            
        # Verify that inputed tests is valid
        if set(tests).issubset(set(self.ALL_TESTS)):
            # Run each test and send outputs in combined list
            outputs = []
            output_logs = []
            threshold = None
            selected_columns = None

            for test in tests:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                test_log_csv = None # Ensure it exists even if errors occur
                overall_relevance_score = {"test": None, "value": None}  # Ensure it exists even if errors occur
                try:
                    # Get any user-defined parameters for this test
                    params = self.test_params.get(test, {}) 

                    # Get the test class corresponding to this test
                    TestClass = self.ALL_TESTS[test]

                    # Create an instance of the test class
                    test_instance = TestClass(
                        dataset_path = self.dataset_path,
                        return_type = self.return_type,
                        logging_path = self.logging_path,
                        **params
                    )

                    threshold = test_instance.threshold
                    selected_columns = test_instance.selected_columns

                    overall_relevance_score["test"] = test
                    relevance_score, test_log_csv = test_instance.run_test()
                    overall_relevance_score["value"] = relevance_score
                
                except KeyError as e:
                    print(f'{core_operations.RED}Issue with column names, are you sure you entered them correctly?{core_operations.RESET}')
                    print(f'Column name that fails: {e}')
                    print(f'List of all detected column names: {list(core_operations.read_data(self.dataset_path).columns)}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e) + ' column not found in dataset.'
                except Exception as e:
                    print(f'{core_operations.RED} {type(e).__name__} error has occured!{core_operations.RESET}')
                    print(e)
                    errors = type(e).__name__  
                    test_fail_comment = str(e)

                outputs.append(overall_relevance_score)
               
                # Output report of results
                test_output_log = core_operations.output_log_score(
                    test_name = test, 
                    dataset_name = self.uploaded_file_name if self.uploaded_file_name else core_operations.get_dataset_name(self.dataset_path), 
                    score = overall_relevance_score, 
                    selected_columns = selected_columns,
                    excluded_columns = [''],
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Relevance", 
                    threshold= threshold,
                    test_log_csv = test_log_csv,
                    return_log = return_logs)
                output_logs.append(test_output_log)
         
            # Only return outputs logs if output_log_score has returned logs in memory 
            if return_logs:
                return outputs, output_logs
            return outputs
        else:
            print(f'{core_operations.RED}Non valid entry for tests.{core_operations.RESET}')
            print(f'Test options: {self.ALL_TESTS}, inputted tests: {tests}')
            return -1