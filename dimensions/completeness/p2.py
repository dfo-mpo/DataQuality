from dython.nominal import associations
from utils import core_operations, table_operations
from ui_tool.metadata import TestMetadata, ParameterType

TEST = "P2"

""" Class to represent an individual test for the Completeness dimension.

    Goal: Ensure that all required data is available and that there are no missing values. 
    Complete data includes all necessary records and fields needed for the intended use.

dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only test scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
p2_threshold: threshold for correlation coefficient that is acceptable for P2 test.
"""
class Test:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, p2_threshold=0.5, threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        self.p2_threshold = p2_threshold

        self.threshold = self.p2_threshold
        self.selected_columns = None
    
    """ Completeness Type 2 (P2): Finds column pairs with missing values whose correlation coefficient is higher than 0.5 (or any threshold).
    Given that correlation ranges from -1 to 1 (1 suggests perfect association, 0 suggests no relation), 0.5 will be used as a midpoint threshold to investigate whether an association exists. 
    """ 
    def run_test(self):
        df = core_operations.read_data(self.dataset_path)

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
        corrs_thr = table_operations.filter_corrs(corrs, self.p2_threshold)

        # Compute score 
        completeness_score = (1 - (len(corrs_thr) / n_pairs)) if corrs_thr is not None else None
        
        # Conditional return logic
        if self.return_type == "score":
            return completeness_score, None
        elif self.return_type == "dataset":
            if not completeness_score: 
                return "No valid p2 results generated", None
            
            pdf = corrs_thr
            output_file = core_operations.df_to_csv(self.logging_path, test=TEST.lower(), final_df=pdf)
            return completeness_score, output_file  # Return the file name
            
        else:
            return df, None  # Default return value (DataFrame)
       
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Completeness"

    # Define instance for test
    p2_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    p2_metadata.add_parameter('p2_threshold', 'P2 Threshold', ParameterType.DECIMAL, value='0.5', step = 0.05)
    
    return p2_metadata 