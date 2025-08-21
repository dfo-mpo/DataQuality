# Data Quality Tests  
This repository contains scripts needed to obtain a data quality grade for a dataset used for the PSSI Data Portal. The tests are broken down into 8 dimensions, each with its own metrics. The final outputs are used to calculate a dimension score, and all dimension scores are used to calculate a final grade.  
  
There is a notebook and UI that can be used to run all desired tests and obtain a final data quality grade for a given dataset (CSV or XLSX format).  
  
Additional metrics can also be added to the existing dimensions, with steps to do so explained in the [adding new metrics section](#adding-new-metrics).  
  
## Dimensions and Metric Tests  
All tests used to calculate a data quality score are divided into 8 dimensions: accessibility, accuracy, completeness, consistency, interdependency, relevance, timeliness, and uniqueness. Metrics are tests used to determine how well a given dataset satisfies the given dimension.  
  
Each dimension has its own Python file in the dimensions folder containing all relevant metrics and documentation on what the goal of the given dimension is.  

### Logging Outputs
Results from each individual metric run is outputed into the file 'DQS_Output_Log_Test.xlsx'. It contains the score with details to provide insight on why the metric gave the score it did. The information that is logged for each metric run are: "Dataset", "Dimension", "Test", "Selected_Columns", "Threshold", "Score", "Run_Time_and_Date", "New_or_Existing_Test", "One_Line_Summary", "Errors", "Why_Did_the_Test_Fail". To generate the "One_Line_Summary" for the metric result, the return type for the dimension must be set to 'Dataset' rather than 'Score'.
  
### Code Structure  
Each dimension file describes a Python class with the initializer containing the parameters needed for all of the dimension's metric tests. Each metric test exists in the class as numbered private methods with a public method (`run_metrics`) that can be used to run all or specific metrics.  
  
Each metric returns a final score but can also return a CSV with the values that were used to calculate the score. This is used to create a one-sentence summary in the output report giving more insights into how the dataset produced the score for the given metric.  

### Accessibility
Goal: Ensure that data is easily accessible to authorized users when needed. Accessible data is stored in a way that makes it easy to retrieve and use, while also being secure from unauthorized access.

#### Accessibility Type 1 (S1):
Currently an empty template test.

### Accuracy
Goal: Ensure that the data correctly represents the real-world values it is intended to model. Accurate data is free from errors and is a true reflection of the actual values.

#### Accuracy Type 1 (A1): 
Determines whether there are symbols in numerics. Make the column a string, find symbols, and calculate the accuracy scores for multiple columns.

#### Accuracy Type 2 (A2): 
Find outliers that are 1.5 (or any threshold) times away from the inter-quartile range. The threshold for how many inter-quartile range is considered to be an outlier and percentage of the column selected that passes can be customized.

### Completeness
Goal: Ensure that all required data is available and that there are no missing values. Complete data includes all necessary records and fields needed for the intended use.

#### Completeness Type 1 (P1): 
Checks for whether there are blanks in the entire dataset.

### Consistancy
Goal: Ensure that data is consistent across different datasets and systems. Consistent data follows the same formats, standards, and definitions, and there are no contradictions within the dataset.

#### Consistency Type 1 (C1): 
Determines the similarity between string values in specified columns. Process the dataset, normalize the text, and calculate the similarity scores for multiple columns.

Limitations: It will not check for differences in capitalization of the same word (since all the words will be changed to lower case before the similarity score is calculated)

#### Consistency Type 2 (C2): 
Compares reference data and string values in specified columns. The compared columns in question must be identical to the ref list, otherwise they will be penalized more harshly.

### Interdependency
Goal: Ensure that data across different systems and datasets are harmonized and can be integrated. Interdependent data can be effectively combined and used together without discrepancies.

#### Interdependency Type 1 (I1):
Currently an empty template test.

### Relevance
Goal: Ensure that the data is relevant and useful for the intended purposes. Relevant data meets the needs of the users and supports the business processes.

#### Relevance Type 1 (R1):
Currently an empty template test.

### Timeliness
Goal: Ensure that the data is up-to-date and available when needed. Timely data is delivered at the right time to support decision-making processes.

#### Timeliness Type 1 (T1):
Currently an empty template test.

### Unigueness
Goal: Ensure that each record in the dataset is unique and there are no duplicate entries. Unique data means there are no redundant records.

#### Uniqueness Type 1 (U1):
Find duplicated rows.

## Running Tests with the Notebook  
You can run all of the tests using the notebook [Data Quality Complete](/Data%20Quality%20Complete.ipynb).  
  
1. Install the following libraries using the following command in a terminal:  
    ```sh  
    pip install numpy pandas scikit-learn nbformat openpyxl
    ```  
  
2. Connect your dataset (CSV or XLSX file):  
    - Ensure the data is on the **first sheet** in the Excel document.  
    - Ensure the **first row** contains the column names.  
    - Ensure the Excel file is not open, or else the tests won't run.  
    - Go to the last code cell under Setup and set `GLOBAL_USER` and `DATA_FILE_PATH` to your dataset.  
  
3. Prepare metrics to run. There is a code cell for each dimension. For each cell:  
    - Ensure in class initialization desired values are added to given parameters.  
    - For `calculate_dimension_score()`:  
        - Ensure `run_metrics()` has the name of all metrics you wish to run as a list (e.g., `['C1', 'C3']`) or leave the input empty to run all metrics for the given dimension.  
        - Default weighting for metrics under a dimension makes them all equal. To change this, you can specify the weights with a dictionary with metric names that match those used for `run_metrics()` (e.g., `{'C1': 0.3, 'C3': 0.7}`). Note that weights must add up to 1, or the default will be used.  
  
4. Run the entire notebook.  
  
5. Check results:  
    - For the Data Quality grade, see the output at the last cell in the notebook.  
    - For individual metrics and dimensions, see the output below each code cell for the given dimension.  
  
## Running Tests with the UI
The UI tool allows users to calculate Data Quality scores by entering the given feilds and pressing the 'Calaculate Grade' button.
### First time setup
1. Open a terminal at the root of this project and run the following commands to setup the enviroment:
```
py -m venv streamlit-env
streamlit-env\Scripts\activate
pip install streamlit numpy pandas scikit-learn nbformat openpyxl
```
2. Then to run the UI, run the following command
```
streamlit run dq_ui.py
```

### Running script after setup
1. Open a terminal at the root of this project and active the virtual enviroment with the command: `streamlit-env\Scripts\activate`
2. Run the UI with the command: `streamlit run dq_ui.py`
  
## Adding New Metrics  
You may want to run your own test or implementation of an existing test. In this case, you can add a metric to the code. This will involve adding the new test code and updating global parts of a dimension Python file, then making possible updates to the Notebook and UI to run the new metric properly.  
  
### Adding Test to Dimension File  
Once you have determined which dimension the metric fits under, open its Python file under the dimensions folder and follow the steps below:  
  
1. Add a new private method into the class for your test:  
    - Ensure it follows the naming convention of other tests (following the format `_x#_metric()`).  
    - If it uses helper functions, you can add them to the file `dimensions/utils.py` and then add them as an import to the dimension file.  
    - Add a comment above the method describing what the test does.  
    - Parameters should only be `self` and `metric`. If any other parameters are needed, use `self.` instead. In the next step, these will be added to the class.  
  
2. If there are any additional parameters the test needs, add them as parameters in the `__init__()` method and use the `self` keyword to assign the new parameters to the class instance so your new metric can use them.  
  
3. At the top of the file, there is a variable called `ALL_METRICS`. Add the name for your new metric following the naming convention of X#.  
  
4. In the method `run_metrics()`:  
    - In the `thresholds` dictionary, add an entry using your new metric name. If your test does not use one, set the value to `None`.  
    - In the `columns` dictionary, add an entry using your new metric. If your test does not select specific columns, set the value to `None`.  
    - Go to the try/catch block and add a new `elif` case where if `metric` equals your new test name, it will match the other if blocks except it will call your new metric test.  
  
### Changes in Notebook for New Metric  
Before re-running the notebook, you may need to apply the following changes to the cell for the dimension you have modified:  
- Update the class initialization with any new parameters added to the dimension class.  
- In the function call `calculate_dimension_score()`, update the input for `run_metrics` if you specified which tests to run.  
- In the function call `calculate_dimension_score()`, update the weights if you have specified what weights to use.  
  
### Changes in UI File for New Metric  
To be added soon, the tool is still under development. Code standards need to be set first.