# Run Tests
This page explains how to run data quality tests and what information is included in the output log for each test run.

## How to Run Tests
You can run data quality tests using either of the following:
- [Jupyter Notebook](#Run-Tests-with-the-Notebook) — set test parameters in code
- [UI Tool](#Run-Tests-with-the-UI) — ideal for quick, no-code execution


### Run Tests with the Notebook
1. **Install required libraries**

   In a terminal, run:
   ```  
   pip install -r requirements.txt
   ```  
2. **Open the notebook**

    Open the [Data Quality Complete](Data%20Quality%20Complete.ipynb) notebook and choose your dataset file (CSV or XLSX):
    - In **Setup**, set `DATA_FILE_PATH` in the last code cell.
    - Dataset requirements:
        - The data must be on the **first sheet** in the Excel document.  
        - The **first row** must be the column names.  
        - The test won't run if the Excel file is open.

3. **Prepare tests**

    - Under **Prepare [Dimension] Tests**:
       - Define or update parameters for each test in the `test_params` dictionary.
    - Under **Run [Dimension] Tests**:
        - Specify which tests to run as a list with `run_tests()` (e.g., `['C1', 'C3']`). Leave the input empty to run all tests for the given dimension.
        - By default, `calculate_dimension_score()` uses equal weights. To change this, set weights in a dictionary (e.g., `weights = {'C1': 0.3, 'C3': 0.7}`). Note that weights must add up to 1, otherwise the default is used.
  
4. **Run the entire notebook**

   Go to **Run** (top lefthand corner) >  **Run All Cells**
  
5. **View Results**
    - For the calculated Data Quality, see the output at the last cell in the notebook.  
    - For individual test and dimension scores, see the output below each code cell for the given dimension.

### Run Tests with the UI

1. **First time setup**

   In a terminal at the project root, run:
   ```
   py -m venv streamlit-env
   streamlit-env\Scripts\activate
   pip install -r requirements.txt
   pip install streamlit
   ```

2. **Launch the UI**

   - Activate the virtual environment (required for all launches after the first time setup):
     ```
     streamlit-env\Scripts\activate
     ```
    - Launch the UI:
      ```
      streamlit run ui_tool/dq_ui.py
      ```

3. **Run Tests in the UI**

    - Choose your dataset file (CSV or XLSX).
    - Select dimensions and tests to run.
    - Enter any required fields and press **Calculate Data Quality**.

## Logging Outputs
Each test run generates results in the [DQS_Output_Log_Test.xlsx](DQS_Output_Log_Test.xlsx) file. The log includes the test score and details on why the score was assigned. Each row contains the following information:
- **Dataset**: Dataset name
- **Dimension**: Dimension name
- **Test**: Test name
- **Selected_Columns**: Column(s) tested
- **Threshold**: Threshold used or `no threshold`
- **Score**: Test score
- **Run_time_and_Date**: Timestamp of the run
- **New_or_Existing_Test**: `Standard` for tests included in the framework; `Custom` for tests not created by the SDPA team 
- **One_Line_Summary**: Concise summary for framework tests only; custom tests do not include a summary
- **Errors**: Error type, if any
- **Why_Did_the_Test_Fail**: Reason for error, if any
  
To generate the "One_Line_Summary" in the notebook, set the dimenion's return type to `"Dataset"` instead of `"Score"`.