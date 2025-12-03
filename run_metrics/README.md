# Run Metrics
This page explains how to run data quality metrics and what information is included in the output log for each metric run.

## How to Run Metrics
You can run data quality metrics using either of the following:
- [Jupyter Notebook](#Run-Metrics-with-the-Notebook) — set metrics parameters in code
- [UI Tool](#Run-Metrics-with-the-UI) — ideal for quick, no-code execution


### Run Metrics with the Notebook
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

3. **Prepare metrics**

    - Under **Prepare [Dimension] Metrics**:
       - Define or update parameters for each metric in the `metric_params` dictionary.
    - Under **Run [Dimension] Metrics**:
        - Specify which metrics to run as a list with `run_metrics()` (e.g., `['C1', 'C3']`). Leave the input empty to run all metrics for the given dimension.
        - By default, `calculate_dimension_score()` uses equal weights. To change this, set weights in a dictionary (e.g., `weights = {'C1': 0.3, 'C3': 0.7}`). Note that weights must add up to 1, otherwise the default is used.
  
4. **Run the entire notebook**

   Go to **Run** (top lefthand corner) >  **Run All Cells**
  
5. **View Results**
    - For the calculated Data Quality, see the output at the last cell in the notebook.  
    - For individual metric and dimension scores, see the output below each code cell for the given dimension.

### Run Metrics with the UI

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

3. **Run Metrics in the UI**

    - Choose your dataset file (CSV or XLSX).
    - Select dimensions and metrics to run.
    - Enter any required fields and press **Calculate Data Quality**.

## Logging Outputs
Each metric run generates results in the [DQS_Output_Log_Test.xlsx](DQS_Output_Log_Test.xlsx) file. The log includes the metric score and details on why the score was assigned. Each row contains the following information:
- **Dataset**: Dataset name
- **Dimension**: Dimension name
- **Test**: Metric name
- **Selected_Columns**: Column(s) tested
- **Threshold**: Threshold used or `no threshold`
- **Score**: Metric score
- **Run_time_and_Date**: Timestamp of the run
- **New_or_Existing_Test**: `Standard` for metrics included in the framework; `Custom` for metrics not created by the SDPA team 
- **One_Line_Summary**: Concise summary for framework metrics only; custom metrics do not include a summary
- **Errors**: Error type, if any
- **Why_Did_the_Test_Fail**: Reason for error, if any
  
To generate the "One_Line_Summary" in the notebook, set the dimenion's return type to `"Dataset"` instead of `"Score"`.