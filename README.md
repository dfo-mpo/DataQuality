# Data Quality Test Generator
The Data Quality Test Generator is a no-code tool that generates Python-based data quality tests for the Data Quality Framework. 

It creates a completed test template based on user input, allowing contributors to define new tests or tailor existing ones to their dataset without manually writing code. The generated template can be downloaded and added to a local clone of the Data Quality Framework repository and run using the framework notebook or UI tool.

For instructions on using the generator, see the [Using the Test Generator](#Using-the-Test-Generator).


## Features
The generator allows users to:
- Select one of the data quality [dimension](#dimensions)
- Describe the logic of a test
- Automatically generate a completed Python test template
- Download or copy the generated template 

## Dimensions
This tool supports the 8 dimensions used in the Data Quality Framework:
- Accessibility
- Accuracy
- Completeness
- Consistency
- Interdependency
- Relevance
- Timeliness
- Uniqueness

For examples of existing tests in the framework, see the [Detailed Tests](https://github.com/dfo-mpo/DataQuality/blob/main/run_tests/DETAILED_TESTS.md) page.

## Using the Test Generator
Open the tool: [Link to be added once deployed]

### Generate a Filled Test Template

1. **Select** a data quality dimension from the **dropdown**.
2. **Describe** the test, including:
    - Whether the test applies to one column, multiple columns, or the entire dataset
    - Edge cases or special conditions
    - Scoring logic *(e.g., proportion of non-null values in a column)*
3. Click the **up arrow** button to generate code.
4. Click the **Download File** button to download the filled template or copy the generated code in code block.

**Output:** A Python test template compatible with the Data Quality Framework.

## Using the Filled Test Template in the Framework
After generating a template, add it to your **local clone of the Data Quality Framework repository**. 

See the [Getting Started](https://github.com/dfo-mpo/DataQuality?tab=readme-ov-file#Getting-Started) section for setup instructions.

### Add the Test to the Framework
1. **Open your local clone** of the Data Quality Framework repository.
2. **Navigate** to the appropriate dimension folder under [dimensions/](https://github.com/dfo-mpo/DataQuality/tree/main/dimensions).
3. **Add the test:**
    - **Move the downloaded file** into the folder, or 
    - **Copy the test template** `test_template.py` and paste the generated code into the new file
4. **Rename the file** using the framework's naming convention (e.g., `a5.py`, `c3.py`).

For more details, see:
- [Code Structure](https://github.com/dfo-mpo/DataQuality/tree/main?tab=contributing-ov-file#code-structure)
- [Adding New Tests](https://github.com/dfo-mpo/DataQuality/tree/main?tab=contributing-ov-file#adding-new-tests) 

**Important:** The generator uses an AI model and may make mistakes. Always review the generated code to ensure correct naming conventions, parameters, and test logic.

### Running the Test
Once your test is added to the framework repository, run it using the framework [notebook](#run-a-single-test-notebook) or [UI tool](#run-a-single-test-ui-tool).

#### Run a Single Test (Notebook)
This workflow is intended for running and validating your newly generated test within the notebook.

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

3. **Register your new test** under its dimension in the notebook:
   - Set test-specific parameters in the `test_params` dictionary.
        - These should match the parameters defined in the `__init__` header in your template file.
   - Set `run_tests()` to include only your test.
        - Example: `run_tests(['C3'])`

4. **Restart the kernel**

    Go to **Kernel** (top lefthand corner) > **Restart Kernel**

5. **Run the required sections**

    Run a selected cell using **Shift+Enter** or go to **Run** (top lefthand corner) > **Run Selected Cell**.
    
    Run:
    - **Setup** section
    - Your test's **dimension** section
    - **Determine Overall Data Quality** section
  
6. **View Results**
    - For the calculated Data Quality, see the output at the last cell in the notebook.  
    - For individual test and dimension scores, see the output below each code cell for the given dimension.
    - Confirm the test runs and behaves as expected.

#### Run a Single Test (UI Tool)
This workflow is intended for running and validating your newly generated test through the UI tool.

Before re-launching the UI tool:
1. **Ensure** `create_metadata()` **is updated** with test metadata and parameter types.
2. **Stop any previous UI session**
   
    In a terminal running the UI, press **CTRL + C**
3. **Re-launch the UI tool**:
   ```
   streamlit run ui_tool/dq_ui.py
   ```
4. **Run Test in the UI**

    - Choose your dataset file (CSV or XLSX).
    - Select the dimension and your new test to run.
    - Enter any required fields and press **Calculate Data Quality**.

5. **View Results**
    - For the calculated Data Quality and individual test scores, see the output log under **Calculated Data Quality:**.
    - Confirm the test runs and behaves as expected.
