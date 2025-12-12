# Contributing to the Data Quality Framework
Thank you for considering contributing! Whether you are adding a new test or tailoring existing ones to your data and validation needs, your contributions help make the framework more flexible, robust, and useful for a wide range of datasets.

## Getting Started
Before contributing, ensure you have forked the repository and cloned it locally. 

### Basic Workflow
1. **Create a new branch**

   In a terminal at the project root:
   ```
   git checkout -b branch_name
   ```
2. **Make your changes** following the guidelines in [Adding New Tests](#Adding-New-Tests)
3. **Add your changes** (only add the files you want included in the pull request)

   In a terminal at the project root:
   ```
   git add x#.py # e.g., git add a1.py
   ```
4. **Commit and push your changes**

    In a terminal at the project root:
    ```
    git commit -m "Added new test X# for [dimension]"
    git push -u origin branch_name
    ```   
5. **Open a pull request** to merge your changes into the main repository

   Describe the test and include examples if helpful.
   
   TODO: Set a standard for merging changes to main repo
   
## Code Structure
The framework organizes tests into dimensions, where tests that evaluate similar aspects of data quality are conceptually grouped as *metrics*. Each dimension is stored in its own folder, containing all files needed to define, manage, and load its tests.

### Files in each Dimension Folder
- `dimension_reference.py`: Manages and loads all tests in a dimension, collects their metadata, and provides `run_tests()` to execute selected or all tests with their specific parameters.
- Test file (e.g., `a1.py`, `c2.py`): Defines a single tests's parameters, logic, and metadata. Each test returns a score and optionally a CSV used for reporting.
- `test_template.py`: Provides a template for contributors to add a new test, including placeholders for parameters, logic, and metadata. 

## Adding New Tests
Add a new test or customize an existing one by copying the test template and filling in the `# TODO` sections.

### Code Standards
- Test names must follow the `X#` naming convention (e.g., `A1`).
- Any new library added for a test is to be added to the [requirements.txt](requirements.txt) file with the package version specified. 

### Steps to Add a New Test
1.  **Identify the dimension** your test belongs to. See the [Tests Reference Table](REFERENCE_TABLE.md) for reference.
2. **Navigate** to the corresponding dimension folder under [dimensions/](dimensions/).
3. **Copy the test template** `test_template.py` and rename it to your test name (e.g., `a5.py`). 
4. **Edit the template** following the `# TODO` comments:
   - Define the test name and test specific parameters.
   - Assign test specific attributes to `self` variables.
   - Set `self.threshold` (use `None` if not applicable).
   - Set `self.selected_columns` (use `None` if your test does not specify specific columns).
   - Implement your test logic in `run_test()`.
   - **Optional**: Define test metadata and parameter types in `create_metadata()` to run your test in the **UI tool**. Skip this step if using **only the notebook**.
        - Set each parameter's type using `ParameterType.[TYPE]` (see available types [here](TODO:add-link)).
5. **Import any required operation modules** from [utils/](/utils) at the top of your test file (e.g., `from utils import item_operations, column_operations, table_operations`).
   - See [Operations](utils/README.md) for a list of available operations.
   - See [Creating Custom Operations](utils/README.md#Creating-Custom-Operations) to add custom operations.


## Testing your Changes

### Testing Your Test in the Notebook
Before re-running the [notebook](run_tests/Data%20Quality%20Complete.ipynb):
1. **Restart the kernel**
   
   Go to **Kernel** (top lefthand corner) > **Restart Kernel**
2. **Add your new test** under its dimension in the notebook:
   - Set or update any test-specific parameters.
   - Include your test in the `run_tests()` list if not running all tests.
   - Adjust weights if your test requires custom weighting.

### Testing Your Test in the UI Tool
Before re-launching the UI tool:
1. **Ensure** `create_metadata()` **is updated** with test metadata and parameter types.
2. **Stop any previous UI session**
   
    In a terminal running the UI, press **CTRL + C**
3. **Re-launch the UI tool**:
   ```
   streamlit run ui_tool/dq_ui.py
   ```