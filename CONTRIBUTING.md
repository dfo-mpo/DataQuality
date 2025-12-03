# Contributing to the Data Quality Framework
Thank you for considering contributing! Whether you are adding a new metric or tailoring existing ones to your data and validation needs, your contributions help make the framework more flexible, robust, and useful for a wide range of datasets.

## Getting Started
Before contributing, ensure you have forked the repository and cloned it locally. 

### Basic Workflow
1. **Create a new branch**

   In a terminal at the project root:
   ```
   git checkout -b branch_name
   ```
2. **Make your changes** following the guidelines in [Adding New Metrics](#Adding-New-Metrics)
3. **Add your changes** (only add the files you want included in the pull request)

   In a terminal at the project root:
   ```
   git add x#.py # e.g., git add a1.py
   ```
4. **Commit and push your changes**

    In a terminal at the project root:
    ```
    git commit -m "Added new metric X# for [dimension]"
    git push -u origin branch_name
    ```   
5. **Open a pull request** to merge your changes into the main repository
   Describe the metric and include examples if helpful.
   
   TODO: Set a standard for merging changes to main repo
   
## Code Structure
The framework organizes metrics into dimensions, with each dimension stored in its own folder. Each folder contains all files needed to define, manage, and load its metrics.

### Files in each Dimension Folder
- `dimension_reference.py`: Manages and loads all metrics in a dimension, collects their metadata, and provides `run_metrics()` to execute selected or all metrics with their specific parameters.
- Metric file (e.g., `a1.py`, `c2.py`): Defines a single metric's parameters, logic, and metadata. Each metric returns a score and optionally a CSV used for reporting.
- `metric_template.py`: Provides a template for contributors to add a new metric, including placeholders for parameters, logic, and metadata. 

## Adding New Metrics
Add a new metric or customize an existing one by copying the metric template and filling in the `# TODO` sections.

### Code Standards
- Metric names must follow the `X#` naming convention (e.g., `A1`).
- Any new library added for a metric is to be added to the [requirements.txt](requirements.txt) file with the package version specified. 

### Steps to Add a New Metric
1.  **Identify the dimension** your metric belongs to. See the [Metrics Reference Table](REFERENCE_TABLE.md) for reference.
2. **Navigate** to the corresponding dimension folder under [dimensions/](dimensions/).
3. **Copy the metric template** `metric_template.py` and rename it to your metric name (e.g., `a5.py`). 
4. **Edit the template** following the `# TODO` comments:
   - Define the metric name and metric specific parameters.
   - Assign metric specific attributes to `self` variables.
   - Set `self.threshold` (use `None` if not applicable).
   - Set `self.selected_columns` (use `None` if your metric does not specify specific columns).  
   - Implement your metric logic in `run_metric()`.
   - **Optional**: Define metric metadata and parameter types in `create_metadata()` to run your metric in the **UI tool**. Skip this step if using **only the notebook**.
        - Set each parameter's type using `ParameterType.[TYPE]` (see available types [here](TODO:add-link)).
5. **Import any required operation modules** from [utils/](/utils) at the top of your metric file (e.g., `from utils import item_operations, column_operations, table_operations`).
   - See [Operations](utils/OPERATIONS.md) for a list of available operations.
   - See [Adding New Operations](utils/OPERATIONS.md#Adding-New-Operations) to add custom operations.


## Testing your Changes

### Testing Your Metric in the Notebook
Before re-running the [notebook](run_metrics/Data%20Quality%20Complete.ipynb):
1. **Restart the kernel**
   
   Go to **Kernel** (top lefthand corner) > **Restart Kernel**
2. **Add your new metric** under its dimension in the notebook:
   - Set or update any metric-specific parameters.
   - Include your metric in the `run_metrics()` list if not running all metrics.
   - Adjust weights if your metric requires custom weighting.

### Testing Your Metric in the UI Tool
Before re-launching the UI tool:
1. **Ensure** `create_metadata()` **is updated** with metric metadata and parameter types.
2. **Stop any previous UI session**
   
    In a terminal running the UI, press **CTRL + C**
3. **Re-launch the UI tool**:
   ```
   streamlit run ui_tool/dq_ui.py
   ```