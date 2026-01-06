# Operations
This page provides an overview of the operations used throughout the framework. Operations are reusable helper and utility functions that support test logic and repetitive processing tasks.

Contributors can reuse existing operations or create custom ones. For guidelines on adding custom operations, see the [Creating Custom Operations](#Creating-Custom-Operations) section.

## Types of Operations
Operations are organized by level of application:
| Type         | Application                                  |
|--------------|----------------------------------------------|
| [Item](#Item-Operations) | Applied to an individual value or cell in a dataset. |
| [Column](#Column-Operations) | Applied to an entire column or list of values. |
| [Table](#Table-Operations) | Applied to an entire dataset or multiple rows and columns at once. |
| [Core](#Core-Operations) | Handles tasks such as reading/writing data, grading, and logging results |

Each type of operation is implemented in its respective files under the [utils/](../utils) folder:
 - [items_operations.py](item_operations.py)
 - [column_operations.py](column_operations.py)
 - [table_operations.py](table_operations.py)
 - [core_operations.py](core_operations.py)

## Using Operations
1. Import the operation module at the top of your test file:
    ```
    from utils import item_operations # or column_operations, table_operations
    ```
2. Call the desired operations using the module name as a prefix:
    ```
    numbers = item_operations.extract_numbers("ID 123")
    print(numbers) # Output: ['123']
    ```
This same approach works for Column and Table operations; use the corresponding module name.

## Creating Custom Operations
Contributors can add custom operations to support the test they implement.

1. Choose the appropriate operation type:
   - `Item` for individual values or cells
   - `Column` for single columns or lists of values
   - `Table` for entire datasets of multiple rows/columns
2. Implement your operation in the chosen file under [utils/](../utils).
3. Use your custom operations in a test following the steps in [Using Operations](#Using-Operations).

## Item Operations
Item operations are applied to individual values or cells in a dataset.

| Operation               | Description                                   |
|------------------------|-----------------------------------------------|
| `extract_numbers` | Extracts all numbers from the input text and returns them as a list of strings. |
| `inconsistent_datetime` | Checks whether a given string date-time entry matches a specified format. | 
| `normalize_text` | Normalizes input text by converting to lowercase, stripping whitespace, replacing province abbreviations with full names, and removing non-alphanumeric characters. Optionally removes numbers based on the `remove_numbers` flag. |
| `remove_short_numbers` | Removes numbers with one to four digits from the input text. |
| `province_abbreviations` | Dictionary mapping province abbreviations to their full names. |
| `string_similarity` | Calculates the similarity between two strings using `SequenceMatcher` from `difflib` and returns the similarity ratio. |              

## Column Operations
Column operations are applied to entire columns or lists of values.

| Operation               | Description                                   |
|------------------------|-----------------------------------------------|
| `calculate_cosine_similarity` | Calculates the cosine similarity between lists of texts using TF-IDF vectorization. |
| `calculate_levenshtein_similarity` | Calculates the levenshtein similarity ratio between lists of texts. |        
| `compare_datasets` | Compares whether a column being tested resembles a reference data column and creates a DataFrame with new column(s) added. |
| `contains_short_number` | Checks whether any number in the list has one to four digits. |
| `find_non_digits` | Uses the new null flag column to find symbols in numeric columns, changes existing nulls to `True` to prepare any output dataset that only flags symbols in numeric columns, and replaces symbols in numerics with `NaN`. |
| `get_names_used_for_column` | Extracts unique non-null values from a specified column. |        
| `new_column` | Creates a new column that flags previously existing nulls and empty strings, preventing false positives by not counting them as symbols in numeric columns. |
| `numbers_match` | Checks whether any number in the first list is present in the second list. |
| `numeric_similarity` | Calculates the similarity between two lists of numbers by comparing each digit and returns the proportion of matching digits. |

## Table Operations
Table operations are applied to entire datasets or multiple rows and columns at once.

| Operation               | Description                                   |
|------------------------|-----------------------------------------------|
| `add_only_numbers_columns` | Creates a new column(s) in the output report based on the original input dataset with additional generated columns. |
| `average_c1_consistency_score` | Calculates the average consistency score for C1 using the cosine similarity matrix and a specified threshold.  |        
| `average_c2_consistency_score` | Calculates the average consistency score for C2 using the cosine similarity matrix and a specified threshold. |
| `average_c3_consistency_score` | Calculates the average consistency score for C3 using the levenshtein similarity ratio matrix and a specified threshold. |
| `calculate_combined_similarity` | Combines the text and numeric similarities into a single similarity matrix. |
| `filter_corrs` | Filters column pairs that meet the specified threshold, removing duplicates and same column pairings. |        
| `get_max_similarity_values` | Extracts the maximum similarity values with their corresponding project names and creates a DataFrame with this information. |

## Core Operations
Core operations handle tasks such as reading/writing data, grading, and logging results.

**Note:** This file is read-only. Please do not modify or add operations to this file.

| Operation               | Description                                   |
|------------------------|-----------------------------------------------|
| `are_weights_valid` | Determines whether the given set of weights meet the following criteria: the number of weights match the number of tests/dimensions, and the weights sum up to 1. |
| `calculate_dimension_score` | Calculates the total score for a dimension from a list of test scores and returns a dictionary with the dimension and overall score. |        
| `calculate_DQ_grade` | Determines a data quality grade from a list of dimension score, where each entry contains the dimension name and total score. |
| `df_to_csv` | Converts a DataFrame to CSV, returning the CSV filename if a logging path is defined, otherwise returns the CSV in memory. |
| `list_test_names` | Gets a list of all test names from filenames in the specified folder, excluding the `.py` extension. |
| `output_log_score` | Logs a new row into the `DQS_Output_Log_xx.xlsx` file |        
| `read_data` |  Reads data from a CSV or XLSX file or returns the input if already a DataFrame. |
| `RED` | Colors console output as red text. |
| `RESET` |  Resets console output as white text (default). |