# Data Quality Framework
This repository contains scripts to calculate data quality for datasets used in the Pacific Salmon Strategy Initiative (PSSI) Data Portal. Tests are organized into dimensions, and tests that evaluate similar aspects of data quality are conceptually grouped as *metrics*. Each test can be run using a Jupyter Notebook or a no-code UI tool on CSV or XLSX datasets. 

For instructions on setting up the repository and running tests, see the [Getting Started](#Getting-Started) section.

This framework is **open source**, and contributions are welcome. Check out the [CONTRIBUTING](CONTRIBUTING.md) page to add new tests or customize existing ones.  
  
## Dimensions and Tests 
Data quality tests are divided into 8 dimensions:
1. Accessibility
2. Accuracy
3. Completeness
4. Consistency
5. Interdependency
6. Relevance
7. Timeliness
8. Uniqueness

See the [Tests Reference Table](REFERENCE_TABLE.md) for a complete list of runnable tests.

For full details on each test, see the [Detailed Tests](run_tests/DETAILED_TESTS.md) page.

## Getting Started
Set up the repository and run metrics.

### Prerequisites
- Python 3.10 or later. See [instructions](https://www.python.org/downloads/).
- Git. See [instructions](https://git-scm.com/install/)
- Jupyter Notebook or Jupyterlab. See [instructions](https://jupyter.org/install).

### Clone the Repository

1. **Fork the repository and clone it locally**

    In a terminal, run:
    ```
    git clone https://github.com/dfo-mpo/DataQuality.git
    cd DataQuality
    ```
2. **Choose how to run tests**
    - [Run tests with the notebook](run_tests/README.md#Run-Tests-with-the-Notebook)
    - [Run tests with the UI tool](run_tests/README.md#Run-Tests-with-the-UI)
