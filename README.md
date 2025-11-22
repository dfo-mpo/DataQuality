# Data Quality Framework
This repository contains scripts to calculate data quality for datasets used in the Pacific Salmon Strategy Initiative (PSSI) Data Portal. Metrics are organized into dimensions, and each metric can be run using a Jupyter Notebook or a no-code UI tool on CSV or XLSX datasets.

For instructions on setting up the repository and running metrics, see the [Getting Started](#Getting-Started) section.

This framework is **open source**, and contributions are welcome. Check out the [CONTRIBUTING](CONTRIBUTING.md) page to add new metrics or customize existing ones.  
  
## Dimensions and Metrics 
Data quality metrics are divided into 8 dimensions:
1. Accessibility
2. Accuracy
3. Completeness
4. Consistency
5. Interdependency
6. Relevance
7. Timeliness
8. Uniqueness

See the [Metrics Reference Table](REFERENCE_TABLE.md) for a complete list of runnable metrics.

For full details on each metric, see the [Detailed Metrics](run_metrics/DETAILED_METRICS_TESTS.md) page.

## Getting Started
Set up the repository and run metrics.

1. **Fork the repository and clone it locally**

    In a terminal, run:
    ```
    git clone https://github.com/dfo-mpo/DataQuality.git
    cd DataQuality
    ```
2. **Choose how to run metrics**
    - [Run metrics with the notebook](run_metrics/RUN_METRICS.md#Run-Metrics-with-the-Notebook)
    - [Run metrics with the UI tool](run_metrics/RUN_METRICS.md#Run-Metrics-with-the-UI)
