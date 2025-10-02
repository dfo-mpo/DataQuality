# Detailed Metrics/Tests
This page provides an in-depth overview of the data quality metrics/tests, along with variables that users can define. All tests used to calculate a data quality score are divided into 8 dimensions: accessibility, accuracy, completeness, consistency, interdependency, relevance, timeliness, and uniqueness.

To compute the final data quality grade for a given dataset, weights can be assigned to each dimension:
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `Dimension Weights` | Weights assigned to each dimension. By default, all dimensions are weighted equally. All weights must add up to 1. | `{'Accessibility': 0.3,'Consistency': 0.4,'Uniqueness': 0.3}` |

## Metric Tests 
Metrics are tests used to determine how well a given dataset satisfies the given dimension. Each metric returns a final score and can optionally generate a CSV file containing a subset of the data that is non-compliant. This is used to create a one-sentence summary in the output report, giving more insights into how the dataset produced the score for the given metric. 

For tests applied across multiple columns, the returned CSV includes additional indicator columns to identify the source of errors. These can be filtered to look at column-specific information. In the case of Completeness (P2) and Interdependency (I1), the output report includes pairs of column names along with their corresponding correlation coefficients, which meet or exceed a defined threshold.

The following variables are generic and can be defined for each dimension:
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `Metrics` | Metrics to run within a dimension. By default, all metrics are run. | `A1` `A2` |
| `Return Type` | Defines what a metric returns. Set to `dataset` to generate a one-line summary. Default is `score`.  | `score` |
| `Weights` | Weights assigned to metrics within a dimension. By default, all metrics are weighted equally. Weights must add up to 1. | `{'A1': 0.3,'A2': 0.7}` |

Additional variables specific to each metric will be described in their respective sections below.

### Accessibility
Goal: Ensure that data is easily accessible to authorized users when needed. Accessible data is stored in a way that makes it easy to retrieve and use, while also being secure from unauthorized access.

#### Accessibility Type 1 (S1)
Currently an empty template test.

### Accuracy
Goal: Ensure that the data correctly represents the real-world values it is intended to model. Accurate data is free from errors and is a true reflection of the actual values.

#### Accuracy Type 1 (A1) 
A1 checks for letters or symbols in columns that should contain only numeric values. All entries in the selected columns are converted to strings and examined character by character. This test can be applied to one or more columns at the same time, with scores averaged across selected columns. 

A record is flagged as a data entry mistake if any non-numeric character is found within a numeric column.

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `Selected Columns` | Column(s) to test for non-numeric characters. | `col1` `col2` |

#### Accuracy Type 2 (A2)
A2 detects outliers in numeric columns using the interquartile range (IQR) method. A value is considered an outlier if it falls below `Q1 - (threshold * IQR)` or above `Q3 + (threshold * IQR)`. The default threshold is 1.5, but it can be adjusted based on the dataset. This test can be applied to one or more numeric columns at the same time, with scores averaged across selected columns. Users can also specify grouping columns to find outliers within subgroups, with results computed per group and then averaged. 

A column or group is flagged as inaccurate if the proportion of non-outliers falls below or above the specified range. 

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `A2 Minimum Score` | Minimum acceptable proportion of non-outliers in a column or group. Default is `0.85`. | `0.85` |
| `A2 Threshold` | Threshold multiplier for IQR-based outlier detections. Default is `1.5`. | `1.5` |
| `Groupby Columns` | Column(s) to group data by. | `groupby_col` |
| `Selected Columns` | Column(s) to test for outliers. | `col1` `col2` |

#### Accuracy Type 3 (A3) 
A3 checks whether values in an aggregated column (e.g., Total) match the sum of their respective component columns. Blank or NULL entries in selected columns are replaced with zeros for calculation and comparison purposes. This test can be applied to one aggregated column against multiple component columns at the same time. 

A record is flagged as inaccurate if the aggregated value does not equal the sum of its components.

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `A3 Aggregated Column` | Column containing the total value to validate. | `total_col` |
| `A3 Column Names` | Component columns expected to sum up to the aggregated column. | `col1` `col2` |

#### Accuracy Type 4 (A4) 
A4 verifies whether related timestamp columns are in chronological order (i.e., the start timestamp occurs on or before the end timestamp). Missing values in these columns are treated as valid to account for records that are still in progress, awaiting updates, or have no start date on record. This test can be applied to one or more column pairs at the same time, with scores averaged across selected pairs.

A record is flagged as inaccurate if the start timestamp is later than the corresponding end timestamp.

**Variables Users Can Define:**
| Variable          | Description                                   | Example         |
|-------------------|-----------------------------------------------|-----------------|
| `A4 Column Pairs` | List of timestamp column pairs to check. The first column in each pair is assumed to be the start timestamp. | `[('start_col1', 'end_col1'), ('start_col2', 'end_col2')]` |

### Completeness
Goal: Ensure that all required data is available and that there are no missing values. Complete data includes all necessary records and fields needed for the intended use.

#### Completeness Type 1 (P1) 
P1 identifies blank or NULL values in a dataset. By default, all columns are tested, except for:
- Columns explicitly excluded by the user (e.g., columns intentionally left blank)
- Columns intended for comments
- Columns exceeding a specified missing value threshold

The default threshold is 0.75, meaning columns with more than 75% missing values are excluded from the test. Users can adjust this threshold to better suit their dataset. 

A record is flagged as incomplete if it contains a blank or NULL value in any of the tested columns.

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `Exclude Columns` | Column(s) to exclude from the test. | `col1` `col2` |
| `P1 Threshold` | Maximum allowable proportion of missing values in a column. Default is `0.75`. | `0.75` |

#### Completeness Type 2 (P2)
P2 finds pairs of columns whose missing values tend to occur together, with an absolute value of the correlation exceeding a specified threshold. Each data point is labelled `true` if it is missing and `false` otherwise. The correlation coefficient is then calculated between these true/false values for each pair of columns. This test is applied to columns with one or more missing values, excluding comments intended for comments, which are considered less informative.

The correlation coefficient ranges from -1 to 1, where 1 suggests a perfect association and 0 suggests no relationship. The default threshold is 0.5, which serves as a midpoint to detect potential associations. Users may increase this threshold (e.g., to 0.75) to explore stronger missingness correlations.

P2 serves as a secondary test for users interested in exploring the association between missing values across columns. Investigating such patterns can help guide how missing data should be handled and reveal potential gaps in data collection, such as measurements not being recorded for specific sample types.

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `P2 Threshold` | Correlation coefficient threshold for flagging pairs of columns with associated missingness. Default is `0.75`. | `0.75` |

### Consistency
Goal: Ensure that data is consistent across different datasets and systems. Consistent data follows the same formats, standards, and definitions, and there are no contradictions within the dataset.

#### Consistency Type 1 (C1) 
C1 detects near-duplicate entries in selected columns that likely refer to the same entity despite minor differences in spelling or naming conventions. Before generating a unique list of entries, all text is normalized by:
- Converting to lowercase
- Replacing abbreviations with full province or territory names
- Removing short numbers (fewer than two digits)
- Filtering out stop words to focus on meaningful terms

Cosine similarity combined with sequence matching is then calculated between pairs of unique entries against a threshold. This test can be applied to one or more columns at the same time, with scores averaged across selected columns.

A default threshold of 0.91 was predetermined after manual review of test data. Scores at or above this threshold generally indicate the same entity with minor naming variations, while lower scores suggest distinct entries with similar names.

A record is flagged as inconsistent if it has a near-duplicate entry with a similarity score exceeding the specified threshold.

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `C1 Column Names` | Column(s) to test for similarity. | `col1` `col2` |
| `C1 Stop Words` | Common word(s) to exclude from similarity calculations. Default is `['the', 'and']`. | `['the', 'and']` |
| `C1 Threshold` | Similarity score threshold for flagging inconsistency. Default is `0.91`. | `0.91` |

#### Consistency Type 2 (C2) 
C2 checks whether string values in selected columns consistently follow naming conventions found in a reference dataset. If no reference dataset is provided, the test compares values within the dataset itself. Similarity between test and reference values is measured using cosine similarity based on a user-defined threshold. This test can be applied to one or more columns at the same time, with scores averaged across selected columns. 

A default threshold of 0.91 was predetermined through manual review of test data. Scores at or above this threshold typically indicate a close match to a reference entry, with slight naming variations.

A record is flagged as inconsistent if none of its similarity scores to reference values exceed the specified threshold. 

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `C2 Column Mapping` | Mapping of test column(s) to reference column(s) for comparison. | `{'col1':'ref_col1','col2':'ref_col2'}` |
| `C2 Stop Words` | Common word(s) to exclude from similarity calculations. Default is `['activity']`. | `['activity']` |
| `C2 Threshold` | Similarity score threshold for flagging inconsistency. Default is `0.91`. | `0.91` |
| `Reference Dataset File` | File containing reference values for comparison (CSV, XLSX). | `reference_data.csv` |

#### Consistency Type 3 (C3)
C3 compares string values to official province or territory names using the Levenshtein similarity ratio. This ratio measures the similarity between two strings based on the number of character edits required to transform one into the other, where a score of 1 indicates an exact match. Before calculating similarity scores, all text is normalized by:
- Converting to lowercase
- Replacing abbreviations with full province or territory names
- Stripping whitespaces
- Removing non-alphanumeric characters

Each entry is then compared against all official names, and the highest similarity score is used for evaluation against a defined threshold. This test can be applied to one or more columns at the same time, with scores averaged across selected columns. 

A default threshold of 0.91 was chosen after manual review of test data. Scores at or above this threshold largely resemble an official name with minor spelling differences.

A record is flagged as inconsistent if its similarity to any official province or territory name does not exceed the specified threshold.

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `C3 Column Names` | Column(s) to test for similarity to official province/territory names. | `col1` `col2` |
| `C3 Threshold` | Similarity score threshold for flagging inconsistency. Default is `0.91`. | `0.91` |

#### Consistency Type 4 (C4)
C4 checks whether date-time values in selected columns follow a specified format, such as the standard ISO 18601 format (YYYY-MM-DD HH:MM:SS) or any other format appropriate for the dataset. The expected format must be provided as a Python datetime format specifier (e.g., %Y%m%d represents YYYYMMDD). This test can be applied to one or more columns at the same time, with scores averaged across selected columns.

A record is flagged as inconsistent if it does not match the specified date-time format or contains out of bounds values, such as a month greater than 12 or a day outside the valid range. 
 
**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `C4 Column Names` | Column(s) to test for inconsistent date-time formatting. | `col1` `col2` |
| `C4 Format` | Python datetime format specifier to compare entries against. Default is `%Y-%m-%d %H:%M:%S`. | `%Y-%m-%d %H:%M:%S` |

#### Consistency Type 5 (C5) 
C5 verifies that geographic coordinates follow Decimal Degrees (DD) formatting and represent valild latitude and longitude values. Users can optionally restrict validation to coordinates that fall within DFO's administrative Pacific Region. This test can be applied to one or more columns at the same time, with scores averaged across selected columns.

A record is flagged as inconsistent if its geographic coordinate falls outside the defined bounds.

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `C5 Column Names` | Column(s) to test for invalid coordinates. | `col1` `col2` |
| `C5 Region` | Restricts test to check within DFO's administrative Pacific Region by setting to `Pacific`. Default is `None`. | `Pacific` |

### Interdependency
Goal: Ensure that data across different systems and datasets are harmonized and can be integrated. Interdependent data can be effectively combined and used together without discrepancies.

#### Interdependency Type 1 (I1) 
I1 identifies proxy variables, which indirectly capture information about sensitive features and are often used as substitutes for other variables. Sensitive features refer to protected data that, if exposed or mishandled, can lead to legal consequences. For this test, these features can include personal or business identifiable information such as name, address, or licence number. Non-sensitive features do not require the same level of protection and cannot be used to uniquely identify an individual or business. This test is applied to user-defined sensitive features, while all other columns are treated as non-sensitive features. Comment columns are excluded by default, as they are considered less informative. 

The test flags pairs of columns where the absolute value of the correlation coefficient between a non-sensitive and a sensitive feature exceeds 0.75, or any user-defined threshold. Since correlation ranges from -1 to 1, where 1 suggests a perfect association and 0 suggests no relationship, a threshold of 0.75 suggests a high level of association. 

I1 serves to identify proxy variables that capture similar patterns, enabling meaningful analysis without directly using sensitive data.

**Variables Users Can Define:**
| Variable               | Description                                   | Example         |
|------------------------|-----------------------------------------------|-----------------|
| `I1 Sensitive Columns` | Sensitive column(s) to test. | `col1` `col2` |
| `I1 Threshold` | Correlation threshold for flagging proxy variables. Default is `0.75`. | `0.75` |

### Relevance
Goal: Ensure that the data is relevant and useful for the intended purposes. Relevant data meets the needs of the users and supports the business processes.

#### Relevance Type 1 (R1)
Currently an empty template test.

### Timeliness
Goal: Ensure that the data is up-to-date and available when needed. Timely data is delivered at the right time to support decision-making processes.

#### Timeliness Type 1 (T1)
Currently an empty template test.

### Uniqueness
Goal: Ensure that each record in the dataset is unique and there are no duplicate entries. Unique data means there are no redundant records.

#### Uniqueness Type 1 (U1)
U1 finds duplicate rows across the entire dataset. This test is applied to all columns.

A record is flagged as not unique if it exactly matches another row in the data.

## Documentation
- [Data Quality Blog Post](https://086gc.sharepoint.com/:w:/r/sites/PacificSalmonTeam/_layouts/15/Doc.aspx?sourcedoc=%7BA9C80E30-C66C-4C4A-8ED7-E262D58D376F%7D&file=Data%20Quality%20Blog%20v2%20DSComments.docx&wdLOR=c7EBAB712-CB8F-4FE0-B18D-56C95CFBCCA2&action=default&mobileredirect=true)
- [Additional Data Quality Tests](https://086gc.sharepoint.com/:w:/r/sites/PacificSalmonTeam/Shared%20Documents/General/02%20-%20PSSI%20Secretariat%20Teams/04%20-%20Strategic%20Salmon%20Data%20Policy%20and%20Analytics/02%20-%20Data%20Governance/00%20-%20Projects/10%20-%20Data%20Quality/Additional%20Tests/Data%20Quality%20Additional%20Tests_internal.docx?d=w8ce8e77317f44eea8bd076beadab7e2a&csf=1&web=1&e=sKb9He)