# Metrics Reference Table

| Dimension     | Metric | Description                                  |
|---------------|--------|----------------------------------------------|
| Accessibility | S1     | Gives a score based on whether a metadata file exists for the specified dataset. |
| Accuracy | A1     | Checks whether letters or symbols exist in the specified numeric column(s). |
| Accuracy | A2     | Detects outliers in the specified numeric column(s) using the interquartile range (IQR) method. |
| Accuracy | A3     | Checks whether values in the specified aggregated column (e.g., Total) equal the sum of their respective component columns. |
| Accuracy | A4     | Checks whether related timestamp columns are in chronological order, with the first specified column in each pair assumed to be the start timestamp. |
| Completeness | P1     | Checks whether there are blanks or NULL values in the entire specified dataset. |
| Completeness | P2     | Finds column pairs whose missing values are correlated above a specified threshold, exploring whether missingness occurs independently or follows consistent patterns. |
| Consistency | C1     | Detects near-duplicate entries in the specified column(s) that likely refer to the same entity despite minor differences in spelling or naming conventions. |
| Consistency | C2     | Checks whether string values in the specified column(s) consistently follow naming conventions found in the specified reference dataset. |
| Consistency | C3     | Checks whether string values in the specified column(s) are similar to official province or territory names using the Levenshtein similarity ratio. |
| Consistency | C4     | Checks whether date-time values in the specified column(s) follow a specified format, such as ISO 8601. |
| Consistency | C5     | Checks whether geographic coordinates in the specified column(s) follow Decimal Degrees (DD) format and represent valid latitude and longitude values, optionally restricting validation to DFO’s Pacific Region. |
| Interdependency | I1     | Identifies proxy variables by finding non-sensitive features whose correlation with the specified sensitive feature(s) exceed a specified threshold. |
| Relevance | R1     | Currently an empty template metric. |
| Timeliness | T1     | Currently an empty template metric. |
| Uniqueness | U1     | Detects duplicate rows across the entire specified dataset. |