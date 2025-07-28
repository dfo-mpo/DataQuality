import numpy as np  
import pandas as pd 
import re
from . import utils

ALL_METRICS = ['C1', 'C2', 'C3', 'C4', 'C5']

""" Class to represent all metric tests for the Consistency dimension
    Goal: Ensure that data is consistent across different datasets and systems. Consistent data follows the same formats, standards, 
    and definitions, and there are no contradictions within the dataset.

dataset_path: path of the csv/xlsx to evaluate.
c1_column_names: columns used from the dataset for the C1 metric.
c2_column_mapping: columns used from the dataset for the C2 metric.
c3_column_names: columns used from the dataset for the C3 metric.
c4_column_names: columns used from the dataset for the C4 metric.
c5_column_names: columns used from the dataset for the C5 metric.
c1_threshold: threshold for simulatrity score that is acceptable for C1 metric.
c2_threshold: threshold for consistency score that is acceptable for C2 metric.
c3_threshold: threshold for simulatrity score that is acceptable for C3 metric.
c1_stop_words: Words filtered for C1 metric simularity calculations, purpose is to remove common words and focus on more meaningful words in the text that can better represent the content and context.
c2_stop_words: Words filtered for C2 metric simularity calculations, purpose is to remove common words and focus on more meaningful words in the text that can better represent the content and context.
c4_format: date-time format that selected dataset columns are compared to in C4 metric.
ref_dataset_path: Reference dataset that selected dataset columns are compared to in C2 metric.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
"""
class Consistency:
    def __init__(self, dataset_path, c1_column_names, c2_column_mapping, c3_column_names, c4_column_names, c5_column_names, c1_threshold=0.91, c2_threshold=0.91, c3_threshold=0.91, c1_stop_words=["the", "and"], c2_stop_words=["activity"], c4_format='%Y-%m-%d %H:%M:%S', ref_dataset_path=None, return_type="score", logging_path=None):
        self.dataset_path = dataset_path  
        self.c1_column_names = c1_column_names 
        self.c2_column_mapping = c2_column_mapping
        self.c3_column_names = c3_column_names
        self.c4_column_names = c4_column_names
        self.c5_column_names = c5_column_names
        self.c1_threshold = c1_threshold
        self.c2_threshold = c2_threshold
        self.c3_threshold = c3_threshold
        self.c1_stop_words = c1_stop_words 
        self.c2_stop_words = c2_stop_words
        self.c4_format = c4_format
        self.ref_dataset_path = ref_dataset_path
        self.return_type = return_type 
        self.logging_path = logging_path

    """ Consistency Type 1 (C1): Determines the similarity between string values in specified columns.
    Process the dataset, normalize the text, and calculate the similarity scores for multiple columns.
    """    
    consistency_score_list=[]
    
    def _c1_metric(self, metric):
        # Read the dataset from the provided Excel file path
        df = utils.read_data(self.dataset_path)
        overall_consistency_scores = []
        consistency_score_list =[]

        # Iterate over each specified column
        for column_name in self.c1_column_names:
            # Normalize the text in the specified column and store the results in a new column
            df[f"Normalized {column_name}"] = df[column_name].apply(utils.normalize_text)

            # Get unique normalized observations by removing duplicates and NaN values
            unique_observations = pd.unique(df[f"Normalized {column_name}"].dropna().values.ravel())

            # Calculate the cosine similarity matrix for the unique normalized observations
            text_sim_matrix = utils.calculate_cosine_similarity(
                unique_observations.tolist(), unique_observations.tolist(), self.c1_stop_words)

            # Set the diagonal of the similarity matrix to 0 to ignore self-similarity
            np.fill_diagonal(text_sim_matrix, 0)

            # Combine text similarity with numeric similarity to get a final similarity matrix
            combined_sim_matrix = utils.calculate_combined_similarity(unique_observations, text_sim_matrix)
            
            # Output the results of combined_sim_matrix into a dataframe with column names, and the next most similar column names
            max_values_df = utils.get_max_similarity_values(combined_sim_matrix, unique_observations, column_name)
            overall_consistency_scores.append(max_values_df)

            # Initialize columns in the dataframe to store the recommended organization matches and all matches
            df[f"Recommended {column_name}"] = None
            df[f"All Matches {column_name}"] = None

            # Iterate over each normalized organization in the dataframe
            for i, norm_org in enumerate(df[f"Normalized {column_name}"]):
                # Find the index of the current normalized organization in the unique observations
                try:
                    current_index = np.where(unique_observations == norm_org)[0][0]
                except IndexError:
                    df.at[i, f"Recommended {column_name}"] = "No significant match"
                    df.at[i, f"All Matches {column_name}"] = []
                    continue

                # Get the similarities for the current organization from the combined similarity matrix
                similarities = combined_sim_matrix[current_index]

                # Find the indices and values of all matching organizations
                matched_indices = np.where(similarities >= self.c1_threshold)[0]
                all_matches = [unique_observations[idx] for idx in matched_indices]
                all_match_scores = [similarities[idx] for idx in matched_indices]

                best_score = 0
                best_match = "No significant match"

                # Extract numbers from the current organization
                num_list_current = utils.extract_numbers(norm_org)

                for idx in matched_indices:
                    candidate_match = unique_observations[idx]
                    num_list_candidate = utils.extract_numbers(candidate_match)

                    if utils.contains_short_number(num_list_current) or utils.contains_short_number(
                        num_list_candidate
                    ):
                        # If short numbers are present, ensure they match; otherwise, skip this match
                        if not utils.numbers_match(num_list_current, num_list_candidate):
                            continue
                        # Recalculate similarity excluding short numbers
                        norm_org_no_nums = utils.remove_short_numbers(norm_org)
                        candidate_no_nums = utils.remove_short_numbers(candidate_match)
                        recalculated_similarity = utils.string_similarity(
                            norm_org_no_nums, candidate_no_nums
                        )
                        if recalculated_similarity > best_score:
                            best_score = recalculated_similarity
                            best_match = candidate_match
                    else:
                        if similarities[idx] > best_score:
                            best_score = similarities[idx]
                            best_match = candidate_match

                # Assign the best match to the dataframe
                if best_score > self.c1_threshold:
                    df.at[i, f"Recommended {column_name}"] = (
                        f"{best_match} ({best_score:.2f})"
                    )
                else:
                    df.at[i, f"Recommended {column_name}"] = "No significant match"

                # Store all matches
                df.at[i, f"All Matches {column_name}"] = ", ".join(
                    [
                        f"{match} ({score:.2f})"
                        for match, score in zip(all_matches, all_match_scores)
                        if score > self.c1_threshold
                    ]
                )

            # Calculate the overall consistency score for the current column
            consistency_score = utils.average_c1_consistency_score(text_sim_matrix, self.c1_threshold)
            consistency_score_list.append(consistency_score)

        # Calculate the overall consistency score as the average of individual consistency scores
        overall_consistency_score = np.mean(consistency_score_list)
        df['Overall Consistency Score'] = overall_consistency_score
        
        # add conditional return logic
        if self.return_type == "score":
            return overall_consistency_score, None
        elif self.return_type == "dataset":
            if not overall_consistency_scores:
                return "No valid c1 results generated", None
            
            final_df = pd.concat(overall_consistency_scores, ignore_index=True)  # Merge all results
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return overall_consistency_score, output_file  # Return the file name, add return for score
        else:
            return df, None  # Default return value (DataFrame)    

    """ Consistency Type 2 (C2): Compares reference data and string values in specified columns.
    The compared columns in question must be identical to the ref list, otherwise they will be penalized more harshly.
    """
    def _c2_metric(self, metric):
        # Read the data file
        df = utils.read_data(self.dataset_path)

        # Initialize ref_df if a ref dataset is provided
        if self.ref_dataset_path:
            df_ref = utils.read_data(self.ref_dataset_path)
            ref_data = True  # Flag to indicate we are using a ref dataset
        else:
            ref_data = False  # No ref dataset, compare within the same dataset

        all_consistency_scores = []

        for selected_column, m_selected_column in self.c2_column_mapping.items():
            if ref_data:
                # Compare to ref dataset
                unique_observations = utils.get_names_used_for_column(df_ref, m_selected_column)
            else:
                # Use own column for comparison
                unique_observations = utils.get_names_used_for_column(df, selected_column)

            cosine_sim_matrix = utils.calculate_cosine_similarity(
                df[selected_column].dropna(), unique_observations, stop_words=self.c2_stop_words
            )
            column_consistency_score = utils.average_c2_consistency_score(
                cosine_sim_matrix, self.c2_threshold
            )
            all_consistency_scores.append(column_consistency_score)

        # Calculate the average of all consistency scores
        overall_avg_consistency = (
            sum(all_consistency_scores) / len(all_consistency_scores)
            if all_consistency_scores
            else None
        )
        
        # add conditional return logic
        if self.return_type == "score":
            return overall_avg_consistency, None
        elif self.return_type == "dataset":
            if not overall_avg_consistency :
                return "No valid c2 results generated", None
            
            final_df = utils.compare_datasets(df, selected_column, unique_observations)  
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return overall_avg_consistency, output_file  # Return the file name
        else:
            return df, None  # Default return value (DataFrame)

    """ Consistency Type 3 (C3): Compares province/territory names (reference data) and string values in specified columns using Levenshtein Similarity Ratio.
    Levenshtein Similarity Ratio = 1 - (normalized Levenshtein Distance), where a score of 1 means the strings are identical.
    """
    def _c3_metric(self, metric):
        df = utils.read_data(self.dataset_path) 
        all_consistency_scores = []
        compare_df = pd.DataFrame()
    
        # Initialize reference data (lowercased province/territory names)
        arr_ref_normalized = np.array([name.lower() for name in utils.province_abbreviations.values()])
    
        for column in self.c3_column_names:
    
            # Normalize entries 
            df[f"Normalized {column}"] = df[column].apply(utils.normalize_text)
    
            # Calculate Levenshtein Similarity Ratio matrix and average consistency score based on matrix and threshold
            levenshtein_sim_matrix = utils.calculate_levenshtein_similarity(df[f"Normalized {column}"].dropna(), arr_ref_normalized)    
            column_consistency_score = utils.average_c3_consistency_score(levenshtein_sim_matrix, self.c3_threshold)
            all_consistency_scores.append(column_consistency_score)
    
            # Compare to reference data and add comparison column to dataset
            compare_df = utils.compare_datasets(df, f"Normalized {column}", arr_ref_normalized)

        # Drop normalized columns for output report
        columns_to_drop = [f"Normalized {col}" for col in compare_df.columns]
        compare_df = compare_df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Take subset of data inconsistent with reference data 
        comparison_columns = [f"Normalized {col}_comparison" for col in self.c3_column_names]
        inconsistent = ~compare_df[comparison_columns].all(axis=1)
        inconsistent_df = compare_df[inconsistent].copy() 
            
        # Compute average score
        avg_score = (
            sum(all_consistency_scores) / len(all_consistency_scores)
            if all_consistency_scores
            else None
        )
        
        # add conditional return logic
        if self.return_type == "score":
            return avg_score, None
        elif self.return_type == "dataset":
            if not avg_score: 
                return "No valid C3 results generated", None
                    
            final_df = inconsistent_df
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return avg_score, output_file  # Return the file name
                    
        else:
            return df, None  # Default return value (DataFrame) 
        
    """ Consistency Type 4 (C4): Checks whether the dataset follows standard date-time ISO 8601 formatting (or any format defined by the user).
    """
    def _c4_metric(self, metric):
        df = utils.read_data(self.dataset_path)
        results = df.copy()
        all_consistency_scores = {}

        # Check date-time formating on the whole column
        for column in self.c4_column_names:
            # Remove NA values
            df_clean = df.dropna(subset=[column])
    
            # Calculate proportion of incorrectly formated values in each column
            results[f"{column}_inconsistent"] = df_clean[column].apply(lambda x: utils.inconsistent_datetime(str(x), self.c4_format))
            all_consistency_scores[column] = 1 - results[f"{column}_inconsistent"].mean()
    
        # Take subset of data with inconsistent date-time formatting
        comparison_columns = [f"{col}_inconsistent" for col in self.c4_column_names]
        inconsistent = results[comparison_columns].any(axis=1)
        inconsistent_df = results[inconsistent].copy() 
        
        # Compute average score  
        overall_consistency_score = sum(all_consistency_scores.values()) / len(all_consistency_scores)
    
        # add conditional return logic
        if self.return_type == "score":
            return overall_consistency_score, None
        elif self.return_type == "dataset":
            if not overall_consistency_score: 
                return "No valid C4 results generated", None
                
            final_df = inconsistent_df
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return overall_consistency_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame) 

    """ Consistency Type 5 (C5): Checks whether the dataset follows Decimal Degrees (DD) formatting and has valid latitude & longitude coordinates.
    """
    def _c5_metric(self, metric):
        df = utils.read_data(self.dataset_path)
        results = df.copy()
        all_consistency_scores = {}

        # Compile regex patterns to detect latitude and longitude column names
        lat_pattern = re.compile(r'(lat|latitude)', flags=re.IGNORECASE)
        long_pattern = re.compile(r'(long|longitude)', flags=re.IGNORECASE)
    
        for column in self.c5_column_names:
            # Remove NA values
            df_clean = df.dropna(subset=[column])
            # Normalize column names by converting to lowercase and stripping whitespaces
            lower = column.lower().strip()
    
            # Check validity of coordinates depending on if latitude or longitude (flags those out of bounds)
            if lat_pattern.search(column):
                results[f"{column}_invalid"] = df_clean[column].apply(lambda x: False if -90 <= x <= 90 else True)
                all_consistency_scores[column] = 1 - results[f"{column}_invalid"].mean()
    
            elif long_pattern.search(column):
                results[f"{column}_invalid"] = df_clean[column].apply(lambda x: False if -180 <= x <= 180 else True)
                all_consistency_scores[column] = 1 - results[f"{column}_invalid"].mean()

        # Take subset of data with invalid coordinates 
        comparison_columns = [f"{col}_invalid" for col in self.c5_column_names]
        invalid = results[comparison_columns].any(axis=1)
        invalid_df = results[invalid].copy()
    
        # Compute average score
        overall_consistency_score = sum(all_consistency_scores.values()) / len(all_consistency_scores)
    
        # add conditional return logic
        if self.return_type == "score":
            return overall_consistency_score, None
        elif self.return_type == "dataset":
            if not overall_consistency_score: 
                return "No valid c5 results generated", None
                
            final_df = invalid_df
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return overall_consistency_score, output_file  # Return the file name
                
        else:
            return df, None  # Default return value (DataFrame)
            
    """ Run metrics: Will run specified metrics or all consistency metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"C1": self.c1_threshold, "C2": self.c2_threshold, "C3": self.c3_threshold, "C4": None, "C5": None}
            columns = {"C1": self.c1_column_names, "C2": self.c2_column_mapping, "C3": self.c3_column_names, "C4": self.c4_column_names, "C5": self.c5_column_names}

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                metric_log_csv = None # Ensure it exists even if errors occur
                overall_consistency_score = {"metric": None, "value": None}  # Ensure it exists even if errors occur

                try:
                    if metric == 'C1':
                        overall_consistency_score["metric"] = metric
                        consistency_score, metric_log_csv = self._c1_metric(metric.lower())
                        overall_consistency_score["value"] = consistency_score
                    elif metric == 'C2':
                        overall_consistency_score["metric"] = metric
                        consistency_score, metric_log_csv = self._c2_metric(metric.lower())
                        overall_consistency_score["value"] = consistency_score
                    elif metric == 'C3':
                        overall_consistency_score["metric"] = metric
                        consistency_score, metric_log_csv = self._c3_metric(metric.lower())
                        overall_consistency_score["value"] = consistency_score
                    elif metric == 'C4':
                        overall_consistency_score["metric"] = metric
                        consistency_score, metric_log_csv = self._c4_metric(metric.lower())
                        overall_consistency_score["value"] = consistency_score 
                    elif metric == 'C5':
                        overall_consistency_score["metric"] = metric
                        consistency_score, metric_log_csv = self._c5_metric(metric.lower())
                        overall_consistency_score["value"] = consistency_score

                except MemoryError as e:
                    print(f'{utils.RED}Dataset is too large for this test, out of memory!{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e) + '. Dataset is too large for this test.'
                except KeyError as e:
                    print(f'{utils.RED}Issue with column names, are you sure you entered them correctly?{utils.RESET}')
                    print(f'Column name that fails: {e}')
                    print(f'List of all detected column names: {list(utils.read_data(self.dataset_path).columns)}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e) + ' column not found in dataset.'
                except FileNotFoundError as e:
                    print(f'{utils.RED}Did not find dataset, make sure you have provided the correct name.{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)
                except Exception as e:
                    print(f'{utils.RED} {type(e).__name__} error has occured!{utils.RESET}')
                    errors = type(e).__name__  
                    test_fail_comment = str(e)
                
                outputs.append(overall_consistency_score)

                # output report of results
                utils.output_log_score(
                    test_name = metric, 
                    dataset_name = utils.get_dataset_name(self.dataset_path), 
                    score = overall_consistency_score, 
                    selected_columns = columns[metric],
                    excluded_columns=[''], 
                    isStandardTest = True, 
                    test_fail_comment = test_fail_comment, 
                    errors = errors, 
                    dimension = "Consistency", 
                    threshold= thresholds[metric],
                    metric_log_csv = metric_log_csv)
            return outputs
        else:
            print(f'{utils.RED}Non valid entry for metrics.{utils.RESET}')
            print(f'Metric options: {ALL_METRICS}, inputted metrics: {metrics}')
            return -1
        