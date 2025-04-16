import numpy as np  
import pandas as pd 
from . import utils

ALL_METRICS = ['C1', 'C2']

""" Class to represent all metric tests for the Consistency dimension """
class Consistency:
    def __init__(self, dataset_path, c1_column_names, c2_column_mapping, c1_threshold=0.91, c2_threshold=0.91, c1_stop_words=["the", "and"], c2_stop_words="activity", ref_dataset_path=None, return_type="score", logging_path=None):
        self.dataset_path = dataset_path  
        self.c1_column_names = c1_column_names 
        self.c2_column_mapping = c2_column_mapping 
        self.c1_threshold = c1_threshold
        self.c2_threshold = c2_threshold
        self.c1_stop_words = c1_stop_words 
        self.c2_stop_words = c2_stop_words
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
            consistency_score = utils.average_consistency_score(text_sim_matrix, self.c1_threshold)
            consistency_score_list.append(consistency_score)

        # Calculate the overall consistency score as the average of individual consistency scores
        overall_consistency_score = np.mean(consistency_score_list)
        df['Overall Consistency Score'] = overall_consistency_score
        
        # add conditional return logic
        if self.return_type == "score":
            return overall_consistency_score, None
        elif self.return_type == "dataset":
            if not overall_consistency_scores:
                return "No valid c1 results generated"
            
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
                df[selected_column].dropna(), unique_observations, Stop_Words=self.c2_stop_words
            )
            column_consistency_score = utils.average_consistency_score(
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
                return "No valid c2 results generated"
            
            final_df = utils.compare_datasets(df, selected_column, unique_observations)  
            output_file = utils.df_to_csv(self.logging_path, metric=metric, final_df=final_df)
            return overall_avg_consistency, output_file  # Return the file name
        else:
            return df, None  # Default return value (DataFrame)
    
    """ Run metrics: Will run specified metrics or all consistency metrics by default
    """
    def run_metrics(self, metrics=ALL_METRICS):
        # Verify that inputed metrics is valid
        if set(metrics).issubset(set(ALL_METRICS)):
            # Run each metric and send outputs in combined list
            outputs = []
            thresholds = {"C1": self.c1_threshold, "C2": self.c2_threshold}
            columns = {"C1": self.c1_column_names, "C2": self.c2_column_mapping}

            for metric in metrics:
                # Variables that prepare for output reports
                errors = None
                test_fail_comment = None
                overall_consistency_score = None  # Ensure it exists even if errors occur

                try:
                    if metric == 'C1':
                        overall_consistency_score, metric_log_csv = self._c1_metric(metric.lower())
                    elif metric == 'C2':
                        overall_consistency_score, metric_log_csv = self._c2_metric(metric.lower())     

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
        