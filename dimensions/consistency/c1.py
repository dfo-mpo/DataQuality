import numpy as np  
import pandas as pd
from utils import core_operations, table_operations, column_operations, item_operations
from ui_tool.metadata import MetricMetadata, ParameterType

METRIC = "C1"

""" Class to represent an individual metric for the Consistency dimension.
    
    Goal: Ensure that data is consistent across different datasets and systems. 
    Consistent data follows the same formats, standards, and definitions, and there are no contradictions within the dataset.
    
dataset_path: path of the csv/xlsx to evaluate.
return_type: either score to return only metric scores, or dataset to also return a csv used to calculate the score (is used for one line summary in output logs).
logging_path: path to store csv of what test used to calculate score, if set to None (default) it is kept in memory only.
uploaded_file_name: stores the name of the file uploaded when using the UI tool.
c1_column_names: columns used from the dataset for the C1 metric.
c1_threshold: threshold for simulatrity score that is acceptable for C1 metric.
c1_stop_words: Words filtered for C1 metric simularity calculations, purpose is to remove common words and focus on more meaningful words in the text that can better represent the content and context.
"""
class Metric:
    def __init__(self, dataset_path, return_type="score", logging_path=None, uploaded_file_name=None, c1_column_names=[], c1_threshold=0.91, c1_stop_words=["the", "and"], threshold=None, selected_columns=None):
        self.dataset_path = dataset_path  
        self.return_type = return_type
        self.logging_path = logging_path
        self.uploaded_file_name = uploaded_file_name
        
        self.c1_column_names = c1_column_names 
        self.c1_threshold = c1_threshold
        self.c1_stop_words = c1_stop_words 

        self.threshold = self.c1_threshold
        self.selected_columns = self.c1_column_names
    
    """ Consistency Type 1 (C1): Determines the similarity between string values in specified columns.
    Process the dataset, normalize the text, and calculate the similarity scores for multiple columns.
    Limitations: It will not check for differences in capitalization of the same word (since all the words will be changed to lower case before the similarity score is calculated).
    """     
    def run_metric(self):    
        # Read the dataset from the provided Excel file path
        df = core_operations.read_data(self.dataset_path)
        overall_consistency_scores = []
        consistency_score_list =[]

        # Iterate over each specified column
        for column_name in self.c1_column_names:
            # Normalize the text in the specified column and store the results in a new column
            df[f"Normalized {column_name}"] = df[column_name].apply(item_operations.normalize_text)

            # Get unique normalized observations by removing duplicates and NaN values
            unique_observations = pd.unique(df[f"Normalized {column_name}"].dropna().values.ravel())

            # Calculate the cosine similarity matrix for the unique normalized observations
            text_sim_matrix = column_operations.calculate_cosine_similarity(
                unique_observations.tolist(), unique_observations.tolist(), self.c1_stop_words)

            # Set the diagonal of the similarity matrix to 0 to ignore self-similarity
            np.fill_diagonal(text_sim_matrix, 0)

            # Combine text similarity with numeric similarity to get a final similarity matrix
            combined_sim_matrix = table_operations.calculate_combined_similarity(unique_observations, text_sim_matrix)
            
            # Output the results of combined_sim_matrix into a dataframe with column names, and the next most similar column names
            max_values_df = table_operations.get_max_similarity_values(combined_sim_matrix, unique_observations, column_name)
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
                num_list_current = item_operations.extract_numbers(norm_org)

                for idx in matched_indices:
                    candidate_match = unique_observations[idx]
                    num_list_candidate = item_operations.extract_numbers(candidate_match)

                    if column_operations.contains_short_number(num_list_current) or column_operations.contains_short_number(
                        num_list_candidate
                    ):
                        # If short numbers are present, ensure they match; otherwise, skip this match
                        if not column_operations.numbers_match(num_list_current, num_list_candidate):
                            continue
                        # Recalculate similarity excluding short numbers
                        norm_org_no_nums = item_operations.remove_short_numbers(norm_org)
                        candidate_no_nums = item_operations.remove_short_numbers(candidate_match)
                        recalculated_similarity = item_operations.string_similarity(
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
            score = table_operations.average_c1_consistency_score(text_sim_matrix, self.c1_threshold)
            consistency_score_list.append(score)

        # Calculate the overall consistency score as the average of individual consistency scores
        consistency_score = np.mean(consistency_score_list)
        df['Overall Consistency Score'] = consistency_score
        
        # Conditional return logic
        if self.return_type == "score":
            return consistency_score, None
        elif self.return_type == "dataset":
            if not overall_consistency_scores:
                return f"No valid {METRIC} results generated", None
            
            cdf = pd.concat(overall_consistency_scores, ignore_index=True)  # Merge all results
            output_file = core_operations.df_to_csv(self.logging_path, metric=METRIC.lower(), final_df=cdf)
            return consistency_score, output_file  # Return the file name, add return for score
        else:
            return df, None  # Default return value (DataFrame)
       
""" Create metadata: Will create instances of metadata classes for each metric's parameters to allow the UI tool to generate input feilds.
Returns list of MetricMetadata objects or [] if there are no addtional input parameters required for this dimension
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for metric
    c1_metadata = MetricMetadata(dimension, METRIC)
    # Define each parameter needed for metric, use ParameterType when defining type
    c1_metadata.add_parameter('c1_column_names', 'C1 Column Names', ParameterType.MULTI_SELECT, default=[])
    c1_metadata.add_parameter('c1_threshold', 'C1 Threshold', ParameterType.DECIMAL, value='0.91', step = 0.01)
    c1_metadata.add_parameter('c1_stop_words', 'C1 Stop Words', ParameterType.STRING_LIST, value=["the", "and"], suggestions=["the", "and"], hint="Words filtered for C1 metric simularity calculations")
    
    return c1_metadata 