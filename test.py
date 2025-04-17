import pandas as pd  
import numpy as np  
  
def compare_datasets(df, selected_column, unique_observations):  
    # Iterate over each row in the selected column  
    column_results = []  
    for value in df[selected_column]:  
        # Check if the value is in unique_observations or is NaN  
        if pd.isnull(value):  
            column_results.append(True)  
        else:  
            column_results.append(value in unique_observations)  
      
    # Add the results as a new column in the DataFrame  
    df[selected_column + '_comparison'] = column_results  
      
    return df  
  
# Example usage  
data = {  
    'ColumnA': ['apple', 'banana', 'cherry', None, 'apple'],  
    'ColumnB': [1, 2, 3, 4, 5]  
}  
unique_observations = ['apple', 'cherry']  
  
df = pd.DataFrame(data)  
result_df = compare_datasets(df, 'ColumnA', unique_observations)  
  
# Save the result to a CSV file  
result_df.to_csv('output.csv', index=False)  
  
# Print the resulting DataFrame  
print(result_df)  