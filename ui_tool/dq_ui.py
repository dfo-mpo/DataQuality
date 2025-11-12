import streamlit as st
import pandas as pd  
from functools import reduce
import ast
from metadata import ParameterType
import components as components

# Import dimensions
import dimensions.accessibility as accessibility
import dimensions.consistency as consistency
import dimensions.accuracy as accuracy
import dimensions.completeness as completeness
import dimensions.interdependency as interdependency
import dimensions.relevance as relevance
import dimensions.timeliness as timeliness
import dimensions.uniqueness as uniqueness
from dimensions.utils import calculate_dimension_score, calculate_DQ_grade, read_data, are_weights_valid

# Title of the web app  
st.title("Data Quality Calculator", anchor=False)

# Set the page layout to wide mode
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:57rem;
        }
    </style>
    """
)

DIMENSION_SCORES = [] # Stores the final score for each dimension used to calculate final grade
dimensions = { # Nested dictionary to keep track of values for each given dimension durring runtime
    "Accessibility": { "all_metrics": accessibility.ALL_METRICS, 'metadata': accessibility.create_metadata(), "parameters": {}, "instantiate": accessibility.Accessibility},
    "Accuracy": { "all_metrics": accuracy.ALL_METRICS, 'metadata': accuracy.create_metadata(), "parameters": {}, "instantiate": accuracy.Accuracy },
    "Completeness": { "all_metrics": completeness.ALL_METRICS, 'metadata': completeness.create_metadata(), "parameters": {}, "instantiate": completeness.Completeness},
    "Consistency": { "all_metrics": consistency.ALL_METRICS, 'metadata': consistency.create_metadata(), "parameters": {}, "instantiate": consistency.Consistency},
    "Interdependency": { "all_metrics": interdependency.ALL_METRICS, 'metadata': interdependency.create_metadata(), "parameters": {}, "instantiate": interdependency.Interdependency},
    "Relevance": { "all_metrics": relevance.ALL_METRICS, 'metadata': relevance.create_metadata(), "parameters": {}, "instantiate": relevance.Relevance},
    "Timeliness": { "all_metrics": timeliness.ALL_METRICS, 'metadata': timeliness.create_metadata(), "parameters": {}, "instantiate": timeliness.Timeliness},
    "Uniqueness": { "all_metrics": uniqueness.ALL_METRICS, 'metadata': uniqueness.create_metadata(), "parameters": {}, "instantiate": uniqueness.Uniqueness},
}

# Instructions for the dataset upload section  
st.subheader("Some requirements for the datasets:", anchor=False)
st.markdown("""   
- The data must be on the **first sheet** in the Excel document.  
- The **first row** must be the column names.  
- The test won't run if the Excel file is open.  
""") 

# File upload  
uploaded_file = st.file_uploader("Choose your dataset file (CSV or XLSX)", type=["csv", "xlsx"])  
  
if uploaded_file is not None:  
    final_grade = None

    # Convert the uploaded file to a dataframe  
    df = read_data(uploaded_file, uploaded_file.name) 
      
    # Display the dataframe in a scrollable expander  
    with st.expander("View Uploaded Dataset"):  
        st.dataframe(df)  
    
    # Selection for dimensions to use for tests
    st.subheader("Select Dimensions to Include ", anchor=False)
    st.markdown("""  
      Documentation for dimensions and the metric tests within can be found in the [README file](https://github.com/dfo-mpo/DataQuality/blob/main/README.md#dimensions-and-metric-tests).  
    """)
    selected_dimensions = st.multiselect("Choose dimensions to use", dimensions.keys())


    # CHange metrics to only show params when selected, remove hint for metric, and check that a metric is selected for check dimension before allowing a data quality run.
    # Iterate through each dimension
    for dimension in dimensions:
        if dimension in selected_dimensions:
            with st.expander(f"{dimension} Dimension", expanded=True):  
                # Create intial row that each dimension needs, adds return_type, metrics, and weights to each dimension
                components.generateFirstDimensionRow(dimension_dict=dimensions[dimension])

                # Iterate through each metric that has metadata, parameters will be grouped together for each metric
                for metric in dimensions[dimension]["metadata"]: # metric is of type MetricMetadata from metadata.py
                    if metric.name in dimensions[dimension]["metrics"]:
                        try:
                            components.generateDimensionRow(dimension_dict=dimensions[dimension], parameters=metric.parameters, df_columns=df.columns.tolist())
                        except Exception as e:
                            st.error(f"Error encountered when generating fields for the {metric.name} metric!")
                            st.error(e)
                    
    # Run Tests button
    dim_weights = st.text_input("Dimension Weights", value="", 
                                          placeholder="e.g., {'Accessibility': 0.3, 'Consistency': 0.4, 'Uniqueness': 0.3}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
    if st.button("Calculate Data Quality"):  
        # st.write(f"Running the following dimensions: {selected_dimensions}")
        output_logs = []

        # Run each selected dimension by creating class instance and running selected metrics to get dimension scores
        for dimension in selected_dimensions:
            dimension_dict = dimensions[dimension]

            # Add checks to ensure parameters are entered correctly or apply defaults
            for metric in dimension_dict["metadata"]: # metric is of type MetricMetadata from metadata.py
                # Only check metrics that have been selected for the given dimension
                if dimensions[dimension]["metrics"] == [] or metric.name in dimensions[dimension]["metrics"]:
                    for parameter in metric.parameters: # parameter is of type ParameterMetadata from metadata.py
                        if parameter.type == ParameterType.TEXT_INPUT: # Appy transformation into object type if possible
                            try:
                                parameter_value = dimension_dict['parameters'][parameter.name].strip()
                                if parameter_value == "":
                                    dimension_dict['parameters'][parameter.name] = parameter.default
                                else:
                                    parameter_value = parameter_value.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"') # sanitize quates to prevent syntax errors
                                    parameter_value = ast.literal_eval(parameter_value)
                                    if len(parameter_value) < 1:
                                        dimension_dict['parameters'][parameter.name] = parameter.default
                                    else:
                                        dimension_dict['parameters'][parameter.name] = parameter_value
                            except:
                                st.error(f"Invalid {parameter.title} format provided, defaulting value to {parameter.default} for calculation.")
                                dimension_dict['parameters'][parameter.name] = parameter.default

                        # Due to format supported in dimension classes, data read must take place before passing to run metrics
                        elif parameter.type == ParameterType.FILE_UPLOAD: 
                            dataset_path = dimension_dict['parameters'][parameter.name]
                            try:
                                if dataset_path:
                                    dimension_dict['parameters'][parameter.name] = read_data(dataset_path, dataset_path.name)
                            except:
                                st.error(f"Error processing uploaded file for {parameter.title}, defaulting value to {parameter.default} for calculation.")
                                dimension_dict['parameters'][parameter.name] = parameter.default
                        
                        # If left blank update Multi select with default (which is None if not defined in the dimension class)
                        elif parameter.type == ParameterType.MULTI_SELECT:
                            if dimension_dict['parameters'][parameter.name] == []:
                                dimension_dict['parameters'][parameter.name] = parameter.default
                
            # Instanciate class instance using generated parameter fields
            dimension_tests = dimension_dict["instantiate"](dataset_path=df, return_type='dataset', uploaded_file_name=uploaded_file.name, **dimension_dict["parameters"])

            # Run all of the metrics if not specified dimensions[dimension]["metrics"]
            scores, logs = dimension_tests.run_metrics(return_logs=True) if dimension_dict["metrics"] == [] else dimension_tests.run_metrics(dimension_dict["metrics"], return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(dimension_dict["weights"], scores)
            if not valid:
                st.error(f'Weights entered for {dimension} are not valid, using defualt weights intead.')
            accessibility_score = calculate_dimension_score(dimension, scores=scores, weights=weights)
            DIMENSION_SCORES.append(accessibility_score)
        
        # Calculate final grade using dimension outputs
        # First check if weights are valid, if not use default weights
        weights, valid = are_weights_valid(dim_weights, scores)
        if not valid:
            st.error('Dimension weights entered are not valid, using default weights instead.')
        final_grade = calculate_DQ_grade(DIMENSION_SCORES, weights=dim_weights)
    
    if final_grade != None:
        st.markdown(f"### Calculated Data Quality: {final_grade}") 
        st.write("See output logs below for results from each metric.")

        with st.expander("Output Logs"):
            merged_df = pd.concat(output_logs, ignore_index=True)
            st.dataframe(merged_df)

else:  
    # Disabled Run Tests button  
    st.button("Calculate Data Quality", disabled=True)  