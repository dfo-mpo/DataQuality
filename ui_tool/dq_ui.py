import streamlit as st
# Overwrite root path set by streamlit so files from sibling folders (dimensions) can be accessed
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd  
from functools import reduce
import ast
from ui_tool.metadata import ParameterType
import ui_tool.components as components

# Import dimensions
from dimensions.accessibility.dimension_reference import Accessibility 
from dimensions.accuracy.dimension_reference import Accuracy
from dimensions.completeness.dimension_reference import Completeness 
from dimensions.consistency.dimension_reference import Consistency 
from dimensions.interdependency.dimension_reference import Interdependency 
from dimensions.relevance.dimension_reference import Relevance 
from dimensions.timeliness.dimension_reference import Timeliness 
from dimensions.uniqueness.dimension_reference import Uniqueness 
from utils.core_operations import calculate_dimension_score, calculate_DQ_grade, read_data, are_weights_valid

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
    "Accessibility": { "all_tests": Accessibility.ALL_TESTS, 'metadata': Accessibility.collect_metadata(), "parameters": {}, "instantiate": Accessibility},
    "Accuracy": { "all_tests": Accuracy.ALL_TESTS, 'metadata': Accuracy.collect_metadata(), "parameters": {}, "instantiate": Accuracy},
    "Completeness": { "all_tests": Completeness.ALL_TESTS, 'metadata': Completeness.collect_metadata(), "parameters": {}, "instantiate": Completeness},
    "Consistency": { "all_tests": Consistency.ALL_TESTS, 'metadata': Consistency.collect_metadata(), "parameters": {}, "instantiate": Consistency},
    "Interdependency": { "all_tests": Interdependency.ALL_TESTS, 'metadata': Interdependency.collect_metadata(), "parameters": {}, "instantiate": Interdependency},
    "Relevance": { "all_tests": Relevance.ALL_TESTS, 'metadata': Relevance.collect_metadata(), "parameters": {}, "instantiate": Relevance},
    "Timeliness": { "all_tests": Timeliness.ALL_TESTS, 'metadata': Timeliness.collect_metadata(), "parameters": {}, "instantiate": Timeliness},
    "Uniqueness": { "all_tests": Uniqueness.ALL_TESTS, 'metadata': Uniqueness.collect_metadata(), "parameters": {}, "instantiate": Uniqueness},
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
      Documentation for dimensions and the test tests within can be found in the [README file](https://github.com/dfo-mpo/DataQuality/blob/main/README.md#dimensions-and-test-tests).  
    """)
    selected_dimensions = st.multiselect("Choose dimensions to use", dimensions.keys())


    # CHange tests to only show params when selected, remove hint for test, and check that a test is selected for check dimension before allowing a data quality run.
    # Iterate through each dimension
    for dimension in dimensions:
        if dimension in selected_dimensions:
            with st.expander(f"{dimension} Dimension", expanded=True):  
                # Create intial row that each dimension needs, adds return_type, tests, and weights to each dimension
                components.generateFirstDimensionRow(dimension_dict=dimensions[dimension])

                # Iterate through each test that has metadata, parameters will be grouped together for each test
                for test in dimensions[dimension]["metadata"]: # test is of type TestMetadata from metadata.py
                    if test.name in dimensions[dimension]["tests"]:
                        try:
                            components.generateDimensionRow(dimension_dict=dimensions[dimension], test=test.name, parameters=test.parameters, df_columns=df.columns.tolist())
                        except Exception as e:
                            st.error(f"Error encountered when generating fields for the {test.name} test!")
                            st.error(e)
                    
    # Run Tests button
    dim_weights = components.generateDimensionWeights(selected_dimensions)
    if st.button("Calculate Data Quality"):  
        # st.write(f"Running the following dimensions: {selected_dimensions}")
        output_logs = []

        # Run each selected dimension by creating class instance and running selected tests to get dimension scores
        for dimension in selected_dimensions:
            dimension_dict = dimensions[dimension]
            
            # Add checks to ensure parameters are entered correctly or apply defaults
            for test in dimension_dict["metadata"]: # test is of type TestMetadata from metadata.py
                # Only check tests that have been selected for the given dimension
                if dimensions[dimension]["tests"] == [] or test.name in dimensions[dimension]["tests"]:
                    for parameter in test.parameters: # parameter is of type ParameterMetadata from metadata.py
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

                        # Due to format supported in dimension classes, data read must take place before passing to run tests
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

                        # Need to convert stringified tuples into objects within the list
                        elif parameter.type == ParameterType.PAIRS:
                            pairs = dimension_dict['parameters'][parameter.name]
                            dimension_dict['parameters'][parameter.name] = [ast.literal_eval(pair) for pair in pairs]
                
            # Create parameter dictionary for each test (grouped by test name)
            test_params = {}
            for test_meta in dimension_dict["metadata"]:
                test_name = test_meta.name
                test_params[test_name] = {}
                
                for param in test_meta.parameters:
                    param_name = param.name
                    test_params[test_name][param_name] = dimension_dict["parameters"].get(param_name)
                    
            # Instanciate class instance using generated parameter fields        
            dimension_tests = dimension_dict["instantiate"](dataset_path=df, return_type='dataset', uploaded_file_name=uploaded_file.name, test_params=test_params)

            # Run all of the tests if not specified dimensions[dimension]["tests"]
            scores, logs = dimension_tests.run_tests(return_logs=True) if dimension_dict["tests"] == [] else dimension_tests.run_tests(dimension_dict["tests"], return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(dimension_dict["weights"], scores)
            if not valid:
                st.error(f'Weights entered for {dimension} are not valid, using defualt weights intead.')
            dimension_score = calculate_dimension_score(dimension, scores=scores, weights=weights)
            DIMENSION_SCORES.append(dimension_score)
        
        # Calculate final grade using dimension outputs
        # First check if weights are valid, if not use default weights
        weights, valid = are_weights_valid(dim_weights, scores)
        if not valid:
            st.error('Dimension weights entered are not valid, using default weights instead.')
        final_grade = calculate_DQ_grade(DIMENSION_SCORES, weights=dim_weights)
    
    if final_grade != None:
        st.markdown(f"### Calculated Data Quality: {final_grade}") 
        st.write("See output logs below for results from each test.")

        with st.expander("Output Logs"):
            merged_df = pd.concat(output_logs, ignore_index=True)
            st.dataframe(merged_df)

else:  
    # Disabled Run Tests button  
    st.button("Calculate Data Quality", disabled=True)  