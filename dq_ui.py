import streamlit as st
import pandas as pd  
from functools import reduce

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
st.title("Data Quality Grade Calculator")
DIMENSION_SCORES = [] # Stores the final score for each dimension used to calculate final grade

# Instructions for the dataset upload section  
st.markdown("""  
### Some requirements for the datasets:  
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
    st.markdown("""  
    ### Select Dimensions to Include  
      Documentation for dimensions and the metric tests within can be found in the [README file](https://github.com/dfo-mpo/DataQuality/blob/main/README.md#dimensions-and-metric-tests).  
    """)
    selected_dimensions = st.multiselect("Choose dimensions to use", ["Accessibility", "Accuracy", "Completeness", "Consistency", "Interdependency", "Relevance", "Timeliness", "Uniqueness"])

    # Accessibility configuration block
    if "Accessibility" in selected_dimensions:
        with st.expander("Accessiblity Dimension", expanded=True):  
            # Create columns for the input fields                  
            col_s1, col_s2, col_s3 = st.columns(3)  
            with col_s1:  
                s_return_type = st.selectbox("Return Type", ["score", "dataset"], key="Accessibility")
            with col_s2:  
                s_metrics = st.multiselect("Metrics", accessibility.ALL_METRICS, help="Runs all metrics by default.")
            with col_s3:  
                s_weights = st.text_input("Weights", value="", 
                                          placeholder="e.g., {'S1': 0.3, 'S2': 0.7}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
    
    # Accuracy configuration block
    if "Accuracy" in selected_dimensions:
        with st.expander("Accuracy Dimension", expanded=True):  
            # Create columns for the input fields                  
            col_a1, col_a2,  = st.columns(2)  
            with col_a1:  
                a_selected_columns = st.multiselect("Selected Columns", df.columns.tolist())
            with col_a2: 
                a_groupby_columns = st.multiselect("Groupby Column(s)", df.columns.tolist(), help="TODO: Add comment on purpose. Used in A2 metric.") 
            
            # Row 2
            col_a3, col_a4, col_a5 = st.columns(3)
            with col_a3:  
                a_return_type = st.selectbox("Return Type", ["score", "dataset"], key="Accuracy")
            with col_a4:  
                a2_threshold = st.number_input("A2 Threshold", value=1.5, step=0.1)
            with col_a5:
                a2_minimum_score = st.number_input("A2 Minimum Score", value=0.85, step=0.05)

            # Row 3
            col_a6, col_a7 = st.columns(2)    
            with col_a6:  
                a_metrics = st.multiselect("Metrics", accuracy.ALL_METRICS, help="Runs all metrics by default.")
            with col_a7:  
                a_weights = st.text_input("Weights", value="", 
                                          placeholder="e.g., {'A1': 0.3, 'A2': 0.7}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
    
    # Completeness configuration block
    if "Completeness" in selected_dimensions:
        with st.expander("Completeness Dimension", expanded=True):  
            # Create columns for the input fields                  
            col_p1, col_p2 = st.columns(2)  
            with col_p1: 
                p_exclude_columns = st.multiselect("Exclude Columns", df.columns.tolist(), default=[])
            with col_p2:  
                p1_threshold = st.number_input("P1 Threshold", value=0.75, step=0.05) 
            # Row 2
            col_p3, col_p4, col_p5 = st.columns(3)  
            with col_p3:  
                p_return_type = st.selectbox("Return Type", ["score", "dataset"],  key="Completeness")
            with col_p4:  
                p_metrics = st.multiselect("Metrics", completeness.ALL_METRICS, help="Runs all metrics by default.")
            with col_p5:  
                p_weights = st.text_input("Weights", value="", 
                                          placeholder="e.g., {'P1': 0.3, 'P2': 0.7}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
    
    # Consistency configuration block
    if "Consistency" in selected_dimensions:
        with st.expander("Consistency Dimension", expanded=True):  
            # Create columns for the input fields 
            # Rows 1 and 2                 
            col_c1, col_c2 = st.columns(2)  
            with col_c1:  
                c1_column_names = st.multiselect("C1 Column Names", df.columns.tolist())
                c_metrics = st.multiselect("Metrics", consistency.ALL_METRICS, help="Runs all metrics by default.")
            with col_c2:  
                c_ref_dataset_path = st.file_uploader("Reference Dataset File", type=["csv", "xlsx"])
            
            # Row 3 
            c2_column_mapping = st.text_input("C2 Column Mapping", value='',
                                              placeholder="e.g., {'Column1': 'Reference1', 'Column2': 'Reference2',}")             
            
            # Row 4
            col_c5, col_c6, col_c7 = st.columns(3)  
            with col_c5:  
                c1_threshold = st.number_input("C1 Threshold", value=0.91, step=0.01)
            with col_c6:  
                c2_threshold = st.number_input("C2 Threshold", value=0.91, step=0.01)
            with col_c7:  
                c_return_type = st.selectbox("Return Type", ["score", "dataset"], key="Consistency")
            
            # Row 5
            col_c8, col_c9, col_c10 = st.columns(3)
            with col_c8:  
                c1_stop_words = st.text_input("C1 Stop Words", value='["the", "and"]', help="Words filtered for C1 metric simularity calculations")
            with col_c9:  
                c2_stop_words = st.text_input("C2 Stop Words", value='["activity"]', help="Words filtered for C2 metric simularity calculations")
            with col_c10:
                c_weights = st.text_input("Weights", value="", 
                                          placeholder="e.g., {'C1': 0.3, 'C2': 0.7}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
            
    # Interdependency configuration block
    if "Interdependency" in selected_dimensions:
        with st.expander("Interdependency Dimension", expanded=True):  
            # Create columns for the input fields                  
            col_i1, col_i2, col_i3 = st.columns(3)  
            with col_i1:  
                i_return_type = st.selectbox("Return Type", ["score", "dataset"], key="Interdependency")
            with col_i2:  
                i_metrics = st.multiselect("Metrics", interdependency.ALL_METRICS, help="Runs all metrics by default.")
            with col_i3:  
                i_weights = st.text_input("Weights", value="", 
                                          placeholder="e.g., {'I1': 0.3, 'I2': 0.7}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
    
    # Relevance configuration block
    if "Relevance" in selected_dimensions:
        with st.expander("Relevance Dimension", expanded=True):  
            # Create columns for the input fields                  
            col_r1, col_r2, col_r3 = st.columns(3)  
            with col_r1:  
                r_return_type = st.selectbox("Return Type", ["score", "dataset"], key="Relevance")
            with col_r2:  
                r_metrics = st.multiselect("Metrics", relevance.ALL_METRICS, help="Runs all metrics by default.")
            with col_r3:  
                r_weights = st.text_input("Weights", value="", 
                                          placeholder="e.g., {'R1': 0.3, 'R2': 0.7}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
    
    # Timeliness configuration block
    if "Timeliness" in selected_dimensions:
        with st.expander("Timeliness Dimension", expanded=True):  
            # Create columns for the input fields                  
            col_t1, col_t2, col_t3 = st.columns(3)  
            with col_t1:  
                t_return_type = st.selectbox("Return Type", ["score", "dataset"],  key="Timeliness")
            with col_t2:  
                t_metrics = st.multiselect("Metrics", timeliness.ALL_METRICS, help="Runs all metrics by default.")
            with col_t3:  
                t_weights = st.text_input("Weights", value="", 
                                          placeholder="e.g., {'T1': 0.3, 'T2': 0.7}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
    
    # Uniqueness configuration block
    if "Uniqueness" in selected_dimensions:
        with st.expander("Uniqueness Dimension", expanded=True):  
            # Create columns for the input fields                  
            col_u1, col_u2, col_u3 = st.columns(3)  
            with col_u1:  
                u_return_type = st.selectbox("Return Type", ["score", "dataset"], key="Uniqueness")
            with col_u2:  
                u_metrics = st.multiselect("Metrics", uniqueness.ALL_METRICS, help="Runs all metrics by default.")
            with col_u3:  
                u_weights = st.text_input("Weights", value="", 
                                          placeholder="e.g., {'U1': 0.3, 'U2': 0.7}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
      
    # Run Tests button
    dim_weights = st.text_input("Dimension Weights", value="", 
                                          placeholder="e.g., {'Accessibility': 0.3, 'Consistency': 0.4, 'Uniqueness': 0.3}", 
                                          help="If left empty, weighting will be equal. Weights must add up to 1.")
    if st.button("Calculate Grade"):  
        # st.write(f"Running the following dimensions: {selected_dimensions}")
        output_logs = []

        # Run each selected dimension by creating class instance and running selected metrics to get dimension scores
        if "Accessibility" in selected_dimensions:
            accessibility_tests = accessibility.Accessibility(dataset_path=df, return_type=s_return_type, uploaded_file_name=uploaded_file.name)

            # Run all of the metrics if not specified
            scores, logs = accessibility_tests.run_metrics(return_logs=True) if s_metrics == [] else accessibility_tests.run_metrics(s_metrics, return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(s_weights, scores)
            if not valid:
                st.error('Weights entered for Accessibility are not valid, using defualt weights intead.')

            accessibility_score = calculate_dimension_score("Accessibility", scores=scores, weights=weights)
            DIMENSION_SCORES.append(accessibility_score)

        if "Accuracy" in selected_dimensions:
            accuracy_tests = accuracy.Accuracy(dataset_path=df, selected_columns=a_selected_columns, a2_threshold=a2_threshold, a2_minimum_score=a2_minimum_score,
                return_type=a_return_type, groupby_columns=a_groupby_columns, uploaded_file_name=uploaded_file.name
            )

            # Run all of the metrics if not specified
            scores, logs = accuracy_tests.run_metrics(return_logs=True) if a_metrics == [] else accuracy_tests.run_metrics(a_metrics, return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(a_weights, scores)
            if not valid:
                st.error('Weights entered for Accuracy are not valid, using defualt weights intead.')

            accuracy_score = calculate_dimension_score("Accuracy", scores=scores, weights=weights)
            DIMENSION_SCORES.append(accuracy_score)

        if "Completeness" in selected_dimensions:
            completeness_tests = completeness.Completeness(dataset_path=df, exclude_columns=p_exclude_columns, p1_threshold=p1_threshold, return_type=p_return_type, uploaded_file_name=uploaded_file.name)

            # Run all of the metrics if not specified
            scores, logs = completeness_tests.run_metrics(return_logs=True) if p_metrics == [] else completeness_tests.run_metrics(p_metrics, return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(p_weights, scores)
            if not valid:
                st.error('Weights entered for Completeness are not valid, using defualt weights intead.')

            completeness_score = calculate_dimension_score("Completeness", scores=scores, weights=weights)
            DIMENSION_SCORES.append(completeness_score)
        
        if "Consistency" in selected_dimensions:
            consitancy_tests = consistency.Consistency(dataset_path=df, c1_column_names=c1_column_names, c2_column_mapping=c2_column_mapping, 
                ref_dataset_path=read_data(c_ref_dataset_path, c_ref_dataset_path.name), c1_threshold=c1_threshold, 
                c2_threshold=c2_threshold, return_type=c_return_type, c1_stop_words=c1_stop_words, c2_stop_words=c2_stop_words, uploaded_file_name=uploaded_file.name
            )

            # Run all of the metrics if not specified
            scores, logs = consitancy_tests.run_metrics(return_logs=True) if c_metrics == [] else consitancy_tests.run_metrics(c_metrics, return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(c_weights, scores)
            if not valid:
                st.error('Weights entered for Consistency are not valid, using defualt weights intead.')

            consistancy_score = calculate_dimension_score("Consistency", scores=scores, weights=weights)
            DIMENSION_SCORES.append(consistancy_score)
        
        if "Interdependency" in selected_dimensions:
            interdependency_tests = interdependency.Interdependency(dataset_path=df, return_type=i_return_type, uploaded_file_name=uploaded_file.name)

            # Run all of the metrics if not specified
            scores, logs = interdependency_tests.run_metrics(return_logs=True) if i_metrics == [] else interdependency_tests.run_metrics(i_metrics, return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(i_weights, scores)
            if not valid:
                st.error('Weights entered for Interdependency are not valid, using defualt weights intead.')

            interdependency_score = calculate_dimension_score("Interdependency", scores=scores, weights=weights)
            DIMENSION_SCORES.append(interdependency_score)
        
        if "Relevance" in selected_dimensions:
            relevance_tests = relevance.Relevance(dataset_path=df, return_type=r_return_type, uploaded_file_name=uploaded_file.name)

            # Run all of the metrics if not specified
            scores, logs = relevance_tests.run_metrics(return_logs=True) if r_metrics == [] else relevance_tests.run_metrics(r_metrics, return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(r_weights, scores)
            if not valid:
                st.error('Weights entered for Relevance are not valid, using defualt weights intead.')

            relevance_score = calculate_dimension_score("Relevance", scores=scores, weights=weights)
            DIMENSION_SCORES.append(relevance_score)

        if "Timeliness" in selected_dimensions:
            timeliness_tests = timeliness.Timeliness(dataset_path=df, return_type=t_return_type, uploaded_file_name=uploaded_file.name)

            # Run all of the metrics if not specified
            scores, logs = timeliness_tests.run_metrics(return_logs=True) if t_metrics == [] else timeliness_tests.run_metrics(t_metrics, return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(t_weights, scores)
            if not valid:
                st.error('Weights entered for Timeliness are not valid, using defualt weights intead.')

            timeliness_score = calculate_dimension_score("Timeliness", scores=scores, weights=weights)
            DIMENSION_SCORES.append(timeliness_score)
        
        if "Uniqueness" in selected_dimensions:
            uniqueness_tests = uniqueness.Uniqueness(dataset_path=df, return_type=u_return_type, uploaded_file_name=uploaded_file.name)

            # Run all of the metrics if not specified
            scores, logs = uniqueness_tests.run_metrics(return_logs=True) if u_metrics == [] else uniqueness_tests.run_metrics(u_metrics, return_logs=True)
            output_logs.extend(logs)

            # Check if weights are valid, if not use default weights
            weights, valid = are_weights_valid(u_weights, scores)
            if not valid:
                st.error('Weights entered for Uniqueness are not valid, using defualt weights intead.')

            uniqueness_score = calculate_dimension_score("Uniqueness", scores=scores, weights=weights)
            DIMENSION_SCORES.append(uniqueness_score)
        
        # Calculate final grade using dimension outputs
        # First check if weights are valid, if not use default weights
        weights, valid = are_weights_valid(dim_weights, scores)
        if not valid:
            st.error('Dimension weights entered are not valid, using defualt weights intead.')
        final_grade = calculate_DQ_grade(DIMENSION_SCORES, weights=dim_weights)
    
    if final_grade != None:
        st.markdown(f"### Calculated Data Quality Grade is: {final_grade}") 
        st.write("See output logs below for results from each metric.")

        with st.expander("Output Logs"):
            merged_df = pd.concat(output_logs, ignore_index=True)
            st.dataframe(merged_df)

else:  
    # Disabled Run Tests button  
    st.button("Calculate Grade", disabled=True)  