from metadata import ParameterMetadata, ParameterType
import streamlit as st

""" Generate first row containing parameters required for all dimensions, inputs are stored in the provided dictionary object. 
dimension: The name of the given dimension.
dimension_dict: The dictionary representing the given dimension.
"""
def generateFirstDimensionRow(dimension_dict):
    all_metrics = dimension_dict["all_metrics"]
    example_weights = f"'{all_metrics[0]}': 0.3, '{all_metrics[0][0]}': 0.7"
    col_1, col_2 = st.columns(2)

    with col_1:  
        dimension_dict["metrics"] = st.multiselect("Metrics", all_metrics, help="Runs all metrics by default.")
    with col_2:  
        dimension_dict["weights"] = st.text_input("Weights", value="", 
                                    placeholder="e.g., {"+example_weights+"}", 
                                    help="If left empty, weighting will be equal. Weights must add up to 1.")
        

def generateDimensionRow(dimension_dict, parameters: list[ParameterMetadata], df_columns):
    if len(parameters) == 0:
        return

    # Check if any of the first 3 existing parameters is type FileUpload
    containsFileUpload = False
    if (parameters[0].type == ParameterType.FILE_UPLOAD) or (len(parameters) > 1 and parameters[1].type == ParameterType.FILE_UPLOAD) or (len(parameters) > 2 and parameters[2].type == ParameterType.FILE_UPLOAD):
        containsFileUpload = True

    # Determine the number of columns that will be generated for the following row 
    numOfColumns = 3 # Default to 3

    # To ensure instead of have a row of 3 then 1, will do a row of 2 then 2
    if len(parameters) == 4:
        numOfColumns = 2
    elif len(parameters) <= 3:
        numOfColumns = len(parameters)

    # File upload does not properly fit in 3 column row
    if containsFileUpload and len(parameters) > 1:
        numOfColumns = 2

    # Loop though to generate column content one at a time
    cols = st.columns(numOfColumns)
    currentParameterIndex = 0
    doubleColumn = False
    for col in cols:
        parameter = parameters[currentParameterIndex]
        with col:
            # Generate parameter
            dimension_dict["parameters"][parameter.name] = generateParameterField(parameter, df_columns)

            # If there is an upload file in row but not current column. File uploads have a hight of 2 rows hence 2 parameters can fit in the column in this case.
            if containsFileUpload and parameters[currentParameterIndex].type != ParameterType.FILE_UPLOAD:
                # If next parameter is not a file Upload, generate it.
                if len(parameters) > currentParameterIndex + 1 and parameters[currentParameterIndex + 1].type != ParameterType.FILE_UPLOAD:
                    dimension_dict["parameters"][parameters[currentParameterIndex+1].name] = generateParameterField(parameters[currentParameterIndex+1], df_columns)
                    currentParameterIndex += 1
                    doubleColumn = True
                # Case if second parameter is a file upload while the first and third are not
                elif currentParameterIndex == 0 and len(parameters) > 2 and parameters[2].type != ParameterType.FILE_UPLOAD:
                    dimension_dict["parameters"][parameters[2].name] = generateParameterField(parameters[2], df_columns)
                    doubleColumn = True

        currentParameterIndex += 1
    
    # If more parameters are left to generate, do recursive call otherwise terminate
    numOfParamUsed = numOfColumns if not doubleColumn else numOfColumns + 1
    if len(parameters) > numOfParamUsed:
        generateDimensionRow(dimension_dict, parameters=parameters[numOfParamUsed:], df_columns=df_columns)
    else:
        return


def generateParameterField(parameter: ParameterMetadata, df_columns: list):
    match parameter.type:
        case ParameterType.MULTI_SELECT:
            options = parameter.value if parameter.value else df_columns
            return st.multiselect(parameter.title, options=options, default=[])
        case ParameterType.SINGLE_SELECT:
            options = parameter.value if parameter.value else df_columns
            return st.selectbox(parameter.title, options=options)
        case ParameterType.DECIMAL:
            return st.number_input(parameter.title, value=float(parameter.value), step=parameter.step) 
        case ParameterType.TEXT_INPUT | ParameterType.STRING: # Difference between the 2 is when sanitizing fields before running metrics
            return st.text_input(parameter.title, value=parameter.value, placeholder=parameter.placeholder, help=parameter.hint)
        case ParameterType.FILE_UPLOAD:
            return st.file_uploader(parameter.title, type=["csv", "xlsx"])
        # Fall through if invalid ParameterType found
        case _:
            st.error("None valid ParameterType found when generating fields from dimension class metadata.")