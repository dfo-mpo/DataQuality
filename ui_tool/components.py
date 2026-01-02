import sys  
from pathlib import Path  
from natsort import natsorted
  
# Set path for local custom components 
BASE_DIR = Path(__file__).resolve().parent  
COMPONENTS_DIR = BASE_DIR / "custom_components"  
if str(COMPONENTS_DIR) not in sys.path:  
    sys.path.insert(0, str(COMPONENTS_DIR))  

from ui_tool.metadata import ParameterMetadata, ParameterType
from streamlit_tags import st_tags # Community component
from streamlit_pairs import st_pairs # Custom component
from streamlit_weights import st_weights # Custom component
import streamlit as st

""" Generate first row containing parameters required for all dimensions, inputs are stored in the provided dictionary object. 
dimension: The name of the given dimension.
dimension_dict: The dictionary representing the given dimension.
"""
def generateFirstDimensionRow(dimension_dict):
    all_tests = dimension_dict["all_tests"]
    col_1, col_2 = st.columns(2)

    with col_1:  
        dimension_dict["tests"] = st.multiselect("Tests", natsorted(all_tests))
    with col_2:  
        weights = generateWeightsDict(dimension_dict["metrics"], dimension_dict.get("weights",{}))
        dimension_dict["weights"] = st_weights(key=f"{all_tests[0]}", placeholder="First select the metrics you wish to run.", label="Weights", value=weights, step=0.05, min=0, max=1.0 )

 
def generateWeightsDict(keys, oldDict):
    newWeights = {}
    for key in keys:
        if not key in oldDict:
            newWeights[key] = 0
        else:
            newWeights[key] = oldDict[key]
    
    return newWeights

def generateDimensionWeights(selected_dimensions):
    weights = generateWeightsDict(selected_dimensions, {})
    return st_weights(key=f"Dimensions", placeholder="First select the dimensions you wish to run.", label="Dimension Weights", value=weights, step=0.05, min=0, max=1.0 )

def generateDimensionRow(dimension_dict, test, parameters: list[ParameterMetadata], df_columns):
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
        generateDimensionRow(dimension_dict, test=test, parameters=parameters[numOfParamUsed:], df_columns=df_columns)
    else:
        return


def generateParameterField(parameter: ParameterMetadata, df_columns: list):
    match parameter.type:
        case ParameterType.MULTI_SELECT:
            # If value is not provided, use provided dataset columns as select options
            options = parameter.value if parameter.value else df_columns

            return st.multiselect(parameter.title, options=options, default=[], help=parameter.hint)
        case ParameterType.SINGLE_SELECT:
            # If value is not provided, use provided dataset columns as select options
            options = parameter.value if parameter.value else df_columns
            # If options is a dictionary rather than a list, create a list using the keys
            options = list(options.keys()) if isinstance(options, dict) else options
            
            selectbox_value =  st.selectbox(parameter.title, placeholder=parameter.placeholder, options=options, index=parameter.index, accept_new_options=parameter.accept_new_options, help=parameter.hint)
            return options[selectbox_value] if isinstance(options, dict) and selectbox_value in options else selectbox_value
        case ParameterType.DECIMAL:
            return st.number_input(parameter.title, value=float(parameter.value), step=parameter.step, help=parameter.hint) 
        case ParameterType.TEXT_INPUT | ParameterType.STRING: # Difference between the 2 is when sanitizing fields before running tests
            return st.text_input(parameter.title, value=parameter.value, placeholder=parameter.placeholder, help=parameter.hint)
        case ParameterType.CHECKBOX:
            return st.checkbox(parameter.title, value=parameter.value, help=parameter.hint)
        case ParameterType.FILE_UPLOAD:
            return st.file_uploader(parameter.title, type=["csv", "xlsx"], help=parameter.hint)
        case ParameterType.STRING_LIST:
            return st_tags(label=parameter.title, text='Press enter to add more', value=parameter.value, suggestions=parameter.suggestions)
        case ParameterType.PAIRS:
            suggestions = parameter.suggestions if parameter.suggestions != [] else df_columns
            return st_pairs(label=parameter.title, text='Enter column name', value=parameter.value, suggestions=suggestions)
        case ParameterType.WEIGHTS: # TODO: update with new feilds needed
            return st_weights(label=parameter.title, text='Enter column name', value=parameter.value)
        # Fall through if invalid ParameterType found
        case _:
            st.error("None valid ParameterType found when generating fields from dimension class metadata.")