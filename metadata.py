from enum import Enum

""" Class to represent the different parameter types that can be generated using the UI tool.
""" # TODO: Add coments about each type note select types use uploaded columns as options unless custom is provided in value of metadata parameter

# file upload is only csv or xlsx types
class ParameterType(Enum):
    MULTI_SELECT = "multi-select" # uses dataset as options if no value is provided
    SINGLE_SELECT = "single-select"
    DECIMAL = "decimal"
    TEXT_INPUT = "text"
    STRING = "string"
    CHECKBOX = "checkbox"
    FILE_UPLOAD = "file-upload"
    STRING_LIST = "string-list" # output is a list of strings
    PAIRS = "pairs" # output list of string pairs, each value list entry is structured as a string double, ('col1', 'col2')

""" Class to represent the properties of the parameters required for a given metric. Used by the UI tool to generate parameter input boxes/feilds.

dimension_name: Name of the dimension the metric is under.
name: Name of the metric for the metadata class.
parameters: List of ParameterMetadata objects for each parameter in the given metric.
"""
class MetricMetadata:
    def __init__(self, dimension_name, name):
        self.dimension_name = dimension_name
        self.name = name
        self.parameters = []

    # Creates a new ParameterMetadata instance and appends it to the parameters list
    def add_parameter(self, name, title, type: ParameterType, value = "", default = None, placeholder = None, suggestions = [], step = 0.01, hint = None):

        self.parameters.append(ParameterMetadata(name, title, type, value, default, placeholder, suggestions, step, hint))


""" Class to represent the properties of a parameter. Used by the UI tool to generate parameter input boxes/feilds.

name: Parameter represented, must match parameter name use in class definition.
title: Title for parameter shown in UI.
type: Type of parameter input to generate in the UI. Can be any option defined in the ParameterType class.
value: Value is the inital value used by the metric, it is changed based on user input. For select options value is the set of available options. 
       Note that ParameterType MULTI_SELECT will use the dataset column names if no value is provided.
default: Value used if input provided is not in a valid format (ParameterType TEXT_INPUT) or if no user input is provided (ParameterType MULTI_SELECT).
placeholder: If type is Text, placeholder is a greyed out text shown when box is empty.
suggestions: List of autofill options for string inputs for ParameterTypes STRING_LIST and PAIRS. For PAIRS default value will use dataset columns unless default is overwritten with any list or None.
step: If type is Decimal, step is the increment/decrement amounts when the arrow keys are used to change the value.
hint: Helper message that can provide more context to how the user should enter this parameter value (not available for ParameterTypes STRING_LIST and PAIRS).
"""
class ParameterMetadata:
    def __init__(self, name, title, type: ParameterType, value = "", default = None, placeholder = None, suggestions = [], step = 0.01, hint = None):
        self.name = name
        self.title = title
        self.type = type
        self.value = value
        self.default = default
        self.placeholder = placeholder
        self.suggestions = suggestions
        self.step = step
        self.hint = hint