from enum import Enum

""" Class to represent the different parameter types that can be generated using the UI tool.
""" # TODO: Add coments about each type note select types use uploaded columns as options unless custom is provided in value of metadata parameter
# file upload is only csv or xlsx types
class ParameterType(Enum):
    MULTI_SELECT = "multi-select"
    SINGLE_SELECT = "single-select"
    DECIMAL = "decimal"
    TEXT_INPUT = "text"
    STRING = "string"
    FILE_UPLOAD = "file-upload"

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
    def add_parameter(self, name, title, type: ParameterType, value = "", default = None, placeholder = None, step = 0.01, hint = None):
        self.parameters.append(ParameterMetadata(name, title, type, value, default, placeholder, step, hint))


""" Class to represent the properties of a parameter. Used by the UI tool to generate parameter input boxes/feilds.

name: Parameter represented, must match parameter name use in class definition.
title: Title for parameter shown in UI.
type: Type of parameter input TODO: Add in options that are allowed.
value: Value is the default value used by the metric, it is changed based on user input.
default: Value used if input provided is not in a valid format, specifically for ParameterType TEXT_INPUT.
placeholder: If type is Text, placeholder is a greyed out text shown when box is empty.
step: If type is Decimal, step is the increment/decrement amounts when the arrow keys are used to change the value.
hint: Helper message that can provide more context to how the user should enter this parameter value.
"""
class ParameterMetadata:
    def __init__(self, name, title, type: ParameterType, value = "", default = None, placeholder = None, step = 0.01, hint = None):
        self.name = name
        self.title = title
        self.type = type
        self.value = value
        self.default = default
        self.placeholder = placeholder
        self.step = step
        self.hint = hint


# IMPORTANT: multibox will assume the input is all columns from uploaded dataset.
# IMPORTANT: verifiy is default = [] can be used for all multibox inputs.

