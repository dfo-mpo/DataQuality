# Writing Parameter Metadata for UI
Adding a parameter input to the UI tool, requires using the `add_parameter()` function from the `TestMetadata` object found in the test template inside the `create_metadata()` method. The type of UI component generated and parameters needed when using `add_parameter()` depends on the `ParameterType` chosen.

## ParameterType choices
**List of ParameterType that can be used:**
| ParameterType               | Purpose                                  | ParameterMetadata Parameters for Given ParameterType      |
|------------------------|-----------------------------------------------|-----------------|
  <a href="#multi_select" onclick="document.getElementById('multi_select_details').open = true;">MULTI_SELECT</a> | Select multiple options | `name`, `title`, `type`, `value`, `default`, `hint`  |
| <a href="#single_select" onclick="document.getElementById('single_select_details').open = true;">SINGLE_SELECT</a> | Select a single option | `name`, `title`, `type`, `value`, `placeholder`, `index`, `accept_new_options`, `hint` |
| <a href="#decimal" onclick="document.getElementById('decimal_select_details').open = true;">DECIMAL</a> | Numeric input (`int` or `float`) | `name`, `title`, `type`, `value`, `step`, `min`, `max`, `hint` |
  <a href="#string" onclick="document.getElementById('string_select_details').open = true;">STRING</a> | Single-line text input | `name`, `title`, `type`, `value`, `placeholder`, `max`, `hint` |
| <a href="#text_input" onclick="document.getElementById('text_input_select_details').open = true;">TEXT_INPUT</a> | Structured / object-like text input | `name`, `title`, `type`, `value`, `placeholder`, `max`, `hint` |
| <a href="#checkbox" onclick="document.getElementById('checkbox_select_details').open = true;">CHECKBOX</a> | Boolean input | `name`, `title`, `type`, `value`, `hint` |
| <a href="#file_upload" onclick="document.getElementById('file_upload_select_details').open = true;">FILE_UPLOAD</a> |  CSV / XLSX file upload and conversion to pandas DataFrame | `name`, `title`, `type`, `hint` |
| <a href="#string_list" onclick="document.getElementById('string_list_select_details').open = true;">STRING_LIST</a> | List of user-defined strings (tags) | `name`, `title`, `type`, `value`, `placeholder`, `suggestions` |
| <a href="#pairs" onclick="document.getElementById('pairs_select_details').open = true;">PAIRS</a> | List of user-defined tuple pairs | `name`, `title`, `type`, `value`, `placeholder`, `suggestions` |
| <a href="#weights" onclick="document.getElementById('weights_select_details').open = true;">WEIGHTS</a> | Weighted numeric inputs with sum indicator | `name`, `title`, `type`, `value`, `placeholder`, `step`, `min`, `max` |

### Detailed documentation
<a id="multi_select"></a> 
<details id="multi_select_details">  
<summary><strong>MULTI_SELECT</strong></summary>  
  
### Description  
Generate a selection box, allowing a user to pick multiple options.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.MULTI_SELECT, value="", default=None, hint=None)
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | [str] \| str \| None | The set of available options for the user to choose from. By default (or if set to either `None` or an empty string) the uploaded dataset's columns will be used as the value. |
|`default` | [str] \| str \| None | Intial selected option(s) when generated on the UI. By default, nothing is selected. |
|`hint` | str \| None | Helper message tooltip that can provide more context to how the user should enter this parameter value/field. If this is `None` (default), no tooltip is displayed. |
  
### Returns from UI to test
Return a list containing strings (`[str]`) for each option selected by the user.  
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Accuracy"

    # Define instance for test
    a1_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    a1_metadata.add_parameter('a1_column_names', 'A1 Column Names', ParameterType.MULTI_SELECT, default=[])
    
    return a1_metadata 
```
</details> 

<a id="single_select"></a>  
<details id="single_select_details">  
<summary><strong>SINGLE_SELECT</strong></summary>  
  
### Description  
Generate a selection box, allowing a user to pick a single option.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.SINGLE_SELECT, value="", placeholder=None, index=None, accept_new_options=False, hint=None)
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | [str] \| str \| dict[str, str] \| None | The set of available options for the user to choose from. By default (or if set to either `None` or an empty string) the uploaded dataset's columns will be used as the value. If using a dictionary (instead of a list), the keys will be presented as options to the user.|
|`placeholder` | str \| None | A string to display when no options are selected. If this is `None` (default), the widget displays placeholder text based on the widget's configuration:<ul><li>"Choose an option" is displayed when options are available and `accept_new_options=False`.</li><li>"Choose or add an option" is displayed when options are available and `accept_new_options=True`.</li><li>"Add an option" is displayed when no options are available and `accept_new_options=True`.</li><li>"No options to select" is displayed when no options are available and `accept_new_options=False`. The widget is also disabled in this case.</li></ul> |
|`index` | int \| None | The index of the preselected option on first render. Defaults to `None`, which will initialize empty and return `None` until the user selects an option. |
|`accept_new_options` | int \| None | Whether the user can add a selection that isn't included in `value`. If this is `False` (default), the user can only select from the items in `value`. If this is `True`, the user can enter a new item that doesn't exist in `value`. |
|`hint` | str \| None | Helper message tooltip that can provide more context to how the user should enter this parameter value/field. If this is `None` (default), no tooltip is displayed. |
  
### Returns from UI to test
Return a string (`str`) for the chosen option. If a dictionary was used as the `value`, the chosen key's value is returned.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for test
    c4_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    c4_metadata.add_parameter('c4_format', 'C4 Format', ParameterType.SINGLE_SELECT, placeholder="Choose option or enter custom date-time format...",
                                value={'2001 (YYYY)': '%Y', '2001-03-14 (YYYY-MM-DD)': '%Y-%m-%d', '14-Mar-01 (DD-MMM-YY)': '%d-%b-%y', '03/14/2001 (MM/DD/YYYY)': '%m/%d/%Y',
                                    '14/03/2001 (DD/MM/YYYY)': '%d/%m/%Y', '20010314 (YYYYMMDD)': '%Y%m%d', '2001-03-14 13:30:55 (YYYY-MM-DD HH:MM:SS)': '%Y-%m-%d %H:%M:%S',
                                    '14-Mar-01 13:30:55 (DD-MMM-YY HH:MM:SS)': '%d-%b-%y %H:%M:%S', '03/14/2001 13:30:55 (MM/DD/YYYY HH:MM:SS)': '%m/%d/%Y %H:%M:%S',
                                    '14/03/2001 13:30:55 (DD/MM/YYYY HH:MM:SS)': '%d/%m/%Y %H:%M:%S', '20010314 13:30:55 (YYYYMMDD HH:MM:SS)': '%Y%m%d %H:%M:%S'
                                }, 
                                accept_new_options=True,
                                hint=" Enter a Python date-time format string using strftime codes (e.g., %Y-%m-%d %H:%M:%S). \n For a full list of format codes: \n https://docs.python.org/3.11/library/datetime.html?utm_source=chatgpt.com#format-codes" )
    
    return c4_metadata 
```
</details> 

<a id="decimal"></a> 
<details id="decimal_select_details">  
<summary><strong>DECIMAL</strong></summary>  
  
### Description  
Generate an input box, allowing a user to enter a number.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.DECIMAL, value="", step=0.01, min=None, max=None, hint=None)
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | str \| int \| float | The numeric value of the input when it first renders. By default it is 0 or 0.0, but if a `min` is provided then that will become the initial value.  |
|`step` | int \| float | The stepping interval. Defaults to 0.01. |
|`min` | int \| float \| None | The minimum permitted value. If this is `None` (default), there will be no minimum for float values and a minimum of `- (1<<53) + 1` for integer values. |
|`max` | int \| float \| None | The maximum permitted value. If this is `None` (default), there will be no maximum for float values and a maximum of `(1<<53) - 1` for integer values. |
|`hint` | str \| None | Helper message tooltip that can provide more context to how the user should enter this parameter value/field. If this is `None` (default), no tooltip is displayed. |
  
### Returns from UI to test
Returns the number (int or float) entered in the input box.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for test
    c1_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    c1_metadata.add_parameter('c1_threshold', 'C1 Threshold', ParameterType.DECIMAL, value='0.91', step = 0.01)
    
    return c1_metadata 
```
</details>

<a id="string"></a>
<details id="string_select_details">  
<summary><strong>STRING</strong></summary>  
  
### Description  
Generate a text input box, allowing a user to enter a string.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.STRING, value="", placeholder=None, max=None, hint=None)
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | str | The string value of the input when it first renders. By default it is an empty string.  |
|`placeholder` | str \| None | An optional string displayed when the text input is empty. If `None`, no text is displayed. |
|`max` | int \| None | Max number of characters allowed in text input. |
|`hint` | str \| None | Helper message tooltip that can provide more context to how the user should enter this parameter value/field. If this is `None` (default), no tooltip is displayed. |
  
### Returns from UI to test
Returns the text (str) entered in the input box.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for test
    c2_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    c2_metadata.add_parameter('c2_search_word', 'C2 Search Word', ParameterType.STRING, placeholder="Enter the search word", max=24, hint="The search word is used to filter which columns are evaluated by this test.")
    
    return c2_metadata 
```
</details>

<a id="text_input"></a> 
<details id="text_input_select_details">  
<summary><strong>TEXT_INPUT</strong></summary>  
  
### Description  
Generate a text input box, allowing a user to enter structured text.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.TEXT_INPUT, value="", placeholder=None, max=None, hint=None)
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | str | The string value of the input when it first renders. By default it is an empty string.  |
|`placeholder` | str \| None | An optional string displayed when the text input is empty. If `None`, no text is displayed. |
|`max` | int \| None | Max number of characters allowed in text input. |
|`hint` | str \| None | Helper message tooltip that can provide more context to how the user should enter this parameter value/field. If this is `None` (default), no tooltip is displayed. |
  
### Returns from UI to test
Returns the text entered in the input box after it is parsed (str or obj). If the inputted text is properly structured as an object, it will be converted into whichever type of object it is structuring.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for test
    c2_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    c2_metadata.add_parameter('c2_column_mapping', 'C2 Column Mapping', ParameterType.TEXT_INPUT, placeholder="e.g., {'Column1': 'Reference1', 'Column2': 'Reference2'}")
    
    return c2_metadata 
```
</details>

<a id="checkbox"></a>
<details id="checkbox_select_details">  
<summary><strong>CHECKBOX</strong></summary>  
  
### Description  
Generate a checkbox.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.CHECKBOX, value="", hint=None)
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | bool | Preselect the checkbox when it first renders. This will be cast to `bool` internally.  |
|`hint` | str \| None | Helper message tooltip that can provide more context to how the user should enter this parameter value/field. If this is `None` (default), no tooltip is displayed. |
  
### Returns from UI to test
Returns `True` if the box is selected, otherwise `False`.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Accessibility"

    # Define instance for test, replace with test that requires parameters
    s1_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    s1_metadata.add_parameter("s1_has_metadata", "S1 Has Metadata", ParameterType.CHECKBOX, value=False)
    
    return s1_metadata 
```
</details>

<a id="file_upload"></a>
<details id="file_upload_select_details">  
<summary><strong>FILE_UPLOAD</strong></summary>  
  
### Description  
Generate a file upload box. Allows `csv` or `xlsx` file inputs, which are limited to 200 MB each.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.FILE_UPLOAD, hint=None)
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`hint` | str \| None | Helper message tooltip that can provide more context to how the user should enter this parameter value/field. If this is `None` (default), no tooltip is displayed. |
  
### Returns from UI to test
Returns the pandas DataFrame of the uploaded file.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for test
    c2_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    c2_metadata.add_parameter('ref_dataset_path', 'C2 Reference Dataset File', ParameterType.FILE_UPLOAD)
    
    return c2_metadata 
```
</details>

<a id="string_list"></a>
<details id="string_list_select_details">  
<summary><strong>STRING_LIST</strong></summary>  
  
### Description  
Generate a text input used to create strings (shown as tags). Allows a user to input as many strings as they want.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.STRING_LIST, value="", placeholder=None, suggestions=[])
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | [str] \| str \| None | The inital set of strings that exists durring the first render. |
|`placeholder` | str \| None | Text that appears in the input box when it is empty. By default, this is 'Press enter to add more'. |
|`suggestions` | [str] \| None | List of fill in options when a user starts typing a string in the input box. |
  
### Returns from UI to test
Returns list (`[str]`) of all existing string (tags) in the component.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Consistency"

    # Define instance for test
    c1_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    c1_metadata.add_parameter('c1_stop_words', 'C1 Stop Words', ParameterType.STRING_LIST, value=["the", "and"], placeholder="Enter a stop word", suggestions=["the", "and"])
    
    return c1_metadata 
```
</details>

<a id="pairs"></a>
<details id="pairs_select_details">  
<summary><strong>PAIRS</strong></summary>  
  
### Description  
Generate a text input used to create strings (shown as tags). Allows a user to input as many strings as they want.
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.PAIRS, value=[], placeholder=None, suggestions=[])
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | [str] \| None | The inital set of pairs that exists durring the first render. Entry strings should be formatted as: `(FirstPairString, SecondPairString)`. |
|`placeholder` | str \| None | Text that appears in the input box when it is empty. By default, this is 'Enter column name'. |
|`suggestions` | [str] \| None | List of fill in options when a user starts typing a string in the input box. If `None` (default), all columns from uploaded dataset is used as suggestions. |
  
### Returns from UI to test
Returns list of tuples (`Tuple[str,str]`) of all existing string pairs in the component.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Accuracy"

    # Define instance for test
    a4_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    a4_metadata.add_parameter('a4_column_pairs', 'A4 Column Pairs', ParameterType.PAIRS, value=['(Chum_count, total_count)', '(Chinook_count, total_count)'])
    
    return a4_metadata 
```
</details>

<a id="weights"></a>
<details id="weights_select_details">  
<summary><strong>WEIGHTS</strong></summary>  
  
### Description  
Generate a set of number inputs to generate weight values using a dictionary input (keys are names with values as the inital weights). Provides indication of the sum of all weights. 
  
### TestMetadata.add_parameter() function signature 
```python
TestMetadata.add_parameter(name, title, type=ParameterType.WEIGHTS, value=[], placeholder=None, step=0.01, min=None, max=None)
```

### Parameters
| Name | Type | Description 
|-------|----|---------------|
|`name` | str | Parameter represented, must match parameter name used in test's class definition (the class `__init__`). |
|`title` | str | Title for parameter shown in UI. |
|`value` | dict[str, float \| int] \| None | The intial value for this component on the first render. Each key corrisponds to 1 generated weight name while its value is the intial weight value. |
|`placeholder` | str \| None | Text that appears in the component when no weights are in `value`. By default, this is 'Once valid options a selected weight selections will appear here.'. |
|`step` | int \| float | The stepping interval. Defaults to 0.01. |
|`min` | int \| float \| None | The minimum permitted value for a given weight. If this is `None` (default), there will be no minimum for float values and a minimum of `- (1<<53) + 1` for integer values. |
|`max` | int \| float \| None | The maximum permitted value for a given weight and weight sum limit used for indicator of total weight value. If this is `None` (default), there will be no maximum for float values and a maximum of `(1<<53) - 1` for integer values. |

  
### Returns from UI to test
Returns a copy of the inputted `value` but with the user entered weights.
  
### Example  
```python  
""" Creates a TestMetadata instance for a single test, defining any parameters used by the UI to generate input fields.
"""
def create_metadata():
    dimension = "Accuracy"

    # Define instance for test
    a5_metadata = TestMetadata(dimension, TEST)
    # Define each parameter needed for test, use ParameterType when defining type
    a5_metadata.add_parameter('a5_weights', 'A5 Weights', ParameterType.WEIGHTS, value={'a1': 0, 'a2': 0.5, 'a3': 0.5}, step=0.05, min=0, max=1.0 )
    
    return a5_metadata 
```
</details>