import datetime
import streamlit as st
import random
import time
import pandas as pd
from pathlib import Path

TIMEOUT_SECONDS = 600

st.set_page_config(page_title="Data Quality Coding Agent", layout="wide")


@st.dialog("Chat expired")
def expire():
    st.write("Chat expired, please refresh the page.")
    # if "messages" in st.session_state:
    #     st.session_state.messages = []
    # st.rerun()
    
col1, col2 = st.columns([4, 1])

# col1.subheader("Data Quality Code Assistant")

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.datetime.now()

if "agent" not in st.session_state:
    from src.graph import graph
    st.session_state['agent'] = graph()

# Check for timeout on each rerun
current_time = datetime.datetime.now()
time_since_last_activity = (current_time - st.session_state.last_activity).total_seconds()

# if time_since_last_activity > TIMEOUT_SECONDS:
    
    
#     del st.session_state["thread"]
#     del st.session_state["last_activity"]
#     del st.session_state["messages"]
#     expire()



if "function" not in st.session_state:
        
        
    st.session_state['function'] ={}
    
if "list_instantiated"not in st.session_state:
        
        
    st.session_state['list_instantiated'] =[] 
if "dataset" not in st.session_state:
        
        
    st.session_state['dataset'] = None

with col1:
        
    # st.header("Data Quality Coding Agent")   
    # if "client" not in st.session_state:
        
    #     from langgraph_sdk import get_sync_client
    #     st.session_state['client'] = get_sync_client(url="http://localhost:2024")
        
        

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        
    dimensions = [
        "Accessibility",
        "Accuracy",
        "Consistency",
        "Completeness",
        "Interdependency",
        "Relevance",
        "Timeliness",
        "Uniqueness"
    ]
    dimension_selected = st.selectbox("Select a Data Quality Dimension", dimensions)
    st.session_state['selected_dimension'] = dimension_selected


    def generate_response_from_graph(text: str, dimension: str, template_file: str):
        
        agent = st.session_state['agent']

        # load template
        template_path = Path(template_file)
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_file}")
        
        template_content = template_path.read_text(encoding="utf-8")

        # build prompt with constraints
        full_prompt = f"""
        You are a STRICT code-generation agent. Follow these rules exactly.

        1. Template Rules:
        - Return the template exactly as provided.
        - Fill in all TODO blocks.
        - Do not change function names, parameters, imports, or structure.
        - Do not add or remove any functions.
        - Output ONLY valid Python code.
        - Do not include explanations, markdown, or comments outside the template.

        2. Test Naming:
        - Each test MUST have a unique name.
        - Check the knowledge base to see which test names already exist.
        - Test names MUST follow the format <DimensionCode><Number>.
        - Dimension codes: Accessibility:S, Accuracy:A, Completeness:P, Consistency:C, Interdependency:I, Relevance:R, Timeliness:T, Uniqueness:U.
        - Example: A1, C2, U3.
        - Do NOT add words, reuse existing names, or change the format.
        - Assign the next available number for the dimension.

        3. Threshold Parameter:
        - Default threshold = None.
        - Only set a value if the user specifies it.
        - Do NOT guess or infer threshold values.

        4. UI Parameters:
        - Use the `.add_parameter(...)` method from the knowledge base.
        - Parameter types MUST be one of: MULTI_SELECT, SINGLE_SELECT, DECIMAL, TEXT_INPUT, STRING, CHECKBOX, FILE_UPLOAD, STRING_LIST, PAIRS.
        - Do NOT invent new parameter types.
        - Each parameter MUST have a name, title, value, default, and description (hint).
        - The description must explain the parameter clearly.

        5. Initialization Parameters:
        - Initialization parameter names MUST follow the convention `<test_name>_parameter_name`.
        - Example: for test name A5, a column list parameter could be named `A5_column_list`.
        - `x#` is a placeholder in the template; replace it with the actual test name.
        - All parameters MUST have a description using the `hint` argument.
        - Do NOT leave parameters undocumented or ambiguously named.

        6. Column List Parameters:
        - Any column parameter MUST be a list of strings: ['col1'] or ['col1', 'col2'].
        - Single columns must still be a list.
        - Column list parameters MUST use `ParameterType.SINGLE_SELECT' for a single column or `ParameterType.MUTLI_SELECT' for multiple columns.
        - Always iterate over the list in the logic.

        7. Score:
        - Return a score as a proportion (0.0 to 1.0).
        - Score = proportion of records passing the test.
        - Do NOT return percentages, counts, or booleans.

        Dimension:
        {dimension}

        User intent:
        {text}

        TEMPLATE (DO NOT MODIFY STRUCTURE):
        ----------------------------------
        {template_content}
        ----------------------------------
        """


        inputs = {"messages": [{"role": "user", "content": full_prompt}]}
        
        function_list = []
        response = agent.invoke(inputs)

        try:
            for f in response['structured_response'].function_list:
                function_list.append(f.python_function)
        except:
            function_list.append("# No functions generated")

        return function_list
        
        

    # Accept user input
    if prompt := st.chat_input("Asking data quality code?"):
        
        # Display user message in chat message container
        # with st.chat_message("user"):
        #     st.markdown(prompt)
            
        # Add user message to chat history
        # st.session_state.messages.append({"role": "user", "content": prompt})
        #pass

        # Get selected dimension, default to first in list (Accessibility)
        dimension = st.session_state.get('selected_dimension', 'Accessibility')

        # Map to corresponding template file
        template_file = f"templates/{dimension.lower()}_test_template.py"

        
        if prompt:
            with st.chat_message("assistant"):
                with st.spinner("Thinking...", show_time=True):
                    # Generate function using full prompt
                    dimension = st.session_state.get('selected_dimension', 'Accessibility')
                    template_file = f"templates/{dimension.lower()}_test_template.py"
                    #full_prompt = f"Dimension: {dimension}\nUser request: {prompt}\nTemplate file: {template_file}"
                    
                    functions = generate_response_from_graph(prompt, dimension, template_file)

                    if functions != "# No functions generated":
                        st.write("GENERATED FUNCTION:")
                        for f in functions:
                            st.code(f)
                            f_name = f.split("def ")[-1].split("(")[0]
                            st.session_state['function'][f_name] = f
        elif "function" in st.session_state and st.session_state['function']:
            st.write("STORED FUNCTION:")
            for stored_f in st.session_state['function'].values():
                st.code(stored_f)
          
                        # response = st.write_stream(stream(generate_response(prompt)))
            # Add assistant response to chat history
                    # st.session_state.messages.append({"role": "assistant", "content": res})


            

    # def generate_response_from_client(text):
        
    #     response = []
    #     client = st.session_state['client']
    #     for chunk in client.runs.stream(
    #         None,  # Threadless run
    #         "agent", # Name of assistant. Defined in langgraph.json.
    #         input={
    #             "messages": [{
    #                 "role": "human",
    #                 "content": text,
    #             }],
    #         },
    #         stream_mode="messages-tuple",
    #     ):
        
    #         if chunk.event == "messages":
    #             response.append(chunk.data)

    #     function_list = []
        
    #     try:
    #         for f in response[0][0]['tool_calls'][0]['args']['function_list']:
    #             function_list.append(f['python_function'])
    #     except:
            
    #         function_list.append("# No functions generated")
            
    #     # function_list = "\n\n __FUNCTION__ \n\n".join(function_list)
            
    #     return function_list


    def stream(resp):
        import time 
        
        for word in resp.split(" "):
            yield word + " "
            time.sleep(0.005)

    # if not prompt and "last_activity" in st.session_state:
    #     st.session_state.last_activity = datetime.datetime.now()

        
    
with col2:
     
    
    if st.button("Delete all functions",type="primary") and "function" in st.session_state:
        #  import pyautogui
        #  pyautogui.hotkey("ctrl","F5")
         button_labels =[]
         st.session_state['function'] ={}
         for stored_f in st.session_state['function'].keys():
             if stored_f in list(globals().keys()):
                 del globals()[stored_f]
                 
         st.rerun()
                 
        
    st.header("Generated functions:")
    
    uploaded_file = st.file_uploader("Upload a CSV dataset", type="csv")
    
    if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        st.session_state['dataset'] = df
        st.success("Dataset uploaded successfully!")
                     
                     
    if "function" in st.session_state:
        
        df= pd.DataFrame(
        list(st.session_state['function'].keys()),
        columns=["Function Name"],)
        st.dataframe(df, use_container_width=True)
  

        # Define a list of button labels
        button_labels = list(st.session_state['function'].keys())

        st.write("Run a function on the uploaded dataset:")

        # Iterate through the labels and create a button for each
        for label in button_labels:
            if st.button(label):
                exec(st.session_state['function'][label])
                st.write(f"You have run: {label}, output below:")
                st.write(locals()[label](st.session_state['dataset']))
                # st.write(f"You have run: {label}")
        
    
        
        # for f in st.session_state['function'].keys():
        #     if f in list(locals().keys()):
        #         st.session_state['list_instantiated'].append(f)
                
        # df2= pd.DataFrame(
        # st.session_state['list_instantiated'],
        # columns=["List of instantiated functions"],)
        # st.dataframe(df2, use_container_width=True)
        
        # f_instance = st.session_state['list_instantiated']
        # if f_instance!=[]:
            
        #     for instance in f_instance:
        #         if st.button(instance):
        #             if st.session_state['dataset']:
        #                 st.write(f"You have run: {instance} on the dataset, output below:")
        #                 st.write(locals()[instance](st.session_state['dataset']))
                    
    