import datetime
import streamlit as st
import random
import time
import pandas as pd
from src.graph import graph, load_test_template

TIMEOUT_SECONDS = 600

st.set_page_config(page_title="Data Quality Test Generator", layout="wide")


@st.dialog("Chat expired")
def expire():
    st.write("Chat expired, please refresh the page.")
    
col1, col2 = st.columns([0.8, 0.2], gap="large")

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.datetime.now()

if "model" not in st.session_state:
    model, system_prompt = graph()
    st.session_state['model'] = model
    st.session_state['system_prompt'] = system_prompt

if "generated_code" not in st.session_state:
    st.session_state.generated_code = None

# Check for timeout on each rerun
current_time = datetime.datetime.now()
time_since_last_activity = (current_time - st.session_state.last_activity).total_seconds()
    

with col1:
        
    st.markdown("### Data Quality Test Generator")   

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])    


    # Accept user input
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

    if prompt := st.chat_input("Describe your test...", key="chat_input"):

        st.text_area("Your Test Description:", st.session_state.chat_input, disabled=True)
        pass


    def generate_response_from_graph(text, dimension):
        
        template = load_test_template(dimension)
        model = st.session_state['model']
        system_prompt = st.session_state['system_prompt']

        full_prompt = f"""{system_prompt}

        Template: 
        {template}

        User test description: 
        {text}
        """
        response = model.invoke(full_prompt)

        return response.python_file
    

    def stream(resp):
        import time 
        
        for word in resp.split(" "):
            yield word + " "
            time.sleep(0.005)



    if prompt:
        with st.chat_message("assistant"):
            with st.spinner("Generating test template...", show_time=True):
                dimension = st.session_state.get('selected_dimension')
                code = generate_response_from_graph(prompt, dimension)
                st.session_state.generated_code = code

                st.write("##### Filled Test Template")
                st.code(code, language='python')


with col2:
    
    st.markdown("### Download Filled Test Template")
    st.download_button(label=f"Download File",
        data=st.session_state.generated_code if st.session_state.generated_code else "",
        file_name=f"{st.session_state.get('selected_dimension').lower()}_test_filled.py",
        mime="text/x-python",
        disabled=st.session_state.generated_code is None 
    )
        
    st.markdown(
        """
        **Instructions**
        1. **Select a data quality dimension** from the dropdown.
        2. **Describe your test** (e.g., how many columns the test applies to, edge cases, etc.).
        3. **Click the arrow button** to generate the test code.
        4. **Download or copy the code**:
            - Press **Download File** to save the file and upload it to your system.
            - Or copy the code and manually paste it into your template. 

        For more information: [link]
        """)


