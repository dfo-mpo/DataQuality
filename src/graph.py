import os
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

DIMENSION_NAME_MAP = {
    "Accessibility": "S",
    "Accuracy": "A",
    "Completeness": "P",
    "Consistency": "C",
    "Interdependency": "I",
    "Relevance": "R",
    "Timeliness": "T",
    "Uniqueness": "U",
}

from pydantic import BaseModel, Field
class FillTestTemplate(BaseModel):
    python_file: str = Field(..., description="Filled out python test template")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.load_local(
    "vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True
)

@tool
def query_doc(query_string: str) -> str:
   """Query data quality documentation to find relevant information. Call this function only when you need further information about data quality"""
   
   return vector_store.similarity_search(query_string)[0].page_content

# added: helper function
def load_test_template(dimension: str) -> str:
    filename = f"{dimension.lower()}_test_template.py"
    template_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "templates",
        filename,
    )

    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"Template not found for dimension: {dimension}"
        )

    return open(template_path, "r", encoding="utf-8").read()

def graph():
    DEFAULT_SYSTEM_PROMPT = (
    f"""You will be given a Python template under the key 'template'. Only fill in TODO sections as a STRICT code-generation agent. Follow ALL rules exactly.

    1. Template:
    - Return the template exactly as provided.
    - Do NOT add or remove functions or classes.
    - ONLY fill in TODO blocks.
    - Do NOT add comments, explanations, or markdown outside the template.
    - IMPORTANT: Remove comments starting with TODO or comments that describe rules once code is filled out. 
      You may keep comments that describe the test logic, parameters, or docstrings.

    2. Placeholders:
    - Replace all placeholders like X# (uppercase) and x# (lowercase) with the test name.
    - Respect casing: X# → TEST variable, x# → metadata variable names and parameters.

    3. Test Naming:
    - Each test MUST have a unique name.
    - You have access to previous test documentation through the query_doc(query_string) tool.
    - Use query_doc to:
    1. Check which tests already exist in the framework.
    2. Understand how previous tests are structured logically and mathematically.
    3. Reference existing parameter types, column usage, or scoring methodology.
    - Only use information returned by query_doc to guide test creation. Do not invent structures or parameters not in the knowledge base.
    - Example query: query_doc("Which Accuracy tests exist?") returns ["A1", "A2", "A3"]. Use this to pick the next available number.
    - Pick the next available number based on the existing test names returned.
    - Dimension code names MUST follow {DIMENSION_NAME_MAP}.
    - Test names MUST follow <DimensionCode><Number> format (e.g., A5, C6, U3).
    - Do NOT reuse names or invent new words.

    4. Parameters:
    - Fill all test-specific __init__ parameters (exclude threshold and selected_columns).
    - Parameter names MUST match exactly the __init__ argument names.
    - Do NOT invent parameters or types.
    - Column parameters MUST be lists, SINGLE_SELECT for one column, MULTI_SELECT for multiple.

    5. Threshold:
    - Default = None.
    - Only set if the user specifies.
    - Do NOT guess thresholds.

    6. UI Parameters (create_metadata):
    - Fully populate the `create_metadata()` function.
    - Use `.add_parameter(name, title, ParameterType, value (optional), default (optional), hint (required))`.
    - The hint MUST clearly describe the input.
    - Required arguments MUST NOT be omitted.
    - Do NOT add parameters not present in Test.__init__ (excluding threshold and selected_columns).

    7. Initialization Parameters:
    - Names MUST follow `<test_name>_parameter_name`.
    - `x#` placeholders MUST be replaced with the actual test name.
    - All parameters MUST have a description using `hint`.
    - Do NOT leave parameters undocumented or ambiguously named.

    8. Column List Parameters:
    - Any column parameter MUST always be a list, even if it contains only one column.
    - In run_test, iterate over all columns; do NOT assume a single string.
    - SINGLE_SELECT for single columns, MULTI_SELECT for multiple columns.

    9. Score:
    - Return a float between 0.0 and 1.0.
    - Name the variable `accuracy_score`.
    - Do NOT return percentages, counts, booleans, or None.

    10. General:
    - Do not change template formatting, spacing, or docstrings.
    - Follow the naming style and examples in the template.
    """
    )


    model = init_chat_model(
        model="gemini-2.5-flash", #gemini-2.5-pro gemini-2.5-flash #gpt-5.1-codex-mini
        model_provider= "google_genai" #"google_genai #"azure_ai"
    ).with_structured_output(FillTestTemplate)
  
    return model, DEFAULT_SYSTEM_PROMPT
