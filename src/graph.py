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


from pydantic import BaseModel, Field
class DataQualityFunction(BaseModel):
    """Generate python functions to calculate data quality score"""
    python_function: str = Field(..., description="python function to calculate data quality score")

class ListPythonFunction(BaseModel):
    """List of python functions"""
    function_list: List[DataQualityFunction] = Field(description="A list of python functions to calculate data quality score")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.load_local(
    "vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True
)

@tool
def query_doc(query_string: str) -> str:
   """Query data quality documentation to find relevant information. Call this function only when you need further information about data quality"""
   
   return vector_store.similarity_search(query_string)[0].page_content



def graph():
    DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant that generates python functions to calculate data quality score. You only generate code about data quality. Do not do anything else"
)
    tools = [query_doc]



    model = init_chat_model(
        model="gemini-2.5-flash", #gemini-2.5-pro gemini-2.5-flash #gpt-5.1-codex-mini
        model_provider= "google_genai" #"google_genai #"azure_ai"
    )#.with_structured_output(AllergenResponse)
  
    
    return create_agent(

        model=model,
        tools=tools,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=ListPythonFunction
    )

