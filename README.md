# Data Quality Coding Agent
Set up virtual evironment, activate it, install required packages, set env variable in .env.example and rename it to .env

## To Run App:
```
streamlit run app.py
```
## System Architecture
```mermaid
graph TB

subgraph User Interaction
    A[User using Web App] --> D[Prompt]
end



subgraph DQ Agent
    D --> F[Coding Agent] --> K[Generate Functions]
    F --> T[Langgraph Tool Decoration]
    T --> R[RAG ] --> V[FAISS vectorstore storing Data Quality Framework]
    R --"Retrieve Relevant Info"--> F[Coding Agent]
end

subgraph Realtime result
    A -- "Upload Dataset" --> K[Generate Functions] --"Run Function in Realtime"--> M[Display Result]
    
end

```

## Update/Build vectorstore

To update or build vectorstore so Agent has more domain knowledge:

Use notebook Notebook_to_build_vectorstore.ipynb
