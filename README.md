# Data Quality Coding Agent
Install required packages

## TO Run App:
```
streamlit run app.py
```

```mermaid
graph TB

subgraph User Interaction
    A[User using Web App] --> D[Prompt]
end



subgraph RAG
    D --> F[Coding Agent] --> K[Generate Functions]
    F --> T[Tool]
    T --> R[RAG to Data Quality Framework] --> V[FAISS vectorstore]
    R --"Retrieve Relevant Info"--> F[Coding Agent]
end

subgraph Insurance Analysis
    A -- "Upload Dataset" --> K[Generate Functions] --"Run Function in Realtime"--> M[Display Result]
    
end

```

## To update vectorstore so Agent has more domain knowledge:

Use notebook Notebook_to_build_vectorstore