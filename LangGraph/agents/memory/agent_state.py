import operator
from langchain.schema import Document
from typing import Annotated, TypedDict, List, Dict, Any


class AgentState(TypedDict):
    user_query: str
    database_retrieval: List[Document]
    graph_retrieval: Dict[str, List]
    context_combination: Dict[str, Any]
    data_csv: Annotated[Dict[str, List], operator.add]

