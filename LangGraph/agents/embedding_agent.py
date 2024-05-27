from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.utils_agent import UtilsAgent

class EmbeddingAgent:
    def __init__(self, db_path):
        openai_api_key = UtilsAgent.get_env_variable("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
        self.memory = SqliteSaver.from_conn_string(db_path)
        self.graph = MessageGraph()
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("embed_documents", ToolNode([self.embed_documents]))
        self.graph.add_node("embed_query", ToolNode([self.embed_query]))

        self.graph.add_edge("embed_documents", "embed_query")

        self.graph.set_entry_point("embed_documents")

    def embed_documents(self, documents):
        """Method for embedding documents."""
        return self.embeddings.embed_documents(documents)

    def embed_query(self, query):
        """Method for embedding queries."""
        return self.embeddings.embed_query(query)
