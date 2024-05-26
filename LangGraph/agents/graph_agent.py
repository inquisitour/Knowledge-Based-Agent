from langgraph.graph import MessageGraph
from agents.utils_agent import UtilsAgent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.embedding_agent import EmbeddingAgent
from data_processing.database_ops import DBops
from tools.graph_embedding_retriever import GraphEmbeddingRetriever

class GraphAgent:
    def __init__(self, db_path):
        openai_api_key = UtilsAgent.get_env_variable("OPENAI_API_KEY")
        neo4j_credentials = UtilsAgent.load_db_credentials('neo4j')
        embeddings = EmbeddingAgent(db_path)
        self.graph_embedding_retriever = GraphEmbeddingRetriever(
            neo4j_credentials['uri'],
            neo4j_credentials['username'],
            neo4j_credentials['password'],
            openai_api_key,
            db_path,
            embeddings
        )
        self.memory = SqliteSaver.from_conn_string(db_path)
        self.graph = MessageGraph()
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("create_knowledge_graph", ToolNode([self.create_knowledge_graph]))
        self.graph.add_node("query_knowledge_graph", ToolNode([self.query_knowledge_graph]))

        self.graph.add_conditional_edges("check_data_hash", DBops.check_data_hash)
        self.graph.add_edge("check_data_hash", "create_knowledge_graph")
        self.graph.add_edge("create_knowledge_graph", "query_knowledge_graph")

        self.graph.set_entry_point("check_data_hash")

    def create_knowledge_graph(self, state):
        """Method for creating a knowledge graph."""
        data_csv = state["data_csv"]
        self.graph_embedding_retriever.create_knowledge_graph(data_csv)

    def query_knowledge_graph(self, query):
        """Method for querying a knowledge graph."""
        return self.graph_embedding_retriever.query_knowledge_graph(query)

