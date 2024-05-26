from langgraph.graph import MessageGraph
from agents.utils_agent import UtilsAgent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.embedding_agent import EmbeddingAgent
from data_processing.database_ops import DBops
from agents.query_generation_agent import QueryGenerationAgent
from tools.graph_embedding_retriever import GraphEmbeddingRetriever

class GraphAgent:
    def __init__(self, db_path):
        openai_api_key = UtilsAgent.get_env_variable("OPENAI_API_KEY")
        neo4j_credentials = UtilsAgent.load_db_credentials('neo4j')
        self.query_generation_agent = QueryGenerationAgent(db_path=db_path)
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
        self.graph.add_node("create_knowledge_graph", ToolNode([self.create_kg]))
        self.graph.add_node("query_knowledge_graph", ToolNode([self.query_kg]))

        self.graph.add_conditional_edges("check_data_hash", DBops.check_data_hash)
        self.graph.add_edge("check_data_hash", "create_knowledge_graph")
        self.graph.add_edge("create_knowledge_graph", "query_knowledge_graph")

        self.graph.set_entry_point("check_data_hash")

    def create_kg(self, state):
        """Method for creating a knowledge graph."""
        data_csv = state["data_csv"]
        self.graph_embedding_retriever.create_knowledge_graph(data_csv)

    def query_kg(self, state):
        """Method for querying a knowledge graph."""
        user_query = state["user_query"]
        cypher_query = self.query_generation_agent.generate_cypher_query(state)
        return self.graph_embedding_retriever.query_knowledge_graph(user_query, cypher_query)

