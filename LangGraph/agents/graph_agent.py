from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from tools.graph_embedding_retriever import GraphEmbeddingRetriever
from agents.utils_agent import UtilsAgent

class GraphAgent:
    def __init__(self, db_path):
        openai_api_key = UtilsAgent.get_env_variable("OPENAI_API_KEY")
        neo4j_credentials = UtilsAgent.load_db_credentials('neo4j')
        self.graph_embedding_retriever = GraphEmbeddingRetriever(
            neo4j_credentials['uri'],
            neo4j_credentials['username'],
            neo4j_credentials['password'],
            openai_api_key
        )
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.graph = MessageGraph(memory=self.memory)
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("create_knowledge_graph", ToolNode(self.create_knowledge_graph))
        self.graph.add_node("query_knowledge_graph", ToolNode(self.query_knowledge_graph))
        self.graph.set_entry_point("query_knowledge_graph")

    def create_knowledge_graph(self, data):
        self.graph_embedding_retriever.create_knowledge_graph(data)

    def query_knowledge_graph(self, query):
        return self.graph_embedding_retriever.query_knowledge_graph(query)

    def process_create_knowledge_graph(self, data):
        return self.graph.run("create_knowledge_graph", data=data)

    def process_query_knowledge_graph(self, query):
        return self.graph.run("query_knowledge_graph", query=query)

# Example usage:
if __name__ == "__main__":
    agent = GraphAgent(db_path="graph_memory.db")
    data = {"nodes": [], "edges": []}  # Example data structure
    query = "Find connections between X and Y."
    agent.process_create_knowledge_graph(data)
    result = agent.process_query_knowledge_graph(query)
    print(result)
