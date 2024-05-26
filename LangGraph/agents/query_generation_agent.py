from langchain.schema import HumanMessage
from langgraph.graph import MessageGraph
from agents.utils_agent import UtilsAgent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.chat_models import ChatOpenAI

class QueryGenerationAgent:
    def __init__(self, db_path):
        openai_api_key = UtilsAgent.get_env_variable("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
        self.memory = SqliteSaver.from_conn_string(db_path)
        self.graph = MessageGraph()
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("generate_cypher_query", ToolNode([self.generate_cypher_query]))
        self.graph.set_entry_point("generate_cypher_query")

    def generate_cypher_query(self, user_query):
        """Method for generating a cypher query."""
        prompt = f"Given the user query: {user_query}, generate a Cypher query to retrieve relevant information from the Neo4j knowledge graph."
        messages = HumanMessage(content=prompt)
        response = self.llm([messages])
        cypher_query = response.content
        return cypher_query

