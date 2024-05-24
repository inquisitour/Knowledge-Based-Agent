from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.utils_agent import UtilsAgent

class QueryGenerationAgent:
    def __init__(self, db_path):
        openai_api_key = UtilsAgent.get_env_variable("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.graph = MessageGraph(memory=self.memory)
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("generate_cypher_query", ToolNode(self.generate_cypher_query))
        self.graph.set_entry_point("generate_cypher_query")

    def generate_cypher_query(self, user_query):
        prompt = f"Given the user query: {user_query}, generate a Cypher query to retrieve relevant information from the Neo4j knowledge graph."
        messages = HumanMessage(content=prompt)
        response = self.llm([messages])
        cypher_query = response.content
        return cypher_query

    def process_generate_cypher_query(self, user_query):
        return self.graph.run("generate_cypher_query", user_query=user_query)

# Example usage:
if __name__ == "__main__":
    agent = QueryGenerationAgent(db_path="query_generation_memory.db")
    user_query = "Find all nodes connected to Node X."
    cypher_query = agent.process_generate_cypher_query(user_query)
    print(cypher_query)
