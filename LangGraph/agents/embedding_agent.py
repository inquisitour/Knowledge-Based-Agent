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
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.graph = MessageGraph(memory=self.memory)
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("embed_documents", ToolNode(self.embed_documents))
        self.graph.add_node("embed_query", ToolNode(self.embed_query))
        self.graph.set_entry_point("embed_query")

    def embed_documents(self, documents):
        return self.embeddings.embed_documents(documents)

    def embed_query(self, query):
        return self.embeddings.embed_query(query)

    def process_documents(self, documents):
        return self.graph.run("embed_documents", documents=documents)

    def process_query(self, query):
        return self.graph.run("embed_query", query=query)

    def get_graph(self):
        return self.graph

# Example usage:
if __name__ == "__main__":
    agent = EmbeddingAgent(db_path="embeddings_memory.db")
    documents = ["Document 1 content", "Document 2 content"]
    query = "What is the content of Document 1?"
    print(agent.process_documents(documents))
    print(agent.process_query(query))