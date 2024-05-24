import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolNode
# from langgraph.core import messages
from data_processing.database_ops import DBops
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

class DatabaseAgent:
    def __init__(self, db_path):
        self.db_path = db_path
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.db_ops = None

    def get_graph(self, embeddings):
        self.db_ops = DBops(db_path=self.db_path, embeddings=embeddings)
        graph = MessageGraph()

        graph.add_node("initialize", ToolNode(self.db_ops.setup_database))
        graph.add_node("get_db_connection", ToolNode(self.db_ops.get_database_connection))
        graph.add_node("process_file", ToolNode(self.db_ops.process_local_file))
        graph.add_node("check_data_hash", ToolNode(self.db_ops.check_data_hash))
        graph.add_node("delete_all_data_hashes", ToolNode(self.db_ops.delete_all_data_hashes))
        graph.add_node("update_data_hash", ToolNode(self.db_ops.update_data_hash))

        graph.add_edge("initialize", "get_db_connection")
        graph.add_edge("get_db_connection", "process_file")
        graph.add_edge("process_file", "check_data_hash")
        # graph.add_edge("check_data_hash", "delete_all_data_hashes", messages.ConditionalEdge(self.condition_check))
        graph.add_conditional_edges("check_data_hash", self.condition_check)
        graph.add_edge("delete_all_data_hashes", "update_data_hash")

        graph.set_entry_point("initialize")

        return graph

    def condition_check(self, state):
        if not state["check_data_hash"]:
            return "delete_all_data_hashes"
        return "__end__"
        # return not state["check_data_hash"]

    def execute(self, params):
        # Setup the database
        self.db_ops.setup_database()

        data_csv = params["data_csv"]
        if data_csv is not None:
            self.db_ops.process_local_file(data_csv)
        else:
            raise ValueError("data_csv parameter is missing")

# Example usage
if __name__ == "__main__":
    db_agent = DatabaseAgent(db_path="my_database.db")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    graph = db_agent.get_graph(embeddings)
    params = {"data_csv": pd.DataFrame({"questions": ["What is AI?"], "answers": ["AI is artificial intelligence."]})}
    db_agent.execute(params)
    print(graph)