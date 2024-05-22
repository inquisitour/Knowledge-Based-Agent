import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolNode
from langgraph.core import messages
from data_processing.database_ops import DBops, with_connection
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

class DatabaseAgent:
    def __init__(self, db_path):
        self.db_path = db_path
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")
        self.db_ops = DBops(db_path=db_path)
        self.graph = MessageGraph()

        self.graph.add_node("initialize", ToolNode(self.db_ops.setup_db_node))
        self.graph.add_node("fetch_data", ToolNode(self.fetch_data))
        self.graph.add_node("process_file", ToolNode(self.db_ops.process_file_node))

        self.graph.add_edge("initialize", "fetch_data", messages.ConditionalEdge(self.condition_check))

        self.graph.add_edge("fetch_data", "process_file")

        self.graph.set_entry_point("initialize")

    @with_connection
    def fetch_data(self, params, conn):
        query = params.get("query")
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
        return {"data": result}

    def condition_check(self, params):
        return True

    def execute(self, start_node, params):
        result = self.graph.run(start_node, params)
        return result

# Example usage
if __name__ == "__main__":
    db_agent = DatabaseAgent(db_path="my_database.db")
    params = {"data_csv": pd.DataFrame({"questions": ["What is AI?"], "answers": ["AI is artificial intelligence."]})}
    result = db_agent.execute("initialize", params)
    print(result)