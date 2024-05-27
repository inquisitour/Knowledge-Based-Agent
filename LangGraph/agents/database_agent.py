import pandas as pd
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessageGraph
from data_processing.database_ops import DBops
from langgraph.checkpoint.sqlite import SqliteSaver

class DatabaseAgent:
    def __init__(self, db_path):
        self.memory = SqliteSaver.from_conn_string(db_path)
        self.graph = MessageGraph()
        self.db_ops = DBops(db_path)
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("execute", ToolNode([self.execute]))
        self.graph.set_entry_point("execute")

    def execute(self, state):
        """Executes the necessary database operations."""
        data_csv = state["data_csv"]
        if data_csv is not None:
            df = pd.DataFrame(data_csv)
            self.db_ops.process_local_file(df)
        else:
            raise ValueError("No data_csv found in the state.")
