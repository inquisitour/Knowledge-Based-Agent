from langgraph.prebuilt import ToolNode
from langgraph.graph import MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver

class UserQueryAgent:
    def __init__(self, db_path):
        self.memory = SqliteSaver.from_conn_string(db_path)
        self.graph = MessageGraph()
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("get_user_query", ToolNode([self.get_user_query]))
        self.graph.set_entry_point("get_user_query")

    def get_user_query(self, state):
        """Executes the necessary database operations."""
        user_query = state["user_query"]
        if user_query is not None:
            return user_query
        else:
            raise ValueError("Please enter a valid query.")